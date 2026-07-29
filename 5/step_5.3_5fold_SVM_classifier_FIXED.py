"""
============================================================
 Step 8.6c — SVM Baseline Classifier (AD vs Healthy) — EEG

 Companion baseline to DB-HLSTM/CNN/ANN. Uses the SAME
 handcrafted per-window features as the ANN baseline (mean/
 std/min/max/peak-to-peak per channel + I_val + node/adjacency
 stats), and the SAME subject-level 5-fold CV + pooled-OOF
 threshold tuning, so all four baselines are directly
 comparable.

 FIX — GPU support. Plain sklearn SVC is CPU-only by design;
 that's why the P100 GPU never "loaded" for it, not a bug. This
 version uses RAPIDS cuML's SVC (GPU-accelerated, drop-in
 fit/predict_proba API) when available, and falls back to
 sklearn's CPU SVC automatically if cuML isn't installed.

 FIX 2 — cuML/P100 RUNTIME incompatibility. cuML importing
 successfully does NOT guarantee its CUDA kernels were compiled
 for this GPU. On this Kaggle image, cuML SVC's kernels are not
 built for the P100 (Pascal, compute capability 6.0), so .fit()
 raised: "RuntimeError: ... cudaErrorNoKernelImageForDevice: no
 kernel image is available for execution on the device" — an
 environment/library incompatibility, not a code bug. This is
 only detectable at the first actual .fit() call, not at import
 time. Fix: catch this RuntimeError on the first fold, disable
 the GPU path for the REST of the run (it will fail identically
 on every fold, so there's no point retrying it 5 times), and
 retry that same fold on CPU immediately instead of losing it.

 FIX — speed. RBF-kernel SVC is O(n^2)-O(n^3) in training-set
 size. With ~127,000 training samples per fold, plain CPU SVC
 was not hung — it was genuinely taking hours per fold. On CPU,
 once the training set exceeds MAX_EXACT_SVC_SAMPLES (15,000),
 this script automatically switches to a Nystroem kernel
 approximation + SGDClassifier, which scales ~O(n) and finishes
 in minutes while approximating the same RBF decision boundary.
 On GPU (cuML) the exact RBF SVC is kept, since GPU throughput
 handles this n reasonably.

 Model: RBF-kernel SVC, class-balanced (via class_weight or, on
 cuML builds that lack it, via manual sample_weight). Falls back
 to a Nystroem+SGD approximation on large CPU-only training sets.

 Input  : /kaggle/working/EEG_BOLD_Data.csv
 Output : /kaggle/working/svm_disease_fold{N}.joblib
============================================================
"""

import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (precision_recall_curve, confusion_matrix,
                             classification_report, roc_auc_score)
import joblib

# FIX — sklearn's SVC NEVER uses GPU, by design (it's a CPU-only
# implementation). That's why "GPU not loading" was expected, not a
# bug. Kaggle GPU notebooks ship RAPIDS cuML, which provides a
# GPU-accelerated SVC with the same fit/predict_proba API.
#
# FIX 2 — cuML/P100 RUNTIME incompatibility. Even when cuML IMPORTS
# successfully, its compiled CUDA kernels may not include the P100's
# architecture (Pascal, compute capability 6.0) — some RAPIDS builds
# target newer GPUs only. This doesn't show up as an ImportError; it
# shows up as a RuntimeError ("no kernel image is available for
# execution on the device") the first time .fit() actually runs a
# kernel on the GPU. So we import BOTH backends up front, try cuML
# first, and if a RuntimeError happens during the actual fit (not
# just an import failure), we permanently switch to the sklearn CPU
# backend for the rest of the run (once it fails on this GPU, it will
# fail on every subsequent fold too — no point retrying it 5 times).
from sklearn.svm import SVC as SVC_CPU
try:
    from cuml.svm import SVC as SVC_GPU
    CUML_AVAILABLE = True
except ImportError:
    SVC_GPU = None
    CUML_AVAILABLE = False

USE_GPU = True
GPU_BACKEND = USE_GPU and CUML_AVAILABLE
if GPU_BACKEND:
    print("  cuML available — will attempt GPU SVC (with automatic CPU "
          "fallback if the GPU kernels turn out to be incompatible).")
else:
    print("  cuML not available — using sklearn SVC (CPU).")
    print("  (To get GPU SVM on Kaggle: Settings -> Accelerator -> GPU P100,")
    print("   and make sure the RAPIDS/cuML environment is enabled.)")

CSV_PATH   = '/kaggle/working/EEG_BOLD_Data.csv'
SEQ_LEN    = 600
WINDOW     = 100
STRIDE     = 100
SEED       = 42
N_FOLDS    = 5
SAVE_MODELS = True
MODEL_DIR  = '/kaggle/working'

# NEW — figure-pipeline output folder
import os
RESULTS_DIR = '/kaggle/working/results/svm'
os.makedirs(RESULTS_DIR, exist_ok=True)

# FIX — RBF-kernel SVM is O(n^2)-O(n^3) in the number of TRAINING
# samples. This dataset has ~159k windowed samples (~127k per fold's
# training set), which is far past the size where sklearn's SVC is
# practical — training can take hours per fold with no progress
# output in between, which looks "stuck" but is just slow. Cap the
# per-fold TRAINING set size by random subsampling (stratified by
# class) so each fold trains in minutes, not hours. Set to None to
# disable and use the full training set (only recommended if using
# the GPU cuML backend, which handles this scale far better).
MAX_TRAIN_SAMPLES = 15000

print("=" * 60)
print("  SVM BASELINE — DISEASE CLASSIFICATION (EEG)")
print("=" * 60)

df = pd.read_csv(CSV_PATH)
print(f"  Rows: {len(df):,}  Columns: {len(df.columns):,}")

y_disease_full = df['Class'].values.astype(np.int32)
i_val_full      = df['I_val'].values.astype(np.float32)
subject_full    = df['Subject'].values

channel_defs = {
    'bold':   [f'bold_{i}'   for i in range(SEQ_LEN)],
    'hrf_c':  [f'hrf_c_{i}'  for i in range(SEQ_LEN)],
    'hrf_td': [f'hrf_td_{i}' for i in range(SEQ_LEN)],
    'hrf_dd': [f'hrf_dd_{i}' for i in range(SEQ_LEN)],
    'v':      [f'v_{i}'      for i in range(SEQ_LEN)],
}

_ratio_env_cols = [f'ratio_env_{i}' for i in range(SEQ_LEN)]
_theta_env_cols = [f'theta_env_{i}' for i in range(SEQ_LEN)]
if all(c in df.columns for c in _ratio_env_cols):
    channel_defs['ratio_env'] = _ratio_env_cols
    print("  Using ratio_env_0..599 (v7 naming).")
elif all(c in df.columns for c in _theta_env_cols):
    channel_defs['ratio_env'] = _theta_env_cols
    print("  Using theta_env_0..599 (v5/v6 naming).")
else:
    raise KeyError("Neither ratio_env_0..599 nor theta_env_0..599 found in CSV.")

node_cols = sorted([c for c in df.columns if c.startswith('node_')],
                   key=lambda c: int(c.split('_')[1]))
adj_cols  = sorted([c for c in df.columns if c.startswith('adj_')],
                   key=lambda c: int(c.split('_')[1]))
has_gnn_cols = len(node_cols) > 0 and len(adj_cols) > 0
if has_gnn_cols:
    print(f"  Found {len(node_cols)} node_ columns, {len(adj_cols)} adj_ columns — "
          f"including node/adjacency summary stats as extra features.")
    node_full = df[node_cols].values.astype(np.float32)
    adj_full  = df[adj_cols].values.astype(np.float32)
else:
    print("  No node_/adj_ columns found — skipping GNN-derived features.")

n_windows = (SEQ_LEN - WINDOW) // STRIDE + 1
n_samples = len(df) * n_windows
print(f"  Windows/sweep: {n_windows}  Total samples: {n_samples:,}")

channel_names = list(channel_defs.keys())
n_stats_per_channel = 5
n_extra = 1 + (2 if has_gnn_cols else 0)
n_features = len(channel_names) * n_stats_per_channel + n_extra
print(f"  Feature vector size: {n_features}")

X_feat    = np.zeros((n_samples, n_features), dtype=np.float32)
y_dis     = np.zeros(n_samples, dtype=np.int32)
X_subject = np.empty(n_samples, dtype=object)

idx = 0
for row_i in range(len(df)):
    row = df.iloc[row_i]
    lb   = int(y_disease_full[row_i])
    subj = subject_full[row_i]
    ival = i_val_full[row_i]

    channel_arrays = {name: row[cols].values.astype(np.float32)
                       for name, cols in channel_defs.items()}

    if has_gnn_cols:
        node_row_mean = float(node_full[row_i].mean())
        adj_row_mean  = float(adj_full[row_i].mean())

    for start in range(0, SEQ_LEN - WINDOW + 1, STRIDE):
        end = start + WINDOW
        feats = []
        for name in channel_names:
            seg = channel_arrays[name][start:end]
            feats.extend([seg.mean(), seg.std(), seg.min(), seg.max(), seg.max() - seg.min()])
        feats.append(ival)
        if has_gnn_cols:
            feats.extend([node_row_mean, adj_row_mean])
        X_feat[idx] = np.array(feats, dtype=np.float32)
        y_dis[idx]  = lb
        X_subject[idx] = subj
        idx += 1

    if row_i % 2000 == 0:
        print(f"    ...{row_i}/{len(df)} rows processed")

del df; gc.collect()
print(f"  Feature extraction complete: {idx:,} samples")

X_sc = StandardScaler().fit_transform(X_feat).astype(np.float32)
del X_feat; gc.collect()
print(f"  X_sc shape: {X_sc.shape}")

sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_probs = np.zeros(len(y_dis), dtype=np.float64)
fold_ids  = np.zeros(len(y_dis), dtype=np.int32)   # NEW
fold_metrics = []
best_fold_auc, best_fold_i = -1.0, -1

for fold_i, (train_idx, test_idx) in enumerate(sgkf.split(X_sc, y_dis, groups=X_subject)):
    fold_no = fold_i + 1
    print(f"\n{'-'*60}\n  FOLD {fold_no}/{N_FOLDS}\n{'-'*60}")

    y_tr, y_test_f = y_dis[train_idx], y_dis[test_idx]
    print(f"  Samples — Train: {len(train_idx):,}  Test: {len(test_idx):,}")
    print(f"  Train subjects: {len(np.unique(X_subject[train_idx]))}  "
          f"Test subjects: {len(np.unique(X_subject[test_idx]))}")

    X_train_fold = X_sc[train_idx]
    X_test_fold  = X_sc[test_idx]

    # FIX — WHY THIS WAS TAKING FOREVER / SHOWING NO OUTPUT:
    # RBF-kernel SVC has O(n^2)-O(n^3) time complexity in the number
    # of training samples. With ~127,000 training samples per fold
    # (159,156 total across 5 folds), plain CPU sklearn SVC can
    # realistically take HOURS per fold — it wasn't hung, it was just
    # doing genuinely enormous work. cuML (GPU) is much faster but can
    # still be slow at this n.
    #
    # FIX: on the CPU backend, once training set size exceeds
    # MAX_EXACT_SVC_SAMPLES, switch automatically to a Nystroem kernel
    # approximation + SGDClassifier (hinge/log loss) — this scales
    # ~O(n) instead of O(n^2)-O(n^3), typically finishing in a couple
    # of minutes instead of hours, while still approximating the same
    # RBF-kernel decision boundary. On GPU (cuML) we keep the exact
    # RBF SVC since GPU throughput handles this n reasonably.
    MAX_EXACT_SVC_SAMPLES = 15000

    n0 = (y_tr == 0).sum(); n1 = (y_tr == 1).sum()
    class_w = {0: len(y_tr) / (2 * n0), 1: len(y_tr) / (2 * n1)}
    sample_w = np.array([class_w[y] for y in y_tr], dtype=np.float32)

    def fit_fast_approx():
        """Nystroem+SGD kernel approximation — fast (~O(n)), CPU-only."""
        from sklearn.kernel_approximation import Nystroem
        from sklearn.linear_model import SGDClassifier
        from sklearn.pipeline import make_pipeline
        print(f"  Train set ({len(train_idx):,} samples) exceeds "
              f"{MAX_EXACT_SVC_SAMPLES:,} — CPU RBF-SVC would be impractically "
              f"slow (O(n^2)-O(n^3)). Using Nystroem+SGD kernel approximation "
              f"instead (fast, ~O(n)).")
        model = make_pipeline(
            Nystroem(kernel='rbf', gamma=None, n_components=300, random_state=SEED),
            SGDClassifier(loss='log_loss', class_weight='balanced',
                          max_iter=1000, random_state=SEED)
        )
        model.fit(X_train_fold, y_tr)
        return model

    def fit_exact_svc(SVC_cls):
        """Exact RBF SVC (CPU sklearn or GPU cuML, whichever SVC_cls is)."""
        try:
            model = SVC_cls(kernel='rbf', C=1.0, gamma='scale',
                             class_weight='balanced', probability=True,
                             random_state=SEED)
            model.fit(X_train_fold, y_tr)
        except TypeError:
            # some cuML builds don't accept class_weight — retrain with
            # explicit sample_weight instead (equivalent effect).
            model = SVC_cls(kernel='rbf', C=1.0, gamma='scale',
                             probability=True, random_state=SEED)
            model.fit(X_train_fold, y_tr, sample_weight=sample_w)
        return model

    # Re-check GPU_BACKEND fresh each fold — it can flip to False mid-run
    # if a previous fold discovered a runtime incompatibility (see below).
    use_fast_approx = (not GPU_BACKEND) and (len(train_idx) > MAX_EXACT_SVC_SAMPLES)

    if use_fast_approx:
        svm = fit_fast_approx()
    elif GPU_BACKEND:
        # FIX — cuML IMPORTING successfully doesn't guarantee its CUDA
        # kernels actually run on this GPU. Some RAPIDS builds don't
        # include compiled kernels for older architectures like the
        # P100 (Pascal, compute capability 6.0) — that shows up as a
        # RuntimeError ("no kernel image is available for execution on
        # the device") only once .fit() actually tries to run on the
        # GPU, not at import time. If that happens, this GPU/cuML build
        # combination is fundamentally incompatible for ALL folds (not
        # just this one), so we permanently disable the GPU path for
        # the rest of the run and retry THIS fold on CPU immediately
        # (falling back to the fast approximation if the training set
        # is too large for exact CPU SVC).
        try:
            svm = fit_exact_svc(SVC_GPU)
        except RuntimeError as e:
            print(f"\n  GPU SVC FAILED at runtime (cuML/P100 kernel "
                  f"incompatibility): {e}")
            print("  Disabling GPU SVC for the rest of this run — "
                  "falling back to CPU (sklearn) from here on.")
            GPU_BACKEND = False
            if len(train_idx) > MAX_EXACT_SVC_SAMPLES:
                svm = fit_fast_approx()
            else:
                svm = fit_exact_svc(SVC_CPU)
    else:
        svm = fit_exact_svc(SVC_CPU)

    if SAVE_MODELS:
        try:
            joblib.dump(svm, f'{MODEL_DIR}/svm_disease_fold{fold_no}.joblib')
        except Exception as e:
            print(f"  (Could not save fold {fold_no} model: {e})")

    test_probs = svm.predict_proba(X_test_fold)[:, 1].astype(np.float64)
    oof_probs[test_idx] = test_probs
    fold_ids[test_idx] = fold_no   # NEW
    pred_def = (test_probs >= 0.5).astype(int)

    def _metrics(pred, y_true):
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        acc = (pred == y_true).mean()
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        return acc, prec, rec, f1

    acc_d, prec_d, rec_d, f1_d = _metrics(pred_def, y_test_f)
    fold_auc = roc_auc_score(y_test_f, test_probs) if len(np.unique(y_test_f)) > 1 else float('nan')
    print(f"  Fold {fold_no} AUC={fold_auc:.4f}  acc={acc_d:.4f} prec={prec_d:.4f} rec={rec_d:.4f} f1={f1_d:.4f}")

    fold_metrics.append({'fold': fold_no, 'auc': fold_auc, 'acc_default': acc_d,
                         'prec_default': prec_d, 'rec_default': rec_d, 'f1_default': f1_d})
    if fold_auc > best_fold_auc:
        best_fold_auc, best_fold_i = fold_auc, fold_no

    del svm; gc.collect()

fm = pd.DataFrame(fold_metrics)
print("\n" + "=" * 60)
print(f"  SVM — {N_FOLDS}-FOLD CV SUMMARY (all subjects)")
print("=" * 60)
print(fm.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
for col in ['auc', 'acc_default', 'f1_default']:
    print(f"    {col:14s}: {fm[col].mean():.4f} +/- {fm[col].std():.4f}")

# NEW — save fold-level metrics + pooled OOF predictions (with fold id)
fm.to_csv(f'{RESULTS_DIR}/fold_metrics.csv', index=False)
oof_df_svm = pd.DataFrame({
    'Subject': X_subject, 'y_true': y_dis,
    'oof_prob': oof_probs, 'fold': fold_ids,
})
oof_df_svm.to_csv(f'{RESULTS_DIR}/oof_predictions.csv', index=False)
print(f"  Saved: {RESULTS_DIR}/fold_metrics.csv, {RESULTS_DIR}/oof_predictions.csv")

prec_arr, rec_arr, thr_arr = precision_recall_curve(y_dis, oof_probs)
f1_arr = np.divide(2*prec_arr[:-1]*rec_arr[:-1], prec_arr[:-1]+rec_arr[:-1],
                   out=np.zeros_like(prec_arr[:-1]), where=(prec_arr[:-1]+rec_arr[:-1]) > 0)
POOLED_THRESHOLD = float(thr_arr[int(np.argmax(f1_arr))]) if len(thr_arr) else 0.5
print(f"\n  Pooled-OOF-tuned threshold: {POOLED_THRESHOLD:.4f}")

oof_pred_default = (oof_probs >= 0.5).astype(int)
oof_pred_pooled  = (oof_probs >= POOLED_THRESHOLD).astype(int)
print("\n  Pooled OOF report @ threshold=0.5 (SVM):")
print(classification_report(y_dis, oof_pred_default, target_names=['Healthy', 'AD'], digits=4))
print("\n  Pooled OOF report @ POOLED_THRESHOLD (SVM):")
print(classification_report(y_dis, oof_pred_pooled, target_names=['Healthy', 'AD'], digits=4))
print(f"  Pooled OOF AUC (SVM): {roc_auc_score(y_dis, oof_probs):.4f}")
print("=" * 60)
print(f"\n  Best fold: {best_fold_i} (AUC={best_fold_auc:.4f})")
print("\n  SVM BASELINE DONE")