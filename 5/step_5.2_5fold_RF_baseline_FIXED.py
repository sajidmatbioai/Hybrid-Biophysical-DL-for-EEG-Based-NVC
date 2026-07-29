"""
============================================================
 Step 8.6d — Random Forest Baseline Classifier
 (AD vs Healthy) — EEG

 Companion baseline to DB-HLSTM/CNN/ANN/SVM. Uses the SAME
 handcrafted per-window features as the ANN/SVM baselines
 (mean/std/min/max/peak-to-peak per channel + I_val + node/
 adjacency stats), and the SAME subject-level 5-fold CV +
 pooled-OOF threshold tuning, so all baselines are directly
 comparable. Also reports feature importances where the
 backend supports it — a good sanity check that I_val/
 ratio_env stay high-importance as expected.

 FIX — GPU support. Plain sklearn RandomForestClassifier is
 CPU-only by design; that's why the P100 GPU never "loaded"
 for it, not a bug. This version uses RAPIDS cuML's
 RandomForestClassifier (GPU-accelerated, drop-in fit/
 predict_proba API) when available, and falls back to
 sklearn's CPU version automatically if cuML isn't installed
 OR if its CUDA kernels are incompatible with this GPU
 (see FIX 2 below — this is what was actually crashing).

 Input  : /kaggle/working/EEG_BOLD_Data.csv
 Output : /kaggle/working/rf_disease_fold{N}.joblib
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

# ── BACKEND SETUP ────────────────────────────────────────────
# FIX — plain sklearn RandomForestClassifier is CPU-only by design;
# that's why the P100 GPU never "loaded" for it, not a bug. RAPIDS
# cuML (preinstalled on Kaggle GPU notebooks) provides a
# GPU-accelerated RandomForestClassifier with a similar
# fit/predict_proba API.
#
# FIX 2 — cuML/P100 RUNTIME incompatibility (this was the actual
# crash). cuML IMPORTING successfully does not guarantee its CUDA
# kernels were compiled for this GPU — on this Kaggle image, some
# cuML ops raise "RuntimeError: ... cudaErrorNoKernelImageForDevice:
# no kernel image is available for execution on the device" the
# FIRST TIME they actually run on the P100 (Pascal, compute
# capability 6.0), not at import time. The previous version only
# caught TypeError (cuML's different __init__ signature) — it never
# caught this RuntimeError, so the crash was unhandled. This version
# catches BOTH, and once a GPU failure happens, permanently switches
# to CPU for all remaining folds (instead of re-attempting GPU and
# re-crashing on every fold).
from sklearn.ensemble import RandomForestClassifier as RF_CPU
try:
    from cuml.ensemble import RandomForestClassifier as RF_GPU
    CUML_AVAILABLE = True
except ImportError:
    RF_GPU = None
    CUML_AVAILABLE = False

USE_GPU = True
GPU_BACKEND = USE_GPU and CUML_AVAILABLE  # mutable — flips to False permanently on first GPU failure

if GPU_BACKEND:
    print("  cuML available — will attempt GPU RandomForestClassifier "
          "(with automatic, permanent CPU fallback if the GPU kernels "
          "turn out to be incompatible with this device).")
else:
    print("  cuML not available — using sklearn RandomForestClassifier (CPU).")
    print("  (To get GPU RF on Kaggle: Settings -> Accelerator -> GPU P100,")
    print("   and make sure the RAPIDS/cuML environment is enabled.)")


def make_rf(use_gpu):
    """FIX — single source of truth for building the RF model, so the
    fold loop never references a bare 'RandomForestClassifier' name
    that may or may not be bound depending on which branch ran above."""
    if use_gpu:
        return RF_GPU(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                      random_state=SEED)
    return RF_CPU(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                  class_weight='balanced', n_jobs=-1, random_state=SEED)


CSV_PATH   = '/kaggle/working/EEG_BOLD_Data.csv'
SEQ_LEN    = 600
WINDOW     = 100
STRIDE     = 100
SEED       = 42
N_FOLDS    = 5
SAVE_MODELS = True
MODEL_DIR  = '/kaggle/working'
N_ESTIMATORS = 300
MAX_DEPTH    = 12

# NEW — figure-pipeline output folder (per-model, so all 5 baselines'
# results live in one predictable tree for the plotting script)
import os
RESULTS_DIR = '/kaggle/working/results/rf'
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("  RANDOM FOREST BASELINE — DISEASE CLASSIFICATION (EEG)")
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
feature_names = []
for name in channel_names:
    feature_names.extend([f'{name}_mean', f'{name}_std', f'{name}_min', f'{name}_max', f'{name}_ptp'])
feature_names.append('I_val')
if has_gnn_cols:
    feature_names.extend(['node_mean', 'adj_mean'])
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

# Random Forest doesn't require feature scaling (tree splits are
# scale-invariant), but we scale anyway for consistency with the
# other three baselines' preprocessing / to reuse the same pipeline.
X_sc = StandardScaler().fit_transform(X_feat).astype(np.float32)
del X_feat; gc.collect()
print(f"  X_sc shape: {X_sc.shape}")

sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_probs = np.zeros(len(y_dis), dtype=np.float64)
fold_ids  = np.zeros(len(y_dis), dtype=np.int32)   # NEW — which fold each sample was tested in
fold_metrics = []
best_fold_auc, best_fold_i = -1.0, -1
importances_per_fold = []

for fold_i, (train_idx, test_idx) in enumerate(sgkf.split(X_sc, y_dis, groups=X_subject)):
    fold_no = fold_i + 1
    print(f"\n{'-'*60}\n  FOLD {fold_no}/{N_FOLDS}\n{'-'*60}")

    y_tr, y_test_f = y_dis[train_idx], y_dis[test_idx]
    print(f"  Samples — Train: {len(train_idx):,}  Test: {len(test_idx):,}")
    print(f"  Train subjects: {len(np.unique(X_subject[train_idx]))}  "
          f"Test subjects: {len(np.unique(X_subject[test_idx]))}")

    X_train_fold = X_sc[train_idx]
    X_test_fold  = X_sc[test_idx]

    # FIX — catch BOTH the API-signature TypeError (older cuML builds
    # that don't accept class_weight/n_jobs — make_rf already avoids
    # passing those to GPU, so this branch mainly guards unexpected
    # signature drift) AND the CUDA RuntimeError (kernel incompatible
    # with this GPU — the actual crash seen before). On either, fall
    # back to CPU permanently (GPU_BACKEND flips False) so remaining
    # folds don't re-attempt and re-crash on the GPU.
    try:
        rf = make_rf(GPU_BACKEND)
        rf.fit(X_train_fold, y_tr)
        backend_used = 'GPU (cuML)' if GPU_BACKEND else 'CPU (sklearn)'
    except (TypeError, RuntimeError) as e:
        if GPU_BACKEND:
            print(f"  GPU RandomForest failed ({type(e).__name__}: {e})")
            print("  Falling back to CPU sklearn for this and all remaining folds.")
            GPU_BACKEND = False
            rf = make_rf(GPU_BACKEND)
            rf.fit(X_train_fold, y_tr)
            backend_used = 'CPU (sklearn, after GPU failure)'
        else:
            raise
    print(f"  Backend used: {backend_used}")

    # FIX — feature_importances_ may not exist on some cuML builds;
    # guard so the run doesn't crash on GPU where it's unavailable.
    try:
        importances_per_fold.append(rf.feature_importances_)
    except AttributeError:
        print("  (feature_importances_ not available on this backend — skipping)")

    if SAVE_MODELS:
        try:
            joblib.dump(rf, f'{MODEL_DIR}/rf_disease_fold{fold_no}.joblib')
        except Exception as e:
            print(f"  (Could not save fold {fold_no} model: {e})")

    test_probs = rf.predict_proba(X_test_fold)[:, 1].astype(np.float64)
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

    del rf; gc.collect()

fm = pd.DataFrame(fold_metrics)
print("\n" + "=" * 60)
print(f"  RANDOM FOREST — {N_FOLDS}-FOLD CV SUMMARY (all subjects)")
print("=" * 60)
print(fm.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
for col in ['auc', 'acc_default', 'f1_default']:
    print(f"    {col:14s}: {fm[col].mean():.4f} +/- {fm[col].std():.4f}")

# NEW — save fold-level metrics + pooled OOF predictions (with fold id)
# so the figure-generation script can build Fig 1-6, V1-V2, A1-A3
# without re-running training.
fm.to_csv(f'{RESULTS_DIR}/fold_metrics.csv', index=False)
oof_df_rf = pd.DataFrame({
    'Subject': X_subject, 'y_true': y_dis,
    'oof_prob': oof_probs, 'fold': fold_ids,
})
oof_df_rf.to_csv(f'{RESULTS_DIR}/oof_predictions.csv', index=False)
print(f"  Saved: {RESULTS_DIR}/fold_metrics.csv, {RESULTS_DIR}/oof_predictions.csv")

prec_arr, rec_arr, thr_arr = precision_recall_curve(y_dis, oof_probs)
f1_arr = np.divide(2*prec_arr[:-1]*rec_arr[:-1], prec_arr[:-1]+rec_arr[:-1],
                   out=np.zeros_like(prec_arr[:-1]), where=(prec_arr[:-1]+rec_arr[:-1]) > 0)
# FIX — same degenerate-threshold guard as the DB-HLSTM script:
# clamp the search to [0.05, 0.95] so a spurious extreme (0 or 1)
# from a tiny fold's noise can't be selected as "optimal".
safe_mask = (thr_arr >= 0.05) & (thr_arr <= 0.95)
if safe_mask.any():
    POOLED_THRESHOLD = float(thr_arr[safe_mask][int(np.argmax(f1_arr[safe_mask]))])
else:
    POOLED_THRESHOLD = 0.5
print(f"\n  Pooled-OOF-tuned threshold: {POOLED_THRESHOLD:.4f}")

oof_pred_default = (oof_probs >= 0.5).astype(int)
oof_pred_pooled  = (oof_probs >= POOLED_THRESHOLD).astype(int)
print("\n  Pooled OOF report @ threshold=0.5 (Random Forest):")
print(classification_report(y_dis, oof_pred_default, target_names=['Healthy', 'AD'], digits=4))
print("\n  Pooled OOF report @ POOLED_THRESHOLD (Random Forest):")
print(classification_report(y_dis, oof_pred_pooled, target_names=['Healthy', 'AD'], digits=4))
print(f"  Pooled OOF AUC (Random Forest): {roc_auc_score(y_dis, oof_probs):.4f}")
print("=" * 60)
print(f"\n  Best fold: {best_fold_i} (AUC={best_fold_auc:.4f})")

# Feature importance (averaged across folds) — sanity check that
# I_val/ratio_env-derived features stay high-importance as expected
if importances_per_fold:
    mean_importance = np.mean(importances_per_fold, axis=0)
    imp_df = pd.DataFrame({'feature': feature_names, 'importance': mean_importance})
    imp_df = imp_df.sort_values('importance', ascending=False).reset_index(drop=True)
    print("\n  Feature importances (averaged across folds, top 15):")
    print(imp_df.head(15).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
else:
    print("\n  Feature importances not available on this backend (cuML build without "
          "feature_importances_) — skipping.")

print("\n  RANDOM FOREST BASELINE DONE")
