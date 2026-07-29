"""
============================================================
 Step 8.6c — SVM Baseline Classifier (AD vs Healthy) — EEG
 SINGLE SUBJECT-LEVEL TRAIN / VAL / TEST SPLIT VERSION
 (no cross-validation, no plotting — training/eval only)

 Subject-level split:
     Train ~70%  |  Validation ~10%  |  Test ~20%
 Loaded from the SHARED subject_split.json (generated once by
 generate_subject_split.py) so DB-HLSTM/CNN/ANN/SVM/RF all use
 the exact same subjects in each split — required for a fair
 cross-model comparison. No subject appears in more than one
 split. ONE model is trained and evaluated ONCE on the held-out
 test set.

 All cross-validation machinery has been removed:
 StratifiedGroupKFold, N_FOLDS, the fold loop, pooled-OOF
 predictions, pooled threshold, fold metrics, best-fold
 tracking, and the fold summary table are all gone.

 All plotting has been removed. This script produces plain
 numeric/text outputs only; ROC/PR/confusion-matrix/calibration
 figures belong in a separate publication plotting script that
 consumes predictions.csv.

 Feature extraction (mean/std/min/max/peak-to-peak per channel
 + I_val + node/adjacency stats), StandardScaler normalization,
 and GPU(cuML)/CPU(sklearn) SVC backend selection with automatic
 P100-kernel-incompatibility fallback and the Nystroem+SGD
 large-training-set speed fix are preserved unchanged.

 Input  : /kaggle/working/EEG_BOLD_Data.csv
 Outputs (all in /kaggle/working) — exactly four files:
   - svm_disease_model.joblib   (this is an sklearn/cuML SVC,
                                  not a Keras model, so it is
                                  saved via joblib, not .keras)
   - training_log.csv
   - predictions.csv
   - summary.txt
============================================================
"""

import os
import gc
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import joblib

# ------------------------------------------------------------
# GPU / CPU backend selection — UNCHANGED from the CV version.
# ------------------------------------------------------------
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
SAVE_MODELS = True
MODEL_DIR  = '/kaggle/working/svm'
os.makedirs(MODEL_DIR, exist_ok=True)

# Subject-level split ratios.
# NOTE: these constants are now informational only — the actual
# split is loaded from the shared subject_split.json below, so
# these values must match what generate_subject_split.py used.
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.10
TEST_FRAC  = 0.20   # TRAIN_FRAC + VAL_FRAC + TEST_FRAC == 1.0

MAX_EXACT_SVC_SAMPLES = 15000

print("=" * 60)
print("  SVM BASELINE — DISEASE CLASSIFICATION (EEG)")
print("  Single subject-level Train/Val/Test split")
print("=" * 60)

# ------------------------------------------------------------
# Load data — UNCHANGED
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Feature extraction — UNCHANGED
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# SUBJECT-LEVEL TRAIN / VAL / TEST SPLIT (replaces StratifiedGroupKFold)
#
# Each subject gets one dominant label (its majority Class across all
# its rows) so we can do a stratified split ON SUBJECTS rather than on
# samples — this keeps every window from a given subject entirely
# inside exactly one of Train/Val/Test (no leakage), while still
# balancing AD vs Healthy roughly evenly across the three splits.
# ------------------------------------------------------------
# ------------------------------------------------------------
# Load the SHARED subject-level split (generated once by
# generate_subject_split.py) instead of SVM's own custom
# stratified_subject_split(). This guarantees DB-HLSTM, CNN,
# ANN, SVM, and RF all train/validate/test on the EXACT SAME
# subjects — required for a fair, leakage-free, apples-to-apples
# comparison across models. Run generate_subject_split.py once
# before this script.
# ------------------------------------------------------------
import json
SPLIT_JSON = '/kaggle/working/subject_split.json'
with open(SPLIT_JSON) as f:
    _split = json.load(f)
train_subjects = np.array(_split['train_subjects'])
val_subjects   = np.array(_split['val_subjects'])
test_subjects  = np.array(_split['test_subjects'])
print(f"  Loaded shared subject split from {SPLIT_JSON} "
      f"(seed={_split.get('seed')}, fractions train/val/test="
      f"{_split.get('train_frac')}/{_split.get('val_frac')}/{_split.get('test_frac')})")

print(f"  Train subjects: {len(train_subjects)}  "
      f"Val subjects: {len(val_subjects)}  Test subjects: {len(test_subjects)}")

train_mask = np.isin(X_subject.astype(str), train_subjects)
val_mask   = np.isin(X_subject.astype(str), val_subjects)
test_mask  = np.isin(X_subject.astype(str), test_subjects)

# sanity check — no subject leakage across splits
assert set(X_subject[train_mask]).isdisjoint(set(X_subject[val_mask]))
assert set(X_subject[train_mask]).isdisjoint(set(X_subject[test_mask]))
assert set(X_subject[val_mask]).isdisjoint(set(X_subject[test_mask]))

X_train, y_train = X_sc[train_mask], y_dis[train_mask]
X_val,   y_val    = X_sc[val_mask],   y_dis[val_mask]
X_test,  y_test    = X_sc[test_mask],  y_dis[test_mask]

n_total = len(y_dis)
print(f"\n  Sample split — Train: {len(y_train):,} ({len(y_train)/n_total:.1%})  "
      f"Val: {len(y_val):,} ({len(y_val)/n_total:.1%})  "
      f"Test: {len(y_test):,} ({len(y_test)/n_total:.1%})")

# ------------------------------------------------------------
# Train ONE model on the training split only.
# (SVC itself has no epochs/early-stopping, so val is not used
# inside .fit; only default 0.5 predictions are evaluated below.)
# ------------------------------------------------------------
n0 = (y_train == 0).sum(); n1 = (y_train == 1).sum()
class_w = {0: len(y_train) / (2 * n0), 1: len(y_train) / (2 * n1)}
sample_w = np.array([class_w[y] for y in y_train], dtype=np.float32)

def fit_fast_approx():
    """Nystroem+SGD kernel approximation — fast (~O(n)), CPU-only."""
    from sklearn.kernel_approximation import Nystroem
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import make_pipeline
    print(f"  Train set ({len(X_train):,} samples) exceeds "
          f"{MAX_EXACT_SVC_SAMPLES:,} — CPU RBF-SVC would be impractically "
          f"slow (O(n^2)-O(n^3)). Using Nystroem+SGD kernel approximation "
          f"instead (fast, ~O(n)).")
    model = make_pipeline(
        Nystroem(kernel='rbf', gamma=None, n_components=300, random_state=SEED),
        SGDClassifier(loss='log_loss', class_weight='balanced',
                      max_iter=1000, random_state=SEED)
    )
    model.fit(X_train, y_train)
    return model

def fit_exact_svc(SVC_cls):
    """Exact RBF SVC (CPU sklearn or GPU cuML, whichever SVC_cls is)."""
    try:
        model = SVC_cls(kernel='rbf', C=1.0, gamma='scale',
                         class_weight='balanced', probability=True,
                         random_state=SEED)
        model.fit(X_train, y_train)
    except TypeError:
        model = SVC_cls(kernel='rbf', C=1.0, gamma='scale',
                         probability=True, random_state=SEED)
        model.fit(X_train, y_train, sample_weight=sample_w)
    return model

use_fast_approx = (not GPU_BACKEND) and (len(X_train) > MAX_EXACT_SVC_SAMPLES)

print(f"\n{'-'*60}\n  TRAINING (single model)\n{'-'*60}")

if use_fast_approx:
    svm = fit_fast_approx()
elif GPU_BACKEND:
    try:
        svm = fit_exact_svc(SVC_GPU)
    except RuntimeError as e:
        print(f"\n  GPU SVC FAILED at runtime (cuML/P100 kernel "
              f"incompatibility): {e}")
        print("  Disabling GPU SVC — falling back to CPU (sklearn).")
        GPU_BACKEND = False
        if len(X_train) > MAX_EXACT_SVC_SAMPLES:
            svm = fit_fast_approx()
        else:
            svm = fit_exact_svc(SVC_CPU)
else:
    svm = fit_exact_svc(SVC_CPU)

# ------------------------------------------------------------
# Single evaluation on the held-out TEST split.
# ------------------------------------------------------------
test_probs = svm.predict_proba(X_test)[:, 1].astype(np.float64)
pred_default = (test_probs >= 0.5).astype(int)

def _metrics(pred, y_true):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    acc = (pred == y_true).mean()
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
    return acc, prec, rec, f1

acc_d, prec_d, rec_d, f1_d = _metrics(pred_default, y_test)
test_auc = roc_auc_score(y_test, test_probs) if len(np.unique(y_test)) > 1 else float('nan')

print("\n" + "=" * 60)
print("  TEST SET RESULTS (single held-out evaluation)")
print("=" * 60)
print(f"  AUC: {test_auc:.4f}")
print(f"  Accuracy: {acc_d:.4f}  Precision: {prec_d:.4f}  Recall: {rec_d:.4f}  F1: {f1_d:.4f}")

report_default = classification_report(y_test, pred_default, target_names=['Healthy', 'AD'], digits=4)
print("\n  Classification report:")
print(report_default)

# ------------------------------------------------------------
# Save outputs — exactly four files, no plots.
# ------------------------------------------------------------

# 1) model
if SAVE_MODELS:
    try:
        joblib.dump(svm, f'{MODEL_DIR}/svm_disease_model.joblib')
        print(f"\n  Saved: {MODEL_DIR}/svm_disease_model.joblib")
    except Exception as e:
        print(f"  (Could not save model: {e})")

# 2) training_log.csv — SVM has no epochs, so log the single training run.
training_log = pd.DataFrame([{
    'seed': SEED,
    'n_train_samples': len(y_train),
    'n_val_samples': len(y_val),
    'n_test_samples': len(y_test),
    'n_train_subjects': len(train_subjects),
    'n_val_subjects': len(val_subjects),
    'n_test_subjects': len(test_subjects),
    'backend': 'GPU(cuML)' if GPU_BACKEND else ('Nystroem+SGD' if use_fast_approx else 'CPU(sklearn)'),
    'test_auc': test_auc,
    'test_acc': acc_d,
    'test_prec': prec_d,
    'test_rec': rec_d,
    'test_f1': f1_d,
}])
training_log.to_csv(f'{MODEL_DIR}/training_log.csv', index=False)
print(f"  Saved: {MODEL_DIR}/training_log.csv")

# 3) predictions.csv — raw test-set predictions for a separate plotting
#    script to consume (ROC/PR/confusion-matrix/calibration figures
#    are generated there, not here).
pred_df = pd.DataFrame({
    'subject': X_subject[test_mask],
    'y_true': y_test,
    'y_prob': test_probs,
    'y_pred': pred_default,
})
pred_df.to_csv(f'{MODEL_DIR}/predictions.csv', index=False)
print(f"  Saved: {MODEL_DIR}/predictions.csv")

# 4) summary.txt
with open(f'{MODEL_DIR}/summary.txt', 'w') as f:
    f.write("SVM BASELINE — SINGLE SUBJECT-LEVEL TRAIN/VAL/TEST SPLIT\n")
    f.write("=" * 60 + "\n")
    f.write(f"Seed: {SEED}\n")
    f.write(f"Backend: {'GPU(cuML)' if GPU_BACKEND else ('Nystroem+SGD' if use_fast_approx else 'CPU(sklearn)')}\n\n")
    f.write(f"Subjects — Train: {len(train_subjects)}  Val: {len(val_subjects)}  Test: {len(test_subjects)}\n")
    f.write(f"Samples  — Train: {len(y_train):,}  Val: {len(y_val):,}  Test: {len(y_test):,}\n\n")
    f.write("TEST SET RESULTS\n")
    f.write(f"  AUC: {test_auc:.4f}\n")
    f.write(f"  Accuracy: {acc_d:.4f}  Precision: {prec_d:.4f}  Recall: {rec_d:.4f}  F1: {f1_d:.4f}\n\n")
    f.write("Classification report\n")
    f.write(report_default)
print(f"  Saved: {MODEL_DIR}/summary.txt")

print("\n" + "=" * 60)
print("  SVM BASELINE (single split) DONE")
print("=" * 60)
