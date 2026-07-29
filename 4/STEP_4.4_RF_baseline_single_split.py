"""
============================================================
 Step 8.6d — Random Forest Baseline Classifier
 (AD vs Healthy) — EEG
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
 predictions, fold metrics, best-fold tracking, and the fold
 summary table are all gone. Validation-set threshold tuning
 has also been removed — only default (0.5) model predictions
 are reported.

 All plotting has been removed (this baseline never had any
 plots). Feature importances (a useful sanity check that
 I_val/ratio_env-derived features stay high-importance) are
 kept, written into summary.txt rather than a separate file.

 Uses the SAME handcrafted per-window features as the ANN/SVM
 baselines (mean/std/min/max/peak-to-peak per channel + I_val
 + node/adjacency stats). GPU(cuML)/CPU(sklearn) backend
 selection with automatic, permanent fallback on GPU-kernel
 incompatibility is preserved unchanged.

 Input  : /kaggle/working/EEG_BOLD_Data.csv
 Outputs (all in /kaggle/working) — exactly four files:
   - rf_disease_model.joblib
   - training_log.csv
   - predictions.csv
   - summary.txt   (includes classification report + feature
                     importances)
============================================================
"""

import os
import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import joblib

# ── BACKEND SETUP — UNCHANGED ───────────────────────────────
# FIX — plain sklearn RandomForestClassifier is CPU-only by design;
# that's why the P100 GPU never "loaded" for it, not a bug. RAPIDS
# cuML (preinstalled on Kaggle GPU notebooks) provides a
# GPU-accelerated RandomForestClassifier with a similar
# fit/predict_proba API.
#
# FIX 2 — cuML/P100 RUNTIME incompatibility. cuML IMPORTING
# successfully does not guarantee its CUDA kernels were compiled
# for this GPU — some cuML ops raise a RuntimeError the FIRST TIME
# they actually run on the P100 (Pascal, compute capability 6.0),
# not at import time. This is caught below and permanently falls
# back to CPU for the rest of the run.
from sklearn.ensemble import RandomForestClassifier as RF_CPU
try:
    from cuml.ensemble import RandomForestClassifier as RF_GPU
    CUML_AVAILABLE = True
except ImportError:
    RF_GPU = None
    CUML_AVAILABLE = False

USE_GPU = True
GPU_BACKEND = USE_GPU and CUML_AVAILABLE  # mutable — flips to False permanently on GPU failure

if GPU_BACKEND:
    print("  cuML available — will attempt GPU RandomForestClassifier "
          "(with automatic, permanent CPU fallback if the GPU kernels "
          "turn out to be incompatible with this device).")
else:
    print("  cuML not available — using sklearn RandomForestClassifier (CPU).")
    print("  (To get GPU RF on Kaggle: Settings -> Accelerator -> GPU P100,")
    print("   and make sure the RAPIDS/cuML environment is enabled.)")


def make_rf(use_gpu):
    """Single source of truth for building the RF model."""
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
SAVE_MODELS = True
MODEL_DIR  = '/kaggle/working/rf'
os.makedirs(MODEL_DIR, exist_ok=True)
N_ESTIMATORS = 300
MAX_DEPTH    = 12

# Subject-level split fractions (Train ~70 / Val ~10 / Test ~20)
# NOTE: these constants are now informational only — the actual
# split is loaded from the shared subject_split.json below, so
# these values must match what generate_subject_split.py used.
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.10
TEST_FRAC  = 0.20   # TRAIN_FRAC + VAL_FRAC + TEST_FRAC == 1.0

print("=" * 60)
print("  RANDOM FOREST BASELINE — DISEASE CLASSIFICATION (EEG)")
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
feature_names = []
for name in channel_names:
    feature_names.extend([f'{name}_mean', f'{name}_std', f'{name}_min', f'{name}_max', f'{name}_ptp'])
feature_names.append('I_val')
if has_gnn_cols:
    feature_names.extend(['node_mean', 'adj_mean'])
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

# Random Forest doesn't require feature scaling, but we scale anyway
# for consistency with the other baselines' preprocessing pipeline.
X_sc = StandardScaler().fit_transform(X_feat).astype(np.float32)
del X_feat; gc.collect()
print(f"  X_sc shape: {X_sc.shape}")

# ------------------------------------------------------------
# SUBJECT-LEVEL TRAIN / VAL / TEST SPLIT (replaces StratifiedGroupKFold)
# ------------------------------------------------------------
# ------------------------------------------------------------
# Load the SHARED subject-level split (generated once by
# generate_subject_split.py) instead of RF's own custom
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
# ------------------------------------------------------------
print(f"\n{'-'*60}\n  TRAINING (single model)\n{'-'*60}")

try:
    rf = make_rf(GPU_BACKEND)
    rf.fit(X_train, y_train)
    backend_used = 'GPU (cuML)' if GPU_BACKEND else 'CPU (sklearn)'
except (TypeError, RuntimeError) as e:
    if GPU_BACKEND:
        print(f"  GPU RandomForest failed ({type(e).__name__}: {e})")
        print("  Falling back to CPU sklearn.")
        GPU_BACKEND = False
        rf = make_rf(GPU_BACKEND)
        rf.fit(X_train, y_train)
        backend_used = 'CPU (sklearn, after GPU failure)'
    else:
        raise
print(f"  Backend used: {backend_used}")

# Feature importances — guard for cuML builds that may not expose this.
try:
    importances = rf.feature_importances_
    imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    imp_df = imp_df.sort_values('importance', ascending=False).reset_index(drop=True)
    print("\n  Feature importances (top 15):")
    print(imp_df.head(15).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
except AttributeError:
    imp_df = None
    print("  (feature_importances_ not available on this backend — skipping)")

if SAVE_MODELS:
    try:
        joblib.dump(rf, f'{MODEL_DIR}/rf_disease_model.joblib')
        print(f"  Model saved to {MODEL_DIR}/rf_disease_model.joblib")
    except Exception as e:
        print(f"  (Could not save model: {e})")

# ------------------------------------------------------------
# Single evaluation on the held-out TEST split.
# ------------------------------------------------------------
test_probs = rf.predict_proba(X_test)[:, 1].astype(np.float64)
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

# 1) training_log.csv — RF has no epochs, so log the single training run.
training_log = pd.DataFrame([{
    'seed': SEED,
    'n_train_samples': len(y_train),
    'n_val_samples': len(y_val),
    'n_test_samples': len(y_test),
    'n_train_subjects': len(train_subjects),
    'n_val_subjects': len(val_subjects),
    'n_test_subjects': len(test_subjects),
    'backend': backend_used,
    'n_estimators': N_ESTIMATORS,
    'max_depth': MAX_DEPTH,
    'test_auc': test_auc,
    'test_acc': acc_d,
    'test_prec': prec_d,
    'test_rec': rec_d,
    'test_f1': f1_d,
}])
training_log.to_csv(f'{MODEL_DIR}/training_log.csv', index=False)
print(f"\n  Saved: {MODEL_DIR}/training_log.csv")

# 2) predictions.csv
pred_df = pd.DataFrame({
    'subject': X_subject[test_mask],
    'y_true': y_test,
    'y_prob': test_probs,
    'y_pred': pred_default,
})
pred_df.to_csv(f'{MODEL_DIR}/predictions.csv', index=False)
print(f"  Saved: {MODEL_DIR}/predictions.csv")

# 3) summary.txt — includes classification report + feature importances
with open(f'{MODEL_DIR}/summary.txt', 'w') as f:
    f.write("RANDOM FOREST BASELINE — SINGLE SUBJECT-LEVEL TRAIN/VAL/TEST SPLIT\n")
    f.write("=" * 60 + "\n")
    f.write(f"Seed: {SEED}\n")
    f.write(f"Backend: {backend_used}\n")
    f.write(f"n_estimators: {N_ESTIMATORS}  max_depth: {MAX_DEPTH}\n\n")
    f.write(f"Subjects — Train: {len(train_subjects)}  Val: {len(val_subjects)}  Test: {len(test_subjects)}\n")
    f.write(f"Samples  — Train: {len(y_train):,}  Val: {len(y_val):,}  Test: {len(y_test):,}\n\n")
    f.write("TEST SET RESULTS\n")
    f.write(f"  AUC: {test_auc:.4f}\n")
    f.write(f"  Accuracy: {acc_d:.4f}  Precision: {prec_d:.4f}  Recall: {rec_d:.4f}  F1: {f1_d:.4f}\n\n")
    f.write("Classification report\n")
    f.write(report_default)
    if imp_df is not None:
        f.write("\nFeature importances (top 15)\n")
        f.write(imp_df.head(15).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        f.write("\n")
    else:
        f.write("\nFeature importances not available on this backend.\n")
print(f"  Saved: {MODEL_DIR}/summary.txt")

print("\n" + "=" * 60)
print("  RANDOM FOREST BASELINE (single split) DONE")
print("=" * 60)
