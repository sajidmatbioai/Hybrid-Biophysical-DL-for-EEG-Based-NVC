"""
============================================================
 Step 8.6b — ANN (Feedforward) Baseline Classifier
 (AD vs Healthy) — EEG
 SINGLE SUBJECT-LEVEL TRAIN / VAL / TEST SPLIT VERSION
 (no cross-validation, no plotting — training/eval only)

 Subject-level split:
     Train ~70%  |  Validation ~10%  |  Test ~20%
 No subject appears in more than one split. ONE model is
 trained and evaluated ONCE on the held-out test set.

 All cross-validation machinery has been removed:
 StratifiedGroupKFold, N_FOLDS, the fold loop, per-fold inner
 train/val split, pooled-OOF predictions, fold metrics,
 best-fold tracking, and the fold summary table are all gone.
 No plotting code (this baseline never had any). Validation-set
 threshold tuning has also been removed — only default (0.5)
 model predictions are reported.

 Uses HANDCRAFTED summary features per window (not raw
 sequences) — mean/std/min/max/peak-to-peak for each of the 6
 temporal channels (bold, hrf_c/td/dd, v, ratio_env/theta_env),
 plus I_val and node/adjacency summary stats (if present).

 Architecture (unchanged): Dense(128) -> Dense(64) -> Dense(32) -> sigmoid

 Input  : /kaggle/working/EEG_BOLD_Data.csv
 Output : /kaggle/working/ann_disease_model.keras
          /kaggle/working/training_log.csv
          /kaggle/working/classification_report.txt
          /kaggle/working/predictions.csv
          /kaggle/working/summary.txt
============================================================
"""

import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau, CSVLogger)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

import os
MODEL_DIR  = '/kaggle/working/ann'
os.makedirs(MODEL_DIR, exist_ok=True)

CSV_PATH   = '/kaggle/working/EEG_BOLD_Data.csv'
MODEL_PATH = f'{MODEL_DIR}/ann_disease_model.keras'

SEQ_LEN    = 600
WINDOW     = 100
STRIDE     = 100
BATCH_SIZE = 512
LEARN_RATE = 0.0005
EPOCHS     = 150
PATIENCE_ES = 10
PATIENCE_LR = 3
SEED       = 42

# --- Single subject-level split fractions (Train ~70 / Val ~10 / Test ~20) ---
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.10
TEST_FRAC  = 0.20
assert abs((TRAIN_FRAC + VAL_FRAC + TEST_FRAC) - 1.0) < 1e-9

print("=" * 60)
print("  ANN BASELINE — DISEASE CLASSIFICATION (EEG)  [SINGLE SPLIT]")
print("=" * 60)

# ============================================================
# GPU SETUP — P100  (same as DB-HLSTM scripts)
# ============================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("  GPU: Tesla P100  float16: ON")
else:
    print("  CPU only")
tf.config.optimizer.set_jit(True)

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

# Optional node_/adj_ columns (v6+ GNN-arrange pipeline).
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
n_stats_per_channel = 5   # mean, std, min, max, peak-to-peak
n_extra = 1 + (2 if has_gnn_cols else 0)   # I_val, [node_mean, adj_mean]
n_features = len(channel_names) * n_stats_per_channel + n_extra
print(f"  Feature vector size: {n_features} "
      f"({len(channel_names)} channels x {n_stats_per_channel} stats + {n_extra} extra)")

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


def build_ann(n_features):
    REG = l2(3e-4)
    inp = Input(shape=(n_features,), name='feat_input')
    x = Dense(128, activation='relu', kernel_regularizer=REG)(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.35)(x)
    x = Dense(64, activation='relu', kernel_regularizer=REG)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.35)(x)
    x = Dense(32, activation='relu', kernel_regularizer=REG)(x)
    x = Dropout(0.25)(x)
    out = Dense(1, activation='sigmoid', dtype='float32')(x)

    m = Model(inputs=inp, outputs=out)
    m.compile(optimizer=Adam(LEARN_RATE), loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall')])
    return m


AUTOTUNE = tf.data.AUTOTUNE

# ============================================================
# SINGLE SUBJECT-LEVEL TRAIN / VAL / TEST SPLIT
# (replaces the 5-fold StratifiedGroupKFold loop; no subject
# ever appears in more than one of Train/Val/Test)
# ============================================================
print("\n" + "=" * 60)
print("  DISEASE CLASSIFICATION — AD vs Healthy — SINGLE SPLIT")
print(f"  Target fractions — Train:{TRAIN_FRAC:.0%}  "
      f"Val:{VAL_FRAC:.0%}  Test:{TEST_FRAC:.0%}  (subject-level)")
print("=" * 60)

# ------------------------------------------------------------
# Load the SHARED subject-level split (generated once by
# generate_subject_split.py) instead of computing our own.
# This guarantees DB-HLSTM, CNN, ANN, SVM, and RF all train/
# validate/test on the EXACT SAME subjects — required for a
# fair, leakage-free, apples-to-apples comparison across models.
# Run generate_subject_split.py once before this script.
# ------------------------------------------------------------
import json
SPLIT_JSON = '/kaggle/working/subject_split.json'
with open(SPLIT_JSON) as f:
    _split = json.load(f)
train_subj = list(_split['train_subjects'])
val_subj   = list(_split['val_subjects'])
test_subj  = list(_split['test_subjects'])
print(f"  Loaded shared subject split from {SPLIT_JSON} "
      f"(seed={_split.get('seed')}, fractions train/val/test="
      f"{_split.get('train_frac')}/{_split.get('val_frac')}/{_split.get('test_frac')})")

train_mask = np.isin(X_subject.astype(str), train_subj)
val_mask   = np.isin(X_subject.astype(str), val_subj)
test_mask  = np.isin(X_subject.astype(str), test_subj)

train_idx = np.where(train_mask)[0]
val_idx   = np.where(val_mask)[0]
test_idx  = np.where(test_mask)[0]

# Sanity check — no subject leakage across splits.
assert set(train_subj) & set(val_subj) == set()
assert set(train_subj) & set(test_subj) == set()
assert set(val_subj) & set(test_subj) == set()

n_train_subj, n_val_subj, n_test_subj = len(train_subj), len(val_subj), len(test_subj)
n_total_subj = n_train_subj + n_val_subj + n_test_subj

print(f"  Subjects — Train: {n_train_subj} ({n_train_subj/n_total_subj:.1%})  "
      f"Val: {n_val_subj} ({n_val_subj/n_total_subj:.1%})  "
      f"Test: {n_test_subj} ({n_test_subj/n_total_subj:.1%})")
print(f"  Windows  — Train: {len(train_idx):,}  Val: {len(val_idx):,}  "
      f"Test: {len(test_idx):,}")

y_tr, y_val, y_test = y_dis[train_idx], y_dis[val_idx], y_dis[test_idx]

tr_ds = (tf.data.Dataset.from_tensor_slices((X_sc[train_idx], y_tr))
         .shuffle(len(train_idx), seed=SEED).batch(BATCH_SIZE).prefetch(AUTOTUNE))
vl_ds = (tf.data.Dataset.from_tensor_slices((X_sc[val_idx], y_val))
         .batch(BATCH_SIZE).prefetch(AUTOTUNE))

n0 = (y_tr == 0).sum(); n1 = (y_tr == 1).sum()
print(f"  Train class counts — Healthy(0): {n0:,}  AD(1): {n1:,}")
cw = {0: len(y_tr)/(2*n0), 1: len(y_tr)/(2*n1)}

tf.keras.backend.clear_session()
model = build_ann(X_sc.shape[1])

callbacks = [
    ModelCheckpoint(MODEL_PATH, monitor='val_auc',
                    save_best_only=True, mode='max', verbose=0),
    EarlyStopping(monitor='val_auc', patience=PATIENCE_ES,
                  restore_best_weights=True, mode='max', verbose=0),
    ReduceLROnPlateau(monitor='val_loss', factor=0.4,
                      patience=PATIENCE_LR, min_lr=1e-6, verbose=0),
    CSVLogger(f'{MODEL_DIR}/training_log.csv'),
]

print("\n  Training single ANN model ...")
history = model.fit(tr_ds, validation_data=vl_ds, epochs=EPOCHS,
                     callbacks=callbacks, class_weight=cw, verbose=2)

# ============================================================
# SINGLE, FINAL EVALUATION ON THE TEST SET
# ============================================================
test_probs = model.predict(X_sc[test_idx], verbose=0).ravel().astype(np.float64)

pred_default = (test_probs >= 0.5).astype(int)

def _metrics(pred, y_true):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    acc = (pred == y_true).mean()
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return acc, prec, rec, f1

acc_d, prec_d, rec_d, f1_d = _metrics(pred_default, y_test)
test_auc = roc_auc_score(y_test, test_probs) if len(np.unique(y_test)) > 1 else float('nan')

print("\n" + "=" * 60)
print("  FINAL TEST-SET RESULTS (single held-out split, evaluated once)")
print("=" * 60)
print(f"  Test AUC: {test_auc:.4f}")
print(f"  Accuracy: {acc_d:.4f}  Precision: {prec_d:.4f}  Recall: {rec_d:.4f}  F1: {f1_d:.4f}")

report_default = classification_report(y_test, pred_default,
                                        target_names=['Healthy', 'AD'], digits=4)

print("\n  Test-set classification report:")
print(report_default)

# ============================================================
# SAVE CLASSIFICATION REPORT, PREDICTIONS, SUMMARY
# (no plots — figures are generated in a separate publication
# plotting script that consumes predictions.csv)
# ============================================================
with open(f'{MODEL_DIR}/classification_report.txt', 'w') as f:
    f.write("ANN Baseline — Single Subject-Level Split — Test-Set Classification Report\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Test AUC: {test_auc:.6f}\n\n")
    f.write(report_default)

pred_df = pd.DataFrame({
    'Subject': X_subject[test_idx],
    'y_true': y_test,
    'test_prob': test_probs,
    'y_pred': pred_default,
})
pred_df.to_csv(f'{MODEL_DIR}/predictions.csv', index=False)

with open(f'{MODEL_DIR}/summary.txt', 'w') as f:
    f.write(f"n_train_subjects={n_train_subj}\n")
    f.write(f"n_val_subjects={n_val_subj}\n")
    f.write(f"n_test_subjects={n_test_subj}\n")
    f.write(f"n_train_windows={len(train_idx)}\n")
    f.write(f"n_val_windows={len(val_idx)}\n")
    f.write(f"n_test_windows={len(test_idx)}\n")
    f.write(f"test_auc={test_auc:.6f}\n")
    f.write(f"test_acc={acc_d:.6f}\n")
    f.write(f"test_prec={prec_d:.6f}\n")
    f.write(f"test_rec={rec_d:.6f}\n")
    f.write(f"test_f1={f1_d:.6f}\n")
    f.write(f"model_path={MODEL_PATH}\n")

print(f"\n  Saved: training_log.csv, classification_report.txt, "
      f"predictions.csv, summary.txt")
print(f"  Saved model: {MODEL_PATH}")

del X_sc, y_dis
gc.collect()
tf.keras.backend.clear_session()

print("\n  ANN BASELINE (SINGLE SUBJECT-LEVEL SPLIT) DONE")
