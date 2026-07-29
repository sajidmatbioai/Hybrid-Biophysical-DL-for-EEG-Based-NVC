"""
============================================================
 Step 8.6a — CNN Baseline Classifier (AD vs Healthy) — EEG
 SINGLE SUBJECT-LEVEL TRAIN / VAL / TEST SPLIT VERSION
 Companion baseline to DB-HLSTM, for direct comparison.

 CONVERSION NOTES (what changed vs. the 5-fold CV script):
   - Removed StratifiedGroupKFold and the entire fold loop.
   - Removed pooled out-of-fold (OOF) predictions, the pooled
     threshold, fold metrics, best-fold tracking, and the
     fold-summary table.
   - Added ONE subject-level split: Train ~70%, Val ~10%,
     Test ~20% (stratified by Class at the subject level, no
     subject ever appears in more than one split).
   - Trains exactly ONE CNN model.
   - Threshold tuning has been removed entirely — only default
     (0.5) predictions are evaluated on the test split, once.
   - No plotting code (this baseline never had any).

 Uses the SAME 6 temporal channels as DB-HLSTM's Branch A
 (bold, hrf_c, hrf_td, hrf_dd, v, ratio_env/theta_env), so any
 accuracy difference vs DB-HLSTM is attributable to the
 architecture, not the data or evaluation protocol.

 Architecture (unchanged): pure 1D-CNN (no LSTM, no attention,
 no GNN) —
   Conv1D(64) -> Conv1D(64) -> Conv1D(32) -> GlobalAvgPool -> Dense -> sigmoid

 Input  : /kaggle/working/EEG_BOLD_Data.csv
          (accepts ratio_env_0..599 [v7+] or theta_env_0..599 [v5/v6])
 Output : /kaggle/working/cnn/cnn_disease_model.keras
          /kaggle/working/cnn/training_log.csv
          /kaggle/working/cnn/classification_report.txt
          /kaggle/working/cnn/predictions.csv
          /kaggle/working/cnn/summary.txt
============================================================
"""

import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, BatchNormalization,
                                     Dropout, GlobalAveragePooling1D, Dense)
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau, CSVLogger)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from tqdm.notebook import tqdm

import os
MODEL_DIR  = '/kaggle/working/cnn'
os.makedirs(MODEL_DIR, exist_ok=True)

CSV_PATH   = '/kaggle/working/EEG_BOLD_Data.csv'
MODEL_PATH = f'{MODEL_DIR}/cnn_disease_model.keras'

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
print("  CNN BASELINE — DISEASE CLASSIFICATION (EEG)  [SINGLE SPLIT]")
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
subject_full    = df['Subject'].values

bold_cols   = [f'bold_{i}'   for i in range(SEQ_LEN)]
hrf_c_cols  = [f'hrf_c_{i}'  for i in range(SEQ_LEN)]
hrf_td_cols = [f'hrf_td_{i}' for i in range(SEQ_LEN)]
hrf_dd_cols = [f'hrf_dd_{i}' for i in range(SEQ_LEN)]
v_cols      = [f'v_{i}'      for i in range(SEQ_LEN)]

# ratio_env_ (v7+) / theta_env_ (v5/v6) fallback — same pattern as the
# fixed HRF/BW/disease scripts
_ratio_env_cols = [f'ratio_env_{i}' for i in range(SEQ_LEN)]
_theta_env_cols = [f'theta_env_{i}' for i in range(SEQ_LEN)]
if all(c in df.columns for c in _ratio_env_cols):
    temporal6_cols = _ratio_env_cols
    print("  Using ratio_env_0..599 (v7 naming).")
elif all(c in df.columns for c in _theta_env_cols):
    temporal6_cols = _theta_env_cols
    print("  Using theta_env_0..599 (v5/v6 naming).")
else:
    raise KeyError("Neither ratio_env_0..599 nor theta_env_0..599 found in CSV.")

n_windows = (SEQ_LEN - WINDOW) // STRIDE + 1
n_samples = len(df) * n_windows
print(f"  Windows/sweep: {n_windows}  Total samples: {n_samples:,}")

X_bold   = np.zeros((n_samples, WINDOW), dtype=np.float32)
X_hrf_c  = np.zeros((n_samples, WINDOW), dtype=np.float32)
X_hrf_td = np.zeros((n_samples, WINDOW), dtype=np.float32)
X_hrf_dd = np.zeros((n_samples, WINDOW), dtype=np.float32)
X_volt   = np.zeros((n_samples, WINDOW), dtype=np.float32)
X_temp6  = np.zeros((n_samples, WINDOW), dtype=np.float32)
y_dis     = np.zeros(n_samples, dtype=np.int32)
X_subject = np.empty(n_samples, dtype=object)

idx = 0
for row_i in tqdm(range(len(df)), desc="Windowing"):
    row = df.iloc[row_i]
    bold   = row[bold_cols].values.astype(np.float32)
    hrf_c  = row[hrf_c_cols].values.astype(np.float32)
    hrf_td = row[hrf_td_cols].values.astype(np.float32)
    hrf_dd = row[hrf_dd_cols].values.astype(np.float32)
    volt   = row[v_cols].values.astype(np.float32)
    temp6  = row[temporal6_cols].values.astype(np.float32)
    lb     = int(y_disease_full[row_i])
    subj   = subject_full[row_i]

    for start in range(0, SEQ_LEN - WINDOW + 1, STRIDE):
        end = start + WINDOW
        X_bold[idx]   = bold[start:end]
        X_hrf_c[idx]  = hrf_c[start:end]
        X_hrf_td[idx] = hrf_td[start:end]
        X_hrf_dd[idx] = hrf_dd[start:end]
        X_volt[idx]   = volt[start:end]
        X_temp6[idx]  = temp6[start:end]
        y_dis[idx]    = lb
        X_subject[idx] = subj
        idx += 1

del df; gc.collect()

X_ts = np.stack([X_bold, X_hrf_c, X_hrf_td, X_hrf_dd, X_volt, X_temp6], axis=2)
del X_bold, X_hrf_c, X_hrf_td, X_hrf_dd, X_volt, X_temp6; gc.collect()

X_norm = np.zeros_like(X_ts, dtype=np.float32)
for fi in range(X_ts.shape[2]):
    feat = X_ts[:, :, fi]
    X_norm[:, :, fi] = (feat - feat.mean(axis=1, keepdims=True)) / \
                       (feat.std(axis=1, keepdims=True) + 1e-8)
del X_ts; gc.collect()
print(f"  X_norm shape: {X_norm.shape}")


def build_cnn():
    REG = l2(3e-4)
    inp = Input(shape=(WINDOW, 6), name='ts_input')
    x = Conv1D(64, 3, padding='causal', activation='relu', kernel_regularizer=REG)(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv1D(64, 3, padding='causal', dilation_rate=2, activation='relu', kernel_regularizer=REG)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv1D(32, 3, padding='causal', dilation_rate=4, activation='relu', kernel_regularizer=REG)(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu', kernel_regularizer=REG)(x)
    x = Dropout(0.3)(x)
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

tr_ds = (tf.data.Dataset.from_tensor_slices((X_norm[train_idx], y_tr))
         .shuffle(len(train_idx), seed=SEED).batch(BATCH_SIZE).prefetch(AUTOTUNE))
vl_ds = (tf.data.Dataset.from_tensor_slices((X_norm[val_idx], y_val))
         .batch(BATCH_SIZE).prefetch(AUTOTUNE))

n0 = (y_tr == 0).sum(); n1 = (y_tr == 1).sum()
print(f"  Train class counts — Healthy(0): {n0:,}  AD(1): {n1:,}")
cw = {0: len(y_tr)/(2*n0), 1: len(y_tr)/(2*n1)}

tf.keras.backend.clear_session()
model = build_cnn()

callbacks = [
    ModelCheckpoint(MODEL_PATH, monitor='val_auc',
                    save_best_only=True, mode='max', verbose=0),
    EarlyStopping(monitor='val_auc', patience=PATIENCE_ES,
                  restore_best_weights=True, mode='max', verbose=0),
    ReduceLROnPlateau(monitor='val_loss', factor=0.4,
                      patience=PATIENCE_LR, min_lr=1e-6, verbose=0),
    CSVLogger(f'{MODEL_DIR}/training_log.csv'),
]

print("\n  Training single CNN model ...")
history = model.fit(tr_ds, validation_data=vl_ds, epochs=EPOCHS,
                     callbacks=callbacks, class_weight=cw, verbose=2)

# ============================================================
# SINGLE, FINAL EVALUATION ON THE TEST SET
# ============================================================
test_probs = model.predict(X_norm[test_idx], verbose=0).ravel().astype(np.float64)

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
    f.write("CNN Baseline — Single Subject-Level Split — Test-Set Classification Report\n")
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

del X_norm, y_dis
gc.collect()
tf.keras.backend.clear_session()

print("\n  CNN BASELINE (SINGLE SUBJECT-LEVEL SPLIT) DONE")
