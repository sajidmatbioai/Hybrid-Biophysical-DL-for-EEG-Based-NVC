"""
============================================================
 Step 8.6a — CNN Baseline Classifier (AD vs Healthy) — EEG
 Companion baseline to DB-HLSTM, for direct comparison.

 Uses the SAME 6 temporal channels as DB-HLSTM's Branch A
 (bold, hrf_c, hrf_td, hrf_dd, v, ratio_env/theta_env), the
 SAME subject-level 5-fold CV, and the SAME pooled-OOF
 threshold-tuning approach — so any accuracy difference vs
 DB-HLSTM is attributable to the architecture, not the data
 or evaluation protocol.

 Architecture: pure 1D-CNN (no LSTM, no attention, no GNN) —
   Conv1D(64) -> Conv1D(64) -> Conv1D(32) -> GlobalAvgPool -> Dense -> sigmoid

 Input  : /kaggle/working/EEG_BOLD_Data.csv
          (accepts ratio_env_0..599 [v7+] or theta_env_0..599 [v5/v6])
 Output : /kaggle/working/cnn_disease_fold{N}.keras
============================================================
"""

import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, BatchNormalization,
                                     Dropout, GlobalAveragePooling1D, Dense)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import (precision_recall_curve, confusion_matrix,
                             classification_report, roc_auc_score)
from tqdm.notebook import tqdm

CSV_PATH   = '/kaggle/working/EEG_BOLD_Data.csv'
SEQ_LEN    = 600
WINDOW     = 100
STRIDE     = 100
BATCH_SIZE = 512
LEARN_RATE = 0.0005
EPOCHS     = 150
PATIENCE_ES = 10
PATIENCE_LR = 3
SEED       = 42
N_FOLDS    = 5

# NEW — figure-pipeline output folder
import os
RESULTS_DIR = '/kaggle/working/results/cnn'
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("  CNN BASELINE — DISEASE CLASSIFICATION (EEG)")
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
sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_probs = np.zeros(len(y_dis), dtype=np.float64)
fold_ids  = np.zeros(len(y_dis), dtype=np.int32)   # NEW
fold_metrics = []
best_fold_auc, best_fold_i = -1.0, -1

for fold_i, (tr_full_idx, test_idx) in enumerate(sgkf.split(X_norm, y_dis, groups=X_subject)):
    fold_no = fold_i + 1
    print(f"\n{'-'*60}\n  FOLD {fold_no}/{N_FOLDS}\n{'-'*60}")

    tr_full_subj = X_subject[tr_full_idx]
    subj_y = (pd.DataFrame({'Subject': tr_full_subj, 'y': y_dis[tr_full_idx]})
              .groupby('Subject')['y'].first().reset_index())
    inner_train_subj, inner_val_subj = train_test_split(
        subj_y, test_size=0.15, stratify=subj_y['y'], random_state=SEED + fold_i)

    train_mask = np.isin(tr_full_subj, inner_train_subj['Subject'].values)
    val_mask   = np.isin(tr_full_subj, inner_val_subj['Subject'].values)
    train_idx = tr_full_idx[train_mask]
    val_idx   = tr_full_idx[val_mask]

    y_tr, y_val_f, y_test_f = y_dis[train_idx], y_dis[val_idx], y_dis[test_idx]
    print(f"  Windows — Train: {len(train_idx):,}  Val: {len(val_idx):,}  Test: {len(test_idx):,}")

    tr_ds = (tf.data.Dataset.from_tensor_slices((X_norm[train_idx], y_tr))
             .shuffle(len(train_idx), seed=SEED).batch(BATCH_SIZE).prefetch(AUTOTUNE))
    vl_ds = (tf.data.Dataset.from_tensor_slices((X_norm[val_idx], y_val_f))
             .batch(BATCH_SIZE).prefetch(AUTOTUNE))

    n0 = (y_tr == 0).sum(); n1 = (y_tr == 1).sum()
    cw = {0: len(y_tr)/(2*n0), 1: len(y_tr)/(2*n1)}

    tf.keras.backend.clear_session()
    model = build_cnn()
    fold_path = f'/kaggle/working/cnn_disease_fold{fold_no}.keras'
    cb = [
        ModelCheckpoint(fold_path, monitor='val_auc', save_best_only=True, mode='max', verbose=0),
        EarlyStopping(monitor='val_auc', patience=PATIENCE_ES, restore_best_weights=True, mode='max', verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=PATIENCE_LR, min_lr=1e-6, verbose=0),
    ]
    model.fit(tr_ds, validation_data=vl_ds, epochs=EPOCHS, callbacks=cb, class_weight=cw, verbose=2)

    test_probs = model.predict(X_norm[test_idx], verbose=0).ravel().astype(np.float64)
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

    del model, tr_ds, vl_ds; gc.collect()

fm = pd.DataFrame(fold_metrics)
print("\n" + "=" * 60)
print(f"  CNN — {N_FOLDS}-FOLD CV SUMMARY (all subjects)")
print("=" * 60)
print(fm.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
for col in ['auc', 'acc_default', 'f1_default']:
    print(f"    {col:14s}: {fm[col].mean():.4f} +/- {fm[col].std():.4f}")

# NEW — save fold-level metrics + pooled OOF predictions (with fold id)
fm.to_csv(f'{RESULTS_DIR}/fold_metrics.csv', index=False)
oof_df_cnn = pd.DataFrame({
    'Subject': X_subject, 'y_true': y_dis,
    'oof_prob': oof_probs, 'fold': fold_ids,
})
oof_df_cnn.to_csv(f'{RESULTS_DIR}/oof_predictions.csv', index=False)
print(f"  Saved: {RESULTS_DIR}/fold_metrics.csv, {RESULTS_DIR}/oof_predictions.csv")

prec_arr, rec_arr, thr_arr = precision_recall_curve(y_dis, oof_probs)
f1_arr = np.divide(2*prec_arr[:-1]*rec_arr[:-1], prec_arr[:-1]+rec_arr[:-1],
                   out=np.zeros_like(prec_arr[:-1]), where=(prec_arr[:-1]+rec_arr[:-1]) > 0)
POOLED_THRESHOLD = float(thr_arr[int(np.argmax(f1_arr))]) if len(thr_arr) else 0.5
print(f"\n  Pooled-OOF-tuned threshold: {POOLED_THRESHOLD:.4f}")

oof_pred_default = (oof_probs >= 0.5).astype(int)
oof_pred_pooled  = (oof_probs >= POOLED_THRESHOLD).astype(int)
print("\n  Pooled OOF report @ threshold=0.5 (CNN):")
print(classification_report(y_dis, oof_pred_default, target_names=['Healthy', 'AD'], digits=4))
print("\n  Pooled OOF report @ POOLED_THRESHOLD (CNN):")
print(classification_report(y_dis, oof_pred_pooled, target_names=['Healthy', 'AD'], digits=4))
print(f"  Pooled OOF AUC (CNN): {roc_auc_score(y_dis, oof_probs):.4f}")
print("=" * 60)
print(f"\n  Best fold: {best_fold_i} (AUC={best_fold_auc:.4f})")

del X_norm, y_dis
gc.collect()
tf.keras.backend.clear_session()
print("\n  CNN BASELINE DONE")
