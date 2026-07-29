"""
============================================================
 Step 8.6b — ANN (Feedforward) Baseline Classifier
 (AD vs Healthy) — EEG

 Companion baseline to DB-HLSTM/CNN. Unlike CNN, this uses
 HANDCRAFTED summary features per window (not raw sequences) —
 mean/std/min/max/peak-to-peak for each of the 6 temporal
 channels (bold, hrf_c/td/dd, v, ratio_env/theta_env), plus
 I_val and node/adjacency summary stats (if present, from the
 GNN-arrange pipeline v6+). Same subject-level 5-fold CV and
 pooled-OOF threshold tuning as CNN/DB-HLSTM for a fair
 apples-to-apples comparison.

 Architecture: Dense(128) -> Dense(64) -> Dense(32) -> sigmoid

 Input  : /kaggle/working/EEG_BOLD_Data.csv
 Output : /kaggle/working/ann_disease_fold{N}.keras
============================================================
"""

import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import (precision_recall_curve, confusion_matrix,
                             classification_report, roc_auc_score)

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
RESULTS_DIR = '/kaggle/working/results/ann'
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("  ANN BASELINE — DISEASE CLASSIFICATION (EEG)")
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

# Optional node_/adj_ columns (v6+ GNN-arrange pipeline). Flexible to
# either v6/v7 (19 single-value nodes) or v8 (19 channels x 5 features
# = 95 columns) — detect count from the CSV rather than hardcoding.
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
sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_probs = np.zeros(len(y_dis), dtype=np.float64)
fold_ids  = np.zeros(len(y_dis), dtype=np.int32)   # NEW
fold_metrics = []
best_fold_auc, best_fold_i = -1.0, -1

for fold_i, (tr_full_idx, test_idx) in enumerate(sgkf.split(X_sc, y_dis, groups=X_subject)):
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
    print(f"  Samples — Train: {len(train_idx):,}  Val: {len(val_idx):,}  Test: {len(test_idx):,}")

    tr_ds = (tf.data.Dataset.from_tensor_slices((X_sc[train_idx], y_tr))
             .shuffle(len(train_idx), seed=SEED).batch(BATCH_SIZE).prefetch(AUTOTUNE))
    vl_ds = (tf.data.Dataset.from_tensor_slices((X_sc[val_idx], y_val_f))
             .batch(BATCH_SIZE).prefetch(AUTOTUNE))

    n0 = (y_tr == 0).sum(); n1 = (y_tr == 1).sum()
    cw = {0: len(y_tr)/(2*n0), 1: len(y_tr)/(2*n1)}

    tf.keras.backend.clear_session()
    model = build_ann(X_sc.shape[1])
    fold_path = f'/kaggle/working/ann_disease_fold{fold_no}.keras'
    cb = [
        ModelCheckpoint(fold_path, monitor='val_auc', save_best_only=True, mode='max', verbose=0),
        EarlyStopping(monitor='val_auc', patience=PATIENCE_ES, restore_best_weights=True, mode='max', verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=PATIENCE_LR, min_lr=1e-6, verbose=0),
    ]
    model.fit(tr_ds, validation_data=vl_ds, epochs=EPOCHS, callbacks=cb, class_weight=cw, verbose=2)

    test_probs = model.predict(X_sc[test_idx], verbose=0).ravel().astype(np.float64)
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
print(f"  ANN — {N_FOLDS}-FOLD CV SUMMARY (all subjects)")
print("=" * 60)
print(fm.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
for col in ['auc', 'acc_default', 'f1_default']:
    print(f"    {col:14s}: {fm[col].mean():.4f} +/- {fm[col].std():.4f}")

# NEW — save fold-level metrics + pooled OOF predictions (with fold id)
fm.to_csv(f'{RESULTS_DIR}/fold_metrics.csv', index=False)
oof_df_ann = pd.DataFrame({
    'Subject': X_subject, 'y_true': y_dis,
    'oof_prob': oof_probs, 'fold': fold_ids,
})
oof_df_ann.to_csv(f'{RESULTS_DIR}/oof_predictions.csv', index=False)
print(f"  Saved: {RESULTS_DIR}/fold_metrics.csv, {RESULTS_DIR}/oof_predictions.csv")

prec_arr, rec_arr, thr_arr = precision_recall_curve(y_dis, oof_probs)
f1_arr = np.divide(2*prec_arr[:-1]*rec_arr[:-1], prec_arr[:-1]+rec_arr[:-1],
                   out=np.zeros_like(prec_arr[:-1]), where=(prec_arr[:-1]+rec_arr[:-1]) > 0)
POOLED_THRESHOLD = float(thr_arr[int(np.argmax(f1_arr))]) if len(thr_arr) else 0.5
print(f"\n  Pooled-OOF-tuned threshold: {POOLED_THRESHOLD:.4f}")

oof_pred_default = (oof_probs >= 0.5).astype(int)
oof_pred_pooled  = (oof_probs >= POOLED_THRESHOLD).astype(int)
print("\n  Pooled OOF report @ threshold=0.5 (ANN):")
print(classification_report(y_dis, oof_pred_default, target_names=['Healthy', 'AD'], digits=4))
print("\n  Pooled OOF report @ POOLED_THRESHOLD (ANN):")
print(classification_report(y_dis, oof_pred_pooled, target_names=['Healthy', 'AD'], digits=4))
print(f"  Pooled OOF AUC (ANN): {roc_auc_score(y_dis, oof_probs):.4f}")
print("=" * 60)
print(f"\n  Best fold: {best_fold_i} (AUC={best_fold_auc:.4f})")

del X_sc, y_dis
gc.collect()
tf.keras.backend.clear_session()
print("\n  ANN BASELINE DONE")
