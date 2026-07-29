"""
============================================================
 Step 8.5 (v3.1 — LIGHTER ARCHITECTURE, CAPACITY REDUCTION
 ONLY) — DB-HLSTM v3.1
 Disease Classification ONLY (AD vs Healthy) — EEG data

 v3.1 CHANGES (v3.0 did not improve overall cross-validation
 performance; this is a capacity reduction ONLY — no new
 layers, no deeper network, same overall dual-branch design.
 Dataset, preprocessing, windowing, subject-level K-fold CV,
 feature extraction, threshold tuning, training loop,
 callbacks, and optimizer are ALL UNCHANGED from v3.0):

   1. BiLSTM(64) -> BiLSTM(32)  (1st temporal LSTM layer)
   2. BiLSTM(32) -> BiLSTM(16)  (2nd temporal LSTM layer)
      NOTE: this changes the temporal feature-dim per timestep
      from 64 -> 32 (BiLSTM(16) concat = 16*2).
   3. MultiHeadAttention: num_heads 4 -> 2, key_dim stays 16
      (num_heads*key_dim = 32, matches the new 32-dim
      lstm_output so the residual Add still lines up -- no
      extra projection layer needed).
   4. Classifier head: Dense(128) -> Dense(64); the existing
      Dense(64) output-side layer is kept as-is.
   5. Everything else kept exactly as in v3.0: dilated Conv1D
      block, residual Add + LayerNormalization around the
      attention block, gated fusion block (Dense(64) embedding
      -> sigmoid gate -> Multiply -> Concatenate with scalar
      branch), AdamW optimizer, learning rate, mixed precision,
      callbacks, subject-level 5-fold CV, feature extraction,
      threshold tuning, batch size, and all other
      hyperparameters.

 PRIOR FIXES (v2.0-v3.0, all still in effect, unchanged):
 FIX v2.0.1 -- I_val wired into the scalar branch
 FIX v2.0.2 -- SUBJECT-LEVEL train/val/test split
 FIX v2.0.3 -- theta_env added as 6th temporal channel
 FIX v2.0.5 -- subject-level K-fold CV
 FIX v2.0.6 -- scalar branch stats swapped to theta_env-derived
 FIX v2.1.1 -- LSTM capacity cut, ES/LR patience tightened
 FIX v2.1.2 -- per-fold threshold tuning removed; single pooled-OOF
              global threshold instead
 FIX v2.1.3 -- theta_env_* / ratio_env_* column-name compatibility
 v3.0        -- Bidirectional LSTMs, MultiHeadAttention + residual +
               LayerNorm, gated fusion, upgraded classifier head
 v3.0 tweak  -- fixed AD class weight {0:1.0, 1:2.0}, MHA
               dropout=0.1, Dropout(0.2) added after attention
               LayerNormalization

 ARCHITECTURE (v3.1):
   Branch A: Dilated Conv1D(64,64,64) + SpatialDropout1D + BatchNorm
             -> Bidirectional LSTM(32) -> BatchNorm ->
             Bidirectional LSTM(16) -> BatchNorm
             -> MultiHeadAttention(2 heads, key_dim=16) + residual
                Add + LayerNormalization + Dropout(0.2)
             (temporal: bold, hrf_c/td/dd, v, theta_env)
   Branch B: Dense(64)->BN->Dense(64)->Dropout->Dense(32)
             (scalar: 8 features incl. I_val)
   Fusion:   GlobalAvg+GlobalMax pooling of attended temporal
             sequence -> Dense(64) embedding -> sigmoid gate ->
             Multiply -> Concatenate with scalar branch output ->
             Dense64->BN->Dropout(0.4)->Dense64->Dropout(0.3)->Output
   (GNN Branch C not present in this script)

 Input   : /kaggle/working/EEG_BOLD_Data.csv
           (must be from the theta_env-driven BW pipeline --
           STEP_8.3_BALLOON_WINDKESSEL_v2.py -- for bold/hrf_*
           columns to carry real signal; I_val is independent
           of that fix and already valid either way)
 Output  : /kaggle/working/dbhlstm_v31_disease.keras
============================================================
"""

import os
import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input,
                                     BatchNormalization, Concatenate,
                                     Conv1D, MultiHeadAttention,
                                     GlobalAveragePooling1D,
                                     GlobalMaxPooling1D, Bidirectional,
                                     SpatialDropout1D, Add, Multiply,
                                     LayerNormalization, GaussianNoise)
# Adam -> AdamW. Import path varies by TF version (stable
# namespace in TF 2.11+, experimental in a few in-between versions).
try:
    from tensorflow.keras.optimizers import AdamW
except ImportError:
    from tensorflow.keras.optimizers.experimental import AdamW
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau, CSVLogger)

# NEW — ReduceLROnPlateau changes the optimizer's LR but does not write
# it into the `logs` dict that CSVLogger reads, so the per-epoch log
# CSVs would otherwise have no `lr` column. This callback fixes that,
# enabling the Learning-Rate panel of the training-curves figure.
class LRLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = float(self.model.optimizer.learning_rate.numpy())
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import BinaryCrossentropy
from tqdm.notebook import tqdm

# ============================================================
# GPU SETUP — P100  (unchanged)
# ============================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("  GPU: Tesla P100  float16: ON \u2705")
else:
    print("  CPU only \u26a0\ufe0f")
tf.config.optimizer.set_jit(True)

# ============================================================
# CONFIGURATION  (unchanged)
# ============================================================
CSV_PATH      = '/kaggle/input/datasets/sajidkhan1214/hjjj8999999/EEG_BOLD_Data.csv'
DISEASE_MODEL = '/kaggle/working/dbhlstm_v31_disease.keras'

SEQ_LEN     = 600
WINDOW      = 100
STRIDE      = 100
BATCH_SIZE  = 256
LEARN_RATE  = 0.0003
WEIGHT_DECAY = 3e-4   # v3.2: was 1e-4 in v3.0/v3.1 — stronger regularization
EPOCHS      = 150
PATIENCE_ES = 15
PATIENCE_LR = 3
VAL_FRAC    = 0.15
TEST_FRAC   = 0.10
SEED        = 42

# NEW — figure-pipeline output folder (per-model, matches the other
# 4 scripts so the plotting script can loop the same tree structure)
RESULTS_DIR = '/kaggle/working/results/dbhlstm'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# STEP 1 — LOAD DATA  (unchanged)
# ============================================================
print("=" * 60)
print("  DB-HLSTM v3.1 — DISEASE CLASSIFICATION (EEG)  [LIGHTER ARCH]")
print(f"  Window={WINDOW}  Stride={STRIDE}  Batch={BATCH_SIZE}")
print("=" * 60)

print("\n  Loading EEG_BOLD_Data.csv ...")
df = pd.read_csv(CSV_PATH)
print(f"  Rows     : {len(df):,}")
print(f"  Columns  : {len(df.columns):,}")

if ('theta_env_0' not in df.columns and 'theta_env_599' not in df.columns
        and 'ratio_env_0' not in df.columns and 'ratio_env_599' not in df.columns):
    print("  NOTE: this CSV may be from the pre-ratio-envelope pipeline —")
    print("        bold/hrf_* signal quality depends on the HRF/BW steps")
    print("        having used theta_env(t)/ratio_env(t), not the old s(t).")

# ============================================================
# STEP 2 — DISEASE LABEL + I_val  (unchanged)
# ============================================================
print("\n  Disease label distribution (Class column):")
print(df['Class'].value_counts())
print("  0 = Healthy, 1 = AD")

y_disease_full = df['Class'].values.astype(np.int32)
i_val_full     = df['I_val'].values.astype(np.float32)
subject_full   = df['Subject'].values   # needed for subject-level split

# ============================================================
# STEP 2.5 — DIAGNOSTIC: I_val vs Class (row-level, pre-windowing)  (unchanged)
# ============================================================
print("\n" + "=" * 60)
print("  DIAGNOSTIC — I_val vs Class (row-level, pre-windowing)")
print("=" * 60)
print(f"  I_val  n_unique : {pd.Series(i_val_full).nunique()} / {len(i_val_full)}")
print(f"  I_val  NaN count: {np.isnan(i_val_full).sum()}")
print(f"  I_val  min/max  : {np.nanmin(i_val_full):.4f} / {np.nanmax(i_val_full):.4f}")

ival_healthy = i_val_full[y_disease_full == 0]
ival_ad      = i_val_full[y_disease_full == 1]
print(f"  I_val mean | Healthy: {np.nanmean(ival_healthy):.4f}  "
      f"AD: {np.nanmean(ival_ad):.4f}")

from scipy.stats import pointbiserialr
mask = ~np.isnan(i_val_full)
r_val, p_val = pointbiserialr(y_disease_full[mask], i_val_full[mask])
print(f"  Point-biserial r (I_val vs Class): r={r_val:.4f}  p={p_val:.4g}")
print("=" * 60 + "\n")

# ============================================================
# STEP 3 — WINDOWING (+ track Subject per window)  (unchanged)
# ============================================================
print(f"\n  Windowing (window={WINDOW}, stride={STRIDE}) ...")

bold_cols       = [f'bold_{i}'      for i in range(SEQ_LEN)]
hrf_c_cols      = [f'hrf_c_{i}'     for i in range(SEQ_LEN)]
hrf_td_cols     = [f'hrf_td_{i}'    for i in range(SEQ_LEN)]
hrf_dd_cols     = [f'hrf_dd_{i}'    for i in range(SEQ_LEN)]
v_cols          = [f'v_{i}'         for i in range(SEQ_LEN)]

# theta_env_ / ratio_env_ column-name compatibility (unchanged)
_ratio_env_cols = [f'ratio_env_{i}' for i in range(SEQ_LEN)]
_theta_env_cols = [f'theta_env_{i}' for i in range(SEQ_LEN)]

if all(c in df.columns for c in _ratio_env_cols):
    theta_env_cols = _ratio_env_cols
    print("  Using ratio_env_0..599 as the 6th temporal channel (v7 naming).")
elif all(c in df.columns for c in _theta_env_cols):
    theta_env_cols = _theta_env_cols
    print("  Using theta_env_0..599 as the 6th temporal channel (v5/v6 naming).")
else:
    missing_ratio = [c for c in _ratio_env_cols if c not in df.columns]
    missing_theta = [c for c in _theta_env_cols if c not in df.columns]
    raise KeyError(
        f"  Neither ratio_env_0..599 nor theta_env_0..599 fully present in "
        f"the CSV (missing {len(missing_ratio)}/{SEQ_LEN} of ratio_env_, "
        f"{len(missing_theta)}/{SEQ_LEN} of theta_env_, e.g. "
        f"{(missing_ratio or missing_theta)[:3]}). This script requires the "
        f"theta/alpha ratio envelope as the 6th temporal channel — "
        f"regenerate EEG_BOLD_Data.csv with the ratio-envelope-driven "
        f"pipeline first."
    )

n_windows = (SEQ_LEN - WINDOW) // STRIDE + 1
n_samples = len(df) * n_windows
print(f"  Windows per sweep : {n_windows}")
print(f"  Total samples     : {n_samples:,}")

X_bold    = np.zeros((n_samples, WINDOW), dtype=np.float32)
X_hrf_c   = np.zeros((n_samples, WINDOW), dtype=np.float32)
X_hrf_td  = np.zeros((n_samples, WINDOW), dtype=np.float32)
X_hrf_dd  = np.zeros((n_samples, WINDOW), dtype=np.float32)
X_volt    = np.zeros((n_samples, WINDOW), dtype=np.float32)
X_theta   = np.zeros((n_samples, WINDOW), dtype=np.float32)
y_dis     = np.zeros(n_samples, dtype=np.int32)
X_ival    = np.zeros(n_samples, dtype=np.float32)
X_subject = np.empty(n_samples, dtype=object)

idx = 0
for row_i in tqdm(range(len(df)), desc="Windowing"):
    row = df.iloc[row_i]
    bold   = row[bold_cols].values.astype(np.float32)
    hrf_c  = row[hrf_c_cols].values.astype(np.float32)
    hrf_td = row[hrf_td_cols].values.astype(np.float32)
    hrf_dd = row[hrf_dd_cols].values.astype(np.float32)
    volt   = row[v_cols].values.astype(np.float32)
    theta  = row[theta_env_cols].values.astype(np.float32)
    lb     = int(y_disease_full[row_i])
    ival   = i_val_full[row_i]
    subj_id = subject_full[row_i]

    for start in range(0, SEQ_LEN - WINDOW + 1, STRIDE):
        end = start + WINDOW
        X_bold[idx]    = bold[start:end]
        X_hrf_c[idx]   = hrf_c[start:end]
        X_hrf_td[idx]  = hrf_td[start:end]
        X_hrf_dd[idx]  = hrf_dd[start:end]
        X_volt[idx]    = volt[start:end]
        X_theta[idx]   = theta[start:end]
        y_dis[idx]     = lb
        X_ival[idx]    = ival
        X_subject[idx] = subj_id
        idx += 1

del df; gc.collect()
print(f"  Windowing complete: {idx:,} samples")
print(f"  y_dis unique: {np.unique(y_dis[:idx])}")  # must be [0 1]
print(f"  Unique subjects in windowed data: {len(np.unique(X_subject))}")

# ============================================================
# STEP 4 — FEATURE STACK + NORMALIZE  (unchanged)
# ============================================================
print("\n  Stacking and normalizing features ...")

X_ts = np.stack(
    [X_bold, X_hrf_c, X_hrf_td, X_hrf_dd, X_volt, X_theta], axis=2)
del X_bold, X_hrf_c, X_hrf_td, X_hrf_dd; gc.collect()  # X_theta kept for scalar stats below

N_TS_CHANNELS = X_ts.shape[2]  # 6
X_norm = np.zeros_like(X_ts, dtype=np.float32)
for fi in range(N_TS_CHANNELS):
    feat = X_ts[:, :, fi]
    X_norm[:, :, fi] = (feat - feat.mean(axis=1, keepdims=True)) / \
                       (feat.std(axis=1, keepdims=True) + 1e-8)
del X_ts; gc.collect()

sc = np.column_stack([
    X_volt.mean(axis=1),
    X_volt.std(axis=1),
    X_theta.mean(axis=1),                          # theta_env level
    X_theta.std(axis=1),                            # theta_env variability
    X_theta.max(axis=1),                             # theta_env peak
    X_theta.max(axis=1) - X_theta.min(axis=1),        # theta_env range
    X_theta.sum(axis=1) / WINDOW,                      # theta_env "energy"
    X_ival,   # theta/alpha ratio — validated discriminative feature
]).astype(np.float32)

X_sc = StandardScaler().fit_transform(sc).astype(np.float32)
del X_volt, X_theta, sc; gc.collect()

print(f"  X_norm shape : {X_norm.shape}")
print(f"  X_sc shape   : {X_sc.shape}")

# ============================================================
# STEP 5 — BUILD MODEL (v3.2 — regularization / generalization pass)
# ============================================================
def build_model_v32():
    REG   = l2(4e-4)
    ts_in = Input(shape=(WINDOW, 6), name='ts_input')

    # --- v3.2: GaussianNoise on raw input for robustness (train-time only,
    # Keras automatically disables it at inference) ---
    x_noisy = GaussianNoise(0.02, name='input_gaussian_noise')(ts_in)

    # --- Dilated Conv block (v3.2: filters 64 -> 48, same dilated pattern) ---
    t = Conv1D(48, kernel_size=3, padding='causal', dilation_rate=1,
               activation='relu', kernel_regularizer=REG,
               kernel_initializer='he_normal')(x_noisy)
    t = SpatialDropout1D(0.2)(t)
    t = BatchNormalization()(t)
    t = Conv1D(48, kernel_size=3, padding='causal', dilation_rate=2,
               activation='relu', kernel_regularizer=REG,
               kernel_initializer='he_normal')(t)
    t = BatchNormalization()(t)
    t = Conv1D(48, kernel_size=3, padding='causal', dilation_rate=4,
               activation='relu', kernel_regularizer=REG,
               kernel_initializer='he_normal')(t)
    t = BatchNormalization()(t)

    # --- Temporal stack (v3.2) — 1st LSTM -> Bidirectional LSTM(48),
    # recurrent dropout 0.35 -> 0.40 ---
    x = Bidirectional(LSTM(48, return_sequences=True, dropout=0.40,
                            kernel_regularizer=REG),
                       name='bilstm_48')(t)
    x = BatchNormalization()(x)

    # 2nd LSTM -> Bidirectional LSTM(24), dropout 0.40
    # (output per timestep: 24*2 = 48-dim; MultiHeadAttention's default
    # output_shape falls back to the query's last dim, so it will project
    # back to 48-dim automatically — residual Add below still lines up)
    lstm_output = Bidirectional(LSTM(24, return_sequences=True, dropout=0.40,
                                      kernel_regularizer=REG),
                                 name='bilstm_24')(x)
    lstm_output = BatchNormalization()(lstm_output)

    # --- Multi-Head Self-Attention (v3.2: back to 4 heads, key_dim=16)
    # + residual + LayerNorm + Dropout ---
    attention_output = MultiHeadAttention(
        num_heads=4, key_dim=16, dropout=0.1,
        kernel_initializer='he_normal',
        name='temporal_multihead_attention'
    )(query=lstm_output, value=lstm_output, key=lstm_output)
    attention_output = Add(name='attention_residual_add')(
        [attention_output, lstm_output])
    attention_output = LayerNormalization(name='attention_layernorm')(
        attention_output)
    attention_output = Dropout(0.2, name='attention_output_dropout')(
        attention_output)

    # --- Dual pooling (avg + max) over the attended temporal sequence ---
    temporal_avg = GlobalAveragePooling1D()(attention_output)
    temporal_max = GlobalMaxPooling1D()(attention_output)
    temporal_pool = Concatenate(name='temporal_avg_max_concat')(
        [temporal_avg, temporal_max])

    # --- Scalar branch (unchanged): 8 -> Dense64 -> BN -> Dense64 -> Dropout -> Dense32
    sc_in = Input(shape=(8,), name='scalar_input')
    y_ = Dense(64, activation='relu', kernel_regularizer=REG,
               kernel_initializer='he_normal')(sc_in)
    y_ = BatchNormalization()(y_)
    y_ = Dense(64, activation='relu', kernel_regularizer=REG,
               kernel_initializer='he_normal')(y_)
    y_ = Dropout(0.2)(y_)
    y_ = Dense(32, activation='relu', kernel_regularizer=REG,
               kernel_initializer='he_normal')(y_)   # scalar branch output

    # --- Gated fusion block (unchanged structure) ---
    # temporal_pool -> Dense(64) embedding -> sigmoid gate -> Multiply -> Concat with scalar
    temporal_embed = Dense(64, activation='relu', kernel_regularizer=REG,
                            kernel_initializer='he_normal',
                            name='temporal_embedding')(temporal_pool)
    gate = Dense(64, activation='sigmoid', kernel_regularizer=REG,
                 kernel_initializer='he_normal',
                 name='fusion_gate')(temporal_embed)
    gated_temporal = Multiply(name='gated_temporal_embedding')(
        [temporal_embed, gate])

    z = Concatenate(name='gated_fusion_concat')([gated_temporal, y_])

    # --- Classifier head (unchanged from v3.1) ---
    z = Dense(64, activation='relu', kernel_regularizer=REG,
              kernel_initializer='he_normal')(z)
    z = BatchNormalization()(z)
    z = Dropout(0.4)(z)
    z = Dense(64, activation='relu', kernel_regularizer=REG,
              kernel_initializer='he_normal')(z)
    z = Dropout(0.3)(z)

    out  = Dense(1, activation='sigmoid', dtype='float32')(z)
    # v3.2: label smoothing (0.05) — useful for binary EEG classification
    # where labels are hard 0/1 but the true decision boundary is fuzzy
    loss = BinaryCrossentropy(label_smoothing=0.05)
    mets = ['accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')]

    m = Model(inputs=[ts_in, sc_in], outputs=out)
    # v3.2: weight_decay 1e-4 -> 3e-4 (WEIGHT_DECAY config), + gradient
    # clipping (clipnorm=1.0) for training stability
    m.compile(optimizer=AdamW(learning_rate=LEARN_RATE,
                              weight_decay=WEIGHT_DECAY,
                              clipnorm=1.0),
              loss=loss, metrics=mets)
    return m

AUTOTUNE = tf.data.AUTOTUNE

# ============================================================
# STEP 6 — SUBJECT-LEVEL K-FOLD CROSS-VALIDATION  (unchanged)
# ============================================================
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (precision_recall_curve, confusion_matrix,
                             classification_report, roc_auc_score)

N_FOLDS = 5

print("\n" + "=" * 60)
print(f"  DISEASE CLASSIFICATION — AD vs Healthy — {N_FOLDS}-FOLD CV")
print("  (subject-level; every subject held out exactly once)")
print("=" * 60)

sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_probs = np.zeros(len(y_dis), dtype=np.float64)
fold_ids  = np.zeros(len(y_dis), dtype=np.int32)   # NEW
fold_metrics = []
best_fold_auc = -1.0
best_fold_i = -1

for fold_i, (tr_full_idx, test_idx) in enumerate(
        sgkf.split(X_norm, y_dis, groups=X_subject)):

    fold_no = fold_i + 1
    print(f"\n{'-'*60}")
    print(f"  FOLD {fold_no}/{N_FOLDS}")
    print(f"{'-'*60}")

    tr_full_subj = X_subject[tr_full_idx]
    subj_y = (pd.DataFrame({'Subject': tr_full_subj, 'y': y_dis[tr_full_idx]})
              .groupby('Subject')['y'].first().reset_index())
    inner_train_subj, inner_val_subj = train_test_split(
        subj_y, test_size=0.15, stratify=subj_y['y'],
        random_state=SEED + fold_i)

    inner_train_mask = np.isin(tr_full_subj, inner_train_subj['Subject'].values)
    inner_val_mask   = np.isin(tr_full_subj, inner_val_subj['Subject'].values)
    train_idx = tr_full_idx[inner_train_mask]
    val_idx   = tr_full_idx[inner_val_mask]

    n_test_subj  = len(np.unique(X_subject[test_idx]))
    n_val_subj   = len(inner_val_subj)
    n_train_subj = len(inner_train_subj)
    print(f"  Subjects — Train: {n_train_subj}  Val: {n_val_subj}  "
          f"Test(held-out fold): {n_test_subj}")
    print(f"  Windows  — Train: {len(train_idx):,}  Val: {len(val_idx):,}  "
          f"Test: {len(test_idx):,}")

    y_tr, y_val_f, y_test_f = y_dis[train_idx], y_dis[val_idx], y_dis[test_idx]

    tr_ds = (tf.data.Dataset.from_tensor_slices(
        ({'ts_input': X_norm[train_idx], 'scalar_input': X_sc[train_idx]}, y_tr))
        .shuffle(len(train_idx), seed=SEED).batch(BATCH_SIZE).prefetch(AUTOTUNE))
    vl_ds = (tf.data.Dataset.from_tensor_slices(
        ({'ts_input': X_norm[val_idx], 'scalar_input': X_sc[val_idx]}, y_val_f))
        .batch(BATCH_SIZE).prefetch(AUTOTUNE))

    n0 = (y_tr == 0).sum(); n1 = (y_tr == 1).sum()
    cw = {0: 1.0, 1: 2.0}

    tf.keras.backend.clear_session()
    model_f = build_model_v32()

    fold_model_path = f'{RESULTS_DIR}/dbhlstm_v31_disease_fold{fold_no}.keras'
    cb_f = [
        ModelCheckpoint(fold_model_path, monitor='val_auc',
                        save_best_only=True, mode='max', verbose=0),
        EarlyStopping(monitor='val_auc', patience=PATIENCE_ES,
                      restore_best_weights=True, mode='max', verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=PATIENCE_LR, min_lr=1e-6, verbose=0),
        LRLogger(),  # NEW — must come before CSVLogger so 'lr' is in logs when it writes
        CSVLogger(f'{RESULTS_DIR}/fold{fold_no}_log.csv'),  # per-epoch training curves
    ]

    model_f.fit(tr_ds, validation_data=vl_ds, epochs=EPOCHS,
                callbacks=cb_f, class_weight=cw, verbose=2)

    val_probs = model_f.predict(
        {'ts_input': X_norm[val_idx], 'scalar_input': X_sc[val_idx]},
        verbose=0).ravel().astype(np.float64)

    test_probs = model_f.predict(
        {'ts_input': X_norm[test_idx], 'scalar_input': X_sc[test_idx]},
        verbose=0).ravel().astype(np.float64)
    oof_probs[test_idx] = test_probs
    fold_ids[test_idx] = fold_no   # NEW

    pred_def = (test_probs >= 0.5).astype(int)

    def _metrics(pred, y_true):
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0,1]).ravel()
        acc = (pred == y_true).mean()
        prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1)
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
        return acc, prec, rec, f1

    acc_d, prec_d, rec_d, f1_d = _metrics(pred_def, y_test_f)
    fold_auc = roc_auc_score(y_test_f, test_probs) if len(np.unique(y_test_f)) > 1 else float('nan')

    print(f"  Fold {fold_no} AUC={fold_auc:.4f}  "
          f"[thr=0.5]   acc={acc_d:.4f} prec={prec_d:.4f} rec={rec_d:.4f} f1={f1_d:.4f}")

    fold_metrics.append({
        'fold': fold_no, 'auc': fold_auc,
        'acc_default': acc_d, 'prec_default': prec_d,
        'rec_default': rec_d, 'f1_default': f1_d,
    })

    if fold_auc > best_fold_auc:
        best_fold_auc = fold_auc
        best_fold_i = fold_no

    del model_f, tr_ds, vl_ds
    gc.collect()

# ============================================================
# STEP 7 — CROSS-VALIDATED SUMMARY  (unchanged)
# ============================================================
fm = pd.DataFrame(fold_metrics)
print("\n" + "=" * 60)
print(f"  {N_FOLDS}-FOLD CROSS-VALIDATION SUMMARY (all 65 subjects)")
print("=" * 60)
print(fm.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\n  Mean +/- std across folds:")
for col in ['auc', 'acc_default', 'f1_default']:
    print(f"    {col:14s}: {fm[col].mean():.4f} +/- {fm[col].std():.4f}")

prec_arr, rec_arr, thr_arr = precision_recall_curve(y_dis, oof_probs)
f1_arr = np.divide(
    2*prec_arr[:-1]*rec_arr[:-1], prec_arr[:-1]+rec_arr[:-1],
    out=np.zeros_like(prec_arr[:-1]), where=(prec_arr[:-1]+rec_arr[:-1]) > 0)
POOLED_THRESHOLD = float(thr_arr[int(np.argmax(f1_arr))]) if len(thr_arr) else 0.5
print(f"\n  Pooled-OOF-tuned global threshold (POOLED_THRESHOLD): {POOLED_THRESHOLD:.4f}")
print("  (tuned once on all 65 subjects' pooled out-of-fold probs —")
print("   NOT averaged from unstable per-fold thresholds)")

oof_pred_default = (oof_probs >= 0.5).astype(int)
oof_pred_pooled  = (oof_probs >= POOLED_THRESHOLD).astype(int)

print("\n  Pooled out-of-fold report @ threshold=0.5:")
print(classification_report(y_dis, oof_pred_default,
                            target_names=['Healthy', 'AD'], digits=4))

print("\n  Pooled out-of-fold report @ POOLED_THRESHOLD:")
print(classification_report(y_dis, oof_pred_pooled,
                            target_names=['Healthy', 'AD'], digits=4))

print(f"  Pooled OOF AUC (all subjects): {roc_auc_score(y_dis, oof_probs):.4f}")
print("=" * 60)

print(f"\n  Best single fold: {best_fold_i} (AUC={best_fold_auc:.4f})")
print(f"  Its model is saved at: /kaggle/working/dbhlstm_v31_disease_fold{best_fold_i}.keras")
print(f"  Copy that file to {DISEASE_MODEL} if you want a single")
print(f"  deployable model — recommended deployment threshold: "
      f"{POOLED_THRESHOLD:.4f} (tuned once on pooled OOF, all 65 subjects).")

# ============================================================
# NEW — SAVE RESULTS TO CSV (so a separate plotting script can
# read them without needing to rerun training)
# ============================================================
fm.to_csv(f'{RESULTS_DIR}/fold_metrics.csv', index=False)

oof_df = pd.DataFrame({
    'Subject': X_subject,
    'y_true': y_dis,
    'oof_prob': oof_probs,
    'fold': fold_ids,   # NEW — needed for Appendix fold-wise ROC/PR/CM figures
})
oof_df.to_csv(f'{RESULTS_DIR}/oof_predictions.csv', index=False)

with open(f'{RESULTS_DIR}/summary.txt', 'w') as f:
    f.write(f"POOLED_THRESHOLD={POOLED_THRESHOLD:.6f}\n")
    f.write(f"best_fold={best_fold_i}\n")
    f.write(f"best_fold_auc={best_fold_auc:.6f}\n")
    f.write(f"pooled_oof_auc={roc_auc_score(y_dis, oof_probs):.6f}\n")

print(f"\n  Saved: {RESULTS_DIR}/fold_metrics.csv, {RESULTS_DIR}/oof_predictions.csv, {RESULTS_DIR}/summary.txt")
print(f"  Saved: {RESULTS_DIR}/fold{{1..5}}_log.csv (per-fold training curves, incl. lr column)")

del X_norm, X_sc, y_dis
gc.collect()
tf.keras.backend.clear_session()

print("\n  DISEASE CLASSIFICATION (K-FOLD CV) DONE \u2705")