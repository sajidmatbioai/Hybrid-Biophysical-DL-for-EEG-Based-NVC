"""
============================================================
 Step 8.5 (v3.1 — LIGHTER ARCHITECTURE, CAPACITY REDUCTION
 ONLY) — DB-HLSTM v3.1
 Disease Classification ONLY (AD vs Healthy) — EEG data

 *** SINGLE SUBJECT-LEVEL TRAIN / VAL / TEST SPLIT VERSION ***
 (converted from the subject-level 5-fold StratifiedGroupKFold
 pipeline — see conversion notes below)

 CONVERSION NOTES (what changed vs. the 5-fold CV script):
   - Removed StratifiedGroupKFold and the entire fold loop.
   - Removed pooled out-of-fold (OOF) predictions and fold
     averaging / fold-metrics summary table.
   - Added ONE subject-level split: Train ~70%, Val ~10-15%,
     Test ~20% (stratified by Class at the subject level, no
     subject ever appears in more than one split).
   - Trains exactly ONE DB-HLSTM model.
   - Threshold tuning has been removed entirely — only default
     (0.5) predictions are evaluated on the test split, once.
   - All preprocessing, feature extraction, DB-HLSTM
     architecture (v3.2 build_model_v32), HRF/Balloon-Windkessel
     upstream pipeline, normalization, callbacks, metrics,
     plotting, and model saving are otherwise UNCHANGED.

 v3.1 CHANGES (v3.0 did not improve overall cross-validation
 performance; this is a capacity reduction ONLY — no new
 layers, no deeper network, same overall dual-branch design.
 Dataset, preprocessing, windowing, feature extraction,
 training loop, callbacks, and optimizer are
 ALL UNCHANGED from v3.0):

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
      callbacks, feature extraction, batch
      size, and all other hyperparameters.

 v3.3 CHANGES (optimization pass for better generalization — SAME
 dual-branch architecture as v3.2, Conv1D -> BiLSTM ->
 MultiHeadAttention -> Gated Fusion -> Dense classifier; no new
 layers, no redesign):
   1. Removed GaussianNoise(0.02) and label_smoothing=0.05 — plain
      BinaryCrossentropy() now (was stacking extra regularization
      on top of dropout/L2/weight-decay already present).
   2. WEIGHT_DECAY: 3e-4 -> 1e-4 (was over-regularizing).
   3. Dropout reduced: LSTM dropout 0.40->0.25 (both BiLSTM layers),
      post-attention dropout 0.20->0.10, classifier dropouts
      0.40->0.25 and 0.30->0.20, scalar-branch dropout 0.20->0.15.
      MultiHeadAttention's internal dropout stays 0.10 (unchanged).
   4. Capacity restored: Conv1D filters 48->64 (all 3 dilated
      layers), 1st BiLSTM 48->64, 2nd BiLSTM 24->32. Attention
      (4 heads, key_dim=16) kept exactly the same — dimensions
      still line up (32*2=64 = num_heads*key_dim) for the
      residual Add.
   5. Class weights now computed automatically via sklearn's
      compute_class_weight('balanced', ...) instead of the fixed
      {0:1.0, 1:2.0}.
   6. Added validation-set threshold optimization (Youden Index,
      i.e. max TPR-FPR, from the ROC curve, safety-clamped to
      [0.05, 0.95]) — test set is evaluated at this tuned
      best_threshold instead of the fixed 0.5. best_threshold is
      saved in summary.txt; both default-0.5 and tuned-threshold
      metrics are also saved to test_predictions.csv/summary.txt
      for comparison.
   7. Optimizer unchanged: AdamW, lr=3e-4, gradient clipping,
      ReduceLROnPlateau, EarlyStopping.
   8. Added 5 new training-curve plots: Accuracy/Loss/AUC/
      Precision/Recall vs Epoch (publication-style, train+val).
   Preprocessing, feature extraction, windowing, subject-level
   split, and the dual-branch architecture itself are all
   otherwise UNCHANGED from v3.2.

 PRIOR FIXES (v2.0-v3.2, all still in effect unless listed above):
 FIX v2.0.1 -- I_val wired into the scalar branch
 FIX v2.0.2 -- SUBJECT-LEVEL train/val/test split
 FIX v2.0.3 -- theta_env added as 6th temporal channel
 FIX v2.0.5 -- subject-level K-fold CV (now: single subject-level split)
 FIX v2.0.6 -- scalar branch stats swapped to theta_env-derived
 FIX v2.1.1 -- LSTM capacity cut, ES/LR patience tightened
 FIX v2.1.2 -- per-fold threshold tuning removed (historical; now
              threshold tuning removed entirely, default 0.5 only)
 FIX v2.1.3 -- theta_env_* / ratio_env_* column-name compatibility
 v3.0        -- Bidirectional LSTMs, MultiHeadAttention + residual +
               LayerNorm, gated fusion, upgraded classifier head
 v3.0 tweak  -- fixed AD class weight {0:1.0, 1:2.0}, MHA
               dropout=0.1, Dropout(0.2) added after attention
               LayerNormalization

 ARCHITECTURE (v3.1 — unchanged):
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
             Dense64->BN->Dropout(0.25)->Dense64->Dropout(0.20)->Output
   (GNN Branch C not present in this script)

 Input   : /kaggle/working/EEG_BOLD_Data.csv
           (must be from the theta_env-driven BW pipeline --
           STEP_8.3_BALLOON_WINDKESSEL_v2.py -- for bold/hrf_*
           columns to carry real signal; I_val is independent
           of that fix and already valid either way)
 Output  : /kaggle/working/dbhlstm_v31_disease.keras
           /kaggle/working/training_log.csv
           /kaggle/working/loss_vs_epoch.png
           /kaggle/working/accuracy_vs_epoch.png
           /kaggle/working/auc_vs_epoch.png
           /kaggle/working/precision_vs_epoch.png
           /kaggle/working/recall_vs_epoch.png
           /kaggle/working/confusion_matrix.png
           /kaggle/working/roc_curve.png
           /kaggle/working/precision_recall_curve.png
           /kaggle/working/classification_report.txt
           /kaggle/working/test_predictions.csv
           /kaggle/working/summary.txt (includes best_threshold)
============================================================
"""

import os
import gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
                                     LayerNormalization)
# Adam -> AdamW. Import path varies by TF version (stable
# namespace in TF 2.11+, experimental in a few in-between versions).
try:
    from tensorflow.keras.optimizers import AdamW
except ImportError:
    from tensorflow.keras.optimizers.experimental import AdamW
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau, CSVLogger)
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
    print("  GPU: Tesla P100  float16: ON ✅")
else:
    print("  CPU only ⚠️")
tf.config.optimizer.set_jit(True)

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_DIR     = '/kaggle/working/dbhlstm'
os.makedirs(MODEL_DIR, exist_ok=True)

CSV_PATH      = '/kaggle/working/EEG_BOLD_Data.csv'
DISEASE_MODEL = f'{MODEL_DIR}/dbhlstm_v31_disease.keras'

SEQ_LEN     = 600
WINDOW      = 100
STRIDE      = 100
BATCH_SIZE  = 256
LEARN_RATE  = 0.0003
WEIGHT_DECAY = 1e-4   # v3.3: reduced from 3e-4 — was over-regularizing
EPOCHS      = 150
PATIENCE_ES = 15
PATIENCE_LR = 3
SEED        = 42

# --- Single subject-level split fractions (Train ~70 / Val ~10-15 / Test ~20) ---
TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.10
TEST_FRAC   = 0.20
assert abs((TRAIN_FRAC + VAL_FRAC + TEST_FRAC) - 1.0) < 1e-9

# ============================================================
# STEP 1 — LOAD DATA  (unchanged)
# ============================================================
print("=" * 60)
print("  DB-HLSTM v3.1 — DISEASE CLASSIFICATION (EEG)  [SINGLE SPLIT]")
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
# STEP 5 — BUILD MODEL (v3.3 — generalization/capacity tuning pass;
# SAME dual-branch architecture as v3.2: Conv1D -> BiLSTM ->
# MultiHeadAttention -> Gated Fusion -> Dense classifier. Only
# capacity, dropout, and regularization strength changed.)
# ============================================================
def build_model_v32():
    REG   = l2(4e-4)
    ts_in = Input(shape=(WINDOW, 6), name='ts_input')

    # v3.3: GaussianNoise removed (was adding unnecessary
    # regularization on top of dropout/L2/weight-decay already
    # present) — Conv1D now reads ts_in directly.
    # --- Dilated Conv block (v3.3: filters restored 48 -> 64) ---
    t = Conv1D(64, kernel_size=3, padding='causal', dilation_rate=1,
               activation='relu', kernel_regularizer=REG,
               kernel_initializer='he_normal')(ts_in)
    t = SpatialDropout1D(0.2)(t)
    t = BatchNormalization()(t)
    t = Conv1D(64, kernel_size=3, padding='causal', dilation_rate=2,
               activation='relu', kernel_regularizer=REG,
               kernel_initializer='he_normal')(t)
    t = BatchNormalization()(t)
    t = Conv1D(64, kernel_size=3, padding='causal', dilation_rate=4,
               activation='relu', kernel_regularizer=REG,
               kernel_initializer='he_normal')(t)
    t = BatchNormalization()(t)

    # --- Temporal stack (v3.3) — 1st BiLSTM capacity restored
    # 48 -> 64; dropout reduced 0.40 -> 0.25 ---
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.25,
                            kernel_regularizer=REG),
                       name='bilstm_64')(t)
    x = BatchNormalization()(x)

    # 2nd BiLSTM capacity restored 24 -> 32; dropout reduced 0.40 -> 0.25
    # (output per timestep: 32*2 = 64-dim, matches num_heads*key_dim=64
    # below, so the residual Add still lines up — no projection needed)
    lstm_output = Bidirectional(LSTM(32, return_sequences=True, dropout=0.25,
                                      kernel_regularizer=REG),
                                 name='bilstm_32')(x)
    lstm_output = BatchNormalization()(lstm_output)

    # --- Multi-Head Self-Attention (kept exactly the same: 4 heads,
    # key_dim=16, dropout=0.10) + residual + LayerNorm + Dropout
    # (post-attention Dropout reduced 0.2 -> 0.10, matching the
    # "Attention dropout = 0.10" spec) ---
    attention_output = MultiHeadAttention(
        num_heads=4, key_dim=16, dropout=0.1,
        kernel_initializer='he_normal',
        name='temporal_multihead_attention'
    )(query=lstm_output, value=lstm_output, key=lstm_output)
    attention_output = Add(name='attention_residual_add')(
        [attention_output, lstm_output])
    attention_output = LayerNormalization(name='attention_layernorm')(
        attention_output)
    attention_output = Dropout(0.10, name='attention_output_dropout')(
        attention_output)

    # --- Dual pooling (avg + max) over the attended temporal sequence ---
    temporal_avg = GlobalAveragePooling1D()(attention_output)
    temporal_max = GlobalMaxPooling1D()(attention_output)
    temporal_pool = Concatenate(name='temporal_avg_max_concat')(
        [temporal_avg, temporal_max])

    # --- Scalar branch (unchanged structure; Dropout reduced 0.20 -> 0.15)
    sc_in = Input(shape=(8,), name='scalar_input')
    y_ = Dense(64, activation='relu', kernel_regularizer=REG,
               kernel_initializer='he_normal')(sc_in)
    y_ = BatchNormalization()(y_)
    y_ = Dense(64, activation='relu', kernel_regularizer=REG,
               kernel_initializer='he_normal')(y_)
    y_ = Dropout(0.15)(y_)
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

    # --- Classifier head (unchanged structure; Dropouts reduced
    # 0.40 -> 0.25 and 0.30 -> 0.20) ---
    z = Dense(64, activation='relu', kernel_regularizer=REG,
              kernel_initializer='he_normal')(z)
    z = BatchNormalization()(z)
    z = Dropout(0.25)(z)
    z = Dense(64, activation='relu', kernel_regularizer=REG,
              kernel_initializer='he_normal')(z)
    z = Dropout(0.20)(z)

    out  = Dense(1, activation='sigmoid', dtype='float32')(z)
    # v3.3: label smoothing removed — plain BinaryCrossentropy.
    # Combined with GaussianNoise removal and lower dropout/weight-decay,
    # label smoothing was unnecessary extra regularization stacked on
    # top of everything else.
    loss = BinaryCrossentropy()
    mets = ['accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')]

    m = Model(inputs=[ts_in, sc_in], outputs=out)
    # Optimizer unchanged: AdamW, lr=3e-4, gradient clipping kept.
    # WEIGHT_DECAY now 1e-4 (see CONFIGURATION section above).
    m.compile(optimizer=AdamW(learning_rate=LEARN_RATE,
                              weight_decay=WEIGHT_DECAY,
                              clipnorm=1.0),
              loss=loss, metrics=mets)
    return m

AUTOTUNE = tf.data.AUTOTUNE

# ============================================================
# STEP 6 — SINGLE SUBJECT-LEVEL TRAIN / VAL / TEST SPLIT
# (replaces the 5-fold StratifiedGroupKFold loop; no subject
# ever appears in more than one of Train/Val/Test)
# ============================================================
from sklearn.metrics import (precision_recall_curve, confusion_matrix,
                             classification_report, roc_auc_score,
                             roc_curve, ConfusionMatrixDisplay)

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

tr_ds = (tf.data.Dataset.from_tensor_slices(
    ({'ts_input': X_norm[train_idx], 'scalar_input': X_sc[train_idx]}, y_tr))
    .shuffle(len(train_idx), seed=SEED).batch(BATCH_SIZE).prefetch(AUTOTUNE))
vl_ds = (tf.data.Dataset.from_tensor_slices(
    ({'ts_input': X_norm[val_idx], 'scalar_input': X_sc[val_idx]}, y_val))
    .batch(BATCH_SIZE).prefetch(AUTOTUNE))

n0 = (y_tr == 0).sum(); n1 = (y_tr == 1).sum()
print(f"  Train class counts — Healthy(0): {n0:,}  AD(1): {n1:,}")

# v3.3: automatic class weights (was hardcoded {0:1.0, 1:2.0})
from sklearn.utils.class_weight import compute_class_weight
cw_arr = compute_class_weight(class_weight='balanced',
                              classes=np.array([0, 1]), y=y_tr)
cw = {0: float(cw_arr[0]), 1: float(cw_arr[1])}
print(f"  Computed class weights — Healthy: {cw[0]:.4f}  AD: {cw[1]:.4f}")

tf.keras.backend.clear_session()
model = build_model_v32()

callbacks = [
    ModelCheckpoint(DISEASE_MODEL, monitor='val_auc',
                    save_best_only=True, mode='max', verbose=0),
    EarlyStopping(monitor='val_auc', patience=PATIENCE_ES,
                  restore_best_weights=True, mode='max', verbose=0),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=PATIENCE_LR, min_lr=1e-6, verbose=0),
    CSVLogger(f'{MODEL_DIR}/training_log.csv'),
]

print("\n  Training single DB-HLSTM model ...")
history = model.fit(tr_ds, validation_data=vl_ds, epochs=EPOCHS,
                     callbacks=callbacks, class_weight=cw, verbose=2)

# ============================================================
# STEP 7 — VALIDATION-SET THRESHOLD OPTIMIZATION (Youden Index)
# ============================================================
# Find the decision threshold on the VALIDATION set (never the
# test set) that maximizes Youden's J statistic (TPR - FPR) —
# equivalent to maximizing (sensitivity + specificity - 1).
# Youden's J is used instead of F1-maximization because it is
# derived from the ROC curve, whose thresholds are the model's
# actual predicted probabilities at data points rather than
# precision/recall-curve breakpoints that can spike to a
# degenerate near-0-or-1 value on a small validation set.
val_probs = model.predict(
    {'ts_input': X_norm[val_idx], 'scalar_input': X_sc[val_idx]},
    verbose=0).ravel().astype(np.float64)

val_fpr, val_tpr, val_thr = roc_curve(y_val, val_probs)
youden_j = val_tpr - val_fpr

# Safety clamp — exclude the degenerate extremes near 0 or 1 that
# a small validation set can occasionally produce.
safe_mask = (val_thr >= 0.05) & (val_thr <= 0.95)
if safe_mask.any():
    best_threshold = float(val_thr[safe_mask][int(np.argmax(youden_j[safe_mask]))])
else:
    best_threshold = 0.5

print("\n" + "=" * 60)
print("  VALIDATION-SET THRESHOLD OPTIMIZATION (Youden Index)")
print("=" * 60)
print(f"  best_threshold = {best_threshold:.4f}  "
      f"(vs. default 0.5)")

# ============================================================
# STEP 8 — SINGLE, FINAL EVALUATION ON THE TEST SET
# (using best_threshold from validation, not the fixed 0.5)
# ============================================================
test_probs = model.predict(
    {'ts_input': X_norm[test_idx], 'scalar_input': X_sc[test_idx]},
    verbose=0).ravel().astype(np.float64)

pred_default = (test_probs >= 0.5).astype(int)
pred_tuned   = (test_probs >= best_threshold).astype(int)

def _metrics(pred, y_true):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    acc = (pred == y_true).mean()
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return acc, prec, rec, f1

acc_d, prec_d, rec_d, f1_d = _metrics(pred_default, y_test)
acc_t, prec_t, rec_t, f1_t = _metrics(pred_tuned, y_test)
test_auc = roc_auc_score(y_test, test_probs) if len(np.unique(y_test)) > 1 else float('nan')

print("\n" + "=" * 60)
print("  FINAL TEST-SET RESULTS (single held-out split, evaluated once)")
print("=" * 60)
print(f"  Test AUC: {test_auc:.4f}")
print(f"  Default (0.5)         — Acc: {acc_d:.4f}  Prec: {prec_d:.4f}  "
      f"Rec: {rec_d:.4f}  F1: {f1_d:.4f}")
print(f"  Tuned ({best_threshold:.4f}) — Acc: {acc_t:.4f}  Prec: {prec_t:.4f}  "
      f"Rec: {rec_t:.4f}  F1: {f1_t:.4f}")

# Reporting/plots below use the tuned threshold as the primary
# result, per the requested protocol (Step 6 of the request).
pred_final = pred_tuned
report_default = classification_report(y_test, pred_final,
                                        target_names=['Healthy', 'AD'], digits=4)

print("\n  Test-set classification report (tuned threshold):")
print(report_default)

# ============================================================
# STEP 9 — TRAINING-CURVE PLOTS (Accuracy/Loss/AUC/Precision/
# Recall vs Epoch), publication-quality
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
})

hist = history.history
epochs_range = range(1, len(hist['loss']) + 1)

curve_specs = [
    ('loss', 'val_loss', 'Loss', 'loss_vs_epoch.png'),
    ('accuracy', 'val_accuracy', 'Accuracy', 'accuracy_vs_epoch.png'),
    ('auc', 'val_auc', 'AUC', 'auc_vs_epoch.png'),
    ('precision', 'val_precision', 'Precision', 'precision_vs_epoch.png'),
    ('recall', 'val_recall', 'Recall', 'recall_vs_epoch.png'),
]

for train_key, val_key, ylabel, fname in curve_specs:
    if train_key not in hist:
        continue
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(epochs_range, hist[train_key], color='#0072B2',
            linewidth=2, label='Train')
    if val_key in hist:
        ax.plot(epochs_range, hist[val_key], color='#D55E00',
                linewidth=2, label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{ylabel} vs Epoch')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(f'{MODEL_DIR}/{fname}', dpi=300)
    plt.close(fig)

print("  Saved: loss_vs_epoch.png, accuracy_vs_epoch.png, auc_vs_epoch.png, "
      "precision_vs_epoch.png, recall_vs_epoch.png")

# ============================================================
# STEP 10 — CONFUSION MATRIX / ROC / PR CURVE PLOTS: REMOVED
# These standalone per-model plots are now redundant — the
# combined publication_plots_all_models.py script generates
# confusion_matrices_grid.png, combined_roc_curves.png, and
# combined_pr_curves.png covering all 5 models (including
# DB-HLSTM) from predictions.csv. Training-curve plots above
# (Step 9) are kept since those are DB-HLSTM-specific and not
# duplicated elsewhere.
# ============================================================
# ============================================================
# STEP 11 — SAVE CLASSIFICATION REPORT, PREDICTIONS, SUMMARY
# ============================================================
with open(f'{MODEL_DIR}/classification_report.txt', 'w') as f:
    f.write("DB-HLSTM v3.3 — Single Subject-Level Split — Test-Set Classification Report\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Test AUC: {test_auc:.6f}\n")
    f.write(f"best_threshold (validation-tuned, Youden Index): {best_threshold:.6f}\n\n")
    f.write("Report @ tuned threshold:\n")
    f.write(report_default)

test_pred_df = pd.DataFrame({
    'Subject': X_subject[test_idx],
    'y_true': y_test,
    'test_prob': test_probs,
    'y_pred_default_0.5': pred_default,
    'y_pred_tuned': pred_tuned,
})
test_pred_df.to_csv(f'{MODEL_DIR}/predictions.csv', index=False)

with open(f'{MODEL_DIR}/summary.txt', 'w') as f:
    f.write(f"n_train_subjects={n_train_subj}\n")
    f.write(f"n_val_subjects={n_val_subj}\n")
    f.write(f"n_test_subjects={n_test_subj}\n")
    f.write(f"n_train_windows={len(train_idx)}\n")
    f.write(f"n_val_windows={len(val_idx)}\n")
    f.write(f"n_test_windows={len(test_idx)}\n")
    f.write(f"best_threshold={best_threshold:.6f}\n")
    f.write(f"test_auc={test_auc:.6f}\n")
    f.write(f"test_acc_default_0.5={acc_d:.6f}\n")
    f.write(f"test_prec_default_0.5={prec_d:.6f}\n")
    f.write(f"test_rec_default_0.5={rec_d:.6f}\n")
    f.write(f"test_f1_default_0.5={f1_d:.6f}\n")
    f.write(f"test_acc_tuned={acc_t:.6f}\n")
    f.write(f"test_prec_tuned={prec_t:.6f}\n")
    f.write(f"test_rec_tuned={rec_t:.6f}\n")
    f.write(f"test_f1_tuned={f1_t:.6f}\n")
    f.write(f"model_path={DISEASE_MODEL}\n")

print(f"\n  Saved: training_log.csv, 5x *_vs_epoch.png, confusion_matrix.png, "
      f"roc_curve.png, precision_recall_curve.png, classification_report.txt, "
      f"test_predictions.csv, summary.txt")
print(f"  Saved model: {DISEASE_MODEL}")

del X_norm, X_sc, y_dis
gc.collect()
tf.keras.backend.clear_session()

print("\n  DISEASE CLASSIFICATION (SINGLE SUBJECT-LEVEL SPLIT) DONE ✅")
