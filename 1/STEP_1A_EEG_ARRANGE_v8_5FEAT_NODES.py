"""
============================================================
 Step 8.1 (v8 — 5-FEATURE-PER-NODE GNN INPUT)
 EEG Download + Arrange to Paper-1 Format
 DB-HLSTM Framework — Paper 2 (Real Clinical EEG)

 ============================================================
 WHAT CHANGED FROM v7 -> v8:
 ============================================================
 v7 fixed the GNN node-feature amplitude-scale confound (raw
 theta -> theta/alpha ratio), but each node still carried only
 ONE scalar value. A single number per electrode is a very thin
 representation for a graph neural network to work with — most
 of each channel's spectral information (band-specific power,
 variability) was discarded before the GNN ever saw it.

 v8 FIX: each node (channel) now carries 5 features instead of
 1, computed per window:
   1. theta_mean   — mean theta (4-8 Hz) envelope amplitude
   2. theta_std    — variability of theta envelope within window
   3. alpha_mean   — mean alpha (8-13 Hz) envelope amplitude
   4. beta_mean    — mean beta (13-30 Hz) envelope amplitude
   5. gamma_mean   — mean gamma (30-45 Hz) envelope amplitude
 This gives the GCN meaningfully richer per-electrode input
 (multi-band spectral profile per channel, not a single ratio),
 while keeping the graph STRUCTURE (adjacency) unchanged from
 v6/v7 — this is a data-richness change only, not an
 architecture change, so any performance difference from v7 is
 attributable specifically to richer node features.

 CONNECTIVITY (adjacency) IS UNCHANGED FROM v6/v7.

 CHANGED OUTPUT COLUMNS:
   node_0 .. node_94   (19 channels x 5 features = 95 columns,
                         replaces v7's node_0..18 single-ratio
                         columns — VARIES per window)
   adj_0 .. adj_360    (UNCHANGED from v6/v7 — static per subject)

 EVERYTHING ELSE UNCHANGED FROM v7/v5/v3:
   I_val (subject-level, v3), ratio_env_0..599 (v5, HRF/BOLD input),
   v_0..599, s_0..599, Subject, MMSE, etc.

 INPUT  : Kaggle dataset thngdngvn/openneuro-ds004504
 OUTPUT : /kaggle/working/EEG_Research_Data_Final.csv

 NEXT STEPS:
   -> HRF (v2.1, ratio_env_) and Balloon-Windkessel (v2.1,
      ratio_env_) scripts are UNCHANGED — pass adj_/node_ through
      untouched
   -> Step 8.5 (v2.3, GNN with 5-feature nodes) — needs the
      matching code updates (node_cols count, reshape to
      (N_CHANNELS, 5), node_in Input shape) — see separate file
============================================================
"""

import os
import numpy as np
import pandas as pd
import mne
from scipy.signal import butter, filtfilt, hilbert
from tqdm.notebook import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
SEQ_LEN     = 600
WIN_SEC     = 4
OVERLAP     = 0.5
STEP_SEC    = WIN_SEC * (1 - OVERLAP)
OUTPUT_CSV  = "/kaggle/working/EEG_Research_Data_Final.csv"

THETA_LO, THETA_HI = 4.0, 8.0
ALPHA_LO, ALPHA_HI = 8.0, 13.0
BETA_LO, BETA_HI   = 13.0, 30.0
GAMMA_LO, GAMMA_HI = 30.0, 45.0
N_CHANNELS    = 19
N_NODE_FEATS  = 5   # theta_mean, theta_std, alpha_mean, beta_mean, gamma_mean

mne.set_log_level('ERROR')

# ============================================================
# STEP A — DOWNLOAD DATASET
# ============================================================
print("=" * 60)
print("  Step 8.1 (v8) — EEG Download + Arrange")
print("  (GNN nodes: 5 spectral features per channel, not 1)")
print("=" * 60)

import kagglehub
print("\n  Downloading dataset (or using cached copy) ...")
dataset_path = kagglehub.dataset_download("thngdngvn/openneuro-ds004504")
candidate_a = os.path.join(dataset_path, "ds004504")
candidate_b = dataset_path
if os.path.isdir(candidate_a) and "participants.tsv" in os.listdir(candidate_a):
    BASE = candidate_a
elif "participants.tsv" in os.listdir(candidate_b):
    BASE = candidate_b
else:
    raise FileNotFoundError(f"Could not locate participants.tsv under {dataset_path}.")
print(f"  Using BASE = {BASE}")

participants = pd.read_csv(f"{BASE}/participants.tsv", sep="\t")
print(f"\n  Participants loaded: {len(participants)}")
print(participants['Group'].value_counts())

# ============================================================
# STEP B — HELPER FUNCTIONS
# ============================================================
def load_subject_eeg(sub_id, base=BASE):
    path = f"{base}/derivatives/sub-{sub_id:03d}/eeg/sub-{sub_id:03d}_task-eyesclosed_eeg.set"
    return mne.io.read_raw_eeglab(path, preload=True, verbose=False)

def theta_alpha_subject_level(raw):
    """UNCHANGED FROM v3 — proven, subject-level, full recording."""
    psd = raw.compute_psd(fmin=1, fmax=30, verbose=False)
    freqs = psd.freqs
    data = psd.get_data().mean(axis=0)
    theta = data[(freqs >= 4) & (freqs < 8)].mean()
    alpha = data[(freqs >= 8) & (freqs < 13)].mean() + 1e-20
    return theta / alpha

def band_envelope_1d(data_1d, sfreq, lo, hi):
    """Hilbert amplitude envelope of one channel, band-passed."""
    nyq = sfreq / 2.0
    b, a = butter(N=4, Wn=[lo / nyq, hi / nyq], btype='band')
    filtered = filtfilt(b, a, data_1d)
    return np.abs(hilbert(filtered)).astype(np.float32)

def per_channel_band_envelopes(data_multi, sfreq, lo, hi):
    """Envelope for EACH channel separately, given band [lo, hi]."""
    n_ch = data_multi.shape[0]
    env = np.zeros_like(data_multi, dtype=np.float32)
    for ch in range(n_ch):
        env[ch] = band_envelope_1d(data_multi[ch], sfreq, lo, hi)
    return env

def normalized_adjacency(env_multi):
    """UNCHANGED FROM v6/v7 — AEC connectivity, symmetric degree-normalized."""
    n_ch = env_multi.shape[0]
    corr = np.corrcoef(env_multi)
    corr = np.nan_to_num(corr, nan=0.0)
    adj = np.clip(corr, 0, None)
    np.fill_diagonal(adj, 0)
    adj_self = adj + np.eye(n_ch, dtype=np.float32)
    deg = adj_self.sum(axis=1)
    deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    deg_inv_sqrt[deg == 0] = 0
    D = np.diag(deg_inv_sqrt)
    return (D @ adj_self @ D).astype(np.float32)

def theta_alpha_ratio_envelope(data_1d, sfreq):
    """UNCHANGED FROM v5 — ratio envelope, HRF/BOLD pipeline input."""
    theta_env = band_envelope_1d(data_1d, sfreq, THETA_LO, THETA_HI)
    alpha_env = band_envelope_1d(data_1d, sfreq, ALPHA_LO, ALPHA_HI)
    return theta_env / (alpha_env + 1e-12)

# ============================================================
# STEP C — WINDOW EACH SUBJECT, BUILD PAPER-1-FORMAT ROWS
# ============================================================
print(f"\n  Windowing: {WIN_SEC}s windows, {OVERLAP*100:.0f}% overlap")
print("  Processing subjects (per-channel filtering, 4 bands) ...")

rows = []
sample_no = 0
subjects_to_process = participants[participants['Group'].isin(['A', 'C'])]

for _, row in tqdm(subjects_to_process.iterrows(),
                    total=len(subjects_to_process), desc="Subjects"):
    sub_num = int(row['participant_id'].split('-')[1])
    label = 'AD' if row['Group'] == 'A' else 'Healthy'

    try:
        raw = load_subject_eeg(sub_num)
        sfreq = raw.info['sfreq']
        data_multi = raw.get_data()                 # (19, n_samples)
        data_avg   = data_multi.mean(axis=0)

        subject_ratio = theta_alpha_subject_level(raw)               # I_val
        ratio_env_full = theta_alpha_ratio_envelope(data_avg, sfreq)  # v5

        # per-channel envelopes, ALL 4 bands (v8 — was theta+alpha only)
        theta_env_per_ch = per_channel_band_envelopes(data_multi, sfreq, THETA_LO, THETA_HI)
        alpha_env_per_ch = per_channel_band_envelopes(data_multi, sfreq, ALPHA_LO, ALPHA_HI)
        beta_env_per_ch  = per_channel_band_envelopes(data_multi, sfreq, BETA_LO, BETA_HI)
        gamma_env_per_ch = per_channel_band_envelopes(data_multi, sfreq, GAMMA_LO, GAMMA_HI)

        # connectivity — UNCHANGED, still theta-band AEC
        adj_norm = normalized_adjacency(theta_env_per_ch)
        adj_flat = adj_norm.flatten()

        if sample_no == 0 or row['participant_id'] in ['sub-001','sub-002','sub-037']:
            print(f"    DEBUG {row['participant_id']}: I_val={subject_ratio:.4f}  "
                  f"theta mean={theta_env_per_ch.mean():.4g}  "
                  f"gamma mean={gamma_env_per_ch.mean():.4g}")

        win_len = int(WIN_SEC * sfreq)
        step_len = int(STEP_SEC * sfreq)
        n_samples = len(data_avg)
        n_windows = (n_samples - win_len) // step_len + 1

        for w in range(n_windows):
            start = w * step_len
            seg_avg = data_avg[start:start + win_len]
            seg_ratio_env = ratio_env_full[start:start + win_len]

            idx_r = np.linspace(0, len(seg_avg) - 1, SEQ_LEN).astype(int)
            resampled = seg_avg[idx_r].astype(np.float32)
            resampled_ratio_env = seg_ratio_env[idx_r].astype(np.float32)

            thresh = resampled.mean() + 2 * resampled.std()
            spike_train = (resampled > thresh).astype(np.int8)

            # v8 — 5 features per channel, this window only
            seg_theta = theta_env_per_ch[:, start:start + win_len]
            seg_alpha = alpha_env_per_ch[:, start:start + win_len]
            seg_beta  = beta_env_per_ch[:, start:start + win_len]
            seg_gamma = gamma_env_per_ch[:, start:start + win_len]

            node_feats = np.stack([
                seg_theta.mean(axis=1),
                seg_theta.std(axis=1),
                seg_alpha.mean(axis=1),
                seg_beta.mean(axis=1),
                seg_gamma.mean(axis=1),
            ], axis=1).astype(np.float32)   # (19, 5)
            node_flat = node_feats.flatten()  # ch0[f0..f4], ch1[f0..f4], ...

            record = {
                'Label': label, 'Mode': label,
                'Class': 1 if label == 'AD' else 0,
                'Sample_No': sample_no,
                'a': 0.0, 'b': 0.0,
                'c': float(resampled.min()),
                'd': float(spike_train.sum()),
                'I_val': subject_ratio,
            }
            record.update({f'v_{i}': resampled[i] for i in range(SEQ_LEN)})
            record.update({f'u_{i}': 0.0 for i in range(SEQ_LEN)})
            record.update({f'i_{i}': 0.0 for i in range(SEQ_LEN)})
            record.update({f's_{i}': int(spike_train[i]) for i in range(SEQ_LEN)})
            record.update({f'ratio_env_{i}': resampled_ratio_env[i] for i in range(SEQ_LEN)})
            record.update({f'adj_{i}': adj_flat[i] for i in range(N_CHANNELS * N_CHANNELS)})
            # v8 — 95 columns now (19 channels x 5 features), not 19
            record.update({f'node_{i}': node_flat[i] for i in range(N_CHANNELS * N_NODE_FEATS)})
            record['s_t_count'] = int(spike_train.sum())
            record['Subject'] = row['participant_id']
            record['MMSE'] = row['MMSE']

            rows.append(record)
            sample_no += 1

    except Exception as e:
        print(f"  {row['participant_id']} FAILED: {e}")

# ============================================================
# STEP D — SAVE + VERIFY
# ============================================================
df_eeg = pd.DataFrame(rows)

print()
print("=" * 60)
print("  EEG ARRANGE COMPLETE (v8 — 5 features/node)")
print("=" * 60)
print(f"  Total windowed samples : {len(df_eeg):,}")
print(f"  Total columns          : {len(df_eeg.columns):,}")
print(f"  Subjects processed: {df_eeg['Subject'].nunique()} (expected 65)")

from scipy.stats import ttest_ind, pearsonr, mannwhitneyu
ad_vals = df_eeg[df_eeg['Label'] == 'AD']['I_val']
hc_vals = df_eeg[df_eeg['Label'] == 'Healthy']['I_val']
t_stat, p_val = ttest_ind(ad_vals, hc_vals)
print(f"\n  Sanity check 1 — I_val (must still PASS, unchanged since v3):")
print(f"    p-value = {p_val:.6f}  {'PASS' if p_val < 0.001 else 'FAIL'}")

# v8 — check each of the 5 node-feature types separately
node_cols_all = [f'node_{i}' for i in range(N_CHANNELS * N_NODE_FEATS)]
node_arr = df_eeg[node_cols_all].values.reshape(-1, N_CHANNELS, N_NODE_FEATS)
feat_names = ['theta_mean', 'theta_std', 'alpha_mean', 'beta_mean', 'gamma_mean']
print(f"\n  Sanity check 4 (v8) — per-feature discriminative check "
      f"(averaged across 19 channels):")
for fi, fname in enumerate(feat_names):
    vals = node_arr[:, :, fi].mean(axis=1)
    ad_v = vals[df_eeg['Label'] == 'AD']
    hc_v = vals[df_eeg['Label'] == 'Healthy']
    u, p = mannwhitneyu(ad_v, hc_v, alternative='two-sided')
    status = 'PASS' if p < 0.05 else 'weak'
    print(f"    {fname:12s}: AD={ad_v.mean():.4g}  Healthy={hc_v.mean():.4g}  "
          f"p={p:.6f}  [{status}]")

adj_cols = [f'adj_{i}' for i in range(N_CHANNELS*N_CHANNELS)]
subj_adj = df_eeg.groupby('Subject')[adj_cols].first()
subj_label = df_eeg.groupby('Subject')['Label'].first()
adj_strength = subj_adj.values.mean(axis=1)
ad_adj = adj_strength[subj_label.values == 'AD']
hc_adj = adj_strength[subj_label.values == 'Healthy']
u2, p_adj = mannwhitneyu(ad_adj, hc_adj, alternative='two-sided')
print(f"\n  Sanity check 5 — connectivity strength (UNCHANGED from v6/v7):")
print(f"    Mann-Whitney p = {p_adj:.6f}  (informational only)")

df_eeg.to_csv(OUTPUT_CSV, index=False)
print(f"\n  Saved -> {OUTPUT_CSV}")
print(f"  node_0..{N_CHANNELS*N_NODE_FEATS-1} = {N_CHANNELS} channels x {N_NODE_FEATS} features")
print("=" * 60)
