"""
============================================================
 Step 1.2 (v2.1 — ACCEPTS ratio_env_ (v7) OR theta_env_ (v5/v6))
 HRF Extraction  [KAGGLE VERSION]

 ============================================================
 WHAT CHANGED FROM v2 -> v2.1:
 ============================================================
 FIX — column-name mismatch with the v7 arrange script.
 STEP_8.1_EEG_ARRANGE_v7_RATIO_NODES.py (the current EEG-arrange
 script, used to add ratio-based GNN node features) renamed the
 theta/alpha ratio envelope column from theta_env_0..599 (v5/v6
 name) to ratio_env_0..599 (v7 name) — same signal, new name.
 This HRF script still hard-required theta_env_0..599, so it
 raised a false "columns missing" KeyError on any CSV coming out
 of the v7 pipeline even though the equivalent data (ratio_env_)
 was right there under a different name.

 Fix: accept EITHER name. Prefer ratio_env_ (current/v7 name) if
 present, else fall back to theta_env_ (older v5/v6 pipelines),
 and only raise if NEITHER is found.

 ============================================================
 WHAT CHANGED FROM v1 -> v2 (unchanged from before):
 ============================================================
 v1 convolved s(t) — a threshold-crossing spike train — with the
 3 HRF kernels. s(t) was shown to be non-discriminative for EEG
 (p=0.54), so every downstream HRF/BOLD result inherited that
 noise regardless of AD vs Healthy status.

 v2 convolves the theta/alpha RATIO envelope instead — the
 per-window signal produced by the EEG-arrange script (v5+).
 This is a continuous, physically meaningful, AD-relevant signal
 (validated discriminative, AD > Healthy), so the HRF output now
 carries real signal instead of noise.

 Convolves ratio_env(t) with 3 HRF kernels:
   h_c  (t) = Canonical HRF          -> y_c  = ratio_env(t) * h_c
   h_td (t) = Temporal Derivative    -> y_td = ratio_env(t) * h_td
   h_dd (t) = Dispersion Derivative  -> y_dd = ratio_env(t) * h_dd

 Based on: Friston et al. (1998) — SPM HRF model
   Canonical: h(t) = Gp(t) - C.Gu(t)
   a1=6, a2=16, b1=1, b2=1, C=1/6

 INPUT  : /kaggle/working/EEG_Research_Data_Final.csv
           (must contain ratio_env_0..599 [v7+] or theta_env_0..599
           [v5/v6] — run the EEG arrange script first)
 OUTPUT : /kaggle/working/EEG_HRF_Data.csv
   New columns: hrf_c_0..599, hrf_td_0..599, hrf_dd_0..599
============================================================
"""

import numpy as np
import pandas as pd
from scipy.special import gamma as Gamma
from scipy.signal import convolve
from tqdm.notebook import tqdm
import os, gc

# ── PATHS ───────────────────────────────────────────────────
CSV_IN  = "/kaggle/working/EEG_Research_Data_Final.csv"
CSV_OUT = "/kaggle/working/EEG_HRF_Data.csv"

SEQ_LEN = 600
h_step  = 0.5
t_ms    = np.arange(SEQ_LEN) * h_step
t_s     = t_ms / 1000.0

# ── HRF PARAMETERS (Friston 1998) ───────────────────────────
a1, a2 = 6.0, 16.0
b1, b2 = 1.0,  1.0
C       = 1.0 / 6.0

def h_canonical(t):
    t  = np.maximum(t, 1e-10)
    Gp = (t**(a1-1) * b1**a1 * np.exp(-b1*t)) / Gamma(a1)
    Gu = (t**(a2-1) * b2**a2 * np.exp(-b2*t)) / Gamma(a2)
    return Gp - C * Gu

def h_temporal(t):
    t  = np.maximum(t, 1e-10)
    Gp = (t**(a1-1) * b1**a1 * np.exp(-b1*t)) / Gamma(a1)
    Gu = (t**(a2-1) * b2**a2 * np.exp(-b2*t)) / Gamma(a2)
    return Gp*((a1-1)/t - b1) - C*Gu*((a2-1)/t - b2)

def h_dispersion(t):
    t  = np.maximum(t, 1e-10)
    Gp = (t**(a1-1) * b1**a1 * np.exp(-b1*t)) / Gamma(a1)
    return Gp * (a1/b1 - t)

kernel_c  = h_canonical(t_s)
kernel_td = h_temporal(t_s)
kernel_dd = h_dispersion(t_s)

for k in [kernel_c, kernel_td, kernel_dd]:
    k /= (np.max(np.abs(k)) + 1e-10)

print("=" * 60)
print("  Step 1.2 (v2.1) — HRF Extraction  [KAGGLE]")
print("  Input signal: theta/alpha ratio envelope (NOT s(t))")
print(f"  Input  : {CSV_IN}")
print(f"  Output : {CSV_OUT}")
print("=" * 60)
print(f"\n  HRF kernel peaks (ms):")
print(f"    Canonical  : {t_ms[np.argmax(kernel_c)]:.1f} ms")
print(f"    Temporal D : {t_ms[np.argmax(kernel_td)]:.1f} ms")
print(f"    Dispersion : {t_ms[np.argmax(kernel_dd)]:.1f} ms")

# ── LOAD CSV ─────────────────────────────────────────────────
print(f"\n  Loading CSV ...")
df = pd.read_csv(CSV_IN)
print(f"  Rows : {len(df):,}  |  Cols : {len(df.columns):,}")

# FIX — accept EITHER ratio_env_ (v7 pipeline, current) or
# theta_env_ (v5/v6 pipeline, older). Same signal, renamed in v7.
_ratio_env_cols = [f"ratio_env_{i}" for i in range(SEQ_LEN)]
_theta_env_cols = [f"theta_env_{i}" for i in range(SEQ_LEN)]

if all(c in df.columns for c in _ratio_env_cols):
    theta_env_cols = _ratio_env_cols
    print("  Using ratio_env_0..599 (v7 naming) as the input signal.")
elif all(c in df.columns for c in _theta_env_cols):
    theta_env_cols = _theta_env_cols
    print("  Using theta_env_0..599 (v5/v6 naming) as the input signal.")
else:
    missing_ratio = [c for c in _ratio_env_cols if c not in df.columns]
    missing_theta = [c for c in _theta_env_cols if c not in df.columns]
    raise KeyError(
        f"Neither ratio_env_0..599 nor theta_env_0..599 fully present in "
        f"the input CSV (missing {len(missing_ratio)}/{SEQ_LEN} of "
        f"ratio_env_, {len(missing_theta)}/{SEQ_LEN} of theta_env_, e.g. "
        f"{(missing_ratio or missing_theta)[:3]}). Run the EEG arrange "
        f"script first (v5+ for theta_env_, v7+ for ratio_env_) — the old "
        f"s(t)-only CSV is not compatible with this HRF script."
    )
print(f"  Ratio/theta envelope columns : OK")

# ── SCALE GLOBALLY (NOT per-row) ────────────────────────────
# IMPORTANT: per-window min-max normalization would rescale every
# window to the SAME [0,1] range, which would destroy the very
# amplitude differences between AD and Healthy that make this
# signal useful in the first place. Instead we apply a single
# GLOBAL scale factor (dataset-wide 99th percentile) so all
# windows share the same unit conversion, but relative amplitude
# differences between subjects/groups are fully preserved.
theta_env_raw = df[theta_env_cols].values.astype(np.float32)
global_scale = np.percentile(theta_env_raw, 99) + 1e-10
theta_env_norm = theta_env_raw / global_scale
print(f"  Global scale factor (99th pct): {global_scale:.4g}")

# ── CONVOLUTION ──────────────────────────────────────────────
print(f"\n  Convolving {len(df):,} envelope signals with 3 HRF kernels ...")

yc_all  = np.zeros((len(df), SEQ_LEN), dtype=np.float32)
ytd_all = np.zeros((len(df), SEQ_LEN), dtype=np.float32)
ydd_all = np.zeros((len(df), SEQ_LEN), dtype=np.float32)

for idx in tqdm(range(len(df)), desc="HRF Convolution"):
    x = theta_env_norm[idx]
    yc_all[idx]  = convolve(x, kernel_c,  mode='full')[:SEQ_LEN]
    ytd_all[idx] = convolve(x, kernel_td, mode='full')[:SEQ_LEN]
    ydd_all[idx] = convolve(x, kernel_dd, mode='full')[:SEQ_LEN]

# ── ADD COLUMNS ──────────────────────────────────────────────
print(f"\n  Adding 1,800 new columns ...")
for i in tqdm(range(SEQ_LEN), desc="Adding columns"):
    df[f"hrf_c_{i}"]  = yc_all[:, i]
    df[f"hrf_td_{i}"] = ytd_all[:, i]
    df[f"hrf_dd_{i}"] = ydd_all[:, i]

del yc_all, ytd_all, ydd_all; gc.collect()

# ── SANITY CHECK — HRF output should now be discriminative ────
from scipy.stats import mannwhitneyu
hrf_c_cols = [f"hrf_c_{i}" for i in range(SEQ_LEN)]
peak_amp = df[hrf_c_cols].values.max(axis=1)
ad_peak = peak_amp[df['Class'] == 1]
hc_peak = peak_amp[df['Class'] == 0]
u_stat, p_peak = mannwhitneyu(ad_peak, hc_peak, alternative='two-sided')
print(f"\n  Sanity check — Canonical HRF peak amplitude, AD vs Healthy:")
print(f"    AD mean peak     : {ad_peak.mean():.4g}")
print(f"    Healthy mean peak: {hc_peak.mean():.4g}")
print(f"    Mann-Whitney p   : {p_peak:.6f}")
print(f"    {'PASS — HRF output now carries real group signal' if p_peak < 0.05 else 'Peak alone not significant — check AUC/onset-latency/FWHM features too (see ranking script)'}")

# ── SAVE ─────────────────────────────────────────────────────
print(f"\n  Saving -> {CSV_OUT}")
df.to_csv(CSV_OUT, index=False)

print()
print("=" * 60)
print("  HRF EXTRACTION COMPLETE (v2.1 — ratio_env/theta_env input)")
print("=" * 60)
print(f"  Rows    : {len(df):,}")
print(f"  Columns : {len(df.columns):,}")
print(f"\n  EEG Label distribution:")
print(df['Label'].value_counts())
print(f"  Output  : {CSV_OUT}")
print("=" * 60)
