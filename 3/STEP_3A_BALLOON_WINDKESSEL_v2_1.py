"""
============================================================
 Step 8.3 (v2.1 — ACCEPTS ratio_env_ (v7) OR theta_env_ (v5/v6))
 Balloon-Windkessel BOLD Model  [KAGGLE VERSION]

 ============================================================
 WHAT CHANGED FROM v2 -> v2.1:
 ============================================================
 FIX — column-name mismatch with the v7 arrange script.
 STEP_8.1_EEG_ARRANGE_v7_RATIO_NODES.py (the current EEG-arrange
 script, used to add ratio-based GNN node features) renamed the
 theta/alpha ratio envelope column from theta_env_0..599 (v5/v6
 name) to ratio_env_0..599 (v7 name) — same signal, new name.
 This BW script still hard-required theta_env_0..599, so it
 raised a false "columns missing" KeyError on any CSV coming out
 of the v7 pipeline (via the v2 HRF script) even though the
 equivalent data (ratio_env_) was right there under a different
 name.

 Fix: accept EITHER name. Prefer ratio_env_ (current/v7 name) if
 present, else fall back to theta_env_ (older v5/v6 pipelines),
 and only raise if NEITHER is found.

 ============================================================
 WHAT CHANGED FROM v1 -> v2 (unchanged from before):
 ============================================================
 v1 fed s(t) — the non-discriminative threshold-crossing spike
 train (p=0.54) — directly into the Balloon-Windkessel ODE.
 Even after the HRF script was fixed to use the ratio envelope,
 this BW script was still silently using the OLD s(t) signal, so
 the BOLD output (and everything trained on it) still carried
 the old noise regardless of AD vs Healthy status.

 v2 fix: BW now takes the theta/alpha ratio envelope as its
 neural-drive input — the SAME per-window signal that the HRF
 script uses.

 IMPORTANT — why the ratio envelope, not hrf_c (not a sequential
 cascade): In the original Paper 1 architecture, HRF and
 Balloon-Windkessel are PARALLEL branches that both take the same
 raw neural-drive signal as input independently:

     drive(t) -> HRF convolution      -> hrf_c/td/dd  (branch 1)
     drive(t) -> Balloon-Windkessel   -> cbf/cbv/dhb/bold (branch 2)

 NOT a sequential cascade (drive -> HRF -> BW). Feeding hrf_c into
 BW would change this to a cascade and break architectural
 consistency with Paper 1's "identical pipeline" claim. The
 correct fix keeps both branches parallel — the ratio envelope
 simply replaces s(t) as the shared input to BOTH branches.

 STATE VARIABLES:
   s = vasodilatory signal
   f = normalized cerebral blood flow (CBF)
   v = normalized cerebral blood volume (CBV)
   q = normalized deoxyhemoglobin content (dHb)

 BOLD SIGNAL:
   BOLD(t) = V0 . [k1(1-q) + k2(1-q/v) + k3(1-v)]

 PARAMETERS (from literature, UNCHANGED):
   kappa=0.65, gamma=0.41, tau=0.98, alpha=0.32
   E0=0.40, V0=0.02
   k1=7.E0=2.80, k2=2.0, k3=2.E0-0.2=0.60
   dt=0.05s -> 600 steps x 0.05 = 30 seconds

 INPUT  : /kaggle/working/EEG_HRF_Data.csv
           (must contain ratio_env_0..599 [v7+] or theta_env_0..599
           [v5/v6] — output of the HRF script)
 OUTPUT : /kaggle/working/EEG_BOLD_Data.csv
   New columns: cbf_0..599, cbv_0..599, dhb_0..599, bold_0..599
============================================================
"""

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import os, gc

CSV_IN  = "/kaggle/working/EEG_HRF_Data.csv"
CSV_OUT = "/kaggle/working/EEG_BOLD_Data.csv"

SEQ_LEN = 600

# ── BALLOON-WINDKESSEL PARAMETERS (unchanged) ───────────────
kappa = 0.65    # signal decay rate
gamma = 0.41    # flow-dependent elimination
tau   = 0.98    # hemodynamic transit time (s)
alpha = 0.32    # Grubb's exponent
E0    = 0.40    # resting oxygen extraction fraction
V0    = 0.02    # resting blood volume fraction
k1    = 7.0 * E0           # 2.80
k2    = 2.0                # 2.00
k3    = 2.0 * E0 - 0.2    # 0.60

dt = 0.05   # 50 ms per step -> 30 seconds total

# Initial state: [s, f, v, q] = [0, 1, 1, 1]
S0, F0, V0_init, Q0 = 0.0, 1.0, 1.0, 1.0


def odes(state, u):
    """Balloon-Windkessel differential equations. (unchanged)"""
    s, f, v, q = state
    f    = max(f, 1e-6); v = max(v, 1e-6); q = max(q, 1e-6)
    fout = max(v, 1e-6) ** (1.0 / alpha)
    E    = 1.0 - (1.0 - E0) ** (1.0 / max(f, 1e-6))
    ds   = u - s/kappa - (f-1.0)/gamma
    df   = s
    dv   = (f - fout) / tau
    dq   = (f*E/E0 - fout*q/v) / tau
    return np.array([ds, df, dv, dq])


def rk4(state, u):
    """4th-order Runge-Kutta integration step. (unchanged)"""
    k1_ = odes(state,            u)
    k2_ = odes(state + dt/2*k1_, u)
    k3_ = odes(state + dt/2*k2_, u)
    k4_ = odes(state + dt  *k3_, u)
    return state + (dt/6.0)*(k1_ + 2*k2_ + 2*k3_ + k4_)


def run_balloon(u_arr):
    """Simulate Balloon-Windkessel model for one sample. (unchanged)"""
    state = np.array([S0, F0, V0_init, Q0], dtype=float)
    cbf_out  = np.zeros(SEQ_LEN)
    cbv_out  = np.zeros(SEQ_LEN)
    dhb_out  = np.zeros(SEQ_LEN)
    bold_out = np.zeros(SEQ_LEN)

    for t in range(SEQ_LEN):
        state    = rk4(state, float(u_arr[t]))
        state[1] = max(state[1], 0.1)  # f floor
        state[2] = max(state[2], 0.1)  # v floor
        state[3] = max(state[3], 0.01) # q floor
        s,f,v,q  = state
        cbf_out[t]  = f
        cbv_out[t]  = v
        dhb_out[t]  = q
        bold_out[t] = V0*(k1*(1-q) + k2*(1-q/max(v,1e-6)) + k3*(1-v))

    return cbf_out, cbv_out, dhb_out, bold_out


print("=" * 60)
print("  Step 8.3 (v2.1) — Balloon-Windkessel  [KAGGLE]")
print("  Input signal: theta/alpha ratio envelope (NOT s(t))")
print(f"  dt={dt}s -> {SEQ_LEN*dt:.0f}s total")
print(f"  kappa={kappa}  gamma={gamma}  tau={tau}")
print(f"  alpha={alpha}  E0={E0}  V0={V0}")
print(f"  k1={k1:.2f}  k2={k2:.2f}  k3={k3:.2f}")
print("=" * 60)

print(f"\n  Loading: {CSV_IN}")
df = pd.read_csv(CSV_IN)
print(f"  Rows : {len(df):,}  |  Cols : {len(df.columns):,}")

# FIX — accept EITHER ratio_env_ (v7 pipeline, current) or
# theta_env_ (v5/v6 pipeline, older). Same signal, renamed in v7.
# NOT s_cols (old, non-discriminative) and NOT hrf_c_cols (would
# turn the parallel HRF/BW branches into a sequential cascade).
_ratio_env_cols = [f"ratio_env_{i}" for i in range(SEQ_LEN)]
_theta_env_cols = [f"theta_env_{i}" for i in range(SEQ_LEN)]

if all(c in df.columns for c in _ratio_env_cols):
    theta_env_cols = _ratio_env_cols
    print("  Using ratio_env_0..599 (v7 naming) as the neural-drive input.")
elif all(c in df.columns for c in _theta_env_cols):
    theta_env_cols = _theta_env_cols
    print("  Using theta_env_0..599 (v5/v6 naming) as the neural-drive input.")
else:
    missing_ratio = [c for c in _ratio_env_cols if c not in df.columns]
    missing_theta = [c for c in _theta_env_cols if c not in df.columns]
    raise KeyError(
        f"Neither ratio_env_0..599 nor theta_env_0..599 fully present in "
        f"the input CSV (missing {len(missing_ratio)}/{SEQ_LEN} of "
        f"ratio_env_, {len(missing_theta)}/{SEQ_LEN} of theta_env_, e.g. "
        f"{(missing_ratio or missing_theta)[:3]}). This CSV must be the "
        f"output of the HRF extraction script (v2.1+), which itself "
        f"requires the EEG arrange script's ratio_env_/theta_env_ "
        f"columns. Running this script on an older, s(t)-only HRF CSV "
        f"would silently reintroduce the non-discriminative signal into "
        f"BOLD."
    )
print(f"  Ratio/theta envelope columns : OK")

# Run model
all_cbf, all_cbv, all_dhb, all_bold = [], [], [], []
print(f"\n  Running RK4 on {len(df):,} samples ...")

for _, row in tqdm(df.iterrows(), total=len(df), desc="Balloon-Windkessel"):
    cbf,cbv,dhb,bold = run_balloon(row[theta_env_cols].values.astype(float))
    all_cbf.append(cbf); all_cbv.append(cbv)
    all_dhb.append(dhb); all_bold.append(bold)

all_cbf  = np.array(all_cbf,  dtype=np.float32)
all_cbv  = np.array(all_cbv,  dtype=np.float32)
all_dhb  = np.array(all_dhb,  dtype=np.float32)
all_bold = np.array(all_bold, dtype=np.float32)

print(f"\n  Adding 2,400 columns ...")
for i in range(SEQ_LEN):
    df[f"cbf_{i}"]  = all_cbf[:, i]
    df[f"cbv_{i}"]  = all_cbv[:, i]
    df[f"dhb_{i}"]  = all_dhb[:, i]
    df[f"bold_{i}"] = all_bold[:, i]

print(f"\n  Saving -> {CSV_OUT}")
df.to_csv(CSV_OUT, index=False)

healthy_mask = df['Class'].values==0
ad_mask      = df['Class'].values==1
healthy_peak = all_bold[healthy_mask].max(axis=1).mean()
ad_peak      = all_bold[ad_mask].max(axis=1).mean()

# Sanity check — BOLD output should now be discriminative, since
# it is now driven by the ratio envelope instead of the old s(t)
from scipy.stats import mannwhitneyu
peak_all = all_bold.max(axis=1)
u_stat, p_bold = mannwhitneyu(peak_all[ad_mask], peak_all[healthy_mask],
                               alternative='two-sided')

print()
print("=" * 60)
print("  BOLD GENERATION COMPLETE (v2.1 — ratio_env/theta_env input)")
print("=" * 60)
print(f"  Rows    : {len(df):,}   (expected 26,526)")
print(f"  Columns : {len(df.columns):,}")
print(f"  BOLD max       : {all_bold.max():.6f}")
print(f"  Healthy peak   : {healthy_peak:.6f}")
print(f"  AD peak        : {ad_peak:.6f}")
print(f"\n  Sanity check — BOLD peak amplitude, AD vs Healthy:")
print(f"    Mann-Whitney p-value : {p_bold:.6f}")
print(f"    {'PASS — BOLD output now carries real group signal' if p_bold < 0.05 else 'CHECK — not significant, investigate scaling'}")
print(f"  Output  : {CSV_OUT}")
print("=" * 60)
