"""
============================================================
 EEG Results Plotter — FINAL VERSION
 Paper 2 — Miltiadous EEG (ds004504), DB-HLSTM Framework

 ALL DESIGN DECISIONS FINALIZED IN THIS SCRIPT:
   - Background: very light blue (#eef5fc) throughout, all figures
   - Plot 1  = Option C: PSD overlay (Healthy vs AD), theta band
               shaded GOLD (#f4b400), alpha band shaded PURPLE
               (#7b2d8e), curves in the house blue/red
   - Plot 2  = Option H (finalized): large/clear boxplot + jittered
               points (top) with Mann-Whitney p-value + Cohen's d
               significance bracket, KDE density with group-mean
               dashed lines (bottom)
   - Plot 3  = COMBO (H + F): top panel = scatter + regression line
               + stats box (r, p, n) + legend; bottom panel =
               binned MMSE clinical-severity boxplot
               (Severe <18 / Mild 18-24 / Normal 24-30)
   - Plot 4  = class & subject distribution (pie + bar) — structure
               unchanged, polished styling
   - Plot 5  = multi-sample waveform overlay — structure unchanged,
               polished styling
   - Plot 6  = per-subject mean ratio, sorted lollipop — structure
               unchanged, polished styling
   - All theta/alpha text uses matplotlib mathtext
     ($\\theta$/$\\alpha$) so Greek letters always render correctly
   - dpi = 400 throughout for print-ready quality

 INPUT:  /kaggle/working/EEG_Research_Data_Final.csv
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, pearsonr, linregress, mannwhitneyu
from scipy.signal import welch
import warnings
warnings.filterwarnings('ignore')

# ── GLOBAL STYLE ──────────────────────────────────────────
BG_COLOR = "#eef5fc"   # very light blue — house background, ALL figures

plt.rcParams.update({
    'figure.facecolor':  BG_COLOR,
    'axes.facecolor':    BG_COLOR,
    'axes.edgecolor':    '#222222',
    'axes.linewidth':    1.2,
    'axes.labelweight':  'bold',
    'axes.labelcolor':   '#1a1a1a',
    'text.color':        '#1a1a1a',
    'xtick.color':       '#1a1a1a',
    'ytick.color':       '#1a1a1a',
    'font.size':          12,
    'mathtext.fontset':  'stix',
    'savefig.facecolor': BG_COLOR,
})

# ── CONFIG ──────────────────────────────────────────────────
CSV_PATH = "/kaggle/working/EEG_Research_Data_Final.csv"
SEQ_LEN  = 600
WIN_SEC  = 4
SFREQ_EQUIV = SEQ_LEN / WIN_SEC
T        = np.linspace(0, WIN_SEC, SEQ_LEN)
DPI      = 400

# ── COLORS (finalized) ───────────────────────────────────────
COL_HC     = "#1558b0"   # Healthy — royal blue
COL_AD     = "#c22032"   # AD      — deep crimson red
COL_THETA  = "#f4b400"   # theta band — gold
COL_ALPHA  = "#7b2d8e"   # alpha band — purple
COL_FIT    = "#444444"   # regression / fit lines
COL_GRID   = "#b0b0b0"

V_COLS = [f"v_{i}" for i in range(SEQ_LEN)]

def style_axis(ax):
    ax.grid(True, alpha=0.30, color=COL_GRID, linewidth=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

print("=" * 60)
print("  Loading CSV...")
print("=" * 60)

df = pd.read_csv(CSV_PATH)
print(f"  Shape     : {df.shape}")
print(f"  Labels    : {sorted(df['Label'].unique())}")
print(f"  Subjects  : {df['Subject'].nunique()}")
print()

# ────────────────────────────────────────────────────────────
#  PLOT 1 — PSD Overlay (Option C), theta=gold, alpha=purple
# ────────────────────────────────────────────────────────────
print("  Plot 1: PSD overlay, theta/alpha bands shaded...")

hc_row = df[df['Label'] == 'Healthy'].iloc[0]
ad_row = df[df['Label'] == 'AD'].iloc[0]
v_hc = hc_row[V_COLS].values.astype(float)
v_ad = ad_row[V_COLS].values.astype(float)
ival_hc, ival_ad = hc_row['I_val'], ad_row['I_val']

f_hc, p_hc = welch(v_hc, fs=SFREQ_EQUIV, nperseg=min(len(v_hc), 256))
f_ad, p_ad = welch(v_ad, fs=SFREQ_EQUIV, nperseg=min(len(v_ad), 256))

fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(f_hc, p_hc, color=COL_HC, lw=2.4,
        label=rf'Healthy ($\theta/\alpha$={ival_hc:.2f})', zorder=3)
ax.plot(f_ad, p_ad, color=COL_AD, lw=2.4,
        label=rf'AD ($\theta/\alpha$={ival_ad:.2f})', zorder=3)
ax.axvspan(4, 8, color=COL_THETA, alpha=0.30, label=r'$\theta$ band (4–8 Hz)', zorder=1)
ax.axvspan(8, 13, color=COL_ALPHA, alpha=0.30, label=r'$\alpha$ band (8–13 Hz)', zorder=1)
ax.set_xlim(0, 20)
ax.set_xlabel("Frequency (Hz)", fontsize=14)
ax.set_ylabel("Power Spectral Density", fontsize=14)
ax.set_title(r"Power Spectral Density — AD vs Healthy ($\theta/\alpha$ Bands Highlighted)",
             fontsize=15, fontweight='bold', pad=14)
ax.tick_params(labelsize=12)
ax.legend(fontsize=11.5, framealpha=0.95, loc='upper right')
style_axis(ax)

plt.tight_layout()
plt.savefig("/kaggle/working/eeg_plot1_psd_overlay.png", dpi=DPI, bbox_inches='tight')
plt.show()
print("  Saved: eeg_plot1_psd_overlay.png")

# ────────────────────────────────────────────────────────────
#  PLOT 2 — Boxplot(large/clear) + Points + KDE + Stats (Option H)
# ────────────────────────────────────────────────────────────
print("  Plot 2: theta/alpha ratio — box, points, KDE, stats...")

subj_level = df.groupby('Subject').agg(I_val=('I_val','first'), Label=('Label','first')).reset_index()
hc_vals = subj_level[subj_level['Label'] == 'Healthy']['I_val'].values
ad_vals = subj_level[subj_level['Label'] == 'AD']['I_val'].values

u_stat, p_val = mannwhitneyu(hc_vals, ad_vals, alternative='two-sided')
pooled_std = np.sqrt(((len(hc_vals)-1)*hc_vals.std()**2 + (len(ad_vals)-1)*ad_vals.std()**2)
                      / (len(hc_vals) + len(ad_vals) - 2))
cohens_d = (ad_vals.mean() - hc_vals.mean()) / pooled_std

fig = plt.figure(figsize=(12, 10.5))
gs = gridspec.GridSpec(2, 1, height_ratios=[1.15, 1], hspace=0.32)

# ── Top: large boxplot + jittered points ──────────────────
ax = fig.add_subplot(gs[0])
bp = ax.boxplot([hc_vals, ad_vals], positions=[0, 1], widths=0.42, patch_artist=True,
                showfliers=False, zorder=2,
                medianprops=dict(color='black', lw=2.4),
                whiskerprops=dict(color='#333333', lw=1.6),
                capprops=dict(color='#333333', lw=1.6))
for patch, c in zip(bp['boxes'], [COL_HC, COL_AD]):
    patch.set_facecolor(c)
    patch.set_alpha(0.40)
    patch.set_edgecolor(c)
    patch.set_linewidth(2.2)

rng = np.random.default_rng(42)
for pos, vals, c in zip([0, 1], [hc_vals, ad_vals], [COL_HC, COL_AD]):
    jitter = rng.uniform(-0.14, 0.14, size=len(vals))
    ax.scatter(pos + jitter, vals, s=32, color=c, alpha=0.6,
               edgecolor='#000000', linewidth=0.4, zorder=1)

y_max = max(hc_vals.max(), ad_vals.max())
y_bar = y_max * 1.08
ax.plot([0, 0, 1, 1], [y_bar, y_bar*1.02, y_bar*1.02, y_bar], color='black', lw=1.6)
sig = 'p < 0.001' if p_val < 0.001 else f'p = {p_val:.3g}'
ax.text(0.5, y_bar*1.035, f"{sig}   (Cohen's d = {cohens_d:.2f})",
        ha='center', fontsize=13, fontweight='bold')

ax.set_xticks([0, 1])
ax.set_xticklabels([f'Healthy\n(n={len(hc_vals):,})', f'AD\n(n={len(ad_vals):,})'],
                    fontsize=14, fontweight='bold')
ax.set_ylabel(r"$\theta/\alpha$ Ratio", fontsize=15)
ax.set_title(r"$\theta/\alpha$ Ratio by Group — Distribution & Significance",
             fontsize=17, fontweight='bold', pad=14)
ax.tick_params(axis='y', labelsize=12)
ax.set_ylim(top=y_bar*1.12)
style_axis(ax)

legend_elements = [
    Patch(facecolor=COL_HC, alpha=0.5, edgecolor=COL_HC, label='Healthy'),
    Patch(facecolor=COL_AD, alpha=0.5, edgecolor=COL_AD, label='AD'),
]
ax.legend(handles=legend_elements, fontsize=12.5, loc='upper left', framealpha=0.95)

# ── Bottom: KDE density curve ──────────────────────────────
ax2 = fig.add_subplot(gs[1])
lo = min(hc_vals.min(), ad_vals.min())
hi = np.percentile(np.concatenate([hc_vals, ad_vals]), 99)
ratio_range = np.linspace(lo, hi, 400)
kde_hc = gaussian_kde(hc_vals)(ratio_range)
kde_ad = gaussian_kde(ad_vals)(ratio_range)

ax2.plot(ratio_range, kde_hc, color=COL_HC, lw=2.8, label=f'Healthy (n={len(hc_vals):,})')
ax2.fill_between(ratio_range, kde_hc, color=COL_HC, alpha=0.28)
ax2.plot(ratio_range, kde_ad, color=COL_AD, lw=2.8, label=f'AD (n={len(ad_vals):,})')
ax2.fill_between(ratio_range, kde_ad, color=COL_AD, alpha=0.28)
ax2.axvline(hc_vals.mean(), color=COL_HC, lw=1.8, ls='--', alpha=0.85)
ax2.axvline(ad_vals.mean(), color=COL_AD, lw=1.8, ls='--', alpha=0.85)

ax2.set_xlabel(r"$\theta/\alpha$ Ratio", fontsize=15)
ax2.set_ylabel("Density", fontsize=15)
ax2.set_title(r"$\theta/\alpha$ Ratio Density (KDE) — Dashed Lines = Group Means",
              fontsize=17, fontweight='bold', pad=14)
ax2.tick_params(labelsize=12)
ax2.legend(fontsize=13, framealpha=0.95)
style_axis(ax2)

plt.savefig("/kaggle/working/eeg_plot2_ratio_distribution.png", dpi=DPI, bbox_inches='tight')
plt.show()
print(f"  Saved: eeg_plot2_ratio_distribution.png  (U-test p={p_val:.3g}, d={cohens_d:.2f})")

# ────────────────────────────────────────────────────────────
#  PLOT 3 — COMBO H+F: scatter+regression+stats (top),
#           binned MMSE severity boxplot (bottom)
# ────────────────────────────────────────────────────────────
print("  Plot 3: MMSE vs theta/alpha — combo (scatter + severity bins)...")

if 'MMSE' in df.columns:
    sub_df = df.dropna(subset=['MMSE', 'I_val'])
    subj_avg = sub_df.groupby('Subject').agg(
        MMSE=('MMSE', 'first'), I_val=('I_val', 'mean'), Label=('Label', 'first')
    ).reset_index()

    r, p = pearsonr(subj_avg['MMSE'], subj_avg['I_val'])
    slope, intercept, *_ = linregress(subj_avg['MMSE'], subj_avg['I_val'])
    x_fit = np.linspace(subj_avg['MMSE'].min(), subj_avg['MMSE'].max(), 100)
    colors = subj_avg['Label'].map({'Healthy': COL_HC, 'AD': COL_AD})

    fig = plt.figure(figsize=(10.5, 11))
    gs3 = gridspec.GridSpec(2, 1, height_ratios=[1.2, 1], hspace=0.38)

    # ── Top: scatter + regression + stats box + legend ─────
    ax1 = fig.add_subplot(gs3[0])
    ax1.scatter(subj_avg['MMSE'], subj_avg['I_val'], c=colors, s=110, edgecolor='black',
                linewidth=1.1, alpha=0.85, zorder=3)
    ax1.plot(x_fit, slope*x_fit+intercept, color=COL_FIT, lw=2.4, ls='--', zorder=2)
    stats_txt = f"r = {r:.3f}\np = {p:.2g}\nn = {len(subj_avg)}"
    ax1.text(0.04, 0.06, stats_txt, transform=ax1.transAxes, fontsize=12, va='bottom',
             bbox=dict(boxstyle='round,pad=0.45', facecolor='white', edgecolor='#333333', alpha=0.95))
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COL_HC, markeredgecolor='black', markersize=10, label='Healthy'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COL_AD, markeredgecolor='black', markersize=10, label='AD'),
    ]
    ax1.legend(handles=legend_elements, fontsize=11.5, loc='upper right', framealpha=0.95)
    ax1.set_xlabel("MMSE Score", fontsize=14)
    ax1.set_ylabel(r"Mean $\theta/\alpha$ Ratio", fontsize=14)
    ax1.set_title(r"MMSE vs $\theta/\alpha$ Ratio — Continuous Relationship",
                  fontsize=16, fontweight='bold', pad=12)
    ax1.tick_params(labelsize=11.5)
    style_axis(ax1)

    # ── Bottom: binned MMSE severity boxplot ────────────────
    ax2 = fig.add_subplot(gs3[1])
    bins = [0, 18, 24, 30]
    bin_labels = ['Severe\n(<18)', 'Mild\n(18-24)', 'Normal\n(24-30)']
    subj_avg['bin'] = pd.cut(subj_avg['MMSE'], bins=bins, labels=bin_labels)
    box_data = [subj_avg[subj_avg['bin'] == b]['I_val'].dropna().values for b in bin_labels]
    bp = ax2.boxplot(box_data, patch_artist=True, widths=0.5, showfliers=False,
                      medianprops=dict(color='black', lw=2))
    bin_colors = [COL_AD, "#e0a458", COL_HC]
    for patch, c in zip(bp['boxes'], bin_colors):
        patch.set_facecolor(c); patch.set_alpha(0.4); patch.set_edgecolor(c); patch.set_linewidth(1.8)
    rng2 = np.random.default_rng(3)
    for i, vals in enumerate(box_data):
        jitter = rng2.uniform(-0.08, 0.08, size=len(vals))
        ax2.scatter(np.full(len(vals), i+1)+jitter, vals, s=32, color=bin_colors[i],
                    edgecolor='black', linewidth=0.4, alpha=0.75, zorder=3)
    ax2.set_xticklabels(bin_labels, fontsize=12.5, fontweight='bold')
    ax2.set_ylabel(r"Mean $\theta/\alpha$ Ratio", fontsize=14)
    ax2.set_title("MMSE Severity Groups (Clinical Bins)", fontsize=16, fontweight='bold', pad=12)
    ax2.tick_params(labelsize=11.5)
    style_axis(ax2)

    plt.savefig("/kaggle/working/eeg_plot3_mmse_correlation.png", dpi=DPI, bbox_inches='tight')
    plt.show()
    print(f"  Saved: eeg_plot3_mmse_correlation.png  (r={r:.3f}, p={p:.2g})")
else:
    print("  MMSE column not found — skipping Plot 3.")

# ────────────────────────────────────────────────────────────
#  PLOT 4 — Class & Subject Distribution
# ────────────────────────────────────────────────────────────
print("  Plot 4: class / subject distribution...")

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle("Dataset Composition — AD vs Healthy", fontsize=17, fontweight='bold')

counts = df['Label'].value_counts()
sizes  = [counts.get('Healthy', 0), counts.get('AD', 0)]
labels_pie = [f"Healthy\n{sizes[0]:,}", f"AD\n{sizes[1]:,}"]
wedges, texts, autotexts = axes[0].pie(
    sizes, labels=labels_pie, colors=[COL_HC, COL_AD],
    autopct='%1.1f%%', startangle=90,
    textprops={'fontsize': 12},
    wedgeprops={'edgecolor': 'white', 'lw': 1.5}
)
for at in autotexts:
    at.set_fontsize(12)
    at.set_fontweight('bold')
    at.set_color('white')
axes[0].set_title("Windowed Samples (4 s epochs)", fontsize=14, fontweight='bold')

subj_counts = df.drop_duplicates('Subject')['Label'].value_counts()
bar_labels = ['Healthy', 'AD']
bar_vals = [subj_counts.get(g, 0) for g in bar_labels]
bars = axes[1].bar(bar_labels, bar_vals, color=[COL_HC, COL_AD],
                    edgecolor='#000000', linewidth=1.2, width=0.55)
for b, val in zip(bars, bar_vals):
    axes[1].text(b.get_x() + b.get_width() / 2, val + 0.3, str(val),
                 ha='center', fontsize=14, fontweight='bold')
axes[1].set_ylabel("Number of Subjects", fontsize=14)
axes[1].set_title("Subject-Level Counts", fontsize=14, fontweight='bold')
axes[1].tick_params(labelsize=12)
style_axis(axes[1])

plt.tight_layout()
plt.savefig("/kaggle/working/eeg_plot4_class_distribution.png", dpi=DPI, bbox_inches='tight')
plt.show()
print("  Saved: eeg_plot4_class_distribution.png")

# ────────────────────────────────────────────────────────────
#  PLOT 5 — Multi-Sample Overlay
# ────────────────────────────────────────────────────────────
print("  Plot 5: multi-sample overlay...")

fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
fig.suptitle("5-Window Overlay — Healthy vs AD", fontsize=16, fontweight='bold')

for ax, (lbl, color, mtype) in zip(axes, [('Healthy', COL_HC, 'Healthy'),
                                            ('AD', COL_AD, 'AD')]):
    sub = df[df['Label'] == lbl].head(5)
    for i, (_, row) in enumerate(sub.iterrows()):
        v = row[V_COLS].values.astype(float)
        ax.plot(T, v, color=color, lw=1.6, alpha=0.45 + i * 0.11,
                label=f'Window {i}' if i == 0 else None)
    ax.set_title(f"{mtype} — 5 windows overlay", fontsize=14, fontweight='bold', color=color)
    ax.set_xlabel("Time (s)", fontsize=13)
    ax.set_ylabel("Amplitude ($\\mu V$)", fontsize=13)
    ax.set_xlim(0, WIN_SEC)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=10.5, framealpha=0.95)
    style_axis(ax)

plt.tight_layout()
plt.savefig("/kaggle/working/eeg_plot5_multisample_overlay.png", dpi=DPI, bbox_inches='tight')
plt.show()
print("  Saved: eeg_plot5_multisample_overlay.png")

# ────────────────────────────────────────────────────────────
#  PLOT 6 — Per-Subject Mean Theta/Alpha (sorted lollipop)
# ────────────────────────────────────────────────────────────
print("  Plot 6: per-subject mean ratio (sorted)...")

subj_stats = df.groupby('Subject').agg(
    I_val=('I_val', 'mean'),
    Label=('Label', 'first')
).reset_index().sort_values('I_val')

fig, ax = plt.subplots(figsize=(13, 6.8))
x_pos = np.arange(len(subj_stats))
colors = subj_stats['Label'].map({'Healthy': COL_HC, 'AD': COL_AD})

ax.vlines(x_pos, 0, subj_stats['I_val'], color=colors, lw=1.8, alpha=0.8)
ax.scatter(x_pos, subj_stats['I_val'], c=colors, s=65,
           edgecolor='#000000', linewidth=0.8, zorder=3)

ax.set_xticks([])
ax.set_xlabel(f"Subject (sorted, n={len(subj_stats)})", fontsize=15)
ax.set_ylabel(r"Mean $\theta/\alpha$ Ratio", fontsize=15)
ax.set_title(r"Per-Subject Mean $\theta/\alpha$ Ratio — Sorted",
             fontsize=17, fontweight='bold', pad=14)
ax.tick_params(labelsize=12)
style_axis(ax)

legend_elements = [
    Patch(facecolor=COL_HC, alpha=0.85, edgecolor='#222222', label='Healthy'),
    Patch(facecolor=COL_AD, alpha=0.85, edgecolor='#222222', label='AD'),
]
ax.legend(handles=legend_elements, fontsize=13, loc='upper left', framealpha=0.95)

plt.tight_layout()
plt.savefig("/kaggle/working/eeg_plot6_subject_ratio_sorted.png", dpi=DPI, bbox_inches='tight')
plt.show()
print("  Saved: eeg_plot6_subject_ratio_sorted.png")

# ────────────────────────────────────────────────────────────
#  SUMMARY
# ────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  EEG PLOTTING COMPLETE — FINAL VERSION")
print("=" * 60)
print(f"  Total rows plotted : {len(df):,}")
hc_total = len(df[df['Label'] == 'Healthy'])
ad_total = len(df[df['Label'] == 'AD'])
print(f"  Healthy samples    : {hc_total:,} ({100*hc_total/len(df):.1f}%)")
print(f"  AD samples         : {ad_total:,} ({100*ad_total/len(df):.1f}%)")
print()
print("  Saved files (400 dpi, light-blue background):")
for i in range(1, 7):
    print(f"    /kaggle/working/eeg_plot{i}_*.png")
print("=" * 60)
