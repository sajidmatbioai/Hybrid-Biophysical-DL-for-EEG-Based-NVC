import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

BG_COLOR = "#eef5fc"
plt.rcParams.update({
    'figure.facecolor': BG_COLOR, 'axes.facecolor': BG_COLOR,
    'axes.edgecolor': '#222222', 'axes.linewidth': 1.2,
    'axes.labelweight': 'bold', 'font.size': 12,
    'mathtext.fontset': 'stix', 'savefig.facecolor': BG_COLOR,
})

COL_HC, COL_AD, COL_GRID = "#1558b0", "#c22032", "#b0b0b0"
CSV_PATH = "/kaggle/working/EEG_HRF_Data.csv"
SEQ_LEN, h_step = 600, 0.5
T_MS = np.arange(SEQ_LEN) * h_step
HRF_C = [f"hrf_c_{i}" for i in range(SEQ_LEN)]

df = pd.read_csv(CSV_PATH)
hc_mat = df[df['Class']==0][HRF_C].values.astype(float)
ad_mat = df[df['Class']==1][HRF_C].values.astype(float)

def style(ax):
    ax.grid(True, alpha=0.3, color=COL_GRID, linewidth=0.7)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# ════════════════════════════════════════════════════════════
#  COMPANION PLOT — Canonical HRF, Mean ± SEM, Healthy vs AD
#  (visual curve to accompany the ranking+hero statistical figure)
# ════════════════════════════════════════════════════════════
fig0, ax0 = plt.subplots(figsize=(10, 7.5))
peak_stats_hrf = {}
for mat, c, name in [(hc_mat, COL_HC, 'Healthy'), (ad_mat, COL_AD, 'AD')]:
    m, sem = mat.mean(0), mat.std(0)/np.sqrt(len(mat))
    ax0.plot(T_MS, m, color=c, lw=2.6, label=f'{name} (n={len(mat):,})')
    ax0.fill_between(T_MS, m-sem, m+sem, color=c, alpha=0.30)

    peak_idx = np.argmax(m)
    peak_val, peak_sem, peak_t = m[peak_idx], sem[peak_idx], T_MS[peak_idx]
    peak_stats_hrf[name] = (peak_val, peak_sem, peak_t)
    ax0.scatter([peak_t], [peak_val], color=c, s=80, zorder=5,
                edgecolor='black', linewidth=1.3)

y_range_hrf = max(hc_mat.mean(0).max(), ad_mat.mean(0).max()) - \
              min(hc_mat.mean(0).min(), ad_mat.mean(0).min())

# annotate the value RIGHT AT each peak dot, instead of a combined
# corner stats box
hv, hs, ht = peak_stats_hrf['Healthy']
ax0.annotate(f"Healthy peak\n{hv:.4f} \u00b1 {hs:.4f}",
             xy=(ht, hv), xytext=(ht - 70, hv - 0.20*y_range_hrf),
             fontsize=10.5, fontweight='bold', color=COL_HC, ha='center',
             arrowprops=dict(arrowstyle='->', color=COL_HC, lw=1.4),
             bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                       edgecolor=COL_HC, alpha=0.95))

av, aS, at = peak_stats_hrf['AD']
ax0.annotate(f"AD peak\n{av:.4f} \u00b1 {aS:.4f}",
             xy=(at, av), xytext=(at + 40, av + 0.05*y_range_hrf),
             fontsize=10.5, fontweight='bold', color=COL_AD, ha='center',
             arrowprops=dict(arrowstyle='->', color=COL_AD, lw=1.4),
             bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                       edgecolor=COL_AD, alpha=0.95))

ax0.axhline(0, color='#444444', lw=0.9, ls='--', alpha=0.6)
ax0.set_xlim(0, 300)
ax0.set_xlabel("Time (ms)", fontsize=14)
ax0.set_ylabel("Canonical HRF Amplitude", fontsize=14)
# title moved further above the axes so it never crowds the legend
ax0.set_title("Canonical HRF — Mean ± SEM, Healthy vs AD", fontsize=16, fontweight='bold', pad=28)
ax0.tick_params(labelsize=12)
ax0.legend(fontsize=12, framealpha=0.95, loc='upper left', bbox_to_anchor=(0.01, 0.99))

style(ax0)
plt.tight_layout()
plt.savefig("/kaggle/working/hrf_companion_mean_sem.png", dpi=350, bbox_inches='tight')
print("Saved: hrf_companion_mean_sem.png")

def cohens_d(a, b):
    pooled = np.sqrt(((len(a)-1)*a.std()**2 + (len(b)-1)*b.std()**2) / (len(a)+len(b)-2))
    return (b.mean() - a.mean()) / pooled

# ── Compute all candidate features ──────────────────────────
peak_hc, peak_ad = hc_mat.max(1), ad_mat.max(1)
ttp_hc, ttp_ad   = T_MS[np.argmax(hc_mat,1)], T_MS[np.argmax(ad_mat,1)]
auc_hc, auc_ad   = np.trapezoid(hc_mat, T_MS, axis=1), np.trapezoid(ad_mat, T_MS, axis=1)

def fwhm(mat):
    out=[]
    for row in mat:
        half = row.max()/2
        above = np.where(row>=half)[0]
        out.append((above[-1]-above[0])*h_step if len(above)>1 else np.nan)
    return np.array(out)
fwhm_hc, fwhm_ad = fwhm(hc_mat), fwhm(ad_mat)

def onset(mat, frac=0.2):
    out=[]
    for row in mat:
        idx = np.where(row >= row.max()*frac)[0]
        out.append(T_MS[idx[0]] if len(idx) else np.nan)
    return np.array(out)
ons_hc, ons_ad = onset(hc_mat), onset(ad_mat)

features = {
    'Peak Amplitude':   (peak_hc, peak_ad),
    'Time-to-Peak':     (ttp_hc, ttp_ad),
    'AUC (Total Response)': (auc_hc, auc_ad),
    'FWHM (Peak Width)': (fwhm_hc[~np.isnan(fwhm_hc)], fwhm_ad[~np.isnan(fwhm_ad)]),
    'Onset Latency':    (ons_hc, ons_ad),
}

results = []
for name, (a, b) in features.items():
    d = cohens_d(a, b)
    _, p = mannwhitneyu(a, b, alternative='two-sided')
    results.append({'name': name, 'd': d, 'abs_d': abs(d), 'p': p, 'a': a, 'b': b})
results.sort(key=lambda r: r['abs_d'], reverse=True)

# ══════════════════════════════════════════════════════════
#  FIGURE — ranking (left) + best-feature hero boxplot (right)
# ══════════════════════════════════════════════════════════
fig = plt.figure(figsize=(17, 7.5))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.1, 1], wspace=0.32)

# ── Left: effect-size ranking bar chart ─────────────────────
ax1 = fig.add_subplot(gs[0])
names = [r['name'] for r in results]
d_vals = [r['d'] for r in results]
bar_colors = [COL_AD if d > 0 else COL_HC for d in d_vals]
y_pos = np.arange(len(names))
bars = ax1.barh(y_pos, d_vals, color=bar_colors, edgecolor='black', linewidth=1.3, height=0.6)
for i, r in enumerate(results):
    sig = 'p<0.001' if r['p'] < 0.001 else f"p={r['p']:.2g}"
    x_text = r['d'] + (0.15 if r['d']>=0 else -0.15)
    ax1.text(x_text, i, f"d={r['d']:.2f}, {sig}", va='center',
              ha='left' if r['d']>=0 else 'right', fontsize=10.5, fontweight='bold')
ax1.set_yticks(y_pos); ax1.set_yticklabels(names, fontsize=12.5, fontweight='bold')
ax1.axvline(0, color='black', lw=1.2)
ax1.set_xlabel("Effect Size (Cohen's d, AD − Healthy)", fontsize=13)
ax1.set_title("Which HRF Feature Best Separates AD from Healthy?",
              fontsize=14.5, fontweight='bold', pad=12)
style(ax1)
ax1.set_xlim(min(d_vals)-1, max(d_vals)+1.2)

# ── Right: hero boxplot of the WINNING feature ──────────────
best = results[0]
ax2 = fig.add_subplot(gs[1])
a, b = best['a'], best['b']
bp = ax2.boxplot([a, b], positions=[0,1], widths=0.42, patch_artist=True, showfliers=False,
                 medianprops=dict(color='black', lw=2.4))
for patch, c in zip(bp['boxes'], [COL_HC, COL_AD]):
    patch.set_facecolor(c); patch.set_alpha(0.40); patch.set_edgecolor(c); patch.set_linewidth(2.2)
rng = np.random.default_rng(0)
for pos, vals, c in zip([0,1], [a,b], [COL_HC, COL_AD]):
    jitter = rng.uniform(-0.13, 0.13, size=len(vals))
    ax2.scatter(pos+jitter, vals, s=28, color=c, alpha=0.55, edgecolor='black', linewidth=0.3, zorder=3)
y_max = max(a.max(), b.max()); y_bar = y_max*1.08
ax2.plot([0,0,1,1],[y_bar,y_bar*1.02,y_bar*1.02,y_bar], color='black', lw=1.5)
sig = 'p < 0.001' if best['p']<0.001 else f"p = {best['p']:.3g}"
ax2.text(0.5, y_bar*1.035, f"{sig}  (d = {best['d']:.2f})", ha='center', fontsize=12.5, fontweight='bold')
ax2.set_xticks([0,1]); ax2.set_xticklabels([f'Healthy\n(n={len(a):,})', f'AD\n(n={len(b):,})'],
                                            fontsize=13, fontweight='bold')
ax2.set_ylabel(best['name'], fontsize=14)
ax2.set_title(f"Best Discriminator: {best['name']}", fontsize=14.5, fontweight='bold', pad=12)
ax2.set_ylim(top=y_bar*1.14)
style(ax2)
legend_elements = [Patch(facecolor=COL_HC, alpha=0.5, edgecolor=COL_HC, label='Healthy'),
                    Patch(facecolor=COL_AD, alpha=0.5, edgecolor=COL_AD, label='AD')]
ax2.legend(handles=legend_elements, fontsize=11.5, loc='upper left', framealpha=0.95)

plt.savefig("/kaggle/working/hrf_ranking_and_hero.png", dpi=350, bbox_inches='tight')
print("Saved.")
print()
print("Ranking:")
for r in results:
    print(f"  {r['name']:25s} d={r['d']:+.2f}  p={r['p']:.2g}")
