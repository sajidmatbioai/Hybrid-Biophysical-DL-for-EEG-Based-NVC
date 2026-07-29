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
CSV_PATH = "/kaggle/working/EEG_BOLD_Data.csv"
SEQ_LEN = 600
T = np.linspace(0, 30, SEQ_LEN)

# FIX #1 — column prefixes now match what STEP_8_3 actually writes to
# the CSV (cbf_, cbv_, dhb_, bold_), not the old f_/v_/q_ guesses.
BOLD_COLS = [f"bold_{i}" for i in range(SEQ_LEN)]
CBF_COLS  = [f"cbf_{i}"  for i in range(SEQ_LEN)]
CBV_COLS  = [f"cbv_{i}"  for i in range(SEQ_LEN)]
DHB_COLS  = [f"dhb_{i}"  for i in range(SEQ_LEN)]

df = pd.read_csv(CSV_PATH)
hc_bold = df[df['Class']==0][BOLD_COLS].values.astype(float)
ad_bold = df[df['Class']==1][BOLD_COLS].values.astype(float)
hc_cbf  = df[df['Class']==0][CBF_COLS].values.astype(float)
ad_cbf  = df[df['Class']==1][CBF_COLS].values.astype(float)
hc_cbv  = df[df['Class']==0][CBV_COLS].values.astype(float)
ad_cbv  = df[df['Class']==1][CBV_COLS].values.astype(float)
hc_dhb  = df[df['Class']==0][DHB_COLS].values.astype(float)
ad_dhb  = df[df['Class']==1][DHB_COLS].values.astype(float)

def style(ax):
    ax.grid(True, alpha=0.3, color=COL_GRID, linewidth=0.7)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# ════════════════════════════════════════════════════════════
#  COMPANION PLOT — BOLD Signal, Mean ± SEM, Healthy vs AD
# ════════════════════════════════════════════════════════════
fig0, ax0 = plt.subplots(figsize=(10, 7.5))
peak_stats = {}
for mat, c, name in [(hc_bold, COL_HC, 'Healthy'), (ad_bold, COL_AD, 'AD')]:
    m, sem = mat.mean(0), mat.std(0)/np.sqrt(len(mat))
    ax0.plot(T, m, color=c, lw=2.6, label=f'{name} (n={len(mat):,})')
    ax0.fill_between(T, m-sem, m+sem, color=c, alpha=0.30)

    peak_idx = np.argmax(m)
    peak_val, peak_sem, peak_t = m[peak_idx], sem[peak_idx], T[peak_idx]
    peak_stats[name] = (peak_val, peak_sem, peak_t)
    ax0.scatter([peak_t], [peak_val], color=c, s=80, zorder=5,
                edgecolor='black', linewidth=1.3)

# FIX #2 — explicit headroom above the curves so annotation boxes
# never collide with the title, regardless of data scale.
y_top = max(hc_bold.mean(0).max(), ad_bold.mean(0).max())
y_bot = min(hc_bold.mean(0).min(), ad_bold.mean(0).min())
y_span = y_top - y_bot
ax0.set_ylim(y_bot - 0.05*y_span, y_top + 0.12*y_span)

# FIX #3 — both annotation boxes now sit in the empty lower-right
# region of the plot (curves plateau near the top after t≈5s),
# using axes-fraction coordinates so they never drift into the
# title or overlap each other regardless of data scale.
hv, hs, ht = peak_stats['Healthy']
ax0.annotate(f"Healthy peak\n{hv:.4f} \u00b1 {hs:.4f}",
             xy=(ht, hv), xytext=(0.55, 0.38), textcoords='axes fraction',
             fontsize=10.5, fontweight='bold', color=COL_HC, ha='center',
             arrowprops=dict(arrowstyle='->', color=COL_HC, lw=1.4),
             bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                       edgecolor=COL_HC, alpha=0.95))

av, aS, at = peak_stats['AD']
ax0.annotate(f"AD peak\n{av:.4f} \u00b1 {aS:.4f}",
             xy=(at, av), xytext=(0.55, 0.20), textcoords='axes fraction',
             fontsize=10.5, fontweight='bold', color=COL_AD, ha='center',
             arrowprops=dict(arrowstyle='->', color=COL_AD, lw=1.4),
             bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                       edgecolor=COL_AD, alpha=0.95))

ax0.axhline(0, color='#444444', lw=0.9, ls='--', alpha=0.6)
ax0.set_xlim(0, 30)
ax0.set_xlabel("Time (s)", fontsize=14)
ax0.set_ylabel("BOLD Signal (dS/S)", fontsize=14)
ax0.set_title("BOLD Signal — Mean ± SEM, Healthy vs AD", fontsize=16, fontweight='bold', pad=20)
ax0.tick_params(labelsize=12)
ax0.legend(fontsize=12, framealpha=0.95, loc='upper left', bbox_to_anchor=(0.01, 0.99))

style(ax0)
plt.tight_layout()
plt.savefig("/kaggle/working/bold_companion_mean_sem.png", dpi=350, bbox_inches='tight')
print("Saved: bold_companion_mean_sem.png")

def cohens_d(a, b):
    pooled = np.sqrt(((len(a)-1)*a.std()**2 + (len(b)-1)*b.std()**2) / (len(a)+len(b)-2))
    return (b.mean() - a.mean()) / pooled

def fwhm(mat):
    out=[]
    for row in mat:
        half = row.max()/2
        above = np.where(row>=half)[0]
        out.append((above[-1]-above[0])*(T[1]-T[0]) if len(above)>1 else np.nan)
    return np.array(out)

def onset(mat, frac=0.2):
    out=[]
    for row in mat:
        idx = np.where(row >= row.max()*frac)[0]
        out.append(T[idx[0]] if len(idx) else np.nan)
    return np.array(out)

def undershoot(mat):
    out=[]
    for row in mat:
        peak_idx = np.argmax(row)
        tail = row[peak_idx:]
        out.append(tail.min() if len(tail) else np.nan)
    return np.array(out)

fw_hc, fw_ad = fwhm(hc_bold), fwhm(ad_bold)
fw_hc, fw_ad = fw_hc[~np.isnan(fw_hc)], fw_ad[~np.isnan(fw_ad)]

# ── Candidate features across ALL Balloon-Windkessel variables ──
features = {
    'BOLD Peak Amplitude':  (hc_bold.max(1), ad_bold.max(1)),
    'BOLD AUC (Total Resp.)': (np.trapezoid(hc_bold,T,axis=1), np.trapezoid(ad_bold,T,axis=1)),
    'BOLD Time-to-Peak':     (T[np.argmax(hc_bold,1)], T[np.argmax(ad_bold,1)]),
    'BOLD FWHM (Peak Width)': (fw_hc, fw_ad),
    'BOLD Onset Latency':    (onset(hc_bold), onset(ad_bold)),
    'BOLD Undershoot Depth': (undershoot(hc_bold), undershoot(ad_bold)),
    'CBF Peak':              (hc_cbf.max(1), ad_cbf.max(1)),
    'CBV Peak':              (hc_cbv.max(1), ad_cbv.max(1)),
    'dHb Dip Depth':         (hc_dhb.min(1), ad_dhb.min(1)),
}

results = []
for name, (a, b) in features.items():
    d = cohens_d(a, b)
    _, p = mannwhitneyu(a, b, alternative='two-sided')
    results.append({'name': name, 'd': d, 'abs_d': abs(d), 'p': p, 'a': a, 'b': b})
results.sort(key=lambda r: r['abs_d'], reverse=True)

print("Ranking:")
for r in results:
    print(f"  {r['name']:28s} d={r['d']:+.2f}  p={r['p']:.2g}")

# ══════════════════════════════════════════════════════════
#  FIGURE — ranking (left) + best-feature hero boxplot (right)
# ══════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 9))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.25, 1], wspace=0.3)

ax1 = fig.add_subplot(gs[0])
names = [r['name'] for r in results]
d_vals = [r['d'] for r in results]
bar_colors = [COL_AD if d > 0 else COL_HC for d in d_vals]
y_pos = np.arange(len(names))
ax1.barh(y_pos, d_vals, color=bar_colors, edgecolor='black', linewidth=1.3, height=0.6)

for i, r in enumerate(results):
    sig = 'p<0.001' if r['p'] < 0.001 else f"p={r['p']:.2g}"
    label_txt = f"d={r['d']:.2f}, {sig}"
    if abs(r['d']) > 0.9:
        ax1.text(r['d']/2, i, label_txt, va='center', ha='center',
                  fontsize=10, fontweight='bold', color='white')
    else:
        bar_right_edge = max(r['d'], 0)
        x_text = bar_right_edge + 0.15
        ax1.text(x_text, i, label_txt, va='center', ha='left',
                  fontsize=10, fontweight='bold', color='#1a1a1a')

ax1.set_yticks(y_pos)
ax1.set_yticklabels(names, fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', pad=10)
ax1.axvline(0, color='black', lw=1.2)

# FIX #4 — widen xlim so the "d=.., p=.." text labels (drawn beyond
# the bar edges) stay fully inside this subplot instead of
# overflowing into the boxplot panel on the right.
ax1.set_xlim(min(d_vals) - 0.15, max(d_vals) + 0.45)

ax1.set_xlabel("Effect Size (Cohen's d, AD − Healthy)", fontsize=13)
ax1.set_title("Which Balloon-Windkessel Feature Best Separates AD from Healthy?",
              fontsize=14.5, fontweight='bold', pad=12)
style(ax1)

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
    ax2.scatter(pos+jitter, vals, s=26, color=c, alpha=0.55, edgecolor='black', linewidth=0.3, zorder=3)
y_max = max(a.max(), b.max()); y_bar = y_max*1.1 if y_max>0 else y_max*0.9
y_bar = max(a.max(), b.max()) + 0.08*abs(max(a.max(), b.max()) - min(a.min(), b.min()))
ax2.plot([0,0,1,1],[y_bar,y_bar*1.02,y_bar*1.02,y_bar], color='black', lw=1.5)
sig = 'p < 0.001' if best['p']<0.001 else f"p = {best['p']:.3g}"
ax2.text(0.5, y_bar*1.05, f"{sig}  (d = {best['d']:.2f})", ha='center', fontsize=12.5, fontweight='bold')

# FIX #5 — give the boxplot panel headroom above the significance
# bracket/text so it doesn't get clipped or crowd the title.
ax2.set_ylim(top=y_bar*1.15)

ax2.set_xticks([0,1]); ax2.set_xticklabels([f'Healthy\n(n={len(a):,})', f'AD\n(n={len(b):,})'],
                                            fontsize=13, fontweight='bold')
ax2.set_ylabel(best['name'], fontsize=14)
ax2.set_title(f"Best Discriminator: {best['name']}", fontsize=14.5, fontweight='bold', pad=12)
style(ax2)
legend_elements = [Patch(facecolor=COL_HC, alpha=0.5, edgecolor=COL_HC, label='Healthy'),
                    Patch(facecolor=COL_AD, alpha=0.5, edgecolor=COL_AD, label='AD')]
ax2.legend(handles=legend_elements, fontsize=11.5, loc='upper left', framealpha=0.95)

plt.savefig("/kaggle/working/bold_ranking_and_hero.png", dpi=350, bbox_inches='tight')
print("Saved.")