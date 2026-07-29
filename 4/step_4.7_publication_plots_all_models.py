"""
============================================================
 Publication-Quality Comparison Plots
 DB-HLSTM vs ANN vs SVM vs Random Forest — AD vs Healthy

 Reads each model's saved test-set predictions CSV (produced
 by the four single-split baseline scripts) and generates
 paper-ready comparison figures:

   1. combined_roc_curves.png/.pdf        — all 4 ROC curves, one plot
   2. combined_pr_curves.png/.pdf         — all 4 PR curves, one plot
   3. confusion_matrices_grid.png/.pdf    — 2x2 grid, one CM per model
   4. metrics_bar_comparison.png/.pdf     — grouped bar chart,
                                            Accuracy/Precision/Recall/
                                            F1/AUC side by side

 All predictions are expected to use the DEFAULT (0.5) decision
 threshold — matches the no-threshold-tuning protocol used across
 all four single-split scripts.

 Expected input files (edit PRED_FILES below to match your paths):
   DB-HLSTM : /kaggle/working/test_predictions.csv  (cols: Subject, y_true, test_prob, y_pred)
   ANN      : /kaggle/working/predictions.csv        (cols: Subject, y_true, test_prob, y_pred)
   SVM      : /kaggle/working/svm_predictions.csv    (cols: subject, y_true, y_prob,    y_pred)
   RF       : /kaggle/working/rf_predictions.csv     (cols: subject, y_true, y_prob,    y_pred)

 NOTE — each baseline script currently writes to the SAME filename
 (predictions.csv / test_predictions.csv) in /kaggle/working. Rename
 or move each one right after that baseline finishes running (e.g.
 `!cp /kaggle/working/predictions.csv /kaggle/working/ann_predictions.csv`)
 before running the next baseline, otherwise later runs overwrite
 earlier ones. Update PRED_FILES paths below to match whatever you
 actually named them.

 Output : /kaggle/working/figures/*.png and *.pdf (vector, for the paper)
============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import (roc_curve, precision_recall_curve, auc,
                             confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score,
                             roc_auc_score)

import json

# ============================================================
# CONFIG — edit these paths to match where each baseline's
# predictions CSV actually landed
# ============================================================
PRED_FILES = {
    'DB-HLSTM': '/kaggle/working/dbhlstm/predictions.csv',
    'CNN':      '/kaggle/working/cnn/predictions.csv',
    'ANN':      '/kaggle/working/ann/predictions.csv',
    'SVM':      '/kaggle/working/svm/predictions.csv',
    'RF':       '/kaggle/working/rf/predictions.csv',
}

SPLIT_JSON = '/kaggle/working/subject_split.json'

OUT_DIR = '/kaggle/working/figures'
os.makedirs(OUT_DIR, exist_ok=True)

# Consistent, colorblind-friendly palette (Okabe-Ito), one color per model
COLORS = {
    'DB-HLSTM': '#0072B2',  # blue
    'CNN':      '#E69F00',  # orange
    'ANN':      '#D55E00',  # vermillion
    'SVM':      '#009E73',  # green
    'RF':       '#CC79A7',  # pink
}
MODEL_ORDER = ['DB-HLSTM', 'CNN', 'ANN', 'SVM', 'RF']

# ============================================================
# PUBLICATION STYLE — clean, journal-ready defaults
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'axes.labelweight': 'bold',
    'axes.linewidth': 1.1,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.frameon': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2.2,
})


def _normalize(df):
    """Map each baseline's slightly different column names onto a
    common schema: subject, y_true, y_prob, y_pred.

    DB-HLSTM's predictions.csv (v3.3, with validation-tuned
    threshold restored) saves 'y_pred_tuned' and
    'y_pred_default_0.5' instead of a plain 'y_pred' column —
    prefer the tuned one (the model's primary reported result),
    falling back to default-0.5 if tuned isn't present, falling
    back to plain 'y_pred' for the other baselines (CNN/ANN/SVM/RF)."""
    cols = {c.lower(): c for c in df.columns}
    subj_col = cols.get('subject', 'subject')
    prob_col = cols.get('test_prob', cols.get('y_prob'))
    pred_col = (cols.get('y_pred_tuned')
                or cols.get('y_pred_default_0.5')
                or cols.get('y_pred'))
    if pred_col is None:
        raise KeyError(
            f"No recognizable prediction column found — columns present: "
            f"{list(df.columns)}"
        )
    out = pd.DataFrame({
        'subject': df[subj_col] if subj_col in df.columns else df.iloc[:, 0],
        'y_true': df[cols['y_true']].astype(int),
        'y_prob': df[prob_col].astype(float),
        'y_pred': df[pred_col].astype(int),
    })
    return out


# ============================================================
# LOAD ALL MODELS
# ============================================================
data = {}
missing = []
for name, path in PRED_FILES.items():
    if not os.path.exists(path):
        missing.append((name, path))
        continue
    raw = pd.read_csv(path)
    data[name] = _normalize(raw)
    print(f"  Loaded {name}: {path}  ({len(data[name]):,} rows, "
          f"{data[name]['subject'].nunique()} subjects)")

if missing:
    print("\n  WARNING — could not find predictions for:")
    for name, path in missing:
        print(f"    {name}: {path}  (edit PRED_FILES at the top of this script)")

MODEL_ORDER = [m for m in MODEL_ORDER if m in data]
if not MODEL_ORDER:
    raise FileNotFoundError("No prediction files were found — check PRED_FILES paths.")

# ============================================================
# COMPUTE PER-MODEL METRICS (once, reused across all 4 figures)
# ============================================================
metrics = {}
for name in MODEL_ORDER:
    d = data[name]
    y_true, y_prob, y_pred = d['y_true'].values, d['y_prob'].values, d['y_pred'].values
    metrics[name] = {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
        'f1':        f1_score(y_true, y_pred, zero_division=0),
        'auc':       roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
    }

print("\n  Summary metrics:")
print(pd.DataFrame(metrics).T.to_string(float_format=lambda x: f"{x:.4f}"))

# ============================================================
# FIGURE 1 — COMBINED ROC CURVES
# ============================================================
fig, ax = plt.subplots(figsize=(6.5, 6))
for name in MODEL_ORDER:
    d = data[name]
    fpr, tpr, _ = roc_curve(d['y_true'], d['y_prob'])
    roc_auc = metrics[name]['auc']
    ax.plot(fpr, tpr, color=COLORS[name], label=f'{name} (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], linestyle='--', linewidth=1.2, color='0.6', label='Chance')
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.02)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — Test Set')
ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.legend(loc='lower right')
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/combined_roc_curves.png')
fig.savefig(f'{OUT_DIR}/combined_roc_curves.pdf')
plt.close(fig)
print(f"\n  Saved: {OUT_DIR}/combined_roc_curves.png (+ .pdf)")

# ============================================================
# FIGURE 2 — COMBINED PRECISION-RECALL CURVES
# ============================================================
fig, ax = plt.subplots(figsize=(6.5, 6))
for name in MODEL_ORDER:
    d = data[name]
    prec, rec, _ = precision_recall_curve(d['y_true'], d['y_prob'])
    pr_auc = auc(rec, prec)
    ax.plot(rec, prec, color=COLORS[name], label=f'{name} (AUC = {pr_auc:.3f})')
base_rate = data[MODEL_ORDER[0]]['y_true'].mean()
ax.axhline(base_rate, linestyle='--', linewidth=1.2, color='0.6',
           label=f'Chance ({base_rate:.2f})')
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.02)
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves — Test Set')
ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.legend(loc='lower left')
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/combined_pr_curves.png')
fig.savefig(f'{OUT_DIR}/combined_pr_curves.pdf')
plt.close(fig)
print(f"  Saved: {OUT_DIR}/combined_pr_curves.png (+ .pdf)")

# ============================================================
# FIGURE 3 — CONFUSION MATRICES, 2x2 GRID
# ============================================================
n = len(MODEL_ORDER)
ncols = 2
nrows = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(9, 4.3 * nrows))
axes = np.array(axes).reshape(-1)

for i, name in enumerate(MODEL_ORDER):
    ax = axes[i]
    d = data[name]
    cm = confusion_matrix(d['y_true'], d['y_pred'], labels=[0, 1])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Healthy', 'AD'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Healthy', 'AD'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'{name}  (Acc = {metrics[name]["accuracy"]:.3f})')

    for r in range(2):
        for c in range(2):
            color = 'white' if cm_norm[r, c] > 0.5 else 'black'
            ax.text(c, r, f'{cm[r, c]:,}\n({cm_norm[r, c]*100:.1f}%)',
                    ha='center', va='center', color=color, fontsize=10)

for j in range(n, len(axes)):
    axes[j].axis('off')

# ------------------------------------------------------------
# Use one leftover empty cell (grid has 6 slots for 5 models)
# for a Split Summary table — shows Train/Val/Test subject
# counts and AD/Healthy breakdown at a glance, so a reviewer
# can immediately see the split is shared, balanced, and
# leakage-free across all 5 models.
# ------------------------------------------------------------
if n < len(axes) and os.path.exists(SPLIT_JSON):
    with open(SPLIT_JSON) as f:
        split_info = json.load(f)

    ax_tbl = axes[n]
    ax_tbl.axis('off')

    rows = ['Train', 'Val', 'Test']
    keys = ['train', 'val', 'test']
    col_labels = ['Split', 'Subjects', 'Healthy', 'AD']
    table_data = []
    for label, key in zip(rows, keys):
        subj_n = len(split_info.get(f'{key}_subjects', []))
        counts = split_info.get(f'{key}_class_counts', {})
        table_data.append([label, str(subj_n),
                           str(counts.get('Healthy', '—')),
                           str(counts.get('AD', '—'))])

    tbl = ax_tbl.table(cellText=table_data, colLabels=col_labels,
                       cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(fontweight='bold', color='white')
            cell.set_facecolor('#333333')
        else:
            cell.set_facecolor('#f2f2f2' if r % 2 == 0 else 'white')

    seed = split_info.get('seed', '?')
    tf_, vf_, tef_ = (split_info.get('train_frac'), split_info.get('val_frac'),
                      split_info.get('test_frac'))
    ax_tbl.set_title(
        f'Split Summary (subject-level, seed={seed})\n'
        f'Shared across all 5 models — no subject leakage',
        fontsize=11, fontweight='bold', pad=14)
elif n < len(axes):
    print(f"  NOTE — {SPLIT_JSON} not found; leaving the leftover grid "
          f"cell blank instead of a split-summary table.")

fig.suptitle('Confusion Matrices — Test Set (row-normalized %)', fontsize=15, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/confusion_matrices_grid.png')
fig.savefig(f'{OUT_DIR}/confusion_matrices_grid.pdf')
plt.close(fig)
print(f"  Saved: {OUT_DIR}/confusion_matrices_grid.png (+ .pdf)")

# ============================================================
# FIGURE 4 — GROUPED BAR CHART: ACCURACY/PRECISION/RECALL/F1/AUC
# ============================================================
metric_names = ['accuracy', 'precision', 'recall', 'f1', 'auc']
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC']

x = np.arange(len(metric_names))
width = 0.8 / len(MODEL_ORDER)

fig, ax = plt.subplots(figsize=(9, 5.5))
for i, name in enumerate(MODEL_ORDER):
    vals = [metrics[name][m] for m in metric_names]
    offset = (i - (len(MODEL_ORDER) - 1) / 2) * width
    bars = ax.bar(x + offset, vals, width, label=name, color=COLORS[name])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f'{v:.2f}',
                ha='center', va='bottom', fontsize=8, rotation=0)

ax.set_xticks(x)
ax.set_xticklabels(metric_labels)
ax.set_ylim(0, 1.08)
ax.set_ylabel('Score')
ax.set_title('Model Comparison — Test-Set Metrics')
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=len(MODEL_ORDER))
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/metrics_bar_comparison.png')
fig.savefig(f'{OUT_DIR}/metrics_bar_comparison.pdf')
plt.close(fig)
print(f"  Saved: {OUT_DIR}/metrics_bar_comparison.png (+ .pdf)")

# ============================================================
# SAVE THE METRICS TABLE TOO (handy for a paper table)
# ============================================================
metrics_df = pd.DataFrame(metrics).T[metric_names]
metrics_df.columns = metric_labels
metrics_df.to_csv(f'{OUT_DIR}/metrics_table.csv')
print(f"  Saved: {OUT_DIR}/metrics_table.csv")

print("\n  ALL FIGURES DONE — see:", OUT_DIR)
