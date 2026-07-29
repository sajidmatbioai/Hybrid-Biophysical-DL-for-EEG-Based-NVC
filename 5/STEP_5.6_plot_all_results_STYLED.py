"""
============================================================
 Step 6 — Generate ALL Plots (Post-Training, All 5 Models)
 STYLED VERSION — house design system applied (
 EEG plotting script), matching the polished sample figures:
   - confusion_matrix.png
   - fig4_precision_recall_f1.png
   - fig1_fold_performance.png

 Reads the CSVs that each baseline/model script already saves
 during training — NOTHING is retrained here, this script only
 loads results and plots them. INPUT/READING LOGIC IS UNCHANGED
 from the original Step 6 script — only the plotting code and
 style were updated.

 Expected input (already produced by Steps 8.6a-d / 8.5):
   /kaggle/working/results/dbhlstm/fold_metrics.csv
   /kaggle/working/results/dbhlstm/oof_predictions.csv
   /kaggle/working/results/cnn/fold_metrics.csv
   /kaggle/working/results/cnn/oof_predictions.csv
   /kaggle/working/results/ann/fold_metrics.csv
   /kaggle/working/results/ann/oof_predictions.csv
   /kaggle/working/results/svm/fold_metrics.csv
   /kaggle/working/results/svm/oof_predictions.csv
   /kaggle/working/results/rf/fold_metrics.csv
   /kaggle/working/results/rf/oof_predictions.csv

 fold_metrics.csv columns : fold, auc, acc_default, prec_default,
                            rec_default, f1_default
 oof_predictions.csv cols : Subject, y_true, oof_prob, fold

 Output:
   /kaggle/working/results/plots/<model>/*.png   (per-model plots)
   /kaggle/working/results/plots/comparison/*.png (all-models plots)

 NEW PLOTS ADDED IN THIS VERSION (per model, in plots/<model>/):
   - confusion_matrix_single.png   (large, tuned threshold, count+%)
   - prf_by_class_threshold.png    (Precision/Recall/F1, thr=0.5 vs tuned)
   - fold_performance.png          (AUC/Acc/F1 per fold + mean AUC band)

 NEW PLOTS ADDED IN COMPARISON (plots/comparison/):
   - confusion_matrices_grid.png   (all models, tuned threshold, one figure)
============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                              confusion_matrix, precision_recall_fscore_support)

# ── GLOBAL STYLE (house design system) ───────────────────────
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

DPI = 400  # print-ready, journal quality, applied on every savefig() call

# ── HOUSE COLOR PALETTE ───────────────────────────────────────
COL_HC     = "#1558b0"   # Healthy — royal blue
COL_AD     = "#c22032"   # AD      — deep crimson red
COL_THETA  = "#f4b400"   # gold    — reused for F1-score bars
COL_ALPHA  = "#7b2d8e"   # purple  — reused for AUC bars / mean line
COL_GRID   = "#b0b0b0"


def style_axis(ax):
    """House axis styling: light grid, no top/right spines."""
    ax.grid(True, alpha=0.30, color=COL_GRID, linewidth=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ── CONFIG (unchanged) ────────────────────────────────────────
BASE_RESULTS_DIR = '/kaggle/working/results'
PLOTS_DIR        = f'{BASE_RESULTS_DIR}/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(f'{PLOTS_DIR}/comparison', exist_ok=True)

# model_key -> (folder name under results/, display name for titles/legends)
MODELS = {
    'dbhlstm': 'DB-HLSTM',
    'cnn':     'CNN',
    'ann':     'ANN',
    'svm':     'SVM',
    'rf':      'Random Forest',
}

# Model comparison palette — deliberately reuses the house hues
# (COL_HC / COL_AD / COL_ALPHA) so comparison figures feel like
# part of the same visual family as the class-level EEG figures.
COLORS = {
    'dbhlstm': '#1558b0',   # royal blue   (flagship model)
    'cnn':     '#e08214',   # orange
    'ann':     '#2ca25f',   # green
    'svm':     '#c22032',   # crimson      (== COL_AD)
    'rf':      '#7b2d8e',   # purple       (== COL_ALPHA)
}


# ── HELPERS ──────────────────────────────────────────────────
def pooled_threshold(y_true, y_prob):
    """Same F1-optimal, clamped-to-[0.05,0.95] rule used in the training
    scripts, so plots match the numbers already printed during training."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = np.divide(2 * prec[:-1] * rec[:-1], prec[:-1] + rec[:-1],
                   out=np.zeros_like(prec[:-1]), where=(prec[:-1] + rec[:-1]) > 0)
    mask = (thr >= 0.05) & (thr <= 0.95)
    if mask.any():
        return float(thr[mask][int(np.argmax(f1[mask]))])
    return 0.5


def load_model_results(model_key):
    """Loads fold_metrics.csv + oof_predictions.csv for one model.
    Returns (fm, oof) or (None, None) if not found (skip gracefully).
    UNCHANGED — reads from exactly the same paths as before."""
    folder = f'{BASE_RESULTS_DIR}/{model_key}'
    fm_path  = f'{folder}/fold_metrics.csv'
    oof_path = f'{folder}/oof_predictions.csv'
    if not (os.path.exists(fm_path) and os.path.exists(oof_path)):
        print(f"  [SKIP] {model_key}: CSVs not found in {folder} (model not trained yet?)")
        return None, None
    fm  = pd.read_csv(fm_path)
    oof = pd.read_csv(oof_path)
    return fm, oof


def plot_confusion_house(ax, y_true, y_pred, title, fontsize=15):
    """House-styled confusion matrix: row-normalized % coloring,
    'count\\n(pct%)' annotation — matches confusion_matrices_grid.png."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float),
                        where=row_sums != 0) * 100
    ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Healthy', 'AD'], fontsize=13, fontweight='bold')
    ax.set_yticklabels(['Healthy', 'AD'], fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_ylabel('True', fontsize=14)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=10)
    for i in range(2):
        for j in range(2):
            color = 'white' if cm_pct[i, j] > 50 else '#1a1a1a'
            ax.text(j, i, f"{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)",
                    ha='center', va='center', color=color,
                    fontsize=fontsize, fontweight='bold')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#222222')
        spine.set_linewidth(1.2)


def plot_prf_by_class_threshold(y_true, y_prob, thr, model_label, out_path):
    """Precision / Recall / F1 grouped bar, Healthy vs AD, thr=0.5 vs
    tuned threshold — matches fig4_precision_recall_f1.png style."""
    y_pred_default = (y_prob >= 0.5).astype(int)
    y_pred_tuned   = (y_prob >= thr).astype(int)

    p_d, r_d, f_d, _ = precision_recall_fscore_support(
        y_true, y_pred_default, labels=[0, 1], zero_division=0)
    p_t, r_t, f_t, _ = precision_recall_fscore_support(
        y_true, y_pred_tuned, labels=[0, 1], zero_division=0)

    metrics = ['Precision', 'Recall', 'F1-score']
    hc_default = [p_d[0], r_d[0], f_d[0]]
    ad_default = [p_d[1], r_d[1], f_d[1]]
    hc_tuned   = [p_t[0], r_t[0], f_t[0]]
    ad_tuned   = [p_t[1], r_t[1], f_t[1]]

    x = np.arange(len(metrics))
    width = 0.18

    fig, ax = plt.subplots(figsize=(11, 7))
    groups = [
        (x - 1.5 * width, hc_default, COL_HC, 0.55, COL_HC, f'Healthy (thr=0.5)'),
        (x - 0.5 * width, ad_default, COL_AD, 0.55, COL_AD, f'AD (thr=0.5)'),
        (x + 0.5 * width, hc_tuned,   COL_HC, 1.00, 'black', f'Healthy (tuned={thr:.3f})'),
        (x + 1.5 * width, ad_tuned,   COL_AD, 1.00, 'black', f'AD (tuned={thr:.3f})'),
    ]
    for pos, vals, face, a, edge, lbl in groups:
        bars = ax.bar(pos, vals, width, color=face, alpha=a,
                       edgecolor=edge, linewidth=1.6, label=lbl, zorder=3)
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 0.012, f"{h:.3f}",
                    ha='center', fontsize=10.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=15, fontweight='bold')
    ax.set_ylabel('Score', fontsize=15)
    ax.set_ylim(0, 1.12)
    ax.set_title(f'{model_label} — Pooled OOF Precision / Recall / F1 by Class and Threshold',
                 fontsize=15.5, fontweight='bold', pad=14)
    ax.legend(fontsize=11, framealpha=0.95, loc='lower center',
              ncol=2, bbox_to_anchor=(0.5, -0.30))
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def plot_fold_performance(fm, model_label, out_path):
    """AUC / Accuracy / F1 per fold, with mean AUC dashed line +
    shaded ±std band — matches fig1_fold_performance.png style."""
    folds = fm['fold'].values
    aucs  = fm['auc'].values
    accs  = fm['acc_default'].values
    f1s   = fm['f1_default'].values
    mean_auc, std_auc = aucs.mean(), aucs.std()

    x = np.arange(len(folds))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6.8))
    ax.axhspan(mean_auc - std_auc, mean_auc + std_auc, color=COL_ALPHA, alpha=0.10, zorder=0)
    ax.axhline(mean_auc, color=COL_ALPHA, lw=2, ls='--', zorder=2,
               label=f'Mean AUC = {mean_auc:.3f} \u00B1 {std_auc:.3f}')

    b1 = ax.bar(x - width, aucs, width, color=COL_ALPHA, edgecolor='black', linewidth=1.2, label='AUC', zorder=3)
    b2 = ax.bar(x, accs, width, color=COL_HC, edgecolor='black', linewidth=1.2, label='Accuracy', zorder=3)
    b3 = ax.bar(x + width, f1s, width, color=COL_THETA, edgecolor='black', linewidth=1.2, label='F1-score', zorder=3)

    for bars in (b1, b2, b3):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 0.012, f"{h:.3f}",
                    ha='center', fontsize=10.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {int(f)}" for f in folds], fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=15)
    ax.set_ylim(0, 1.10)
    ax.set_title(f'{model_label} — Per-Fold Performance (5-Fold Subject-Level CV)',
                 fontsize=16, fontweight='bold', pad=14)
    ax.legend(fontsize=11.5, framealpha=0.95, loc='lower left')
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


# ── PER-MODEL PLOTS ──────────────────────────────────────────
# Collected here so the comparison section below can reuse them
# without re-reading CSVs.
all_data = {}

for model_key, model_label in MODELS.items():
    print(f"\nProcessing {model_label} ({model_key}) ...")
    fm, oof = load_model_results(model_key)
    if fm is None:
        continue

    y_true = oof['y_true'].values
    y_prob = oof['oof_prob'].values
    thr    = pooled_threshold(y_true, y_prob)
    y_pred_default = (y_prob >= 0.5).astype(int)
    y_pred_pooled  = (y_prob >= thr).astype(int)

    all_data[model_key] = {
        'label': model_label, 'fm': fm, 'oof': oof,
        'y_true': y_true, 'y_prob': y_prob, 'threshold': thr,
    }

    out_dir = f'{PLOTS_DIR}/{model_key}'
    os.makedirs(out_dir, exist_ok=True)
    c = COLORS[model_key]

    # 1) ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color=c, lw=2.6, label=f'AUC = {roc_auc:.4f}', zorder=3)
    ax.plot([0, 1], [0, 1], color='#555555', ls='--', lw=1.2, alpha=0.7, zorder=1)
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title(f'{model_label} \u2014 ROC Curve (Pooled OOF)', fontsize=15, fontweight='bold', pad=12)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=12, loc='lower right', framealpha=0.95)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(f'{out_dir}/roc_curve.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)

    # 2) Precision-Recall curve
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(rec, prec, color=c, lw=2.6, zorder=3)
    ax.set_xlabel('Recall', fontsize=13)
    ax.set_ylabel('Precision', fontsize=13)
    ax.set_title(f'{model_label} \u2014 Precision-Recall Curve (Pooled OOF)', fontsize=15, fontweight='bold', pad=12)
    ax.tick_params(labelsize=11)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(f'{out_dir}/pr_curve.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)

    # 3) Confusion matrices side-by-side (default 0.5 vs pooled-tuned threshold)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    plot_confusion_house(axes[0], y_true, y_pred_default, 'Threshold = 0.5')
    plot_confusion_house(axes[1], y_true, y_pred_pooled, f'Tuned Threshold = {thr:.3f}')
    fig.suptitle(f'{model_label} \u2014 Confusion Matrices (Pooled OOF)', fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{out_dir}/confusion_matrices.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)

    # 3b) NEW — single large confusion matrix (tuned threshold), headline-figure style
    fig, ax = plt.subplots(figsize=(7, 7))
    plot_confusion_house(ax, y_true, y_pred_pooled,
                          f'{model_label} \u2014 Confusion Matrix (Tuned Threshold = {thr:.3f})',
                          fontsize=18)
    fig.tight_layout()
    fig.savefig(f'{out_dir}/confusion_matrix_single.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)

    # 4) Fold-wise metrics — now styled with mean-AUC band (was plain grouped bar)
    plot_fold_performance(fm, model_label, f'{out_dir}/fold_metrics_bar.png')

    # 4b) NEW — Precision/Recall/F1 by class and threshold
    plot_prf_by_class_threshold(y_true, y_prob, thr, model_label,
                                 f'{out_dir}/prf_by_class_threshold.png')

    print(f"  Saved 6 plots to {out_dir}/")

    # 5) DB-HLSTM only — training curves, if per-fold epoch logs exist
    if model_key == 'dbhlstm':
        log_paths = [f'{BASE_RESULTS_DIR}/{model_key}/fold{n}_log.csv' for n in range(1, 6)]
        existing_logs = [p for p in log_paths if os.path.exists(p)]
        if existing_logs:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for p in existing_logs:
                log_df = pd.read_csv(p)
                fold_n = os.path.basename(p).split('_')[0]
                if 'loss' in log_df.columns:
                    axes[0].plot(log_df['loss'], lw=1.6, label=f'{fold_n} train')
                if 'val_loss' in log_df.columns:
                    axes[0].plot(log_df['val_loss'], '--', lw=1.6, label=f'{fold_n} val')
                auc_col = 'auc' if 'auc' in log_df.columns else None
                val_auc_col = 'val_auc' if 'val_auc' in log_df.columns else None
                if auc_col:
                    axes[1].plot(log_df[auc_col], lw=1.6, label=f'{fold_n} train')
                if val_auc_col:
                    axes[1].plot(log_df[val_auc_col], '--', lw=1.6, label=f'{fold_n} val')
            axes[0].set_title('Loss', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Epoch', fontsize=12)
            axes[0].legend(fontsize=7.5, framealpha=0.95)
            axes[1].set_title('AUC', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].legend(fontsize=7.5, framealpha=0.95)
            for a in axes:
                style_axis(a)
            fig.suptitle(f'{model_label} \u2014 Training Curves (all folds)', fontsize=16, fontweight='bold')
            fig.tight_layout()
            fig.savefig(f'{out_dir}/training_curves.png', dpi=DPI, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved training_curves.png (found {len(existing_logs)} fold logs)")
        else:
            print("  (No fold{N}_log.csv files found — skipping training curves plot)")


# ── COMPARISON PLOTS (ACROSS ALL MODELS) ────────────────────
if len(all_data) == 0:
    print("\nNo model results found at all — nothing to compare. "
          "Make sure Steps 8.5/8.6a-d have been run and saved their CSVs.")
else:
    comp_dir = f'{PLOTS_DIR}/comparison'

    # A) Overlaid ROC curves
    fig, ax = plt.subplots(figsize=(7, 7))
    for model_key, d in all_data.items():
        fpr, tpr, _ = roc_curve(d['y_true'], d['y_prob'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLORS[model_key], lw=2.4,
                label=f"{d['label']} (AUC={roc_auc:.3f})", zorder=3)
    ax.plot([0, 1], [0, 1], color='#555555', ls='--', lw=1.2, alpha=0.7, zorder=1)
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title('ROC Curve Comparison \u2014 All Models (Pooled OOF)', fontsize=16, fontweight='bold', pad=12)
    ax.tick_params(labelsize=11.5)
    ax.legend(loc='lower right', fontsize=10.5, framealpha=0.95)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(f'{comp_dir}/roc_comparison.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)

    # B) Overlaid PR curves
    fig, ax = plt.subplots(figsize=(7, 7))
    for model_key, d in all_data.items():
        prec, rec, _ = precision_recall_curve(d['y_true'], d['y_prob'])
        ax.plot(rec, prec, color=COLORS[model_key], lw=2.4, label=d['label'], zorder=3)
    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title('Precision-Recall Comparison \u2014 All Models (Pooled OOF)', fontsize=16, fontweight='bold', pad=12)
    ax.tick_params(labelsize=11.5)
    ax.legend(loc='lower left', fontsize=10.5, framealpha=0.95)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(f'{comp_dir}/pr_comparison.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)

    # C) Mean +/- std AUC / Acc / F1 bar chart across models
    summary_rows = []
    for model_key, d in all_data.items():
        fm = d['fm']
        summary_rows.append({
            'model': d['label'],
            'auc_mean': fm['auc'].mean(), 'auc_std': fm['auc'].std(),
            'acc_mean': fm['acc_default'].mean(), 'acc_std': fm['acc_default'].std(),
            'f1_mean': fm['f1_default'].mean(), 'f1_std': fm['f1_default'].std(),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f'{comp_dir}/all_models_summary.csv', index=False)

    fig, ax = plt.subplots(figsize=(9.5, 6))
    x = np.arange(len(summary_df))
    width = 0.25
    ax.bar(x - width, summary_df['auc_mean'], width, yerr=summary_df['auc_std'],
           color=COL_ALPHA, edgecolor='black', linewidth=1.1, label='AUC', capsize=4, zorder=3)
    ax.bar(x, summary_df['acc_mean'], width, yerr=summary_df['acc_std'],
           color=COL_HC, edgecolor='black', linewidth=1.1, label='Accuracy', capsize=4, zorder=3)
    ax.bar(x + width, summary_df['f1_mean'], width, yerr=summary_df['f1_std'],
           color=COL_THETA, edgecolor='black', linewidth=1.1, label='F1', capsize=4, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df['model'], fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.10)
    ax.set_ylabel('Score (mean \u00B1 std across folds)', fontsize=14)
    ax.set_title('Model Comparison \u2014 CV Metrics', fontsize=16, fontweight='bold', pad=12)
    ax.tick_params(labelsize=11.5)
    ax.legend(fontsize=12, framealpha=0.95)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(f'{comp_dir}/model_comparison_bar.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)

    # D) NEW — confusion matrices grid, all models, tuned threshold, one figure
    n = len(all_data)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 5.6 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, (model_key, d) in zip(axes, all_data.items()):
        y_pred_tuned = (d['y_prob'] >= d['threshold']).astype(int)
        acc = (y_pred_tuned == d['y_true']).mean()
        plot_confusion_house(ax, d['y_true'], y_pred_tuned,
                              f"{d['label']}  (Acc = {acc:.3f})")
    for ax in axes[n:]:
        ax.axis('off')
    fig.suptitle('Confusion Matrices \u2014 Pooled OOF, Tuned Threshold (row-normalized %)',
                 fontsize=17, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{comp_dir}/confusion_matrices_grid.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)

    print(f"\nSaved comparison plots + all_models_summary.csv to {comp_dir}/")
    print("\nSummary:")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\nSTEP 6 (STYLED) — ALL PLOTS GENERATED")
