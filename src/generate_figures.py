"""
19b_figures_v2.py
Generate updated figures for V2 experiments:

  Fig V2-B: Hurdle V2 variants comparison (NG + RC10 + Stage1 AUC)
  Fig V2-C: CQR coverage with alpha=0.05 (95% PI, q_hat > 0 visible)
  Fig V2-D: SHAP for Hurdle-AllV2 with supervised-selected semantic features
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})


def fig_hurdle_v2():
    print("[V2-B] Hurdle V2 variants comparison...")
    df = pd.read_csv(RESULTS_DIR / 'semantic_walkforward_v2.csv')
    variants = ['Hurdle-RFM', 'Hurdle-Seq', 'Hurdle-SemV2', 'Hurdle-AllV2']
    colors   = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    labels   = ['RFM Only', '+ Sequence', '+ Semantic-V2', '+ Both (AllV2)']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, metric, ylabel, title in [
        (axes[0], 'Norm_Gini',          'Normalized Gini',           'Ranking Quality'),
        (axes[1], 'Revenue_Capture_10', 'Revenue Capture@10% (%)',   'VIP Targeting'),
        (axes[2], 'Stage1_AUC',         'Stage-1 AUC',               'Purchase Classifier'),
    ]:
        means, stds = [], []
        for v in variants:
            sub = df[df['Model'] == v][metric]
            means.append(sub.mean())
            stds.append(sub.std())

        x = np.arange(len(variants))
        bars = ax.bar(x, means, color=colors, alpha=0.85,
                      yerr=stds, capsize=5, width=0.6, edgecolor='white')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(stds) * 0.3,
                    f'{m:.3f}', ha='center', va='bottom', fontsize=9)

        ax.axhline(means[0], color='grey', linestyle='--', alpha=0.5, linewidth=1)

    plt.suptitle('Hurdle V2 Variants: Supervised Dim Selection + Recency-Aware Semantic',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    out = FIGURES_DIR / 'fig_hurdle_v2_variants.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def fig_cqr_alpha05():
    print("[V2-C] CQR alpha=0.05 comparison...")
    df = pd.read_csv(RESULTS_DIR / 'conformal_prediction_alpha05.csv')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: Coverage comparison
    ax = axes[0]
    windows = df['Window'].values
    raw_cov = df['Raw_Cov_Test'].values
    cqr_cov = df['CQR_Coverage'].values
    x = np.arange(len(windows))
    width = 0.35
    bars1 = ax.bar(x - width/2, raw_cov, width, label='Raw Quantile (no CQR)',
                   color='#95a5a6', alpha=0.85)
    bars2 = ax.bar(x + width/2, cqr_cov, width, label='CQR (calibrated)',
                   color='#2ecc71', alpha=0.85)
    ax.axhline(95, color='#e74c3c', linestyle='--', linewidth=2, label='Target 95%')
    ax.axhline(32, color='#3498db', linestyle=':',  linewidth=2, label='MCD-ZILN 95% PI')
    ax.set_xticks(x)
    ax.set_xticklabels([f'W{w}' for w in windows])
    ax.set_xlabel('Walk-forward Window')
    ax.set_ylabel('Empirical Coverage (%)')
    ax.set_title('Coverage: Raw Quantile vs CQR\n(target 95%; MCD-ZILN baseline 32%)')
    ax.set_ylim(0, 110)
    ax.legend(loc='lower right', fontsize=8)
    for bar, c in zip(bars2, cqr_cov):
        ax.text(bar.get_x() + bar.get_width()/2, c + 1,
                f'{c:.1f}%', ha='center', va='bottom', fontsize=8)

    # Right: q_hat values
    ax = axes[1]
    qhat = df['q_hat'].values
    ax.bar(x, qhat, color='#9b59b6', alpha=0.85, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels([f'W{w}' for w in windows])
    ax.set_xlabel('Walk-forward Window')
    ax.set_ylabel(r'$\hat{q}$ (log-scale CQR adjustment)')
    ax.set_title(r'CQR Adjustment $\hat{q}$ per Window')
    for i, q in enumerate(qhat):
        ax.text(i, q + max(qhat)*0.02, f'{q:.3f}',
                ha='center', va='bottom', fontsize=9)

    plt.suptitle(r'Conformalized Quantile Regression ($\alpha$=0.05, 95% PI target)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    out = FIGURES_DIR / 'fig_cqr_alpha05.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def fig_shap_v2():
    print("[V2-D] SHAP for Hurdle-AllV2...")
    shap = pd.read_csv(RESULTS_DIR / 'shap_hurdle_allv2_stage2.csv')
    shap = shap.head(20).copy()

    def feat_group(name):
        n = str(name)
        if n.startswith('sem_full'):  return 'Semantic (full)'
        if n.startswith('sem_rec'):   return 'Semantic (recent)'
        if 'sem_seq' in n:            return 'Sem x Seq Interaction'
        if 'sem_drift' in n:          return 'Semantic (drift)'
        if 'sem_' in n:               return 'Semantic (other)'
        if n.startswith('seq_'):      return 'Sequence'
        if n in ['M_per_F','M_per_T','Active_ratio']: return 'Interaction'
        if n in ['Recency','Frequency','Monetary']:   return 'RFM'
        return 'Behavioral'

    grp_colors = {
        'Semantic (full)':    '#e74c3c',
        'Semantic (recent)':  '#c0392b',
        'Sem x Seq Interaction': '#f39c12',
        'Semantic (drift)':   '#d35400',
        'Sequence':           '#3498db',
        'RFM':                '#2ecc71',
        'Interaction':        '#9b59b6',
        'Behavioral':         '#7f8c8d',
    }
    shap['group'] = shap['feature'].apply(feat_group)
    shap['color'] = shap['group'].map(grp_colors).fillna('#bdc3c7')

    fig, ax = plt.subplots(figsize=(9, 7))
    y = np.arange(len(shap))
    ax.barh(y, shap['mean_abs_shap'], color=shap['color'], alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(shap['feature'], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP value| (Stage 2: Revenue Regressor)')
    ax.set_title('Hurdle-AllV2 Feature Importance (Supervised-Selected Semantic)\nWindow 3 SHAP TreeExplainer')

    patches = [mpatches.Patch(color=c, label=g)
               for g, c in grp_colors.items() if g in shap['group'].values]
    ax.legend(handles=patches, loc='lower right', fontsize=9)

    plt.tight_layout()
    out = FIGURES_DIR / 'fig_semantic_shap_v2.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    fig_hurdle_v2()
    fig_cqr_alpha05()
    fig_shap_v2()
    print("\n[DONE] V2 figures generated")


if __name__ == "__main__":
    main()
