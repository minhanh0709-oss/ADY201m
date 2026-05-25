"""
80_polish_figures.py
Phase Y: Publication-quality figures for paper
- Consistent styling across all figures
- Color-blind friendly palette
- High DPI (300+)
- Vector format option (PDF)
- Bigger fonts for paper readability
"""

import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

DATA_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
PUB_FIGURES_DIR = Path(__file__).parent.parent / "figures_publication"
PUB_FIGURES_DIR.mkdir(exist_ok=True)


# Publication style
PUB_STYLE = {
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
    'lines.linewidth': 2.0,
    'lines.markersize': 8,
}

# Color-blind safe palette (Wong, 2011)
COLORS = {
    'blue': '#0072B2',
    'orange': '#E69F00',
    'green': '#009E73',
    'red': '#D55E00',
    'purple': '#CC79A7',
    'sky': '#56B4E9',
    'yellow': '#F0E442',
    'gray': '#7F7F7F',
}

CATEGORY_COLORS = {
    'Simple Baselines': COLORS['gray'],
    'Probabilistic': COLORS['red'],
    'Linear': COLORS['orange'],
    'Gradient Boosting': COLORS['blue'],
    'Two-Stage (Hurdle)': COLORS['green'],
    'Deep Learning (ZILN-family)': COLORS['purple'],
    'Deep Learning (Sequence)': COLORS['sky'],
}


def categorize_model(model_name):
    name = model_name.lower()
    if 'mean predictor' in name or 'monetary' in name or 'rfm score' in name:
        return 'Simple Baselines'
    if 'bg/nbd' in name:
        return 'Probabilistic'
    if 'linear' in name or 'ridge' in name:
        return 'Linear'
    if 'lightgbm' in name or 'xgboost' in name:
        return 'Gradient Boosting'
    if 'hurdle' in name:
        return 'Two-Stage (Hurdle)'
    if 'ziln' in name or 'optdist' in name or 'mcd' in name:
        return 'Deep Learning (ZILN-family)'
    if 'drnn' in name:
        return 'Deep Learning (Sequence)'
    return 'Other'


def setup_style():
    plt.rcParams.update(PUB_STYLE)


def figure_1_pipeline():
    """Figure 1: Research pipeline (text-based visual)"""
    setup_style()
    fig, ax = plt.subplots(figsize=(14, 6))

    stages = [
        ('Online Retail II\nDataset', '#E0E0E0', 0),
        ('Data Cleaning\n(805K transactions)', '#FFE082', 1),
        ('Feature Engineering\nRFM + Behavioral + Sequence', '#A5D6A7', 2),
        ('Walk-Forward CV\n3 temporal windows', '#90CAF9', 3),
        ('Model Training\n17 models compared', '#CE93D8', 4),
        ('Evaluation\nNorm Gini, Top-K MAPE', '#FFAB91', 5),
        ('VIP Targeting\n+ SHAP Analysis', '#80CBC4', 6),
    ]

    box_w = 1.6
    box_h = 1.2

    for stage, color, i in stages:
        x = i * 2
        rect = mpl.patches.FancyBboxPatch(
            (x, 0), box_w, box_h,
            boxstyle="round,pad=0.05",
            linewidth=1.5, edgecolor='black', facecolor=color
        )
        ax.add_patch(rect)
        ax.text(x + box_w/2, box_h/2, stage,
                ha='center', va='center', fontsize=11, fontweight='bold',
                wrap=True)

        if i < len(stages) - 1:
            ax.annotate('', xy=(x + box_w + 0.3, box_h/2),
                       xytext=(x + box_w + 0.05, box_h/2),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    ax.set_xlim(-0.3, len(stages) * 2)
    ax.set_ylim(-0.2, box_h + 0.2)
    ax.axis('off')
    ax.set_title('Figure 1: Research Pipeline for CLV Prediction', fontsize=15, fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig(PUB_FIGURES_DIR / 'fig1_pipeline.png', dpi=300, bbox_inches='tight')
    plt.savefig(PUB_FIGURES_DIR / 'fig1_pipeline.pdf', bbox_inches='tight')
    plt.close()
    print("  Saved: fig1_pipeline.png/pdf")


def figure_2_temporal_split():
    """Figure 2: Walk-forward window visualization"""
    setup_style()
    fig, ax = plt.subplots(figsize=(13, 5))

    # Timeline
    from datetime import datetime
    start_date = datetime(2009, 12, 1)
    end_date = datetime(2011, 12, 9)
    total_days = (end_date - start_date).days

    windows = [
        ('Window 1', datetime(2009, 12, 1), datetime(2010, 11, 30),
         datetime(2010, 12, 1), datetime(2011, 2, 28), COLORS['blue']),
        ('Window 2', datetime(2009, 12, 1), datetime(2011, 2, 28),
         datetime(2011, 3, 1), datetime(2011, 5, 31), COLORS['green']),
        ('Window 3', datetime(2009, 12, 1), datetime(2011, 5, 31),
         datetime(2011, 6, 1), datetime(2011, 12, 9), COLORS['purple']),
    ]

    for i, (name, obs_s, obs_e, pred_s, pred_e, color) in enumerate(windows):
        y = i + 1
        obs_start_d = (obs_s - start_date).days
        obs_end_d = (obs_e - start_date).days
        pred_start_d = (pred_s - start_date).days
        pred_end_d = (pred_e - start_date).days

        # Observation window
        ax.barh(y, obs_end_d - obs_start_d, left=obs_start_d, height=0.6,
                color=color, alpha=0.7, edgecolor='black', linewidth=1.2,
                label='Observation' if i == 0 else '')

        # Prediction window (hatched)
        ax.barh(y, pred_end_d - pred_start_d, left=pred_start_d, height=0.6,
                color=color, alpha=0.3, edgecolor='black', linewidth=1.2,
                hatch='//', label='Prediction' if i == 0 else '')

        ax.text(-50, y, name, ha='right', va='center', fontsize=12, fontweight='bold')

    # X-axis: months
    import matplotlib.dates as mdates
    from datetime import timedelta
    n_months = 25
    tick_positions = []
    tick_labels = []
    for i in range(n_months):
        d = start_date + timedelta(days=i * 30)
        tick_positions.append(i * 30)
        if i % 3 == 0:
            tick_labels.append(d.strftime('%Y-%m'))
        else:
            tick_labels.append('')

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0, fontsize=9)
    ax.set_yticks([])
    ax.set_xlabel('Time', fontsize=12)
    ax.set_title('Figure 2: Walk-Forward Cross-Validation Windows',
                  fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(-100, total_days + 50)
    ax.set_ylim(0.3, 3.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(PUB_FIGURES_DIR / 'fig2_temporal_split.png', dpi=300, bbox_inches='tight')
    plt.savefig(PUB_FIGURES_DIR / 'fig2_temporal_split.pdf', bbox_inches='tight')
    plt.close()
    print("  Saved: fig2_temporal_split.png/pdf")


def figure_3_master_comparison():
    """Figure 3: Master model comparison (polished)"""
    setup_style()

    df = pd.read_csv(RESULTS_DIR / 'MASTER_TABLE.csv')
    df['Category'] = df['Model'].apply(categorize_model)

    fig, axes = plt.subplots(1, 2, figsize=(16, 9))

    # Sort by Norm Gini
    df_sorted = df.sort_values('Norm_Gini_mean', ascending=True)
    colors = [CATEGORY_COLORS[c] for c in df_sorted['Category']]

    # ===== Left: Norm Gini =====
    ax = axes[0]
    y_pos = np.arange(len(df_sorted))
    bars = ax.barh(y_pos, df_sorted['Norm_Gini_mean'],
                    xerr=df_sorted['Norm_Gini_std'],
                    color=colors, alpha=0.85, edgecolor='black', linewidth=0.5,
                    error_kw={'ecolor': 'black', 'capsize': 4, 'elinewidth': 1.2})

    # Highlight top 3
    for i in [-1, -2, -3]:
        bars[i].set_edgecolor('red')
        bars[i].set_linewidth(2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted['Model'], fontsize=10)
    ax.set_xlabel('Normalized Gini Coefficient (higher = better ranking)', fontsize=12)
    ax.set_title('(a) Discrimination: Normalized Gini', fontsize=13, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.axvline(x=df_sorted['Norm_Gini_mean'].max(), color='red',
                linestyle='--', alpha=0.5, label=f"Best: {df_sorted['Norm_Gini_mean'].max():.3f}")
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(loc='lower right', fontsize=10)

    # ===== Right: Revenue Capture @ 10% =====
    ax = axes[1]
    df_sorted2 = df.sort_values('Revenue_Capture_10_mean', ascending=True)
    colors2 = [CATEGORY_COLORS[c] for c in df_sorted2['Category']]
    y_pos2 = np.arange(len(df_sorted2))

    bars2 = ax.barh(y_pos2, df_sorted2['Revenue_Capture_10_mean'],
                     xerr=df_sorted2['Revenue_Capture_10_std'],
                     color=colors2, alpha=0.85, edgecolor='black', linewidth=0.5,
                     error_kw={'ecolor': 'black', 'capsize': 4, 'elinewidth': 1.2})

    # Highlight top 3
    for i in [-1, -2, -3]:
        bars2[i].set_edgecolor('red')
        bars2[i].set_linewidth(2)

    ax.set_yticks(y_pos2)
    ax.set_yticklabels(df_sorted2['Model'], fontsize=10)
    ax.set_xlabel('Revenue Capture @ Top 10% (%)', fontsize=12)
    ax.set_title('(b) Business Value: Revenue Capture', fontsize=13, fontweight='bold')
    ax.axvline(x=10, color='red', linestyle='--', alpha=0.5, label='Random = 10%')
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(0, 80)

    # Shared legend
    handles = [Patch(color=c, label=cat, alpha=0.85)
               for cat, c in CATEGORY_COLORS.items()]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=10,
                bbox_to_anchor=(0.5, -0.02), frameon=True)

    plt.suptitle('Figure 3: Comparison of 17 CLV Prediction Models\n'
                 '(Mean ± Std across 3 Walk-Forward Windows)',
                 fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(PUB_FIGURES_DIR / 'fig3_master_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(PUB_FIGURES_DIR / 'fig3_master_comparison.pdf', bbox_inches='tight')
    plt.close()
    print("  Saved: fig3_master_comparison.png/pdf")


def figure_4_revenue_capture_curve():
    """Figure 4: Revenue Capture @ K curve (polished)"""
    setup_style()

    df = pd.read_csv(RESULTS_DIR / 'revenue_capture_curve.csv')

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Models to show
    models_show = {
        'Random': COLORS['gray'],
        'Monetary': COLORS['orange'],
        'RFM Score': COLORS['red'],
        'XGBoost': COLORS['blue'],
        'Hurdle (Proposed)': COLORS['green'],
        'Oracle': COLORS['purple'],
    }
    styles = {'Random': '--', 'Oracle': ':'}
    markers = {'Hurdle (Proposed)': 'o', 'Oracle': 's'}

    # ===== Left: Capture curve =====
    ax = axes[0]
    for model, color in models_show.items():
        col_mean = f'{model}_mean'
        col_std = f'{model}_std'
        if col_mean not in df.columns:
            continue

        ls = styles.get(model, '-')
        mk = markers.get(model, None)
        lw = 3 if model == 'Hurdle (Proposed)' else 2

        ax.plot(df['K_percent'], df[col_mean], label=model, color=color,
                linestyle=ls, linewidth=lw, marker=mk, markersize=7)
        ax.fill_between(df['K_percent'],
                          df[col_mean] - df[col_std],
                          df[col_mean] + df[col_std],
                          alpha=0.15, color=color)

    # Add reference line
    ax.plot([0, 100], [0, 100], 'k:', alpha=0.3, linewidth=1)

    ax.set_xlabel('Top K% Customers Targeted', fontsize=12)
    ax.set_ylabel('Revenue Captured (%)', fontsize=12)
    ax.set_title('(a) Revenue Capture @ K Curve', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)

    # Annotate key points
    k10_idx = df.index[df['K_percent'] == 10][0]
    hurdle_10 = df['Hurdle (Proposed)_mean'].iloc[k10_idx]
    random_10 = df['Random_mean'].iloc[k10_idx]
    ax.annotate(f'{hurdle_10:.1f}%',
                xy=(10, hurdle_10), xytext=(20, hurdle_10 - 10),
                arrowprops=dict(arrowstyle='->', color='black', lw=1),
                fontsize=10, fontweight='bold')

    # ===== Right: Lift curve =====
    ax = axes[1]
    for model, color in models_show.items():
        if model == 'Random':
            continue
        col_mean = f'{model}_mean'
        if col_mean not in df.columns:
            continue

        lift = df[col_mean] / df['K_percent']
        ls = styles.get(model, '-')
        mk = markers.get(model, None)
        lw = 3 if model == 'Hurdle (Proposed)' else 2

        ax.plot(df['K_percent'], lift, label=model, color=color,
                linestyle=ls, linewidth=lw, marker=mk, markersize=7)

    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Random (Lift=1)')
    ax.set_xlabel('Top K% Customers Targeted', fontsize=12)
    ax.set_ylabel('Lift (Revenue Capture / K%)', fontsize=12)
    ax.set_title('(b) Targeting Efficiency: Lift', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)

    plt.suptitle('Figure 4: VIP Targeting Performance', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PUB_FIGURES_DIR / 'fig4_revenue_capture.png', dpi=300, bbox_inches='tight')
    plt.savefig(PUB_FIGURES_DIR / 'fig4_revenue_capture.pdf', bbox_inches='tight')
    plt.close()
    print("  Saved: fig4_revenue_capture.png/pdf")


def figure_5_ablation():
    """Figure 5: Ablation study"""
    setup_style()

    df = pd.read_csv(RESULTS_DIR / 'ablation_summary.csv')

    short_labels = ['RFM', '+Behavioral', '+Interactions', '+Sequence']
    bar_colors = [COLORS['gray'], COLORS['orange'], COLORS['blue'], COLORS['green']]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    x = np.arange(len(df))
    bars = ax.bar(x, df['Norm_Gini_mean'], yerr=df['Norm_Gini_std'],
                  color=bar_colors,
                  alpha=0.88, edgecolor='black', linewidth=1.3,
                  error_kw={'ecolor': 'black', 'capsize': 6, 'elinewidth': 1.3})
    for bar, mean_v, std_v in zip(bars, df['Norm_Gini_mean'], df['Norm_Gini_std']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + std_v + 0.004,
                f'{mean_v:.3f}', ha='center', va='bottom',
                fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=13)
    ax.set_ylabel('Normalized Gini', fontsize=14)
    ax.set_title('(a) Discrimination', fontsize=14, fontweight='bold', pad=8)
    ax.set_ylim(0.78, 0.86)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    bars = ax.bar(x, df['Revenue_Capture_10_mean'], yerr=df['Revenue_Capture_10_std'],
                  color=bar_colors,
                  alpha=0.88, edgecolor='black', linewidth=1.3,
                  error_kw={'ecolor': 'black', 'capsize': 6, 'elinewidth': 1.3})
    for bar, mean_v, std_v in zip(bars, df['Revenue_Capture_10_mean'],
                                  df['Revenue_Capture_10_std']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + std_v + 0.25,
                f'{mean_v:.1f}%', ha='center', va='bottom',
                fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=13)
    ax.set_ylabel('Revenue Capture @ Top 10% (%)', fontsize=14)
    ax.set_title('(b) Business Value', fontsize=14, fontweight='bold', pad=8)
    ax.set_ylim(56, 64)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Ablation Study: Cumulative Feature Group Contributions '
                 '(mean ± std across 3 windows)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PUB_FIGURES_DIR / 'fig5_ablation.png', dpi=300, bbox_inches='tight')
    plt.savefig(PUB_FIGURES_DIR / 'fig5_ablation.pdf', bbox_inches='tight')
    plt.close()
    print("  Saved: fig5_ablation.png/pdf")


def figure_6_decile_chart():
    """Figure 6: Decile chart (polished)"""
    setup_style()

    df = pd.read_csv(RESULTS_DIR / 'decile_analysis.csv')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ===== Left: Bar chart predicted vs actual =====
    ax = axes[0]
    x = np.arange(len(df))
    width = 0.4

    bars1 = ax.bar(x - width/2, df['pred_mean'], width,
                    label='Predicted', color=COLORS['blue'],
                    alpha=0.85, edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x + width/2, df['actual_mean'], width,
                    label='Actual', color=COLORS['red'],
                    alpha=0.85, edgecolor='black', linewidth=0.8)

    ax.set_yscale('log')
    ax.set_xlabel('Decile (1 = lowest predicted, 10 = highest)', fontsize=12)
    ax.set_ylabel('Mean CLV ($, log scale)', fontsize=12)
    ax.set_title('(a) Decile Calibration: Hurdle Model', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['decile'])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # ===== Right: Calibration scatter =====
    ax = axes[1]
    sc = ax.scatter(df['pred_mean'], df['actual_mean'],
                    s=200, c=df['decile'], cmap='viridis',
                    edgecolor='black', linewidth=1.5, zorder=3)

    max_val = max(df['pred_mean'].max(), df['actual_mean'].max())
    ax.plot([0, max_val * 1.1], [0, max_val * 1.1],
            'r--', linewidth=2, label='Perfect Calibration')

    for _, row in df.iterrows():
        ax.annotate(f"D{row['decile']}",
                    (row['pred_mean'], row['actual_mean']),
                    textcoords='offset points', xytext=(8, 8),
                    fontsize=10, fontweight='bold')

    ax.set_xlabel('Predicted Mean per Decile ($)', fontsize=12)
    ax.set_ylabel('Actual Mean per Decile ($)', fontsize=12)
    ax.set_title('(b) Calibration Plot', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_val * 1.15)
    ax.set_ylim(0, max_val * 1.15)
    plt.colorbar(sc, ax=ax, label='Decile')

    plt.suptitle('Figure 6: Calibration Analysis (Window 3, n=986 test customers)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PUB_FIGURES_DIR / 'fig6_decile.png', dpi=300, bbox_inches='tight')
    plt.savefig(PUB_FIGURES_DIR / 'fig6_decile.pdf', bbox_inches='tight')
    plt.close()
    print("  Saved: fig6_decile.png/pdf")


def figure_7_permutation_importance():
    """Figure 7: Permutation feature importance"""
    setup_style()

    df = pd.read_csv(RESULTS_DIR / 'permutation_importance.csv')

    # Filter top 15 with positive importance
    df = df[df['importance_mean'] > 0].head(15)

    fig, ax = plt.subplots(figsize=(11, 7))

    # Color by feature type
    def get_color(name):
        if 'seq_' in name:
            return COLORS['sky']
        if 'log_' in name or '_per_' in name or '_ratio' in name:
            return COLORS['orange']
        return COLORS['purple']

    colors = [get_color(f) for f in df['feature']]

    y_pos = np.arange(len(df))[::-1]
    bars = ax.barh(y_pos, df['importance_mean'],
                    xerr=df['importance_std'],
                    color=colors, alpha=0.85, edgecolor='black', linewidth=0.8,
                    error_kw={'ecolor': 'black', 'capsize': 4, 'elinewidth': 1.2})

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['feature'], fontsize=11)
    ax.set_xlabel('Permutation Importance (Norm Gini drop)', fontsize=12)
    ax.set_title('Top 15 Features by Permutation Importance\n'
                  '(Hurdle Model on Window 3)',
                  fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Legend
    handles = [
        Patch(color=COLORS['purple'], label='RFM/Behavioral (raw)', alpha=0.85),
        Patch(color=COLORS['orange'], label='Engineered (log/interactions)', alpha=0.85),
        Patch(color=COLORS['sky'], label='Sequence features', alpha=0.85),
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=11)

    plt.tight_layout()
    plt.savefig(PUB_FIGURES_DIR / 'fig7_importance.png', dpi=300, bbox_inches='tight')
    plt.savefig(PUB_FIGURES_DIR / 'fig7_importance.pdf', bbox_inches='tight')
    plt.close()
    print("  Saved: fig7_importance.png/pdf")


def figure_8_walk_forward_stability():
    """Figure 8: Performance stability across windows"""
    setup_style()

    df_baseline = pd.read_csv(RESULTS_DIR / 'baseline_walkforward.csv')
    df_gbm = pd.read_csv(RESULTS_DIR / 'gbm_walkforward.csv')
    df_hurdle = pd.read_csv(RESULTS_DIR / 'hurdle_walkforward.csv')
    df_ziln = pd.read_csv(RESULTS_DIR / 'ziln_walkforward.csv')
    df_optdist = pd.read_csv(RESULTS_DIR / 'optdist_walkforward.csv')
    df_mcd = pd.read_csv(RESULTS_DIR / 'mcd_walkforward.csv')
    df_drnn = pd.read_csv(RESULTS_DIR / 'drnn_walkforward.csv')
    df_all = pd.concat([df_baseline, df_gbm, df_hurdle, df_ziln,
                          df_optdist, df_mcd, df_drnn], ignore_index=True)

    # Select key models
    key_models = ['Mean Predictor', 'Monetary', 'RFM Score', 'XGBoost (raw)',
                  'Hurdle Model', 'ZILN (Deep Learning)', 'MCD-ZILN',
                  'OptDist (Multi-ZILN)', 'dRNN (Dilated)']

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Norm Gini across windows
    ax = axes[0]
    for model in key_models:
        sub = df_all[df_all['Model'] == model].sort_values('Window')
        if len(sub) > 0:
            cat = categorize_model(model)
            ls = '--' if model in ['Mean Predictor'] else '-'
            lw = 3 if 'Hurdle Model' == model else 1.8
            ax.plot(sub['Window'], sub['Norm_Gini'],
                    marker='o', label=model,
                    color=CATEGORY_COLORS.get(cat, 'black'),
                    linestyle=ls, linewidth=lw, markersize=8, alpha=0.85)

    ax.set_xlabel('Walk-Forward Window', fontsize=12)
    ax.set_ylabel('Normalized Gini', fontsize=12)
    ax.set_title('(a) Discrimination Stability', fontsize=13, fontweight='bold')
    ax.set_xticks([1, 2, 3])
    ax.legend(fontsize=8, loc='lower right', ncol=2)
    ax.grid(True, alpha=0.3)

    # Revenue Capture across windows
    ax = axes[1]
    for model in key_models:
        sub = df_all[df_all['Model'] == model].sort_values('Window')
        if len(sub) > 0:
            cat = categorize_model(model)
            ls = '--' if model in ['Mean Predictor'] else '-'
            lw = 3 if 'Hurdle Model' == model else 1.8
            ax.plot(sub['Window'], sub['Revenue_Capture_10'],
                    marker='s', label=model,
                    color=CATEGORY_COLORS.get(cat, 'black'),
                    linestyle=ls, linewidth=lw, markersize=8, alpha=0.85)

    ax.set_xlabel('Walk-Forward Window', fontsize=12)
    ax.set_ylabel('Revenue Capture @10% (%)', fontsize=12)
    ax.set_title('(b) Business Metric Stability', fontsize=13, fontweight='bold')
    ax.set_xticks([1, 2, 3])
    ax.legend(fontsize=8, loc='lower right', ncol=2)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Figure 8: Model Performance Stability Across Walk-Forward Windows',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PUB_FIGURES_DIR / 'fig8_stability.png', dpi=300, bbox_inches='tight')
    plt.savefig(PUB_FIGURES_DIR / 'fig8_stability.pdf', bbox_inches='tight')
    plt.close()
    print("  Saved: fig8_stability.png/pdf")


def main():
    print("\n" + "="*70)
    print("[PHASE Y] PUBLICATION-QUALITY FIGURES")
    print("="*70)
    print(f"\nOutput directory: {PUB_FIGURES_DIR}")
    print(f"Format: PNG (300 DPI) + PDF (vector)")
    print()

    figure_1_pipeline()
    figure_2_temporal_split()
    figure_3_master_comparison()
    figure_4_revenue_capture_curve()
    figure_5_ablation()
    figure_6_decile_chart()
    figure_7_permutation_importance()
    figure_8_walk_forward_stability()

    print("\n" + "="*70)
    print("[DONE] 8 publication-quality figures generated")
    print(f"Location: {PUB_FIGURES_DIR}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
