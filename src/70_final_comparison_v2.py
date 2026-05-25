"""
70_final_comparison_v2.py
Final comparison including ALL models (classical + ML + DL + SOTA)
17 models total: baselines + GBM + Hurdle + ZILN + OptDist + MCD + dRNN
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"


def load_all_results():
    """Load all model results including SOTA experiments"""
    all_dfs = []
    files = [
        'baseline_walkforward.csv',
        'gbm_walkforward.csv',
        'hurdle_walkforward.csv',
        'ziln_walkforward.csv',
        'optdist_walkforward.csv',
        'mcd_walkforward.csv',
        'drnn_walkforward.csv',
    ]
    for f in files:
        path = RESULTS_DIR / f
        if path.exists():
            df = pd.read_csv(path)
            all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)


def categorize_model(model_name):
    """Add category for grouping"""
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


def create_master_table():
    """Generate paper-ready master table"""
    print("\n" + "="*70)
    print("[FINAL] Master Comparison Table")
    print("="*70)

    all_results = load_all_results()
    print(f"\nTotal: {len(all_results)} rows, {all_results['Model'].nunique()} unique models")

    # Aggregate
    summary = all_results.groupby('Model').agg({
        'MAE': ['mean', 'std'],
        'R2': ['mean', 'std'],
        'Norm_Gini': ['mean', 'std'],
        'Revenue_Capture_10': ['mean', 'std'],
        'Revenue_Capture_20': ['mean', 'std'],
        'Lift_10': ['mean', 'std'],
        'Top5_MAPE': ['mean', 'std'],
    }).round(4)

    summary.columns = ['_'.join(c).strip('_') for c in summary.columns]
    summary = summary.reset_index()
    summary['Category'] = summary['Model'].apply(categorize_model)
    summary = summary.sort_values('Norm_Gini_mean', ascending=False).reset_index(drop=True)
    summary.to_csv(RESULTS_DIR / 'MASTER_TABLE.csv', index=False)

    # Print top 10
    print("\n[Top 10 Models by Norm Gini]")
    print("="*100)
    print(f"{'Rank':<5}{'Model':<35}{'Category':<28}{'Norm Gini':<15}{'Revenue@10':<15}")
    print("="*100)
    for i, row in summary.head(10).iterrows():
        print(f"{i+1:<5}{row['Model']:<35}{row['Category']:<28}"
              f"{row['Norm_Gini_mean']:.4f}±{row['Norm_Gini_std']:.3f}    "
              f"{row['Revenue_Capture_10_mean']:.2f}±{row['Revenue_Capture_10_std']:.2f}%")

    return all_results, summary


def plot_master_comparison(all_results, summary):
    """Polished publication-quality figure"""
    print("\n[Polished Figure] Master comparison...")

    # Setup style
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
    })

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # Color by category
    cat_colors = {
        'Simple Baselines': '#95A5A6',
        'Probabilistic': '#E74C3C',
        'Linear': '#F39C12',
        'Gradient Boosting': '#3498DB',
        'Two-Stage (Hurdle)': '#27AE60',
        'Deep Learning (ZILN-family)': '#9B59B6',
        'Deep Learning (Sequence)': '#1ABC9C',
    }

    # ===== Panel 1: Norm Gini =====
    ax = fig.add_subplot(gs[0, 0])
    df_sorted = summary.sort_values('Norm_Gini_mean', ascending=True)
    colors = [cat_colors[c] for c in df_sorted['Category']]
    bars = ax.barh(df_sorted['Model'], df_sorted['Norm_Gini_mean'],
                    xerr=df_sorted['Norm_Gini_std'],
                    color=colors, alpha=0.85,
                    error_kw={'ecolor': 'black', 'capsize': 3, 'elinewidth': 1})
    ax.set_xlabel('Normalized Gini Coefficient', fontsize=11)
    ax.set_title('(a) Discrimination: Normalized Gini', fontsize=12)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    ax.tick_params(axis='y', labelsize=9)

    # ===== Panel 2: Revenue Capture@10% =====
    ax = fig.add_subplot(gs[0, 1])
    df_sorted = summary.sort_values('Revenue_Capture_10_mean', ascending=True)
    colors = [cat_colors[c] for c in df_sorted['Category']]
    ax.barh(df_sorted['Model'], df_sorted['Revenue_Capture_10_mean'],
             xerr=df_sorted['Revenue_Capture_10_std'],
             color=colors, alpha=0.85,
             error_kw={'ecolor': 'black', 'capsize': 3})
    ax.set_xlabel('Revenue Capture @ Top 10% (%)', fontsize=11)
    ax.set_title('(b) Business Value: Revenue Capture', fontsize=12)
    ax.axvline(x=10, color='red', linestyle='--', alpha=0.5, label='Random (10%)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')
    ax.tick_params(axis='y', labelsize=9)

    # ===== Panel 3: MAE =====
    ax = fig.add_subplot(gs[1, 0])
    df_sorted = summary.sort_values('MAE_mean', ascending=False)
    # Cap MAE for visibility
    mae_capped = df_sorted['MAE_mean'].clip(upper=2000)
    colors = [cat_colors[c] for c in df_sorted['Category']]
    ax.barh(df_sorted['Model'], mae_capped,
             xerr=df_sorted['MAE_std'].clip(upper=500),
             color=colors, alpha=0.85,
             error_kw={'ecolor': 'black', 'capsize': 3})
    ax.set_xlabel('Mean Absolute Error ($, capped at $2000)', fontsize=11)
    ax.set_title('(c) Prediction Accuracy: MAE', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    ax.tick_params(axis='y', labelsize=9)

    # ===== Panel 4: Top 5% MAPE =====
    ax = fig.add_subplot(gs[1, 1])
    df_filtered = summary[summary['Top5_MAPE_mean'].notna()].copy()
    df_sorted = df_filtered.sort_values('Top5_MAPE_mean', ascending=False)
    colors = [cat_colors[c] for c in df_sorted['Category']]
    ax.barh(df_sorted['Model'], df_sorted['Top5_MAPE_mean'].clip(upper=2),
             color=colors, alpha=0.85)
    ax.set_xlabel('Top 5% MAPE (lower = better)', fontsize=11)
    ax.set_title('(d) VIP Identification Error', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    ax.tick_params(axis='y', labelsize=9)

    # Legend (shared)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.85) for c in cat_colors.values()]
    fig.legend(handles, cat_colors.keys(),
                loc='lower center', ncol=4, fontsize=10,
                bbox_to_anchor=(0.5, -0.02),
                frameon=True, fancybox=True, shadow=False)

    plt.suptitle('Comparison of 17 CLV Prediction Models on Online Retail II\n'
                 '(Mean ± Std across 3 Walk-Forward Windows)',
                 fontsize=14, fontweight='bold', y=1.00)

    plt.savefig(FIGURES_DIR / 'master_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: master_comparison.png")


def plot_category_summary(summary):
    """Best model per category"""
    print("\n[Plot] Category-wise best models...")

    best_per_cat = summary.sort_values('Norm_Gini_mean', ascending=False).groupby('Category').first().reset_index()
    best_per_cat = best_per_cat.sort_values('Norm_Gini_mean', ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {
        'Simple Baselines': '#95A5A6',
        'Probabilistic': '#E74C3C',
        'Linear': '#F39C12',
        'Gradient Boosting': '#3498DB',
        'Two-Stage (Hurdle)': '#27AE60',
        'Deep Learning (ZILN-family)': '#9B59B6',
        'Deep Learning (Sequence)': '#1ABC9C',
    }
    bar_colors = [colors[c] for c in best_per_cat['Category']]

    bars = ax.bar(range(len(best_per_cat)), best_per_cat['Norm_Gini_mean'],
                   yerr=best_per_cat['Norm_Gini_std'],
                   color=bar_colors, alpha=0.85,
                   error_kw={'ecolor': 'black', 'capsize': 5, 'elinewidth': 1.5})

    # Annotations
    for i, row in best_per_cat.reset_index(drop=True).iterrows():
        ax.text(i, row['Norm_Gini_mean'] + row['Norm_Gini_std'] + 0.01,
                row['Model'], ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(range(len(best_per_cat)))
    ax.set_xticklabels(best_per_cat['Category'], rotation=15, ha='right', fontsize=10)
    ax.set_ylabel('Normalized Gini (best model per category)', fontsize=11)
    ax.set_title('Best CLV Model in Each Category', fontsize=13, fontweight='bold')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'best_per_category.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: best_per_category.png")


def statistical_test_v2(all_results):
    """Updated statistical testing with all 17 models"""
    print("\n[Statistical Testing] Hurdle vs all others")

    pivot_gini = all_results.pivot_table(
        index='Window', columns='Model', values='Norm_Gini', aggfunc='first'
    )

    best_model = pivot_gini.mean().idxmax()
    print(f"  Best model: {best_model} (mean Norm Gini = {pivot_gini.mean()[best_model]:.4f})")

    test_results = []
    for model in pivot_gini.columns:
        if model == best_model:
            continue

        a = pivot_gini[best_model].values
        b = pivot_gini[model].values
        mask = ~(np.isnan(a) | np.isnan(b))
        if mask.sum() < 2:
            continue
        a, b = a[mask], b[mask]

        try:
            t_stat, p_value = ttest_rel(a, b)
            delta = a.mean() - b.mean()
            sig = p_value < 0.05
            test_results.append({
                'Best': best_model,
                'Compared': model,
                'Delta': delta,
                't_stat': t_stat,
                'p_value': p_value,
                'Significant': sig
            })
        except:
            pass

    df_tests = pd.DataFrame(test_results).sort_values('p_value')
    df_tests.to_csv(RESULTS_DIR / 'final_statistical_tests.csv', index=False)

    # Print significant
    sig_df = df_tests[df_tests['Significant']]
    print(f"\n  Significantly better than {len(sig_df)} models (p<0.05):")
    for _, row in sig_df.iterrows():
        print(f"    vs {row['Compared']:35s} | dGini: {row['Delta']:+.4f} | p={row['p_value']:.4f}")

    # Print non-significant
    nonsig_df = df_tests[~df_tests['Significant']]
    print(f"\n  Statistically equivalent to {len(nonsig_df)} models (p>=0.05):")
    for _, row in nonsig_df.iterrows():
        print(f"    vs {row['Compared']:35s} | dGini: {row['Delta']:+.4f} | p={row['p_value']:.4f}")

    return df_tests


def main():
    print("\n" + "="*70)
    print("[FINAL ANALYSIS] All Models (17 total) Including SOTA")
    print("="*70)

    all_results, summary = create_master_table()
    plot_master_comparison(all_results, summary)
    plot_category_summary(summary)
    df_tests = statistical_test_v2(all_results)

    print("\n" + "="*70)
    print("[DONE] Final comparison complete")
    print("="*70)


if __name__ == "__main__":
    main()
