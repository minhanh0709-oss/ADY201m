"""
40_final_report.py
PHASE D: Generate final comparison tables, figures, and paper content
Combines all model results into a comprehensive report.
"""

import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
exec(open(Path(__file__).parent / '21_utils_cv_metrics.py').read())

from scipy.stats import ttest_rel

DATA_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"


def load_all_results():
    """Load results from all model scripts"""
    all_dfs = []
    files = [
        'baseline_walkforward.csv',
        'gbm_walkforward.csv',
        'hurdle_walkforward.csv',
        'ziln_walkforward.csv',
    ]
    for f in files:
        path = RESULTS_DIR / f
        if path.exists():
            df = pd.read_csv(path)
            all_dfs.append(df)

    if not all_dfs:
        return None
    return pd.concat(all_dfs, ignore_index=True)


def create_comparison_table(all_results):
    """Create comprehensive Table 3"""
    summary = all_results.groupby('Model').agg({
        'MAE': ['mean', 'std'],
        'RMSE': ['mean', 'std'],
        'R2': ['mean', 'std'],
        'Norm_Gini': ['mean', 'std'],
        'Revenue_Capture_10': ['mean', 'std'],
        'Lift_10': ['mean', 'std'],
        'Top5_MAPE': ['mean', 'std'],
        'Decile_MAPE': ['mean', 'std'],
    }).round(4)

    # Flatten columns
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns]
    summary = summary.reset_index()

    # Sort by Norm_Gini (Google's recommendation for CLV)
    summary = summary.sort_values('Norm_Gini_mean', ascending=False).reset_index(drop=True)

    return summary


def format_for_paper(summary):
    """Format for Table 3 in paper (LaTeX-style)"""
    rows = []
    for _, row in summary.iterrows():
        rows.append({
            'Model': row['Model'],
            'MAE': f"{row['MAE_mean']:.2f} ± {row['MAE_std']:.2f}",
            'R²': f"{row['R2_mean']:.4f} ± {row['R2_std']:.4f}",
            'Norm Gini': f"{row['Norm_Gini_mean']:.4f} ± {row['Norm_Gini_std']:.4f}",
            'Revenue@10%': f"{row['Revenue_Capture_10_mean']:.2f}% ± {row['Revenue_Capture_10_std']:.2f}%",
            'Lift@10%': f"{row['Lift_10_mean']:.2f}x",
            'Top5 MAPE': f"{row['Top5_MAPE_mean']:.4f}",
        })
    return pd.DataFrame(rows)


def statistical_testing(all_results, best_model='LightGBM (log)'):
    """Paired t-test comparing best model vs others across windows"""
    print("\n[STATISTICAL TESTING] Paired t-tests vs best model")
    print("-"*70)

    if best_model not in all_results['Model'].unique():
        best_model = all_results['Model'].iloc[0]

    best_scores = all_results[all_results['Model'] == best_model].sort_values('Window')
    print(f"\nBest model: {best_model}")
    print(f"  Norm_Gini per window: {best_scores['Norm_Gini'].values}")

    test_results = []
    for model in all_results['Model'].unique():
        if model == best_model:
            continue

        other_scores = all_results[all_results['Model'] == model].sort_values('Window')

        if len(other_scores) < 2:
            continue

        # Test on Norm_Gini and Revenue_Capture
        for metric in ['Norm_Gini', 'Revenue_Capture_10', 'MAE']:
            best_vals = best_scores[metric].values
            other_vals = other_scores[metric].values

            if len(best_vals) == len(other_vals) and len(best_vals) > 1:
                try:
                    stat, p_value = ttest_rel(best_vals, other_vals)
                    significant = p_value < 0.05

                    test_results.append({
                        'Best': best_model,
                        'Other': model,
                        'Metric': metric,
                        'Best_Mean': best_vals.mean(),
                        'Other_Mean': other_vals.mean(),
                        't_statistic': stat,
                        'p_value': p_value,
                        'Significant': significant
                    })
                except:
                    pass

    df_tests = pd.DataFrame(test_results)
    df_tests.to_csv(RESULTS_DIR / 'statistical_tests.csv', index=False)
    return df_tests


def plot_model_comparison(summary):
    """Plot comprehensive model comparison"""
    print("\n[PLOT] Model comparison...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Sort by Norm Gini for plotting
    df = summary.sort_values('Norm_Gini_mean', ascending=True)

    # Plot 1: Norm Gini (higher is better)
    ax = axes[0, 0]
    bars = ax.barh(df['Model'], df['Norm_Gini_mean'],
                    xerr=df['Norm_Gini_std'], color='#2E86AB',
                    error_kw={'ecolor': 'black', 'capsize': 3})
    ax.set_xlabel('Normalized Gini Coefficient', fontsize=11)
    ax.set_title('Discrimination Ability (Norm Gini)', fontsize=12, fontweight='bold')
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend()

    # Plot 2: Revenue Capture @10% (higher is better)
    ax = axes[0, 1]
    df_sorted = summary.sort_values('Revenue_Capture_10_mean', ascending=True)
    ax.barh(df_sorted['Model'], df_sorted['Revenue_Capture_10_mean'],
            xerr=df_sorted['Revenue_Capture_10_std'], color='#A23B72',
            error_kw={'ecolor': 'black', 'capsize': 3})
    ax.set_xlabel('Revenue Capture @ Top 10% (%)', fontsize=11)
    ax.set_title('Business Value: Revenue Capture @10%', fontsize=12, fontweight='bold')
    ax.axvline(x=10, color='gray', linestyle='--', alpha=0.5, label='Random (10%)')
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend()

    # Plot 3: MAE (lower is better)
    ax = axes[1, 0]
    df_sorted = summary.sort_values('MAE_mean', ascending=False)
    ax.barh(df_sorted['Model'], df_sorted['MAE_mean'],
            xerr=df_sorted['MAE_std'], color='#F18F01',
            error_kw={'ecolor': 'black', 'capsize': 3})
    ax.set_xlabel('Mean Absolute Error ($)', fontsize=11)
    ax.set_title('Prediction Accuracy (MAE)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Plot 4: R² (higher is better, clip for visibility)
    ax = axes[1, 1]
    df_sorted = summary.sort_values('R2_mean', ascending=True)
    r2_clipped = df_sorted['R2_mean'].clip(lower=-1)
    colors = ['#C73E1D' if x < 0 else '#F18F01' if x < 0.5 else '#2E86AB' for x in r2_clipped]
    ax.barh(df_sorted['Model'], r2_clipped, color=colors,
            xerr=df_sorted['R2_std'].clip(upper=1),
            error_kw={'ecolor': 'black', 'capsize': 3})
    ax.set_xlabel('R² Score (clipped to [-1, 1])', fontsize=11)
    ax.set_title('Variance Explained (R²)', fontsize=12, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'final_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: final_model_comparison.png")


def plot_revenue_capture_curve(all_results):
    """Plot Revenue Capture @ K curve"""
    # This needs raw predictions, skip for now or use summary
    pass


def plot_walkforward_stability(all_results):
    """Show how each model performs across windows"""
    print("\n[PLOT] Walk-forward stability...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Norm Gini across windows
    ax = axes[0]
    for model in all_results['Model'].unique():
        df = all_results[all_results['Model'] == model].sort_values('Window')
        ax.plot(df['Window'], df['Norm_Gini'], marker='o', label=model, alpha=0.7)
    ax.set_xlabel('Walk-Forward Window', fontsize=11)
    ax.set_ylabel('Normalized Gini', fontsize=11)
    ax.set_title('Model Stability Across Time Windows', fontsize=12, fontweight='bold')
    ax.set_xticks([1, 2, 3])
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)

    # Plot Revenue Capture across windows
    ax = axes[1]
    for model in all_results['Model'].unique():
        df = all_results[all_results['Model'] == model].sort_values('Window')
        ax.plot(df['Window'], df['Revenue_Capture_10'], marker='s', label=model, alpha=0.7)
    ax.set_xlabel('Walk-Forward Window', fontsize=11)
    ax.set_ylabel('Revenue Capture @10% (%)', fontsize=11)
    ax.set_title('Business Metric Stability', fontsize=12, fontweight='bold')
    ax.set_xticks([1, 2, 3])
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'final_stability.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: final_stability.png")


def main():
    print("\n" + "="*70)
    print("[PHASE D] FINAL REPORT")
    print("="*70)

    all_results = load_all_results()
    if all_results is None:
        print("[ERROR] No results found. Run baseline + GBM + Hurdle + ZILN first.")
        return

    print(f"\n[Data] Loaded results: {len(all_results)} rows, {all_results['Model'].nunique()} models")

    # Create comparison table
    summary = create_comparison_table(all_results)
    summary.to_csv(RESULTS_DIR / 'final_summary.csv', index=False)

    # Format for paper
    paper_table = format_for_paper(summary)
    paper_table.to_csv(RESULTS_DIR / 'TABLE_3_model_comparison.csv', index=False)

    print("\n" + "="*70)
    print("[TABLE 3] FINAL MODEL COMPARISON (Sorted by Norm Gini)")
    print("="*70)
    print(paper_table.to_string(index=False))

    # Statistical testing
    test_results = statistical_testing(all_results)
    if len(test_results) > 0:
        print("\n[STATISTICAL TESTS] Significant differences (p < 0.05):")
        sig = test_results[test_results['Significant']]
        if len(sig) > 0:
            for _, row in sig.iterrows():
                print(f"  {row['Best']} vs {row['Other']} ({row['Metric']}): "
                      f"p={row['p_value']:.4f} *")

    # Plots
    plot_model_comparison(summary)
    plot_walkforward_stability(all_results)

    print("\n" + "="*70)
    print("[DONE] Final report generated")
    print("="*70 + "\n")

    print("Generated files:")
    print(f"  - {RESULTS_DIR / 'final_summary.csv'}")
    print(f"  - {RESULTS_DIR / 'TABLE_3_model_comparison.csv'}")
    print(f"  - {RESULTS_DIR / 'statistical_tests.csv'}")
    print(f"  - {FIGURES_DIR / 'final_model_comparison.png'}")
    print(f"  - {FIGURES_DIR / 'final_stability.png'}")


if __name__ == "__main__":
    main()
