"""
35_statistical_testing.py
PHASE C2: Statistical Testing
Paired t-test comparing all models across walk-forward windows.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ttest_rel, wilcoxon
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_all_results():
    """Load all model results"""
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
    return pd.concat(all_dfs, ignore_index=True)


def run_statistical_tests():
    """Run paired t-tests for all model pairs"""
    print("\n" + "="*70)
    print("[PHASE C2] STATISTICAL TESTING")
    print("="*70)

    all_results = load_all_results()
    print(f"\nLoaded: {len(all_results)} rows, {all_results['Model'].nunique()} models")

    # Pivot: rows = Window, cols = Model, vals = metric
    pivot_gini = all_results.pivot_table(
        index='Window', columns='Model', values='Norm_Gini', aggfunc='first'
    )
    pivot_revenue = all_results.pivot_table(
        index='Window', columns='Model', values='Revenue_Capture_10', aggfunc='first'
    )
    pivot_mae = all_results.pivot_table(
        index='Window', columns='Model', values='MAE', aggfunc='first'
    )

    # Find best model by Norm_Gini
    mean_gini = pivot_gini.mean()
    best_model = mean_gini.idxmax()
    print(f"\nBest model (highest Norm Gini): {best_model}")
    print(f"  Mean Norm Gini: {mean_gini[best_model]:.4f}")

    # Paired t-test best vs each other model
    print("\n" + "="*70)
    print(f"[PAIRED T-TEST] {best_model} vs others (alpha=0.05)")
    print("="*70)

    test_results = []
    for model in pivot_gini.columns:
        if model == best_model:
            continue

        # Get paired values (handle NaN)
        a = pivot_gini[best_model].values
        b = pivot_gini[model].values
        mask = ~(np.isnan(a) | np.isnan(b))

        if mask.sum() < 2:
            continue

        a, b = a[mask], b[mask]

        # T-test
        try:
            t_stat, p_t = ttest_rel(a, b)
        except:
            t_stat, p_t = np.nan, np.nan

        # Wilcoxon (non-parametric)
        try:
            if len(set(a - b)) > 1:
                w_stat, p_w = wilcoxon(a, b, zero_method='wilcox')
            else:
                w_stat, p_w = np.nan, np.nan
        except:
            w_stat, p_w = np.nan, np.nan

        delta = a.mean() - b.mean()
        significant = p_t < 0.05 if not np.isnan(p_t) else False

        marker = " *" if significant else ""
        print(f"\n  {best_model} vs {model}")
        print(f"    Mean diff: {delta:+.4f}")
        print(f"    t-statistic: {t_stat:+.3f}, p-value: {p_t:.4f}{marker}")
        print(f"    Wilcoxon p-value: {p_w:.4f}")

        test_results.append({
            'Best_Model': best_model,
            'Compared_To': model,
            'Best_Mean_Gini': a.mean(),
            'Other_Mean_Gini': b.mean(),
            'Delta': delta,
            't_statistic': t_stat,
            'p_value_ttest': p_t,
            'p_value_wilcoxon': p_w,
            'Significant_p05': significant
        })

    df_tests = pd.DataFrame(test_results)
    df_tests.to_csv(RESULTS_DIR / 'statistical_tests.csv', index=False)

    print("\n" + "="*70)
    print("[SUMMARY] Statistical significance vs best model")
    print("="*70)

    sig_count = df_tests['Significant_p05'].sum()
    total = len(df_tests)
    print(f"\n{best_model} significantly better than {sig_count}/{total} other models (p<0.05)")

    if sig_count > 0:
        print("\nSignificantly outperforms:")
        for _, row in df_tests[df_tests['Significant_p05']].iterrows():
            print(f"  - {row['Compared_To']:30s} (p={row['p_value_ttest']:.4f}, delta={row['Delta']:+.4f})")

    print("\n" + "="*70)
    print("[DONE] Statistical testing complete")
    print("="*70 + "\n")

    return df_tests, best_model


if __name__ == "__main__":
    run_statistical_tests()
