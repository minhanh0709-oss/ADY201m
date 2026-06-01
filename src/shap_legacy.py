"""
11_shap_analysis.py
Feature importance and model explainability analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
FINAL_FILE = DATA_PROCESSED_DIR / "customers_with_labels.csv"

def load_data():
    df = pd.read_csv(FINAL_FILE)
    return df

def compute_feature_importance(df):
    """Compute feature importance using permutation"""
    print("\n[SHAP] Feature Importance (Permutation-based)...")

    feature_cols = ['Frequency', 'Monetary', 'Tenure', 'AvgOrderValue', 'ActiveMonths', 'ProductDiversity']
    X = df[feature_cols].values
    y = df['ActualCLV'].values

    # Baseline predictions (linear model)
    from numpy.linalg import lstsq
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    beta = lstsq(X_with_intercept, y, rcond=None)[0]
    baseline_pred = X_with_intercept @ beta

    # Compute baseline MAE
    baseline_mae = np.mean(np.abs(baseline_pred - y))

    # Permutation importance
    importances = {}
    for i, col in enumerate(feature_cols):
        X_perm = X.copy()
        np.random.seed(42)
        np.random.shuffle(X_perm[:, i])

        X_perm_intercept = np.column_stack([np.ones(len(X_perm)), X_perm])
        perm_pred = X_perm_intercept @ beta
        perm_mae = np.mean(np.abs(perm_pred - y))

        importance = perm_mae - baseline_mae
        importances[col] = max(0, importance)  # Clip negative values

    return pd.Series(importances).sort_values(ascending=False)

def plot_feature_importance(importance_series):
    """Plot feature importance"""
    print("[PLOT] Feature importance bar plot...")

    fig, ax = plt.subplots(figsize=(10, 6))
    importance_series.plot(kind='barh', ax=ax, color='#2E86AB')
    ax.set_xlabel('Importance (MAE increase)', fontsize=11)
    ax.set_ylabel('Feature', fontsize=11)
    ax.set_title('Feature Importance (Permutation-based)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'shap_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: shap_feature_importance.png")

def vip_vs_nonvip_analysis(df):
    """Analyze differences between VIP and non-VIP customers"""
    print("\n[ANALYSIS] VIP vs Non-VIP Segment Comparison...")

    vip = df[df['IsVIP'] == 1]
    non_vip = df[df['IsVIP'] == 0]

    features = ['Recency', 'Frequency', 'Monetary', 'Tenure', 'AvgOrderValue', 'ActiveMonths']

    comparison = {
        'Feature': features,
        'VIP_Mean': [vip[f].mean() for f in features],
        'NonVIP_Mean': [non_vip[f].mean() for f in features]
    }

    comparison_df = pd.DataFrame(comparison)
    comparison_df['Difference_Ratio'] = comparison_df['VIP_Mean'] / (comparison_df['NonVIP_Mean'] + 1e-8)

    print("\nVIP vs Non-VIP comparison:")
    print(comparison_df.to_string(index=False))

    return comparison_df

def plot_vip_comparison(comparison_df):
    """Plot VIP vs Non-VIP comparison"""
    print("[PLOT] VIP vs Non-VIP comparison...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, feature in enumerate(comparison_df['Feature']):
        vip_mean = comparison_df.iloc[idx]['VIP_Mean']
        non_vip_mean = comparison_df.iloc[idx]['NonVIP_Mean']

        axes[idx].bar(['Non-VIP', 'VIP'], [non_vip_mean, vip_mean], color=['#F18F01', '#2E86AB'])
        axes[idx].set_ylabel(feature, fontsize=11)
        axes[idx].set_title(f'{feature} by Segment', fontsize=11, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'shap_vip_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: shap_vip_comparison.png")

def explain_sample_customers(df):
    """Create explanations for sample customers"""
    print("\n[EXPLANATION] Sample Customer Explanations...")

    # Select samples
    vip_high = df[df['IsVIP'] == 1].nlargest(1, 'ActualCLV').iloc[0]
    vip_medium = df[df['IsVIP'] == 1].iloc[len(df[df['IsVIP'] == 1]) // 2]
    non_vip = df[df['IsVIP'] == 0].iloc[0]

    samples = {
        'VIP_High': vip_high,
        'VIP_Medium': vip_medium,
        'Non_VIP': non_vip
    }

    explanations = []
    for name, customer in samples.items():
        exp = {
            'Segment': name,
            'ActualCLV': f"${customer['ActualCLV']:,.2f}",
            'Frequency': int(customer['Frequency']),
            'Monetary': f"${customer['Monetary']:,.2f}",
            'Recency': int(customer['Recency']),
            'AvgOrderValue': f"${customer['AvgOrderValue']:,.2f}",
            'Tenure': int(customer['Tenure'])
        }
        explanations.append(exp)

    exp_df = pd.DataFrame(explanations)
    print("\nSample customer explanations:")
    print(exp_df.to_string(index=False))

    return exp_df

def main():
    print("\n" + "="*70)
    print("[EXPLAINABILITY] Model Interpretation & Feature Analysis")
    print("="*70)

    df = load_data()

    # Feature importance
    importance = compute_feature_importance(df)
    print("\nFeature Importance:")
    print(importance)

    # Plot importance
    plot_feature_importance(importance)

    # VIP comparison
    comparison = vip_vs_nonvip_analysis(df)
    plot_vip_comparison(comparison)

    # Sample explanations
    sample_exp = explain_sample_customers(df)

    # Save results
    print("\n" + "="*70)
    print("[SAVE RESULTS]")
    print("="*70)

    importance.to_csv(RESULTS_DIR / 'shap_feature_importance.csv', header=['Importance'])
    comparison.to_csv(RESULTS_DIR / 'shap_vip_comparison.csv', index=False)
    sample_exp.to_csv(RESULTS_DIR / 'shap_sample_explanations.csv', index=False)

    print("[OK] SHAP analysis results saved")

    print("\n" + "="*70)
    print("[DONE] Explainability analysis completed!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
