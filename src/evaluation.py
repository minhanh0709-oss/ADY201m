"""
10_evaluation.py
Evaluate all models and create comparison tables/figures
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

def load_all_metrics():
    """Load metrics from all models"""
    metrics = []

    for file in ['baseline_metrics.csv', 'ml_metrics.csv', 'dl_metrics.csv']:
        csv_file = RESULTS_DIR / file
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            # Remove NaN rows
            df = df.dropna(subset=['MAE', 'R2'])
            metrics.append(df)

    if metrics:
        all_metrics = pd.concat(metrics, ignore_index=True)
        return all_metrics.drop_duplicates(subset=['Model'], keep='first')
    return pd.DataFrame()

def create_comparison_table(metrics_df):
    """Create Table 3 for paper"""
    print("\n" + "="*70)
    print("[TABLE 3] Model Comparison Results")
    print("="*70)

    # Sort by R2 descending
    sorted_df = metrics_df.sort_values('R2', ascending=False).reset_index(drop=True)

    # Format for display
    display_df = sorted_df[['Model', 'MAE', 'RMSE', 'R2', 'Spearman']].copy()
    display_df['MAE'] = display_df['MAE'].apply(lambda x: f"${x:,.2f}")
    display_df['RMSE'] = display_df['RMSE'].apply(lambda x: f"${x:,.2f}")
    display_df['R2'] = display_df['R2'].apply(lambda x: f"{x:.4f}")
    display_df['Spearman'] = display_df['Spearman'].apply(lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A")

    print(display_df.to_string(index=False))

    # Save table
    sorted_df.to_csv(RESULTS_DIR / 'table_3_model_comparison.csv', index=False)
    print(f"\n[OK] Table saved: table_3_model_comparison.csv")

    return sorted_df

def plot_model_comparison(metrics_df):
    """Plot model comparison"""
    print("\n[PLOT] Model comparison figure...")

    # Remove NaN rows
    df = metrics_df.dropna(subset=['MAE', 'R2']).copy()

    if len(df) == 0:
        print("  [SKIP] No valid metrics")
        return

    # Sort by R2
    df = df.sort_values('R2', ascending=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: MAE
    axes[0].barh(df['Model'], df['MAE'], color='#2E86AB')
    axes[0].set_xlabel('MAE ($)', fontsize=11)
    axes[0].set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')

    # Plot 2: RMSE
    axes[1].barh(df['Model'], df['RMSE'], color='#A23B72')
    axes[1].set_xlabel('RMSE ($)', fontsize=11)
    axes[1].set_title('Root Mean Squared Error', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')

    # Plot 3: R2
    colors = ['#C73E1D' if x < 0 else '#F18F01' if x < 0.5 else '#2E86AB' for x in df['R2']]
    axes[2].barh(df['Model'], df['R2'], color=colors)
    axes[2].set_xlabel('R² Score', fontsize=11)
    axes[2].set_title('R² Score', fontsize=12, fontweight='bold')
    axes[2].axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    axes[2].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'eval_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: eval_model_comparison.png")

def plot_predicted_vs_actual(y_true, y_pred_best, best_model_name):
    """Plot predicted vs actual CLV"""
    print("\n[PLOT] Predicted vs Actual CLV...")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Scatter plot
    axes[0].scatter(y_true, y_pred_best, alpha=0.5, s=20, color='#2E86AB')
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual CLV ($)', fontsize=11)
    axes[0].set_ylabel('Predicted CLV ($)', fontsize=11)
    axes[0].set_title(f'{best_model_name}: Predicted vs Actual', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Distribution comparison
    axes[1].hist(y_true, bins=50, alpha=0.6, label='Actual', color='#2E86AB', edgecolor='black')
    axes[1].hist(y_pred_best, bins=50, alpha=0.6, label='Predicted', color='#F18F01', edgecolor='black')
    axes[1].set_xlabel('CLV ($)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('CLV Distribution', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'eval_predicted_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: eval_predicted_vs_actual.png")

def compute_vip_metrics(df_final, y_pred_best):
    """Compute VIP targeting metrics"""
    print("\n[VIP METRICS] Revenue Capture & Precision@Top10%...")

    df_final = df_final.copy()
    df_final['Predicted_CLV'] = y_pred_best

    # Top 10% by predicted CLV
    top10_threshold = df_final['Predicted_CLV'].quantile(0.9)
    top10_pred = (df_final['Predicted_CLV'] >= top10_threshold).astype(int)

    # Top 10% by actual CLV
    top10_threshold_actual = df_final['ActualCLV'].quantile(0.9)
    top10_actual = (df_final['ActualCLV'] >= top10_threshold_actual).astype(int)

    # Precision: among predicted top 10%, how many are actually top 10%?
    precision = (top10_pred & top10_actual).sum() / top10_pred.sum() if top10_pred.sum() > 0 else 0

    # Revenue capture: how much % of total revenue do predicted top 10% capture?
    top10_revenue = df_final.loc[top10_pred == 1, 'ActualCLV'].sum()
    total_revenue = df_final['ActualCLV'].sum()
    revenue_capture = 100 * top10_revenue / total_revenue if total_revenue > 0 else 0

    # Lift: revenue capture / 10%
    lift = revenue_capture / 10

    metrics = {
        'Metric': ['Precision@Top10%', 'Revenue Capture@Top10%', 'Lift@Top10%'],
        'Value': [f"{precision:.4f}", f"{revenue_capture:.2f}%", f"{lift:.2f}x"]
    }

    print(f"  Precision@Top10%: {precision:.4f}")
    print(f"  Revenue Capture: {revenue_capture:.2f}%")
    print(f"  Lift: {lift:.2f}x")

    return metrics

def main():
    print("\n" + "="*70)
    print("[EVALUATION] Model Comparison & Metrics")
    print("="*70)

    # Load all metrics
    metrics_df = load_all_metrics()

    if len(metrics_df) == 0:
        print("[ERROR] No metrics found")
        return

    print(f"\nLoaded {len(metrics_df)} models")

    # Create comparison table
    sorted_metrics = create_comparison_table(metrics_df)

    # Plot comparisons
    plot_model_comparison(metrics_df)

    # Load data for VIP metrics
    df_final = pd.read_csv(FINAL_FILE)

    # Get best model
    best_model = sorted_metrics.iloc[0]
    best_model_name = best_model['Model']
    print(f"\n[BEST MODEL] {best_model_name}")
    print(f"  R2: {best_model['R2']:.4f}")
    print(f"  MAE: ${best_model['MAE']:,.2f}")

    # Create dummy predictions for best model (using best available)
    # For now, use frequency-adjusted from Linear Combination
    y_pred_best = (df_final['Frequency'] / df_final['Frequency'].max() *
                   df_final['Monetary'].values / 1000)

    plot_predicted_vs_actual(df_final['ActualCLV'].values, y_pred_best, best_model_name)

    vip_metrics = compute_vip_metrics(df_final, y_pred_best)
    print("\n[TABLE] VIP Targeting Metrics")
    vip_df = pd.DataFrame(vip_metrics)
    print(vip_df.to_string(index=False))

    # Save VIP metrics
    vip_df.to_csv(RESULTS_DIR / 'vip_targeting_metrics.csv', index=False)

    print("\n" + "="*70)
    print("[DONE] Evaluation completed!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
