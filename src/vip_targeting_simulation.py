"""
12_vip_targeting_simulation.py
Simulate VIP targeting strategies and measure ROI
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

def simulate_targeting_strategy(df, top_k_pct=0.1):
    """Simulate different targeting strategies"""
    print("\n[SIMULATION] Comparing VIP targeting strategies...")

    n_target = max(1, int(len(df) * top_k_pct))
    total_revenue = df['ActualCLV'].sum()

    strategies = {}

    # Strategy 1: Random targeting
    np.random.seed(42)
    random_indices = np.random.choice(len(df), n_target, replace=False)
    random_revenue = df.iloc[random_indices]['ActualCLV'].sum()
    strategies['Random'] = {
        'revenue': random_revenue,
        'pct': 100 * random_revenue / total_revenue,
        'lift': (100 * random_revenue / total_revenue) / top_k_pct
    }

    # Strategy 2: Monetary-based (simple RFM baseline)
    monetary_top = df.nlargest(n_target, 'Monetary')
    monetary_revenue = monetary_top['ActualCLV'].sum()
    strategies['Monetary-based'] = {
        'revenue': monetary_revenue,
        'pct': 100 * monetary_revenue / total_revenue,
        'lift': (100 * monetary_revenue / total_revenue) / top_k_pct
    }

    # Strategy 3: RFM Score
    r_norm = 1 - (df['Recency'] / df['Recency'].max())
    f_norm = df['Frequency'] / df['Frequency'].max()
    m_norm = df['Monetary'] / df['Monetary'].max()
    rfm_score = r_norm + f_norm + m_norm

    rfm_top_idx = rfm_score.nlargest(n_target).index
    rfm_revenue = df.loc[rfm_top_idx, 'ActualCLV'].sum()
    strategies['RFM Score'] = {
        'revenue': rfm_revenue,
        'pct': 100 * rfm_revenue / total_revenue,
        'lift': (100 * rfm_revenue / total_revenue) / top_k_pct
    }

    # Strategy 4: Frequency-based
    freq_top = df.nlargest(n_target, 'Frequency')
    freq_revenue = freq_top['ActualCLV'].sum()
    strategies['Frequency-based'] = {
        'revenue': freq_revenue,
        'pct': 100 * freq_revenue / total_revenue,
        'lift': (100 * freq_revenue / total_revenue) / top_k_pct
    }

    # Strategy 5: ML Prediction (Linear Combination model)
    # Simple linear prediction
    X = df[['Frequency', 'Monetary', 'Tenure', 'AvgOrderValue']].values
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    from numpy.linalg import lstsq
    X_with_intercept = np.column_stack([np.ones(len(X_norm)), X_norm])
    y = df['ActualCLV'].values
    beta = lstsq(X_with_intercept, y, rcond=None)[0]
    pred = X_with_intercept @ beta

    ml_top_idx = np.argsort(pred)[-n_target:]
    ml_revenue = df.iloc[ml_top_idx]['ActualCLV'].sum()
    strategies['ML Prediction'] = {
        'revenue': ml_revenue,
        'pct': 100 * ml_revenue / total_revenue,
        'lift': (100 * ml_revenue / total_revenue) / top_k_pct
    }

    # Strategy 6: Perfect targeting (oracle - using actual CLV)
    oracle_top = df.nlargest(n_target, 'ActualCLV')
    oracle_revenue = oracle_top['ActualCLV'].sum()
    strategies['Oracle (Perfect)'] = {
        'revenue': oracle_revenue,
        'pct': 100 * oracle_revenue / total_revenue,
        'lift': (100 * oracle_revenue / total_revenue) / top_k_pct
    }

    # Print results
    print(f"\nTargeting top {top_k_pct*100:.0f}% ({n_target:,}) customers:")
    print(f"Total revenue: ${total_revenue:,.2f}\n")

    for strategy, metrics in strategies.items():
        print(f"{strategy:20s} | Revenue: ${metrics['revenue']:>12,.2f} | Capture: {metrics['pct']:>6.2f}% | Lift: {metrics['lift']:>5.2f}x")

    return strategies

def plot_targeting_strategies(strategies):
    """Plot targeting strategy comparison"""
    print("\n[PLOT] Targeting strategy comparison...")

    strategies_list = list(strategies.keys())
    lifts = [strategies[s]['lift'] for s in strategies_list]
    revenues_pct = [strategies[s]['pct'] for s in strategies_list]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Lift chart
    colors = ['#F18F01' if x < 2 else '#A23B72' if x < 4 else '#2E86AB' for x in lifts]
    axes[0].barh(strategies_list, lifts, color=colors)
    axes[0].axvline(x=1.0, color='red', linestyle='--', linewidth=1, label='Baseline')
    axes[0].set_xlabel('Lift vs Random', fontsize=11)
    axes[0].set_title('Targeting Strategy Lift', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    axes[0].legend()

    # Revenue capture
    axes[1].barh(strategies_list, revenues_pct, color='#2E86AB')
    axes[1].axvline(x=10.0, color='red', linestyle='--', linewidth=1, label='Random (10%)')
    axes[1].set_xlabel('Revenue Capture (%)', fontsize=11)
    axes[1].set_title('Revenue Captured by Strategy', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'vip_targeting_strategies.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: vip_targeting_strategies.png")

def create_simulation_table(strategies):
    """Create Table 5 for paper"""
    data = {
        'Strategy': list(strategies.keys()),
        'Revenue_Captured': [f"${strategies[s]['revenue']:,.0f}" for s in strategies.keys()],
        'Percent_Total': [f"{strategies[s]['pct']:.2f}%" for s in strategies.keys()],
        'Lift_vs_Random': [f"{strategies[s]['lift']:.2f}x" for s in strategies.keys()]
    }

    table_df = pd.DataFrame(data)
    table_df.to_csv(RESULTS_DIR / 'table_5_vip_targeting.csv', index=False)

    print("\n" + "="*70)
    print("[TABLE 5] VIP Targeting Strategy Comparison")
    print("="*70)
    print(table_df.to_string(index=False))

    return table_df

def main():
    print("\n" + "="*70)
    print("[VIP TARGETING] Strategy Simulation & ROI Analysis")
    print("="*70)

    df = load_data()
    print(f"\nData: {len(df):,} customers, ${df['ActualCLV'].sum():,.2f} total revenue")

    # Simulate strategies
    strategies = simulate_targeting_strategy(df, top_k_pct=0.1)

    # Plot
    plot_targeting_strategies(strategies)

    # Create table
    table_df = create_simulation_table(strategies)

    print("\n" + "="*70)
    print("[KEY INSIGHTS]")
    print("="*70)

    best_strategy = max(strategies.items(), key=lambda x: x[1]['lift'])
    print(f"Best strategy: {best_strategy[0]}")
    print(f"  - Lift: {best_strategy[1]['lift']:.2f}x vs random")
    print(f"  - Revenue capture: {best_strategy[1]['pct']:.2f}% of total")
    print(f"  - Improvement vs random: {(best_strategy[1]['lift'] - 1) * 100:.1f}%")

    print("\n" + "="*70)
    print("[DONE] VIP targeting simulation completed!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
