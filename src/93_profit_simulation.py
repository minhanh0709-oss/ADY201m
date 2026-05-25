"""
93_profit_simulation.py
Profit-aware targeting simulation.
  Profit@K = margin * captured_revenue - cost_per_customer * n_contacted
Sensitivity over margin=[10%,20%,30%] x cost_per_customer=[$1,$3,$5].
Reads from MASTER_TABLE.csv for Revenue@10%, total revenue estimate.
Saves results/profit_simulation.csv + generates figure.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"

# From dataset: total prediction-window revenue averaged across windows
# Window 1 (3-mo): ~$1.2M, Window 2 (3-mo): ~$1.1M, Window 3 (6-mo): ~$2.4M
# Use Window 3 (most informative) for simulation; average customers tested ~986
TOTAL_REVENUE = 2_400_000   # approx window-3 prediction revenue
N_CUSTOMERS   = 986          # window-3 test set size
K_PCT         = 0.10         # target top 10%

MARGINS = [0.10, 0.20, 0.30]
COSTS   = [1.0,  3.0,  5.0]

# Model names and their Revenue@10% (mean across 3 windows)
MODELS = [
    ('Random',           10.0),
    ('Monetary',         60.02),
    ('RFM Score',        52.68),
    ('XGBoost (Optuna)', 58.85),
    ('LightGBM (Optuna)',58.80),
    ('Hurdle (Proposed)',60.96),
    ('ZILN',             60.20),
    ('MCD-ZILN',         60.68),
    ('Oracle',           72.20),  # estimated from RC@K curve
]


def compute_profit(revenue_capture_pct, total_rev, n_customers,
                   k_pct, margin, cost_per_customer):
    """
    Profit = margin * captured_revenue - cost * n_contacted
    All values in dollars.
    """
    n_contacted       = int(n_customers * k_pct)
    captured_revenue  = (revenue_capture_pct / 100) * total_rev
    gross_profit      = margin * captured_revenue
    campaign_cost     = cost_per_customer * n_contacted
    net_profit        = gross_profit - campaign_cost
    roi               = (net_profit / campaign_cost * 100) if campaign_cost > 0 else float('inf')
    return net_profit, roi


def main():
    print("\n" + "="*70)
    print("[Profit-Aware Targeting Simulation @ Top 10%]")
    print("="*70)
    print(f"  Total revenue (Window 3 approx): ${TOTAL_REVENUE:,}")
    print(f"  N customers in test set: {N_CUSTOMERS}")
    print(f"  N contacted (top 10%): {int(N_CUSTOMERS*K_PCT)}")

    records = []

    for margin in MARGINS:
        for cost in COSTS:
            print(f"\n  Margin={margin*100:.0f}%  Cost/customer=${cost:.0f}")
            print(f"  {'Model':<22} {'Rev@10%':>8}  {'Profit($)':>10}  {'ROI(%)':>8}")
            print(f"  {'-'*55}")
            for model_name, rev_pct in MODELS:
                profit, roi = compute_profit(
                    rev_pct, TOTAL_REVENUE, N_CUSTOMERS, K_PCT, margin, cost
                )
                print(f"  {model_name:<22} {rev_pct:>8.2f}%  {profit:>10,.0f}  {roi:>8.1f}%")
                records.append({
                    'Model':       model_name,
                    'Revenue_10':  rev_pct,
                    'Margin_pct':  margin * 100,
                    'Cost_per_cust': cost,
                    'Net_Profit':  profit,
                    'ROI_pct':     roi,
                })

    df = pd.DataFrame(records)
    out_path = RESULTS_DIR / 'profit_simulation.csv'
    df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    # ── Figure: profit heatmap for Hurdle vs others under medium scenario ────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle('Net Profit at Top-10% Targeting Under Different Margin Assumptions\n'
                 f'(Campaign cost = $3/customer, {N_CUSTOMERS} test customers, '
                 f'${TOTAL_REVENUE/1e6:.1f}M prediction-window revenue)',
                 fontsize=11, fontweight='bold')

    colors = {
        'Random':            '#95A5A6',
        'Monetary':          '#F39C12',
        'RFM Score':         '#E74C3C',
        'XGBoost (Optuna)':  '#3498DB',
        'LightGBM (Optuna)': '#2980B9',
        'Hurdle (Proposed)': '#27AE60',
        'ZILN':              '#9B59B6',
        'MCD-ZILN':          '#8E44AD',
        'Oracle':            '#2C3E50',
    }

    model_names = [m[0] for m in MODELS]
    cost_mid = 3.0

    for ax, margin in zip(axes, MARGINS):
        sub = df[(df['Margin_pct'] == margin * 100) & (df['Cost_per_cust'] == cost_mid)]
        sub = sub.set_index('Model').reindex(model_names)
        bar_colors = [colors.get(m, '#888') for m in model_names]
        bars = ax.barh(model_names, sub['Net_Profit'] / 1000,
                       color=bar_colors, alpha=0.88, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Net Profit ($K)', fontsize=10)
        ax.set_title(f'Margin = {margin*100:.0f}%', fontsize=11, fontweight='bold')
        ax.axvline(0, color='black', linewidth=0.8)
        ax.grid(True, axis='x', alpha=0.3)
        # Annotate Hurdle bar
        hurdle_val = sub.loc['Hurdle (Proposed)', 'Net_Profit'] / 1000
        ax.annotate(f'${hurdle_val:.0f}K',
                    xy=(hurdle_val, model_names.index('Hurdle (Proposed)')),
                    xytext=(5, 0), textcoords='offset points',
                    fontsize=8, color='#27AE60', fontweight='bold', va='center')

    plt.tight_layout()
    fig_path = FIGURES_DIR / 'profit_simulation.png'
    plt.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Figure saved: {fig_path}")

    # ── Insight summary ───────────────────────────────────────────────────────
    print("\n[Key insight — Margin=20%, Cost=$3]")
    sub = df[(df['Margin_pct'] == 20) & (df['Cost_per_cust'] == 3.0)]
    hurdle_profit = sub[sub['Model'] == 'Hurdle (Proposed)']['Net_Profit'].values[0]
    random_profit = sub[sub['Model'] == 'Random']['Net_Profit'].values[0]
    monetary_profit = sub[sub['Model'] == 'Monetary']['Net_Profit'].values[0]
    print(f"  Random:           ${random_profit:,.0f}")
    print(f"  Monetary:         ${monetary_profit:,.0f}")
    print(f"  Hurdle:           ${hurdle_profit:,.0f}")
    print(f"  Hurdle vs Random: +${hurdle_profit - random_profit:,.0f} "
          f"({(hurdle_profit/random_profit - 1)*100:.1f}% more)")
    print(f"  Hurdle vs Monetary: +${hurdle_profit - monetary_profit:,.0f}")


if __name__ == "__main__":
    main()
