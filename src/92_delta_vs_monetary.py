"""
92_delta_vs_monetary.py
Compute delta metrics (NG, Revenue@10%, MAE) vs Monetary baseline for all models.
Reads from MASTER_TABLE.csv and computes differences.
Saves results/delta_vs_monetary.csv and prints LaTeX-ready table.
"""

import pandas as pd
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"


def main():
    master_path = RESULTS_DIR / "MASTER_TABLE.csv"
    if not master_path.exists():
        print(f"MASTER_TABLE.csv not found at {master_path}")
        return

    df = pd.read_csv(master_path)
    print(f"Loaded {len(df)} models from MASTER_TABLE.csv")
    print("Columns:", list(df.columns))

    # Identify Monetary baseline row
    monetary_row = df[df['Model'].str.lower().str.contains('monetary')]
    if monetary_row.empty:
        print("Monetary baseline not found.")
        return

    mon = monetary_row.iloc[0]
    print(f"\nMonetary baseline: NG={mon['Norm_Gini_mean']:.4f}, "
          f"Revenue@10={mon['Revenue_Capture_10_mean']:.2f}%, "
          f"MAE=${mon['MAE_mean']:.0f}")

    # Compute deltas
    df['Delta_NG']         = df['Norm_Gini_mean']           - mon['Norm_Gini_mean']
    df['Delta_Revenue10']  = df['Revenue_Capture_10_mean']  - mon['Revenue_Capture_10_mean']
    df['Delta_MAE']        = df['MAE_mean']                 - mon['MAE_mean']  # negative = better

    # Sort by NG descending
    df_sorted = df.sort_values('Norm_Gini_mean', ascending=False).reset_index(drop=True)
    df_sorted['Rank'] = df_sorted.index + 1

    # Save
    out_path = RESULTS_DIR / "delta_vs_monetary.csv"
    df_sorted[['Rank','Model','Category',
               'Norm_Gini_mean','Norm_Gini_std',
               'Delta_NG',
               'Revenue_Capture_10_mean','Revenue_Capture_10_std',
               'Delta_Revenue10',
               'MAE_mean','MAE_std','Delta_MAE']].to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Print readable table
    print("\n" + "="*95)
    print(f"{'Rank':<5}{'Model':<35}{'Norm Gini':>10}{'dNG':>8}  {'Rev@10%':>8}{'dRev':>8}  {'MAE($)':>8}{'dMAE':>8}")
    print("="*95)
    for _, row in df_sorted.iterrows():
        print(f"{int(row['Rank']):<5}{row['Model']:<35}"
              f"{row['Norm_Gini_mean']:>10.4f}{row['Delta_NG']:>+8.4f}  "
              f"{row['Revenue_Capture_10_mean']:>8.2f}{row['Delta_Revenue10']:>+8.2f}  "
              f"{row['MAE_mean']:>8.0f}{row['Delta_MAE']:>+8.0f}")

    # LaTeX table snippet
    print("\n\n% --- LaTeX table snippet (Delta vs Monetary) ---")
    print(r"\begin{tabular}{llrrrr}")
    print(r"\toprule")
    print(r"Model & Category & Norm Gini & $\Delta$NG & Revenue@10\% & $\Delta$Rev@10 \\")
    print(r"\midrule")
    for _, row in df_sorted.iterrows():
        ng   = f"{row['Norm_Gini_mean']:.4f}"
        dng  = f"{row['Delta_NG']:+.4f}"
        rev  = f"{row['Revenue_Capture_10_mean']:.2f}\\%"
        drev = f"{row['Delta_Revenue10']:+.2f}\\%"
        model_name = row['Model'].replace('&', r'\&').replace('_', r'\_')
        cat = row['Category'].replace('&', r'\&')
        print(f"  {model_name} & {cat} & {ng} & {dng} & {rev} & {drev} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
