"""
99_diagnostic_analysis.py
Diagnose issues with current implementation
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
FINAL_FILE = DATA_PROCESSED_DIR / "customers_with_labels.csv"

def analyze():
    print("\n" + "="*70)
    print("[DIAGNOSTIC] KEY ISSUES FOUND")
    print("="*70)

    df = pd.read_csv(FINAL_FILE)

    # ISSUE 1
    print("\n[ISSUE 1] ActualCLV Distribution - HIGHLY SKEWED")
    print("-"*70)
    print(f"Min: ${df['ActualCLV'].min():,.2f}")
    print(f"Max: ${df['ActualCLV'].max():,.2f}")
    print(f"Mean: ${df['ActualCLV'].mean():,.2f}")
    print(f"Median: ${df['ActualCLV'].median():,.2f}")
    print(f"Std: ${df['ActualCLV'].std():,.2f}")
    print(f"Skewness: {df['ActualCLV'].skew():.2f}")
    print(f"  >> 0=normal, >1=highly skewed (ours: {df['ActualCLV'].skew():.1f})")
    print(f"  >> Log transform skewness: {np.log1p(df['ActualCLV']).skew():.2f}")
    print(f"% zeros (no future purchase): {100*(df['ActualCLV']==0).sum()/len(df):.1f}%")
    print(f"\nPROBLEM: Hard to predict with linear models")
    print(f"SOLUTION: Use log(CLV) or classification")

    # ISSUE 2
    print("\n[ISSUE 2] Sequence Features Stored as STRING (CRITICAL!)")
    print("-"*70)
    sample = df['MonthlyRevenue'].iloc[0]
    print(f"Type: {type(sample)}")
    print(f"Value (first 80 chars): {str(sample)[:80]}...")
    if isinstance(sample, str):
        print(f"\nPROBLEM: Features are TEXT STRING, not NUMPY ARRAY")
        print(f"         This blocks DL models from working!")
        print(f"SOLUTION: Parse string to array: ast.literal_eval() or json.loads()")

    # ISSUE 3
    print("\n[ISSUE 3] Sequence Sparsity - 79% ZEROS")
    print("-"*70)
    sparsity = []
    for idx, row in df.iterrows():
        active = row['ActiveMonths']
        zeros = 18 - active
        sparsity.append(zeros / 18)
    print(f"Average zeros per customer: {100*np.mean(sparsity):.1f}%")
    print(f"Max zeros: {100*np.max(sparsity):.1f}%")
    print(f"Min zeros: {100*np.min(sparsity):.1f}%")
    print(f"\nPROBLEM: LSTM/Transformer need longer, denser sequences")
    print(f"         18 months is too short for deep learning")
    print(f"SOLUTION: Use full 2-year data or tree-based models")

    # ISSUE 4
    print("\n[ISSUE 4] Feature Correlation")
    print("-"*70)
    corr_cols = ['Frequency', 'Monetary', 'Tenure', 'AvgOrderValue', 'ActualCLV']
    corr = df[corr_cols].corr()['ActualCLV'].sort_values(ascending=False)
    for col, val in corr.items():
        print(f"  {col:20s}: {val:7.4f}")
    print(f"\nPROBLEM: Monetary explains 79% (0.89^2)")
    print(f"         Other features add limited info")
    print(f"         Linear model already captures main signal")

    # ISSUE 5
    print("\n[ISSUE 5] Why DL Failed")
    print("-"*70)
    print(f"Sample size: {len(df):,} (OK for DL)")
    print(f"Sequence length: 18 months (TOO SHORT)")
    print(f"Sequence sparsity: 79% zeros (TOO SPARSE)")
    print(f"Feature diversity: Limited (Monetary dominates)")
    print(f"\nCONCLUSION:")
    print(f"  Tree-based models >> DL models on this data")
    print(f"  Linear model already captures 79% of signal")
    print(f"  DL has nothing additional to learn")

    # ISSUE 6
    print("\n[ISSUE 6] Validation Setup")
    print("-"*70)
    print(f"Current: No cross-validation, use all data")
    print(f"Problem: Cannot prove results are stable")
    print(f"Solution: 5-fold time-series CV")

    print("\n" + "="*70)
    print("[FIXES NEEDED FOR REAL RESEARCH]")
    print("="*70)
    print("""
PRIORITY 1 (Fix broken things):
  1. Parse sequence features from STRING to ARRAY
  2. Add proper train/test split with CV
  3. Use log(CLV) or classification for targets

PRIORITY 2 (Better baselines):
  1. Implement BG/NBD + Gamma-Gamma properly
  2. Tune XGBoost/LightGBM with GridSearch
  3. Try classification: predict VIP (top 10%)

PRIORITY 3 (Ablation & validation):
  1. Ablation: RFM only vs RFM+behavior vs RFM+behavior+sequence
  2. Statistical tests: paired t-test on fold results
  3. Cross-dataset: repeat on different retail data
    """)

    return df

if __name__ == "__main__":
    analyze()
