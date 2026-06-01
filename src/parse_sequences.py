"""
20_parse_sequences.py
PHASE A1: Parse sequence features from STRING to numpy ARRAY
Fix critical issue: sequence features were stored as text
"""

import pandas as pd
import numpy as np
import ast
import re
from pathlib import Path

DATA_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
INPUT_FILE = DATA_PROCESSED_DIR / "customers_with_labels.csv"
OUTPUT_FILE = DATA_PROCESSED_DIR / "customers_parsed.parquet"
OUTPUT_CSV = DATA_PROCESSED_DIR / "customers_parsed.csv"

def parse_np_string(s):
    """Parse 'np.float64(x)' style strings to list of floats"""
    if isinstance(s, (list, np.ndarray)):
        return list(s)
    if pd.isna(s):
        return [0.0] * 19

    # Convert 'np.float64(123.45)' -> '123.45'
    cleaned = re.sub(r'np\.float64\(([-\d.e]+)\)', r'\1', str(s))
    try:
        result = ast.literal_eval(cleaned)
        return [float(x) for x in result]
    except:
        print(f"[WARN] Failed to parse: {str(s)[:80]}")
        return [0.0] * 19

def parse_int_list(s):
    """Parse integer list strings"""
    if isinstance(s, (list, np.ndarray)):
        return list(s)
    if pd.isna(s):
        return [0] * 19
    try:
        result = ast.literal_eval(str(s))
        return [int(x) for x in result]
    except:
        return [0] * 19

def main():
    print("\n" + "="*70)
    print("[PHASE A1] PARSE SEQUENCE FEATURES")
    print("="*70)

    df = pd.read_csv(INPUT_FILE)
    print(f"\nLoaded: {len(df):,} customers, {len(df.columns)} columns")

    # Parse sequence columns
    print("\n[Step 1] Parsing MonthlyRevenue...")
    df['MonthlyRevenue_arr'] = df['MonthlyRevenue'].apply(parse_np_string)

    print("[Step 2] Parsing MonthlyFrequency...")
    df['MonthlyFrequency_arr'] = df['MonthlyFrequency'].apply(parse_int_list)

    print("[Step 3] Parsing MonthlyAOV...")
    df['MonthlyAOV_arr'] = df['MonthlyAOV'].apply(parse_np_string)

    # Verify
    sample_rev = df['MonthlyRevenue_arr'].iloc[0]
    sample_freq = df['MonthlyFrequency_arr'].iloc[0]
    sample_aov = df['MonthlyAOV_arr'].iloc[0]

    print(f"\n[Verification]")
    print(f"  MonthlyRevenue type: {type(sample_rev)}, length: {len(sample_rev)}")
    print(f"  MonthlyFrequency type: {type(sample_freq)}, length: {len(sample_freq)}")
    print(f"  MonthlyAOV type: {type(sample_aov)}, length: {len(sample_aov)}")
    print(f"\n  Sample Customer {df['CustomerID'].iloc[0]}:")
    print(f"    MonthlyRevenue: {sample_rev}")
    print(f"    MonthlyFrequency: {sample_freq}")

    # Check sequence length consistency
    rev_lengths = df['MonthlyRevenue_arr'].apply(len).value_counts()
    print(f"\n  MonthlyRevenue length distribution: {dict(rev_lengths)}")

    # Create sequence matrix for DL models
    print("\n[Step 4] Creating sequence matrix...")
    revenue_matrix = np.array([np.array(x[:18]) for x in df['MonthlyRevenue_arr']])
    frequency_matrix = np.array([np.array(x[:18]) for x in df['MonthlyFrequency_arr']])
    aov_matrix = np.array([np.array(x[:18]) for x in df['MonthlyAOV_arr']])

    print(f"  Revenue matrix shape: {revenue_matrix.shape}")
    print(f"  Frequency matrix shape: {frequency_matrix.shape}")
    print(f"  AOV matrix shape: {aov_matrix.shape}")

    # Sanity check
    print(f"\n[Sanity Check]")
    print(f"  Revenue: mean={revenue_matrix.mean():.2f}, max={revenue_matrix.max():.2f}")
    print(f"  Frequency: mean={frequency_matrix.mean():.2f}, max={frequency_matrix.max():.2f}")
    print(f"  AOV: mean={aov_matrix.mean():.2f}, max={aov_matrix.max():.2f}")

    # Save numpy arrays separately (more efficient)
    np.save(DATA_PROCESSED_DIR / 'seq_revenue.npy', revenue_matrix)
    np.save(DATA_PROCESSED_DIR / 'seq_frequency.npy', frequency_matrix)
    np.save(DATA_PROCESSED_DIR / 'seq_aov.npy', aov_matrix)
    print(f"\n[OK] Saved sequence matrices to .npy files")

    # Save parsed dataframe (without complex columns to keep CSV simple)
    df_static = df.drop(columns=['MonthlyRevenue', 'MonthlyFrequency', 'MonthlyAOV',
                                  'MonthlyRevenue_arr', 'MonthlyFrequency_arr', 'MonthlyAOV_arr'])
    df_static.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Saved static features: {OUTPUT_CSV}")
    print(f"     Shape: {df_static.shape}")

    print("\n" + "="*70)
    print("[DONE] Phase A1: Sequence features parsed successfully")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
