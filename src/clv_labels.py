"""
05_clv_labels.py
Calculate actual CLV from prediction window
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
CLEANED_FILE = DATA_PROCESSED_DIR / "online_retail_cleaned.csv"
OBSERVATION_FILE = DATA_PROCESSED_DIR / "customers_observation.csv"
FINAL_FILE = DATA_PROCESSED_DIR / "customers_with_labels.csv"

def load_data():
    """Load data"""
    df = pd.read_csv(CLEANED_FILE)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    features = pd.read_csv(OBSERVATION_FILE)
    return df, features

def calculate_clv_labels(df, features):
    """Calculate actual CLV from prediction window"""
    print("[Step 1] Calculate actual CLV from prediction window...")

    snapshot_date = pd.to_datetime('2011-06-30')
    prediction_end = pd.to_datetime('2011-12-09')

    # Filter to prediction window
    pred_df = df[(df['InvoiceDate'] > snapshot_date) & (df['InvoiceDate'] <= prediction_end)].copy()

    # Calculate CLV for each customer
    clv_by_customer = pred_df.groupby('CustomerID')['TotalPrice'].sum().reset_index()
    clv_by_customer.columns = ['CustomerID', 'ActualCLV']

    print(f"  Calculated CLV for {len(clv_by_customer):,} customers")

    # Merge with features (keeping all customers, filling NaN with 0 for those who didn't buy)
    merged = features.merge(clv_by_customer, on='CustomerID', how='left')
    merged['ActualCLV'] = merged['ActualCLV'].fillna(0)

    print(f"  Total customers with features: {len(merged):,}")
    print(f"  Customers with purchases in prediction: {(merged['ActualCLV'] > 0).sum():,}")
    print(f"  Customers with NO purchases: {(merged['ActualCLV'] == 0).sum():,}")

    return merged

def add_vip_label(df):
    """Add VIP label (top 10%)"""
    print("\n[Step 2] Add VIP labels...")

    threshold = df['ActualCLV'].quantile(0.9)
    df['IsVIP'] = (df['ActualCLV'] >= threshold).astype(int)

    vip_count = (df['IsVIP'] == 1).sum()
    print(f"  VIP threshold (90th percentile): ${threshold:,.2f}")
    print(f"  VIP customers: {vip_count:,} ({100*vip_count/len(df):.1f}%)")

    return df

def add_clv_summary_stats(df):
    """Add summary statistics to dataframe"""
    print("\n[Step 3] Add CLV summary statistics...")

    print(f"\nActual CLV statistics:")
    print(f"  Mean: ${df['ActualCLV'].mean():,.2f}")
    print(f"  Median: ${df['ActualCLV'].median():,.2f}")
    print(f"  Std: ${df['ActualCLV'].std():,.2f}")
    print(f"  Min: ${df['ActualCLV'].min():,.2f}")
    print(f"  Max: ${df['ActualCLV'].max():,.2f}")

    return df

def main():
    print("\n" + "="*70)
    print("[CLV LABELS] Calculate Actual CLV from Prediction Window")
    print("="*70)

    # Load
    df, features = load_data()

    # Calculate CLV
    merged = calculate_clv_labels(df, features)

    # Add VIP label
    merged = add_vip_label(merged)

    # Summary stats
    merged = add_clv_summary_stats(merged)

    # Save
    print("\n" + "="*70)
    print("[STEP 4] Save final dataset with labels")
    print("="*70)

    merged.to_csv(FINAL_FILE, index=False)
    print(f"\n[OK] Final dataset saved: {FINAL_FILE}")
    print(f"     Rows: {len(merged):,}")
    print(f"     Columns: {len(merged.columns)}")

    print("\nFirst few rows:")
    print(merged[['CustomerID', 'Monetary', 'ActualCLV', 'IsVIP']].head(10))

    print("\n" + "="*70)
    print("[DONE] CLV labels calculated!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
