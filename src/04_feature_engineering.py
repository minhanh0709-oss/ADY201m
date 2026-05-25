"""
04_feature_engineering.py
Feature engineering with temporal split and RFM/behavioral features
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Paths
DATA_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
CLEANED_FILE = DATA_PROCESSED_DIR / "online_retail_cleaned.csv"
OBSERVATION_FILE = DATA_PROCESSED_DIR / "customers_observation.csv"

def load_data():
    """Load cleaned data"""
    df = pd.read_csv(CLEANED_FILE)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df

def temporal_split():
    """Define temporal windows"""
    snapshot_date = pd.to_datetime('2011-06-30')
    observation_start = pd.to_datetime('2009-12-01')
    prediction_end = pd.to_datetime('2011-12-09')

    return observation_start, snapshot_date, prediction_end

def extract_observation_features(df, observation_start, snapshot_date):
    """Extract RFM and behavioral features from observation window"""
    print("[Step 1] Extracting RFM and behavioral features...")

    # Filter to observation window
    obs_df = df[(df['InvoiceDate'] >= observation_start) & (df['InvoiceDate'] <= snapshot_date)].copy()

    # Group by customer
    features = []
    for customer_id in obs_df['CustomerID'].unique():
        cust_data = obs_df[obs_df['CustomerID'] == customer_id]

        # RFM features
        recency = (snapshot_date - cust_data['InvoiceDate'].max()).days
        frequency = cust_data['Invoice'].nunique()
        monetary = cust_data['TotalPrice'].sum()

        # Behavioral features
        tenure = (snapshot_date - cust_data['InvoiceDate'].min()).days
        active_months = cust_data['InvoiceDate'].dt.to_period('M').nunique()
        product_diversity = cust_data['StockCode'].nunique()
        avg_order_value = monetary / frequency if frequency > 0 else 0

        # Calculate days between orders
        dates = cust_data.groupby('Invoice')['InvoiceDate'].min().sort_values()
        days_between = dates.diff().dt.days.dropna()
        if len(days_between) > 0:
            avg_days_between = days_between.mean()
            regularity = days_between.std() / days_between.mean() if days_between.mean() > 0 else 0
        else:
            avg_days_between = 0
            regularity = 0

        # Country (UK vs non-UK)
        country = cust_data['Country'].mode()[0] if len(cust_data['Country'].mode()) > 0 else 'Unknown'
        is_uk = 1 if country == 'United Kingdom' else 0

        features.append({
            'CustomerID': customer_id,
            'Recency': recency,
            'Frequency': frequency,
            'Monetary': monetary,
            'Tenure': tenure,
            'ActiveMonths': active_months,
            'ProductDiversity': product_diversity,
            'AvgOrderValue': avg_order_value,
            'AvgDaysBetweenOrders': avg_days_between,
            'Regularity': regularity,
            'IsUK': is_uk,
            'Country': country
        })

    features_df = pd.DataFrame(features)
    print(f"  Extracted features for {len(features_df):,} customers")

    return features_df

def extract_sequence_features(df, observation_start, snapshot_date):
    """Extract monthly sequence features"""
    print("[Step 2] Extracting monthly sequence features...")

    obs_df = df[(df['InvoiceDate'] >= observation_start) & (df['InvoiceDate'] <= snapshot_date)].copy()
    obs_df['YearMonth'] = obs_df['InvoiceDate'].dt.to_period('M')

    # Create 18-month sequence (Dec 2009 - Jun 2011)
    months = pd.period_range(start='2009-12', end='2011-06', freq='M')

    sequence_features = []
    for customer_id in obs_df['CustomerID'].unique():
        cust_data = obs_df[obs_df['CustomerID'] == customer_id]

        monthly_revenue = []
        monthly_frequency = []
        monthly_aov = []

        for month in months:
            month_data = cust_data[cust_data['YearMonth'] == month]
            monthly_revenue.append(month_data['TotalPrice'].sum())
            freq = month_data['Invoice'].nunique()
            monthly_frequency.append(freq)
            aov = month_data['TotalPrice'].sum() / freq if freq > 0 else 0
            monthly_aov.append(aov)

        sequence_features.append({
            'CustomerID': customer_id,
            'MonthlyRevenue': monthly_revenue,
            'MonthlyFrequency': monthly_frequency,
            'MonthlyAOV': monthly_aov
        })

    sequence_df = pd.DataFrame(sequence_features)
    print(f"  Extracted 18-month sequences for {len(sequence_df):,} customers")

    return sequence_df

def merge_features(features_df, sequence_df):
    """Merge static and sequence features"""
    print("[Step 3] Merging features...")

    merged = features_df.merge(sequence_df, on='CustomerID', how='left')
    print(f"  Merged features shape: {merged.shape}")

    return merged

def main():
    print("\n" + "="*70)
    print("[FEATURE ENGINEERING] Temporal Split & Feature Extraction")
    print("="*70)

    # Load data
    df = load_data()
    observation_start, snapshot_date, prediction_end = temporal_split()

    print(f"\nTemporal windows:")
    print(f"  Observation: {observation_start.date()} to {snapshot_date.date()}")
    print(f"  Prediction: {snapshot_date.date()} to {prediction_end.date()}")

    # Extract features
    features_df = extract_observation_features(df, observation_start, snapshot_date)
    sequence_df = extract_sequence_features(df, observation_start, snapshot_date)

    # Merge
    merged_df = merge_features(features_df, sequence_df)

    # Save
    print("\n" + "="*70)
    print("[STEP 4] Save features")
    print("="*70)
    merged_df.to_csv(OBSERVATION_FILE, index=False)
    print(f"\n[OK] Features saved: {OBSERVATION_FILE}")
    print(f"     Rows: {len(merged_df):,}")
    print(f"     Columns: {len(merged_df.columns)}")

    print("\nSample features:")
    print(merged_df[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Tenure', 'ActiveMonths']].head())

    print("\n" + "="*70)
    print("[DONE] Feature engineering completed!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
