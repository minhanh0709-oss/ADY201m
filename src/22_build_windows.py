"""
22_build_windows.py
Build features + labels for each walk-forward window
This enables proper temporal cross-validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

DATA_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
CLEANED_FILE = DATA_PROCESSED_DIR / "online_retail_cleaned.csv"

# Import walk-forward splits
import sys
sys.path.insert(0, str(Path(__file__).parent))


def get_splits():
    """3 walk-forward windows"""
    return [
        ('2009-12-01', '2010-11-30', '2010-12-01', '2011-02-28'),
        ('2009-12-01', '2011-02-28', '2011-03-01', '2011-05-31'),
        ('2009-12-01', '2011-05-31', '2011-06-01', '2011-12-09'),
    ]


def compute_features(df, obs_start, obs_end):
    """Compute RFM + behavioral features for observation window"""
    obs_start = pd.to_datetime(obs_start)
    obs_end = pd.to_datetime(obs_end)

    obs_df = df[(df['InvoiceDate'] >= obs_start) & (df['InvoiceDate'] <= obs_end)].copy()

    features = []
    for customer_id in obs_df['CustomerID'].unique():
        cust_data = obs_df[obs_df['CustomerID'] == customer_id]

        # RFM
        recency = (obs_end - cust_data['InvoiceDate'].max()).days
        frequency = cust_data['Invoice'].nunique()
        monetary = cust_data['TotalPrice'].sum()

        # Behavioral
        first_purchase = cust_data['InvoiceDate'].min()
        tenure = (obs_end - first_purchase).days
        active_months = cust_data['InvoiceDate'].dt.to_period('M').nunique()
        product_diversity = cust_data['StockCode'].nunique()
        avg_order_value = monetary / frequency if frequency > 0 else 0

        # Inter-purchase metrics
        dates = cust_data.groupby('Invoice')['InvoiceDate'].min().sort_values()
        days_between = dates.diff().dt.days.dropna()
        if len(days_between) > 0 and days_between.mean() > 0:
            avg_days_between = days_between.mean()
            regularity = days_between.std() / days_between.mean()
        else:
            avg_days_between = 0
            regularity = 0

        # Country
        country = cust_data['Country'].mode().iloc[0] if len(cust_data) > 0 else 'Unknown'
        is_uk = 1 if country == 'United Kingdom' else 0

        # For BG/NBD: T (age of customer), recency in days from first purchase
        T = tenure
        bgnbd_recency = (cust_data['InvoiceDate'].max() - first_purchase).days

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
            'T_BGNBD': T,  # Age for BG/NBD
            'Recency_BGNBD': bgnbd_recency,  # Recency for BG/NBD (different from R in RFM)
        })

    return pd.DataFrame(features)


def compute_sequence_features(df, obs_start, obs_end, n_months):
    """Compute monthly sequence features"""
    obs_start = pd.to_datetime(obs_start)
    obs_end = pd.to_datetime(obs_end)

    obs_df = df[(df['InvoiceDate'] >= obs_start) & (df['InvoiceDate'] <= obs_end)].copy()
    obs_df['YearMonth'] = obs_df['InvoiceDate'].dt.to_period('M')

    # Generate months in window
    months = pd.period_range(start=obs_start.to_period('M'), end=obs_end.to_period('M'), freq='M')
    months = months[:n_months] if len(months) > n_months else months

    sequence_data = {}
    for customer_id in obs_df['CustomerID'].unique():
        cust_data = obs_df[obs_df['CustomerID'] == customer_id]

        rev_seq = []
        freq_seq = []
        for m in months:
            month_data = cust_data[cust_data['YearMonth'] == m]
            rev_seq.append(month_data['TotalPrice'].sum())
            freq_seq.append(month_data['Invoice'].nunique())

        sequence_data[customer_id] = {
            'revenue_seq': rev_seq,
            'frequency_seq': freq_seq
        }

    return sequence_data


def compute_clv_labels(df, pred_start, pred_end, customer_ids):
    """Compute actual CLV from prediction window"""
    pred_start = pd.to_datetime(pred_start)
    pred_end = pd.to_datetime(pred_end)

    pred_df = df[(df['InvoiceDate'] >= pred_start) & (df['InvoiceDate'] <= pred_end)].copy()
    clv = pred_df.groupby('CustomerID')['TotalPrice'].sum()

    labels = pd.DataFrame({'CustomerID': customer_ids})
    labels = labels.merge(clv.reset_index().rename(columns={'TotalPrice': 'ActualCLV'}),
                          on='CustomerID', how='left')
    labels['ActualCLV'] = labels['ActualCLV'].fillna(0)

    return labels


def build_window(df, obs_start, obs_end, pred_start, pred_end, window_id):
    """Build complete window: features + labels"""
    print(f"\n[Window {window_id}] Building features...")
    print(f"  Observation: {obs_start} to {obs_end}")
    print(f"  Prediction: {pred_start} to {pred_end}")

    # Static features
    features = compute_features(df, obs_start, obs_end)
    print(f"  Customers in observation: {len(features):,}")

    # Sequence features (max 18 months, actual depends on window)
    months_in_obs = (pd.to_datetime(obs_end).to_period('M') -
                     pd.to_datetime(obs_start).to_period('M')).n + 1
    n_months = min(months_in_obs, 24)
    sequence_data = compute_sequence_features(df, obs_start, obs_end, n_months)
    print(f"  Sequence length: {n_months} months")

    # Labels
    labels = compute_clv_labels(df, pred_start, pred_end, features['CustomerID'].values)

    # Merge
    merged = features.merge(labels, on='CustomerID')

    # Add VIP label
    threshold = merged['ActualCLV'].quantile(0.9)
    merged['IsVIP'] = (merged['ActualCLV'] >= threshold).astype(int)

    # Stats
    print(f"  ActualCLV stats:")
    print(f"    Mean: ${merged['ActualCLV'].mean():,.2f}")
    print(f"    Max: ${merged['ActualCLV'].max():,.2f}")
    print(f"    Zero rate: {100*(merged['ActualCLV']==0).sum()/len(merged):.1f}%")
    print(f"    VIP threshold: ${threshold:,.2f}")

    # Build sequence matrices
    n_customers = len(merged)
    rev_matrix = np.zeros((n_customers, n_months))
    freq_matrix = np.zeros((n_customers, n_months))

    for i, cust_id in enumerate(merged['CustomerID'].values):
        if cust_id in sequence_data:
            rev_matrix[i, :] = sequence_data[cust_id]['revenue_seq']
            freq_matrix[i, :] = sequence_data[cust_id]['frequency_seq']

    return {
        'window_id': window_id,
        'obs_start': obs_start,
        'obs_end': obs_end,
        'pred_start': pred_start,
        'pred_end': pred_end,
        'n_months': n_months,
        'features': merged,
        'revenue_seq': rev_matrix,
        'frequency_seq': freq_matrix
    }


def main():
    print("\n" + "="*70)
    print("[PHASE A] BUILD WALK-FORWARD WINDOWS")
    print("="*70)

    print("\n[Step 1] Loading cleaned data...")
    df = pd.read_csv(CLEANED_FILE)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    print(f"  Loaded: {len(df):,} transactions")

    splits = get_splits()
    windows = []

    for i, (a, b, c, d) in enumerate(splits, 1):
        window = build_window(df, a, b, c, d, i)
        windows.append(window)

    # Save all windows
    print("\n[Step 2] Saving windows...")
    with open(DATA_PROCESSED_DIR / 'walk_forward_windows.pkl', 'wb') as f:
        pickle.dump(windows, f)
    print(f"  Saved: walk_forward_windows.pkl")

    # Save individual CSVs for inspection
    for w in windows:
        csv_file = DATA_PROCESSED_DIR / f"window_{w['window_id']}_features.csv"
        w['features'].to_csv(csv_file, index=False)
        np.save(DATA_PROCESSED_DIR / f"window_{w['window_id']}_revenue_seq.npy", w['revenue_seq'])
        np.save(DATA_PROCESSED_DIR / f"window_{w['window_id']}_frequency_seq.npy", w['frequency_seq'])
        print(f"  Window {w['window_id']}: {len(w['features']):,} customers, "
              f"seq shape {w['revenue_seq'].shape}")

    print("\n" + "="*70)
    print("[DONE] Walk-forward windows built")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
