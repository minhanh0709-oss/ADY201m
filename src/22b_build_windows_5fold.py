"""
22b_build_windows_5fold.py
Extended walk-forward CV with 5 windows.

Keeps existing 3 windows intact, adds:
  Window 4: obs Dec'09-Aug'10 (9 mo)  -> pred Sep'10-Nov'10 (3 mo)  [early data]
  Window 5: obs Dec'09-Aug'11 (21 mo) -> pred Sep'11-Nov'11 (3 mo)  [max obs]

All prediction windows = 3 months for comparability
(Window 3 original had 6-month pred; it is KEPT as is in the existing pkl;
 the NEW pkl has W4 and W5 only, merged during evaluation)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

DATA_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
CLEANED_FILE = DATA_PROCESSED_DIR / "online_retail_cleaned.csv"


NEW_SPLITS = [
    # id, obs_start, obs_end, pred_start, pred_end
    (4, '2009-12-01', '2010-08-31', '2010-09-01', '2010-11-30'),
    (5, '2009-12-01', '2011-08-31', '2011-09-01', '2011-11-30'),
]


def compute_features(df, obs_start, obs_end):
    obs_start = pd.to_datetime(obs_start)
    obs_end   = pd.to_datetime(obs_end)
    obs_df = df[(df['InvoiceDate'] >= obs_start) & (df['InvoiceDate'] <= obs_end)].copy()

    features = []
    for cust_id in obs_df['CustomerID'].unique():
        c = obs_df[obs_df['CustomerID'] == cust_id]
        recency   = (obs_end - c['InvoiceDate'].max()).days
        frequency = c['Invoice'].nunique()
        monetary  = c['TotalPrice'].sum()

        first_purchase = c['InvoiceDate'].min()
        tenure         = (obs_end - first_purchase).days
        active_months  = c['InvoiceDate'].dt.to_period('M').nunique()
        product_div    = c['StockCode'].nunique()
        aov            = monetary / frequency if frequency > 0 else 0

        dates = c.groupby('Invoice')['InvoiceDate'].min().sort_values()
        diffs = dates.diff().dt.days.dropna()
        avg_days = diffs.mean() if len(diffs) > 0 else 0
        regularity = (diffs.std() / diffs.mean()) if (len(diffs) > 0 and diffs.mean() > 0) else 0

        country = c['Country'].mode().iloc[0] if len(c) > 0 else 'Unknown'
        is_uk = 1 if country == 'United Kingdom' else 0

        T             = tenure
        bgnbd_recency = (c['InvoiceDate'].max() - first_purchase).days

        features.append({
            'CustomerID': cust_id,
            'Recency': recency, 'Frequency': frequency, 'Monetary': monetary,
            'Tenure': tenure, 'ActiveMonths': active_months,
            'ProductDiversity': product_div, 'AvgOrderValue': aov,
            'AvgDaysBetweenOrders': avg_days, 'Regularity': regularity,
            'IsUK': is_uk, 'T_BGNBD': T, 'Recency_BGNBD': bgnbd_recency,
        })
    return pd.DataFrame(features)


def compute_sequence_features(df, obs_start, obs_end, n_months):
    obs_start = pd.to_datetime(obs_start)
    obs_end   = pd.to_datetime(obs_end)
    obs_df = df[(df['InvoiceDate'] >= obs_start) & (df['InvoiceDate'] <= obs_end)].copy()
    obs_df['YearMonth'] = obs_df['InvoiceDate'].dt.to_period('M')
    months = pd.period_range(start=obs_start.to_period('M'),
                             end=obs_end.to_period('M'), freq='M')[:n_months]
    seq = {}
    for cust_id in obs_df['CustomerID'].unique():
        c = obs_df[obs_df['CustomerID'] == cust_id]
        rev_seq = [c[c['YearMonth'] == m]['TotalPrice'].sum() for m in months]
        frq_seq = [c[c['YearMonth'] == m]['Invoice'].nunique() for m in months]
        seq[cust_id] = {'revenue_seq': rev_seq, 'frequency_seq': frq_seq}
    return seq


def compute_clv_labels(df, pred_start, pred_end, customer_ids):
    pred_start = pd.to_datetime(pred_start)
    pred_end   = pd.to_datetime(pred_end)
    pred_df = df[(df['InvoiceDate'] >= pred_start) & (df['InvoiceDate'] <= pred_end)]
    clv = pred_df.groupby('CustomerID')['TotalPrice'].sum()
    labels = pd.DataFrame({'CustomerID': customer_ids})
    labels = labels.merge(clv.reset_index().rename(columns={'TotalPrice': 'ActualCLV'}),
                          on='CustomerID', how='left')
    labels['ActualCLV'] = labels['ActualCLV'].fillna(0)
    return labels


def build_window(df, wid, obs_start, obs_end, pred_start, pred_end):
    print(f"\n[Window {wid}] obs {obs_start}->{obs_end} | pred {pred_start}->{pred_end}")
    features = compute_features(df, obs_start, obs_end)
    print(f"  Customers: {len(features):,}")

    months_in_obs = (pd.to_datetime(obs_end).to_period('M') -
                     pd.to_datetime(obs_start).to_period('M')).n + 1
    n_months = min(months_in_obs, 24)
    seq_data = compute_sequence_features(df, obs_start, obs_end, n_months)

    labels = compute_clv_labels(df, pred_start, pred_end, features['CustomerID'].values)
    merged = features.merge(labels, on='CustomerID')
    threshold = merged['ActualCLV'].quantile(0.9)
    merged['IsVIP'] = (merged['ActualCLV'] >= threshold).astype(int)

    print(f"  Mean CLV: ${merged['ActualCLV'].mean():,.2f}  "
          f"  Zero rate: {100*(merged['ActualCLV']==0).mean():.1f}%  "
          f"  VIP threshold: ${threshold:,.2f}")

    n = len(merged)
    rev_mat = np.zeros((n, n_months))
    frq_mat = np.zeros((n, n_months))
    for i, cid in enumerate(merged['CustomerID'].values):
        if cid in seq_data:
            rev_mat[i] = seq_data[cid]['revenue_seq']
            frq_mat[i] = seq_data[cid]['frequency_seq']

    return {
        'window_id': wid,
        'obs_start': obs_start, 'obs_end': obs_end,
        'pred_start': pred_start, 'pred_end': pred_end,
        'n_months': n_months,
        'features': merged,
        'revenue_seq': rev_mat,
        'frequency_seq': frq_mat,
    }


def main():
    print("\n" + "="*70)
    print("[5-FOLD] Building 2 new walk-forward windows (W4 and W5)")
    print("="*70)

    df = pd.read_csv(CLEANED_FILE)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    print(f"Loaded {len(df):,} transactions")

    # Load existing 3-window pkl
    existing_pkl = DATA_PROCESSED_DIR / 'walk_forward_windows.pkl'
    with open(existing_pkl, 'rb') as f:
        existing_windows = pickle.load(f)
    print(f"Loaded {len(existing_windows)} existing windows")

    # Build new windows
    new_windows = []
    for (wid, a, b, c, d) in NEW_SPLITS:
        w = build_window(df, wid, a, b, c, d)
        new_windows.append(w)
        # Save individual CSV
        csv_f = DATA_PROCESSED_DIR / f"window_{wid}_features.csv"
        w['features'].to_csv(csv_f, index=False)
        np.save(DATA_PROCESSED_DIR / f"window_{wid}_revenue_seq.npy",  w['revenue_seq'])
        np.save(DATA_PROCESSED_DIR / f"window_{wid}_frequency_seq.npy", w['frequency_seq'])

    # Merge all 5 windows sorted by window_id
    all_windows = sorted(existing_windows + new_windows, key=lambda w: w['window_id'])

    # Save merged pkl
    out_pkl = DATA_PROCESSED_DIR / 'walk_forward_windows_5fold.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump(all_windows, f)
    print(f"\nSaved 5-fold pkl -> {out_pkl}")

    print("\n[Summary] All 5 windows:")
    for w in all_windows:
        print(f"  W{w['window_id']}: obs {w['obs_start']}..{w['obs_end']}  "
              f"pred {w['pred_start']}..{w['pred_end']}  "
              f"n={len(w['features']):,}  seq_len={w['n_months']}")

    print("\n" + "="*70)
    print("[DONE]")
    print("="*70)


if __name__ == "__main__":
    main()
