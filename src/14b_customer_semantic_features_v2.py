"""
14b_customer_semantic_features_v2.py
Recency-aware customer semantic profiles.

For each customer in each walk-forward window, we now compute TWO profiles
plus a drift signal:

  sem_full   = revenue-weighted average over the FULL observation period
               (32-dim, same as v1)
  sem_recent = revenue-weighted average over the LAST 90 DAYS only
               (32-dim — captures current taste)
  sem_drift  = 1 - cos_sim(sem_full, sem_recent)
               (scalar — captures taste change)

Plus aggregate signals:
  sem_recent_count : number of distinct products in recent window
  sem_full_count   : number of distinct products in full window

Total: 65 features (32 + 32 + 1) before any downstream selection.

Scientific motivation:
  Sequence features capture HOW MUCH a customer spends over time;
  recency-aware semantic features capture WHAT THEIR TASTE IS NOW
  versus their historical preference.  A customer drifting from
  seasonal to staple products signals different retention behaviour
  than one continuing the same buying pattern.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
CLEANED_FILE       = DATA_PROCESSED_DIR / "online_retail_cleaned.csv"

RECENT_DAYS = 90   # last 90 days = recent semantic profile


def load_product_embeddings():
    path = DATA_PROCESSED_DIR / 'product_embeddings.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_windows():
    p5 = DATA_PROCESSED_DIR / 'walk_forward_windows_5fold.pkl'
    p3 = DATA_PROCESSED_DIR / 'walk_forward_windows.pkl'
    pkl = p5 if p5.exists() else p3
    with open(pkl, 'rb') as f:
        return pickle.load(f)


def weighted_profile(cust_df, product2idx, embeddings, k):
    """Compute one weighted-average embedding for a given customer slice."""
    if len(cust_df) == 0:
        return np.zeros(k, dtype=np.float32), 0

    prod_spend = cust_df.groupby('StockCode')['TotalPrice'].sum()
    valid_mask = prod_spend.index.isin(product2idx)
    prod_spend = prod_spend[valid_mask]

    if len(prod_spend) == 0:
        return np.zeros(k, dtype=np.float32), 0

    weights = np.maximum(prod_spend.values.astype(np.float32), 0)
    if weights.sum() <= 0:
        return np.zeros(k, dtype=np.float32), 0

    emb_rows = np.array([product2idx[p] for p in prod_spend.index])
    weighted = (embeddings[emb_rows] * weights[:, None]).sum(axis=0)
    return weighted / weights.sum(), len(prod_spend)


def l2_normalize(profile):
    n = np.linalg.norm(profile)
    return profile / n if n > 0 else profile


def cosine_sim(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_recency_aware_profiles(df_obs, customer_ids, emb_data, obs_end):
    """Compute sem_full, sem_recent profiles + sem_drift scalar per customer."""
    embeddings  = emb_data['embeddings']
    product2idx = emb_data['product2idx']
    k           = emb_data['n_components']
    n_cust      = len(customer_ids)

    full_profiles   = np.zeros((n_cust, k), dtype=np.float32)
    recent_profiles = np.zeros((n_cust, k), dtype=np.float32)
    drift           = np.zeros(n_cust,      dtype=np.float32)
    full_counts     = np.zeros(n_cust,      dtype=np.float32)
    recent_counts   = np.zeros(n_cust,      dtype=np.float32)

    cust2row = {cid: i for i, cid in enumerate(customer_ids)}
    obs_end_ts = pd.Timestamp(obs_end)
    recent_cutoff = obs_end_ts - pd.Timedelta(days=RECENT_DAYS)

    grp = df_obs.groupby('CustomerID')
    n_full = 0
    n_recent = 0

    for cid, cust_df in grp:
        if cid not in cust2row:
            continue
        row = cust2row[cid]

        # FULL profile (entire observation period)
        prof_full, cnt_full = weighted_profile(cust_df, product2idx, embeddings, k)
        if cnt_full > 0:
            full_profiles[row] = l2_normalize(prof_full)
            full_counts[row]   = cnt_full
            n_full += 1

        # RECENT profile (last RECENT_DAYS days)
        recent_df = cust_df[cust_df['InvoiceDate'] >= recent_cutoff]
        prof_recent, cnt_recent = weighted_profile(recent_df, product2idx, embeddings, k)
        if cnt_recent > 0:
            recent_profiles[row] = l2_normalize(prof_recent)
            recent_counts[row]   = cnt_recent
            n_recent += 1

        # DRIFT: 1 - cos_sim(full, recent)   (range 0..2)
        # If recent profile is empty, drift = 1 (max separation)
        if cnt_full > 0 and cnt_recent > 0:
            drift[row] = 1.0 - cosine_sim(full_profiles[row], recent_profiles[row])
        else:
            drift[row] = 1.0  # no recent activity -> maximum drift

    print(f"    Full profile coverage:   {n_full:,}/{n_cust:,}  ({n_full/n_cust*100:.1f}%)")
    print(f"    Recent profile coverage: {n_recent:,}/{n_cust:,}  ({n_recent/n_cust*100:.1f}%)")
    print(f"    Mean drift: {drift.mean():.3f}  std: {drift.std():.3f}")

    return {
        'sem_full':       full_profiles,
        'sem_recent':     recent_profiles,
        'sem_drift':      drift,
        'full_count':     full_counts,
        'recent_count':   recent_counts,
    }


def main():
    print("\n" + "="*70)
    print("[14b] Recency-Aware Customer Semantic Profiles")
    print(f"      RECENT_DAYS = {RECENT_DAYS}")
    print("="*70)

    print("\nLoading product embeddings...")
    emb_data = load_product_embeddings()
    print(f"  Embedding dim: {emb_data['n_components']}")

    print("Loading cleaned transactions...")
    df = pd.read_csv(CLEANED_FILE)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    print("Loading walk-forward windows...")
    windows = load_windows()

    for win in windows:
        wid       = win['window_id']
        obs_start = pd.to_datetime(win['obs_start'])
        obs_end   = pd.to_datetime(win['obs_end'])

        print(f"\n[Window {wid}] obs {win['obs_start']} -> {win['obs_end']}")

        df_obs = df[(df['InvoiceDate'] >= obs_start) &
                    (df['InvoiceDate'] <= obs_end)].copy()
        customer_ids = win['features']['CustomerID'].values

        out = compute_recency_aware_profiles(df_obs, customer_ids, emb_data, obs_end)

        np.savez(
            DATA_PROCESSED_DIR / f"semantic_v2_window_{wid}.npz",
            sem_full=out['sem_full'],
            sem_recent=out['sem_recent'],
            sem_drift=out['sem_drift'],
            full_count=out['full_count'],
            recent_count=out['recent_count'],
            customer_ids=customer_ids,
        )
        print(f"    Saved semantic_v2_window_{wid}.npz")

    print("\n" + "="*70)
    print("[DONE] Recency-aware semantic profiles generated")
    print("="*70)


if __name__ == "__main__":
    main()
