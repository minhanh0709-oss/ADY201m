"""
14_customer_semantic_features.py
Compute customer semantic profiles from product embeddings.

For each customer in each walk-forward window's observation period:
  - semantic_profile = weighted_avg(product_embeddings, weights=purchase_amount)
  - weighted by TotalPrice per product to emphasise high-spend items

Output per window:
  data/processed/semantic_features_window_{id}.npy   shape: (n_customers, k)
  data/processed/semantic_customer_ids_window_{id}.npy  shape: (n_customers,)

Scientific framing:
  Two customers with identical RFM can differ in "what" they buy.
  The semantic profile encodes the latent product-taste of the customer
  as a dense knowledge vector, enabling knowledge-driven CLV prediction.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
CLEANED_FILE       = DATA_PROCESSED_DIR / "online_retail_cleaned.csv"


def load_product_embeddings():
    path = DATA_PROCESSED_DIR / 'product_embeddings.pkl'
    if not path.exists():
        raise FileNotFoundError(
            "Run 13_product_graph.py first to generate product_embeddings.pkl"
        )
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_windows():
    p5 = DATA_PROCESSED_DIR / 'walk_forward_windows_5fold.pkl'
    p3 = DATA_PROCESSED_DIR / 'walk_forward_windows.pkl'
    pkl = p5 if p5.exists() else p3
    with open(pkl, 'rb') as f:
        return pickle.load(f)


def compute_customer_semantic_profiles(df_obs, customer_ids, emb_data):
    """
    For each customer, compute weighted-average product embedding.

    Parameters
    ----------
    df_obs : DataFrame  (already filtered to observation window)
    customer_ids : array-like  (ordered list of customers in this window)
    emb_data : dict from load_product_embeddings()

    Returns
    -------
    profiles : np.array  shape (n_customers, k)
    """
    embeddings  = emb_data['embeddings']   # (n_prods, k)
    product2idx = emb_data['product2idx']
    k           = emb_data['n_components']
    n_cust      = len(customer_ids)

    profiles = np.zeros((n_cust, k), dtype=np.float32)
    cust2row = {cid: i for i, cid in enumerate(customer_ids)}

    # Group transactions by customer
    grp = df_obs.groupby('CustomerID')
    n_covered = 0

    for cid, cust_df in grp:
        if cid not in cust2row:
            continue
        row = cust2row[cid]

        # Get products bought and their total spend
        prod_spend = cust_df.groupby('StockCode')['TotalPrice'].sum()
        valid_mask  = prod_spend.index.isin(product2idx)
        prod_spend  = prod_spend[valid_mask]

        if len(prod_spend) == 0:
            continue

        weights = prod_spend.values.astype(np.float32)
        weights = np.maximum(weights, 0)          # ignore refunds
        w_sum   = weights.sum()
        if w_sum <= 0:
            continue

        emb_rows = np.array([product2idx[p] for p in prod_spend.index])
        weighted_emb = (embeddings[emb_rows] * weights[:, None]).sum(axis=0)
        profiles[row] = weighted_emb / w_sum
        n_covered += 1

    coverage = n_covered / n_cust * 100
    print(f"    Semantic profile coverage: {n_covered:,}/{n_cust:,}  ({coverage:.1f}%)")

    # L2-normalize each profile (zero-vectors stay zero)
    norms = np.linalg.norm(profiles, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    profiles = profiles / norms

    return profiles


def main():
    print("\n" + "="*70)
    print("[Phase B] Customer Semantic Profiles per Walk-Forward Window")
    print("="*70)

    # Load resources
    print("\nLoading product embeddings...")
    emb_data = load_product_embeddings()
    k        = emb_data['n_components']
    print(f"  Embedding dim: {k}  |  Vocabulary: {len(emb_data['product2idx']):,} products")

    print("Loading cleaned transactions...")
    df = pd.read_csv(CLEANED_FILE)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    print("Loading walk-forward windows...")
    windows = load_windows()
    print(f"  {len(windows)} windows loaded")

    for win in windows:
        wid        = win['window_id']
        obs_start  = pd.to_datetime(win['obs_start'])
        obs_end    = pd.to_datetime(win['obs_end'])

        print(f"\n[Window {wid}] obs {win['obs_start']} -> {win['obs_end']}")

        # Filter to observation period
        df_obs = df[(df['InvoiceDate'] >= obs_start) &
                    (df['InvoiceDate'] <= obs_end)].copy()

        # Customer list matches the window's feature DataFrame (preserves order)
        customer_ids = win['features']['CustomerID'].values

        # Compute semantic profiles
        profiles = compute_customer_semantic_profiles(df_obs, customer_ids, emb_data)
        print(f"    Profile shape: {profiles.shape}")
        print(f"    Non-zero profiles: {(np.abs(profiles).sum(1) > 0).sum():,}")
        print(f"    Mean L2 norm (should be ~1): {np.linalg.norm(profiles, axis=1).mean():.3f}")

        # Save
        np.save(DATA_PROCESSED_DIR / f"semantic_features_window_{wid}.npy", profiles)
        np.save(DATA_PROCESSED_DIR / f"semantic_customer_ids_window_{wid}.npy", customer_ids)
        print(f"    Saved semantic_features_window_{wid}.npy")

    print("\n" + "="*70)
    print("[DONE] Customer semantic profiles generated for all windows")
    print("="*70)


if __name__ == "__main__":
    main()
