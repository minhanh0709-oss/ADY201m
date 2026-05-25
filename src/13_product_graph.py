"""
13_product_graph.py
Semantic Product Graph via Co-purchase Embeddings

Approach (no special graph library needed):
  1. Build a product-product PPMI (Positive PMI) co-occurrence matrix
     from invoice-level co-purchases.
  2. Apply Truncated SVD  (randomized_svd from sklearn)
     to obtain dense product embeddings (n_products x n_components).
  3. Validate: nearest-neighbour check on a sample product.
  4. Save to data/processed/product_embeddings.pkl

Scientific rationale for SIMC "Semantic Intelligence":
  Products that frequently co-appear in orders share latent semantic meaning
  (same category, complementary use, same buyer profile).  The SVD
  embedding decomposes this shared structure into a low-dimensional
  semantic space — a knowledge representation of the product catalogue.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')

DATA_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
CLEANED_FILE       = DATA_PROCESSED_DIR / "online_retail_cleaned.csv"

N_COMPONENTS    = 32   # embedding dimension
MIN_CO_OCCUR    = 3    # minimum co-occurrence count to retain an edge
MAX_PRODS_INV   = 60   # cap products per invoice (avoids outlier bulk orders)


def build_ppmi_matrix(df, product2idx, n_products):
    """Build PPMI co-occurrence matrix from invoice-level basket pairs."""
    n = n_products

    # Build raw co-occurrence counts
    co_mat = lil_matrix((n, n), dtype=np.float32)
    product_freq = np.zeros(n, dtype=np.float32)

    invoices = df.groupby('Invoice')['StockCode'].apply(list)
    n_invoices = len(invoices)
    print(f"  Processing {n_invoices:,} invoices...")

    for prods in invoices:
        # Deduplicate and cap
        prods_unique = list(dict.fromkeys(prods))[:MAX_PRODS_INV]
        idxs = [product2idx[p] for p in prods_unique if p in product2idx]

        # Update product frequency (marginal)
        for pi in idxs:
            product_freq[pi] += 1

        # Update co-occurrence
        for ii, pi in enumerate(idxs):
            for pj in idxs[ii+1:]:
                co_mat[pi, pj] += 1
                co_mat[pj, pi] += 1

    co_mat = co_mat.tocsr()

    # Filter rare co-occurrences
    co_mat.data[co_mat.data < MIN_CO_OCCUR] = 0
    co_mat.eliminate_zeros()
    print(f"  Non-zero edges after filtering (>={MIN_CO_OCCUR}): {co_mat.nnz:,}")

    # PPMI transformation
    # Correct PMI formula (Bullinaria & Levy 2007):
    #   P(i,j) = co_ij / n_invoices   (joint: invoices containing both i and j)
    #   P(i)   = freq_i / n_invoices  (marginal: invoices containing i)
    #   PMI(i,j) = log(co_ij * n_invoices / (freq_i * freq_j))
    if n_invoices == 0:
        return co_mat

    co_csr = co_mat.copy().astype(np.float64)
    row_idx, col_idx = co_csr.nonzero()

    co_vals  = co_csr.data
    freq_i   = product_freq[row_idx].astype(np.float64)
    freq_j   = product_freq[col_idx].astype(np.float64)

    # PMI(i,j) = log(co_ij * n_invoices / (freq_i * freq_j))
    denom    = freq_i * freq_j
    safe_d   = np.where(denom > 0, denom, 1.0)
    pmi_vals = np.log((co_vals * n_invoices) / safe_d + 1e-9)

    # Positive PMI only (clip negative PMI to 0)
    ppmi_vals = np.maximum(pmi_vals, 0)

    ppmi = csr_matrix((ppmi_vals, (row_idx, col_idx)), shape=(n, n))
    ppmi.eliminate_zeros()
    print(f"  PPMI non-zeros: {ppmi.nnz:,}")
    return ppmi


def main():
    print("\n" + "="*70)
    print("[Phase B] Product Co-purchase Graph -> SVD Embeddings")
    print("="*70)

    # ── Load data ──────────────────────────────────────────────────────────
    df = pd.read_csv(CLEANED_FILE)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    print(f"Loaded {len(df):,} transactions, {df['Invoice'].nunique():,} invoices")

    # Keep only observation-period data (up to 2011-05-31 = end of W3 obs)
    # so embeddings are learned only from training/observation data
    df_obs = df[df['InvoiceDate'] <= '2011-05-31'].copy()
    print(f"Observation data (up to 2011-05-31): {len(df_obs):,} transactions")

    # ── Product vocabulary ─────────────────────────────────────────────────
    # Filter products appearing in < 5 invoices (too rare to embed reliably)
    prod_inv_count = df_obs.groupby('StockCode')['Invoice'].nunique()
    valid_prods = prod_inv_count[prod_inv_count >= 5].index.tolist()
    print(f"Products with >= 5 invoices: {len(valid_prods):,}")

    product2idx = {p: i for i, p in enumerate(valid_prods)}
    idx2product = {i: p for p, i in product2idx.items()}
    n_products  = len(valid_prods)

    df_obs_valid = df_obs[df_obs['StockCode'].isin(valid_prods)].copy()

    # ── Build PPMI matrix ──────────────────────────────────────────────────
    print("\nBuilding PPMI co-occurrence matrix...")
    ppmi = build_ppmi_matrix(df_obs_valid, product2idx, n_products)

    # ── Truncated SVD ──────────────────────────────────────────────────────
    print(f"\nApplying Truncated SVD (n_components={N_COMPONENTS})...")
    k = min(N_COMPONENTS, min(ppmi.shape) - 1)
    U, sigma, Vt = randomized_svd(ppmi, n_components=k,
                                   n_iter=5, random_state=42)
    # Scale by sqrt(sigma) for symmetric factorisation
    embeddings = U * np.sqrt(sigma)            # shape: (n_products, k)
    embeddings = normalize(embeddings, norm='l2')  # unit-norm rows
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Variance explained (top 5 singular values): {sigma[:5].round(1)}")

    # ── Nearest-neighbour sanity check ─────────────────────────────────────
    print("\nNearest-neighbour check (cosine similarity):")
    sample_prods = ['85123A', '22423', '47566', '84029G']
    for sp in sample_prods:
        if sp not in product2idx:
            continue
        si = product2idx[sp]
        sims = embeddings @ embeddings[si]
        top5 = np.argsort(sims)[::-1][1:6]
        nbrs = [(idx2product[j], f"{sims[j]:.3f}") for j in top5]
        # Get product descriptions if available
        desc = df_obs[df_obs['StockCode'] == sp]['Description'].dropna()
        desc_str = desc.iloc[0] if len(desc) > 0 else sp
        print(f"  [{sp}] {desc_str[:40]}")
        for nb_code, nb_sim in nbrs:
            nb_desc = df_obs[df_obs['StockCode'] == nb_code]['Description'].dropna()
            nb_str  = nb_desc.iloc[0][:35] if len(nb_desc) > 0 else nb_code
            print(f"      -> {nb_code:8s} {nb_str:<35} sim={nb_sim}")

    # ── Save ───────────────────────────────────────────────────────────────
    out = {
        'embeddings':   embeddings,      # np.array (n_products, k)
        'product2idx':  product2idx,
        'idx2product':  idx2product,
        'n_components': k,
        'valid_prods':  valid_prods,
    }
    out_path = DATA_PROCESSED_DIR / 'product_embeddings.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(out, f)
    print(f"\nSaved product embeddings -> {out_path}")
    print(f"  {n_products:,} products  x  {k} dimensions")

    # Also save as numpy for easy loading
    np.save(DATA_PROCESSED_DIR / 'product_embeddings.npy', embeddings)

    print("\n" + "="*70)
    print("[DONE] Product graph embeddings complete")
    print("="*70)


if __name__ == "__main__":
    main()
