"""Semantic product-graph features from a normalized observation-window frame.

Pipeline (matches Phase 1's approach, vectorised with sparse incidence):
  1. basket x product incidence -> product-product co-occurrence (B^T B)
  2. PPMI transform -> Truncated SVD -> L2-normalized product embeddings (k dims)
  3. customer taste vector = revenue-weighted mean of bought-product embeddings
     (full window + recent-90d), plus taste drift and diversity.

Embeddings are learned ONLY from observation-window baskets (no leakage)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import normalize


def build_product_embeddings(obs: pd.DataFrame, n_components: int = 32,
                             min_occur: int = 5, min_cooccur: int = 3):
    """Return (product2idx, embeddings[n_products, k])."""
    # keep products with enough support
    occ = obs.groupby("product_id")["basket_id"].nunique()
    keep = occ[occ >= min_occur].index
    o = obs[obs["product_id"].isin(keep)]
    products = np.sort(o["product_id"].unique())
    baskets = o["basket_id"].unique()
    p2i = {p: i for i, p in enumerate(products)}
    b2i = {b: i for i, b in enumerate(baskets)}
    rows = o["basket_id"].map(b2i).to_numpy()
    cols = o["product_id"].map(p2i).to_numpy()
    data = np.ones(len(o), dtype="float32")
    B = sparse.csr_matrix((data, (rows, cols)), shape=(len(baskets), len(products)))
    B.data[:] = 1.0  # presence, not count
    co = (B.T @ B).tocsr().astype("float64")     # product x product co-occurrence
    co.setdiag(0); co.eliminate_zeros()
    # threshold rare co-occurrences
    co.data[co.data < min_cooccur] = 0.0
    co.eliminate_zeros()

    total = co.sum()
    row_sums = np.asarray(co.sum(axis=1)).ravel()
    coo = co.tocoo()
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((coo.data * total) / (row_sums[coo.row] * row_sums[coo.col]))
    pmi[~np.isfinite(pmi)] = 0.0
    pmi[pmi < 0] = 0.0
    ppmi = sparse.csr_matrix((pmi, (coo.row, coo.col)), shape=co.shape)
    ppmi.eliminate_zeros()

    k = min(n_components, min(ppmi.shape) - 1)
    U, sigma, _ = randomized_svd(ppmi, n_components=k, random_state=42)
    emb = normalize(U * np.sqrt(sigma), norm="l2")
    return p2i, emb.astype("float32")


def _weight_matrix(df, c2i, p2i, n_cust, n_prod):
    """Sparse customer x product revenue-weight matrix (clipped >= 0), products limited to p2i."""
    d = df[df["product_id"].isin(p2i)]
    if len(d) == 0:
        return sparse.csr_matrix((n_cust, n_prod), dtype="float32")
    rev = d.groupby(["customer_id", "product_id"])["revenue"].sum().clip(lower=0).reset_index()
    rows = rev["customer_id"].map(c2i).to_numpy()
    cols = rev["product_id"].map(p2i).to_numpy()
    return sparse.csr_matrix((rev["revenue"].to_numpy(dtype="float32"), (rows, cols)),
                             shape=(n_cust, n_prod))


def customer_semantic(obs: pd.DataFrame, snapshot, p2i, emb, recent_days: int = 90):
    """Vectorised: profile = row-normalized customer-product weight matrix @ embeddings."""
    snap = pd.Timestamp(snapshot)
    recent_start = snap - pd.Timedelta(days=recent_days)
    customers = np.sort(obs["customer_id"].unique())
    c2i = {c: i for i, c in enumerate(customers)}
    n_cust, n_prod = len(customers), emb.shape[0]

    W = _weight_matrix(obs, c2i, p2i, n_cust, n_prod)
    Wr = _weight_matrix(obs[obs["date"] >= recent_start], c2i, p2i, n_cust, n_prod)

    def profiles(M):
        rs = np.asarray(M.sum(axis=1)).ravel()
        inv = np.divide(1.0, rs, out=np.zeros_like(rs), where=rs > 0)
        Mn = sparse.diags(inv) @ M
        return (Mn @ emb).astype("float32"), rs

    sem, rs_full = profiles(W)
    sem_recent, rs_recent = profiles(Wr)

    # taste drift = 1 - cosine(full, recent) per customer
    def rownorm(X):
        return np.linalg.norm(X, axis=1)
    nf, nr = rownorm(sem), rownorm(sem_recent)
    denom = nf * nr
    cos = np.einsum("ij,ij->i", sem, sem_recent)
    drift = np.where(denom > 0, 1.0 - cos / np.where(denom > 0, denom, 1.0), np.nan).astype("float32")
    drift[rs_recent <= 0] = np.nan

    # taste diversity = exp(entropy of revenue shares) = effective # products
    W2 = W.tocsr()
    diversity = np.zeros(n_cust, dtype="float32")
    rs = np.asarray(W2.sum(axis=1)).ravel()
    for i in range(n_cust):
        seg = W2.data[W2.indptr[i]:W2.indptr[i + 1]]
        seg = seg[seg > 0]                       # drop zero-value lines (coupon-only items)
        if rs[i] > 0 and seg.size:
            p = seg / seg.sum()
            diversity[i] = float(np.exp(-(p * np.log(p)).sum()))

    tab = pd.DataFrame({"sem_taste_drift": drift, "sem_taste_diversity": diversity},
                       index=customers)
    tab.index.name = "customer_id"
    return customers, sem, tab
