"""Phase 2 — Week 2 — Common feature engineering (RFM + behavioural) for the CLV datasets.

Same pipeline for both contexts via the normalized schema:
  - online_retail : reads Phase-1 cleaned CSV (READ-ONLY reuse) for cross-context CLV (RQ1)
  - dunnhumby     : reads transaction_data.csv from phase2/data/raw/dunnhumby/ (if present)

Writes per-customer feature table + actual_clv label to phase2/data/processed/.

Run:  python phase2/experiments/p2_02_build_features.py --dataset online_retail
      python phase2/experiments/p2_02_build_features.py --dataset dunnhumby
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config as C  # noqa: E402
from features import schema as S  # noqa: E402
from features.rfm import rfm_features  # noqa: E402
from features.behavioral import behavioral_features  # noqa: E402
from features.sequence import trend_features, sequence_matrices  # noqa: E402
from features.semantic_graph import build_product_embeddings, customer_semantic  # noqa: E402
import numpy as np  # noqa: E402


def load_online_retail():
    df = pd.read_csv(C.P1_ONLINE_RETAIL_CLEANED)
    tx = S.normalize_online_retail(df)
    # Phase-1 protocol: observation <= 2011-06-30, prediction horizon ~162 days
    return tx, "2011-06-30", 162, None


def load_dunnhumby():
    hits = list(C.DUNN_RAW.rglob("transaction_data.csv"))
    if not hits:
        sys.exit("  [BLOCKED] transaction_data.csv not found — drop the Kaggle zip into "
                 f"{C.DUNN_RAW} (see p2_01_dunnhumby_audit.py).")
    df = pd.read_csv(hits[0])
    tx = S.normalize_dunnhumby(df)
    # grocery: observation up to ~78 weeks, predict next ~13 weeks; map to synthetic days
    snap = tx["date"].min() + pd.Timedelta(weeks=78)
    prod_cat = None
    prod_hits = list(C.DUNN_RAW.rglob("product.csv"))
    if prod_hits:
        prod = pd.read_csv(prod_hits[0])
        prod.columns = [c.upper() for c in prod.columns]
        if "COMMODITY_DESC" in prod.columns:
            prod_cat = dict(zip(prod["PRODUCT_ID"].astype(str), prod["COMMODITY_DESC"]))
    return tx, snap, 13 * 7, prod_cat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["online_retail", "dunnhumby"], required=True)
    args = ap.parse_args()

    tx, snapshot, horizon, prod_cat = (load_online_retail() if args.dataset == "online_retail"
                                       else load_dunnhumby())
    print(f"=== build features: {args.dataset} ===")
    print(f"  tx lines={len(tx):,}  customers={tx['customer_id'].nunique():,}  "
          f"date span={tx['date'].min().date()}..{tx['date'].max().date()}")

    obs, pred, snap = S.split_observation_prediction(tx, snapshot, horizon)
    customers = obs["customer_id"].unique()
    print(f"  snapshot={snap.date()}  horizon={horizon}d  obs_customers={len(customers):,}  "
          f"obs_lines={len(obs):,}  pred_lines={len(pred):,}")

    rfm = rfm_features(obs, snap)
    beh = behavioral_features(obs, snap, product_category=prod_cat)
    trd = trend_features(obs, snap, n_periods=12, freq="M")

    print("  building semantic product embeddings...")
    p2i, emb = build_product_embeddings(obs, n_components=32)
    sem_customers, sem_mat, sem_tab = customer_semantic(obs, snap, p2i, emb)
    print(f"  product embeddings: {emb.shape}  semantic customer matrix: {sem_mat.shape}")

    label = S.actual_clv(pred, customers)
    feats = rfm.join(beh, how="outer").join(trd, how="left").join(sem_tab, how="left")
    feats = feats.join(label, how="left")
    feats["actual_clv"] = feats["actual_clv"].fillna(0.0)
    feats["is_vip_top10"] = (feats["actual_clv"] >= feats["actual_clv"].quantile(0.90)).astype(int)
    feats = feats.reset_index().rename(columns={"index": "customer_id"})

    out = C.assert_phase2_path(C.PROCESSED / f"{args.dataset}_features.parquet")
    feats.to_parquet(out, index=False)
    feats.to_csv(out.with_suffix(".csv"), index=False)

    # sequence matrices for DL models (saved separately, not in the flat table)
    seq = sequence_matrices(obs, snap, n_periods=12, freq="M")
    seq_path = C.assert_phase2_path(C.PROCESSED / f"{args.dataset}_sequences.npz")
    np.savez_compressed(seq_path, customers=seq["customers"], periods=seq["periods"],
                        spend=seq["spend"], orders=seq["orders"], active=seq["active"])
    print(f"  sequences -> {seq_path}  spend shape={seq['spend'].shape}")

    sem_path = C.assert_phase2_path(C.PROCESSED / f"{args.dataset}_semantic.npz")
    np.savez_compressed(sem_path, customers=sem_customers, embeddings=sem_mat)
    print(f"  semantic -> {sem_path}  shape={sem_mat.shape}")

    zero_rate = (feats["actual_clv"] <= 0).mean()
    print(f"  features: rows={len(feats):,}  cols={feats.shape[1]}  "
          f"zero_clv_rate={zero_rate:.3f}  mean_clv={feats['actual_clv'].mean():.2f}")
    print(f"  cols: {list(feats.columns)}")
    print(f"  -> {out}")
    # quick sanity: no NaN in core RFM
    nan_core = feats[["recency_days", "frequency", "monetary"]].isna().sum().to_dict()
    print(f"  core RFM NaNs: {nan_core}")


if __name__ == "__main__":
    main()
