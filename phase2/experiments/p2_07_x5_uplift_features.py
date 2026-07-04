"""Phase 2 — Week 6 — X5 uplift feature engineering (pre-communication history).

All purchases in X5 are PRIOR to communication, so per-client aggregates are leakage-safe
covariates for uplift. Builds, per client:
  - RFM: recency, frequency (n transactions), monetary (sum transaction purchase_sum)
  - basket: avg_basket_value, total_quantity, n_product_lines, avg_days_between_txn, tenure
  - loyalty engagement: total regular/express points received & spent (signal for who responds)
  - demographics (clients.csv): age, gender, card tenure, has_redeemed, redeem latency
Merged with uplift_train (treatment_flg, target) -> x5_uplift_features.parquet.
The monetary aggregate doubles as the value proxy V̂ for value-adjusted uplift (RQ4).

Run:  python phase2/experiments/p2_07_x5_uplift_features.py
"""
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config as C  # noqa: E402

PURCH = C.X5_RAW / "purchases.csv.gz"
CLIENTS = C.X5_RAW / "clients.csv.gz"
TRAIN = C.X5_RAW / "uplift_train.csv.gz"

USECOLS = ["client_id", "transaction_id", "transaction_datetime", "purchase_sum",
           "product_quantity", "regular_points_received", "express_points_received",
           "regular_points_spent", "express_points_spent"]


def build_transaction_table():
    """Chunk purchases (45.7M lines) -> per-(client,transaction) table (~4.6M rows)."""
    parts = []
    total = 0
    reader = pd.read_csv(PURCH, usecols=lambda c: c in USECOLS, chunksize=5_000_000)
    for i, ch in enumerate(reader):
        total += len(ch)
        for col in ("purchase_sum", "product_quantity", "regular_points_received",
                    "express_points_received", "regular_points_spent", "express_points_spent"):
            ch[col] = pd.to_numeric(ch[col], errors="coerce")
        g = ch.groupby(["client_id", "transaction_id"], sort=False).agg(
            dt=("transaction_datetime", "first"),
            psum=("purchase_sum", "first"),
            qty=("product_quantity", "sum"),
            lines=("transaction_id", "size"),
            rpr=("regular_points_received", "first"),
            epr=("express_points_received", "first"),
            rps=("regular_points_spent", "first"),
            eps=("express_points_spent", "first"),
        ).reset_index()
        parts.append(g)
        print(f"    chunk {i}: lines so far={total:,}  txn rows so far~{sum(len(p) for p in parts):,}")
    txn = pd.concat(parts, ignore_index=True)
    # merge transactions split across chunk boundaries
    txn = txn.groupby(["client_id", "transaction_id"], sort=False).agg(
        dt=("dt", "min"), psum=("psum", "first"), qty=("qty", "sum"), lines=("lines", "sum"),
        rpr=("rpr", "first"), epr=("epr", "first"), rps=("rps", "first"), eps=("eps", "first"),
    ).reset_index()
    txn["dt"] = pd.to_datetime(txn["dt"], errors="coerce")
    print(f"  purchases lines={total:,}  unique transactions={len(txn):,}")
    return txn


def per_client_features(txn):
    ref = txn["dt"].max()  # last observed purchase date = reference "now" (pre-communication)
    g = txn.groupby("client_id")
    first, last = g["dt"].min(), g["dt"].max()
    feats = pd.DataFrame({
        "n_transactions": g["transaction_id"].size().astype("float32"),
        "monetary": g["psum"].sum().astype("float32"),
        "avg_basket_value": g["psum"].mean().astype("float32"),
        "total_quantity": g["qty"].sum().astype("float32"),
        "n_product_lines": g["lines"].sum().astype("float32"),
        "recency_days": (ref - last).dt.days.astype("float32"),
        "tenure_days": (last - first).dt.days.astype("float32"),
        "points_received": (g["rpr"].sum() + g["epr"].sum()).astype("float32"),
        "points_spent": (g["rps"].sum() + g["eps"].sum()).astype("float32"),
    })
    feats["avg_days_between_txn"] = (feats["tenure_days"] /
                                     (feats["n_transactions"] - 1).clip(lower=1)).astype("float32")
    feats["log_monetary"] = np.log1p(feats["monetary"].clip(lower=0))
    feats["log_frequency"] = np.log1p(feats["n_transactions"])
    return feats.reset_index()


def main():
    print("=== X5 uplift feature engineering ===")
    txn = build_transaction_table()
    feats = per_client_features(txn)
    print(f"  per-client features: {feats.shape}")

    clients = pd.read_csv(CLIENTS)
    clients["first_issue_date"] = pd.to_datetime(clients["first_issue_date"], errors="coerce")
    clients["first_redeem_date"] = pd.to_datetime(clients["first_redeem_date"], errors="coerce")
    ref = txn["dt"].max()
    clients["card_tenure_days"] = (ref - clients["first_issue_date"]).dt.days
    clients["has_redeemed"] = clients["first_redeem_date"].notna().astype(int)
    clients["redeem_latency_days"] = (clients["first_redeem_date"] -
                                      clients["first_issue_date"]).dt.days
    clients["gender_code"] = clients["gender"].map({"M": 1, "F": 0, "U": -1}).fillna(-1).astype(int)
    cdemo = clients[["client_id", "age", "gender_code", "card_tenure_days",
                     "has_redeemed", "redeem_latency_days"]]

    train = pd.read_csv(TRAIN)
    df = train.merge(feats, on="client_id", how="left").merge(cdemo, on="client_id", how="left")
    # clients with no purchase history -> 0 activity
    for col in feats.columns:
        if col != "client_id":
            df[col] = df[col].fillna(0)
    out = C.assert_phase2_path(C.PROCESSED / "x5_uplift_features.parquet")
    df.to_parquet(out, index=False)

    print(f"\n  x5_uplift_features: {df.shape}")
    print(f"  cols: {list(df.columns)}")
    print(f"  treatment rate={df['treatment_flg'].mean():.3f}  target rate={df['target'].mean():.3f}")
    print(f"  clients with purchase history: {(df['n_transactions']>0).mean():.3f}")
    print(f"  monetary (value proxy): mean={df['monetary'].mean():.1f} median={df['monetary'].median():.1f}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
