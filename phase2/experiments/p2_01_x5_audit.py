"""Phase 2 — Week 1 — X5 RetailHero dataset audit (feasibility gating).

Checks whether X5 is usable for uplift modeling BEFORE training:
  - treatment/control balance
  - target rate treated vs control  ->  raw uplift = p(target|T=1) - p(target|T=0)
  - purchase history per customer (signal for features)
  - value-proxy availability (purchase_sum) for value-adjusted uplift
  - missing / coverage diagnostics

Outputs:
  tables/x5_audit_statistics.csv
  tables/x5_purchases_per_customer.csv
  figures/audit_x5_uplift_signal.png
  results/x5_feasibility.json   (pass/fail + reasons)

Run:  python phase2/experiments/p2_01_x5_audit.py
"""
import gzip
import json
import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as C  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


EXPECTED = {  # server Content-Length; a partial download is smaller and must be skipped
    "uplift_train.csv.gz": 1182423,
    "clients.csv.gz": 7637755,
    "purchases.csv.gz": 669979268,
}


def _complete(p: Path) -> bool:
    """File present, full size, and a valid gzip (guards against partial downloads)."""
    if not p.exists() or p.stat().st_size < EXPECTED.get(p.name, 1):
        return False
    try:
        with gzip.open(p, "rb") as f:
            while f.read(1 << 20):
                pass
        return True
    except Exception as e:
        print(f"  [warn] {p.name} not a complete gzip: {e}")
        return False


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def main():
    stats = {}
    train_p = C.X5_RAW / "uplift_train.csv.gz"
    clients_p = C.X5_RAW / "clients.csv.gz"
    purch_p = C.X5_RAW / "purchases.csv.gz"

    print("=== X5 audit ===")
    print("files present:",
          {p.name: (p.exists() and round(p.stat().st_size / 1e6, 1)) for p in (train_p, clients_p, purch_p)})

    # ---------- 1. uplift_train: treatment/control + raw uplift ----------
    assert train_p.exists() and _complete(train_p), "uplift_train.csv.gz missing/incomplete"
    train = pd.read_csv(train_p)
    train.columns = [c.strip() for c in train.columns]
    n = len(train)
    n_treat = int((train["treatment_flg"] == 1).sum())
    n_ctrl = int((train["treatment_flg"] == 0).sum())
    tr_treat = train.loc[train.treatment_flg == 1, "target"]
    tr_ctrl = train.loc[train.treatment_flg == 0, "target"]
    rate_t = tr_treat.mean()
    rate_c = tr_ctrl.mean()
    raw_uplift = rate_t - rate_c
    # CI for difference of proportions
    se = math.sqrt(rate_t * (1 - rate_t) / len(tr_treat) + rate_c * (1 - rate_c) / len(tr_ctrl))
    uplift_lo, uplift_hi = raw_uplift - 1.96 * se, raw_uplift + 1.96 * se

    stats.update({
        "train_n_clients": n,
        "train_n_treatment": n_treat,
        "train_n_control": n_ctrl,
        "treatment_ratio": round(n_treat / n, 4),
        "target_rate_overall": round(train["target"].mean(), 4),
        "target_rate_treated": round(rate_t, 4),
        "target_rate_control": round(rate_c, 4),
        "raw_uplift": round(raw_uplift, 4),
        "raw_uplift_ci95_lo": round(uplift_lo, 4),
        "raw_uplift_ci95_hi": round(uplift_hi, 4),
        "uplift_signal_significant": bool(uplift_lo > 0 or uplift_hi < 0),
    })
    print(f"  n={n}  treat_ratio={n_treat/n:.3f}  rate_T={rate_t:.4f}  rate_C={rate_c:.4f}  "
          f"raw_uplift={raw_uplift:.4f} [{uplift_lo:.4f},{uplift_hi:.4f}]")

    # ---------- 2. clients: demographics coverage ----------
    if clients_p.exists() and _complete(clients_p):
        clients = pd.read_csv(clients_p)
        clients.columns = [c.strip() for c in clients.columns]
        stats["clients_n"] = len(clients)
        for col in ("age", "gender", "first_issue_date", "first_redeem_date"):
            if col in clients.columns:
                miss = clients[col].isna().mean()
                stats[f"clients_missing_{col}"] = round(float(miss), 4)
        # coverage of train clients in clients table
        cov = train["client_id"].isin(set(clients["client_id"])).mean()
        stats["train_clients_in_clients_table"] = round(float(cov), 4)
        print(f"  clients_n={len(clients)}  train coverage in clients={cov:.4f}")
    else:
        print("  [skip] clients.csv.gz missing/incomplete")

    # ---------- 3. purchases: history per customer (chunked, vectorised) ----------
    per_cust = None
    if purch_p.exists() and _complete(purch_p):
        print("  scanning purchases (chunked, vectorised)...")
        usecols = ["client_id", "transaction_id", "purchase_sum", "product_quantity"]
        line_parts = []   # per-chunk additive aggregates (lines, qty)
        txn_pairs = []     # unique (client_id, transaction_id) for txn count + txn-level spend
        total_rows = 0
        reader = pd.read_csv(purch_p, usecols=lambda c: c in usecols,
                             chunksize=4_000_000)
        for i, chunk in enumerate(reader):
            total_rows += len(chunk)
            chunk["purchase_sum"] = pd.to_numeric(chunk["purchase_sum"], errors="coerce")
            chunk["product_quantity"] = pd.to_numeric(chunk["product_quantity"], errors="coerce")
            # additive per-client: product lines + quantity
            lp = chunk.groupby("client_id").agg(
                n_product_lines=("transaction_id", "size"),
                total_quantity=("product_quantity", "sum"),
            )
            line_parts.append(lp)
            # transaction-level: purchase_sum is repeated per line within a basket, so take
            # one value per (client,transaction) to get a clean basket value.
            tp = (chunk[["client_id", "transaction_id", "purchase_sum"]]
                  .drop_duplicates(subset=["client_id", "transaction_id"]))
            txn_pairs.append(tp)
            print(f"    chunk {i}: rows so far={total_rows:,}")
        # combine additive line aggregates
        lines = pd.concat(line_parts).groupby(level=0).sum()
        # dedup transactions across chunk boundaries, then per-client basket count + spend
        txns = pd.concat(txn_pairs).drop_duplicates(subset=["client_id", "transaction_id"])
        basket = txns.groupby("client_id").agg(
            n_transactions=("transaction_id", "size"),
            total_purchase_sum=("purchase_sum", "sum"),
        )
        per_cust = basket.join(lines, how="outer").reset_index()
        per_cust["n_transactions"] = per_cust["n_transactions"].fillna(0).astype(int)
        stats["purchases_total_rows"] = int(total_rows)
        stats["purchases_n_clients"] = int(len(per_cust))
        tpc = per_cust["n_transactions"]
        stats["txn_per_customer_mean"] = round(float(tpc.mean()), 3)
        stats["txn_per_customer_median"] = float(tpc.median())
        for k in (2, 5, 10):
            stats[f"pct_clients_ge_{k}_txn"] = round(float((tpc >= k).mean()), 4)
        # train clients with purchase history
        train_with_hist = train["client_id"].isin(set(per_cust["client_id"])).mean()
        stats["train_clients_with_purchase_history"] = round(float(train_with_hist), 4)
        stats["value_proxy_available"] = True
        per_cust.to_csv(C.TABLES / "x5_purchases_per_customer.csv", index=False)
        print(f"  purchases rows={total_rows:,}  clients={len(per_cust):,}  "
              f"txn/cust mean={tpc.mean():.2f} median={tpc.median():.0f}")
    else:
        print("  [skip] purchases.csv.gz missing/incomplete — history features deferred")
        stats["value_proxy_available"] = False

    # ---------- 4. figure ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(["Control", "Treated"], [rate_c, rate_t],
                color=["#9aa7b2", "#2c7fb8"])
    axes[0].set_title(f"Target rate (raw uplift = {raw_uplift:+.4f})")
    axes[0].set_ylabel("P(target=1)")
    for x, v in zip([0, 1], [rate_c, rate_t]):
        axes[0].text(x, v + 0.005, f"{v:.3f}", ha="center")
    if per_cust is not None:
        tpc = per_cust["n_transactions"].clip(upper=60)
        axes[1].hist(tpc, bins=40, color="#41ab5d")
        axes[1].set_title("Transactions per customer (clipped@60)")
        axes[1].set_xlabel("n transactions (pre-communication)")
    else:
        axes[1].text(0.5, 0.5, "purchases.csv.gz\nnot yet available",
                     ha="center", va="center")
        axes[1].set_axis_off()
    fig.tight_layout()
    fig.savefig(C.FIGURES / "audit_x5_uplift_signal.png", dpi=140)
    print(f"  figure -> {C.FIGURES / 'audit_x5_uplift_signal.png'}")

    # ---------- 5. feasibility decision ----------
    reasons = []
    ok = True
    if not (0.3 <= n_treat / n <= 0.7):
        ok = False; reasons.append("treatment ratio far from balanced")
    if not stats["uplift_signal_significant"]:
        ok = False; reasons.append("raw uplift CI includes 0 (weak/absent signal)")
    if raw_uplift <= 0:
        reasons.append("WARNING: raw uplift <= 0 (communication not positively associated)")
    feas = {"dataset": "X5 RetailHero", "decision": "PASS" if ok else "REVIEW",
            "reasons": reasons, "stats": stats}
    (C.RESULTS / "x5_feasibility.json").write_text(json.dumps(feas, indent=2))

    pd.DataFrame(sorted(stats.items()), columns=["metric", "value"]).to_csv(
        C.TABLES / "x5_audit_statistics.csv", index=False)
    print(f"\n  DECISION: {feas['decision']}  reasons={reasons}")
    print(f"  tables -> {C.TABLES / 'x5_audit_statistics.csv'}")


if __name__ == "__main__":
    main()
