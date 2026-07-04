"""Phase 2 — Week 1 — Dunnhumby 'Complete Journey' dataset audit (feasibility gating).

Auto-detects a Kaggle zip dropped in phase2/data/raw/dunnhumby/ (any *.zip),
extracts it, then audits whether the data supports household-level CLV validation:
  - households, baskets, transaction lines, time span (weeks)
  - baskets per household; % households with >=2/5/10 baskets
  - zero future-spend rate + skewness of non-zero future spend (CLV is zero-inflated)
  - demographics coverage; product / commodity (category) counts

A default CLV window (observation weeks 1-78, prediction weeks 79-102) is used ONLY for
the zero-rate / skew diagnostic; the real walk-forward windows are designed in Week 3.

Outputs:
  tables/dunnhumby_audit_statistics.csv
  figures/audit_dunnhumby_zero_skew.png
  results/dunnhumby_feasibility.json

Run:  python phase2/experiments/p2_01_dunnhumby_audit.py
"""
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as C  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def find_csv(name_options):
    """Find a CSV by any of several candidate names, recursively under DUNN_RAW."""
    for opt in name_options:
        hits = list(C.DUNN_RAW.rglob(opt))
        if hits:
            return hits[0]
    return None


def ensure_extracted():
    """Extract any *.zip found in the dunnhumby raw dir if CSVs aren't present yet."""
    has_csv = any(C.DUNN_RAW.rglob("transaction_data.csv"))
    if has_csv:
        print("  transaction_data.csv already present.")
        return True
    zips = list(C.DUNN_RAW.rglob("*.zip"))
    if not zips:
        print(f"  [BLOCKED] No CSVs and no .zip in {C.DUNN_RAW}")
        print("  -> Download Kaggle 'frtgnn/dunnhumby-the-complete-journey' and drop the")
        print("     zip (or extracted CSVs) into that folder, then re-run.")
        return False
    for z in zips:
        print(f"  extracting {z.name} ...")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(C.DUNN_RAW)
    # handle a possible single nested zip (Kaggle sometimes nests)
    for z in C.DUNN_RAW.rglob("*.zip"):
        if z not in zips:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(z.parent)
    return any(C.DUNN_RAW.rglob("transaction_data.csv"))


def main():
    print("=== Dunnhumby audit ===")
    if not ensure_extracted():
        sys.exit(2)

    tx_path = find_csv(["transaction_data.csv"])
    prod_path = find_csv(["product.csv"])
    demo_path = find_csv(["hh_demographic.csv"])
    print(f"  transaction file: {tx_path}")

    tx = pd.read_csv(tx_path)
    tx.columns = [c.strip().upper() for c in tx.columns]
    # canonical column names in the Complete Journey schema
    HH = "HOUSEHOLD_KEY"
    col = {c: c for c in tx.columns}
    assert HH in tx.columns, f"missing {HH}; got {list(tx.columns)[:12]}"
    for need in ("BASKET_ID", "SALES_VALUE", "WEEK_NO", "PRODUCT_ID"):
        assert need in tx.columns, f"missing {need}"

    stats = {}
    stats["n_households"] = int(tx[HH].nunique())
    stats["n_baskets"] = int(tx["BASKET_ID"].nunique())
    stats["n_transaction_lines"] = int(len(tx))
    stats["n_products"] = int(tx["PRODUCT_ID"].nunique())
    stats["week_min"] = int(tx["WEEK_NO"].min())
    stats["week_max"] = int(tx["WEEK_NO"].max())
    stats["total_sales_value"] = round(float(tx["SALES_VALUE"].sum()), 2)

    # baskets per household
    bpc = tx.groupby(HH)["BASKET_ID"].nunique()
    stats["baskets_per_hh_mean"] = round(float(bpc.mean()), 2)
    stats["baskets_per_hh_median"] = float(bpc.median())
    for k in (2, 5, 10):
        stats[f"pct_hh_ge_{k}_baskets"] = round(float((bpc >= k).mean()), 4)

    # ----- CLV window diagnostic: zero future-spend rate + skew -----
    wk_max = int(tx["WEEK_NO"].max())
    obs_end = 78 if wk_max >= 90 else int(wk_max * 0.75)
    obs = tx[tx["WEEK_NO"] <= obs_end]
    fut = tx[tx["WEEK_NO"] > obs_end]
    obs_hh = obs[HH].unique()
    future_spend = (fut.groupby(HH)["SALES_VALUE"].sum()
                    .reindex(obs_hh).fillna(0.0))   # households seen in observation window
    stats["clv_window_obs_end_week"] = obs_end
    stats["clv_window_pred_weeks"] = f"{obs_end+1}-{wk_max}"
    stats["n_hh_in_observation"] = int(len(obs_hh))
    stats["zero_future_spend_rate"] = round(float((future_spend <= 0).mean()), 4)
    nz = future_spend[future_spend > 0]
    stats["nonzero_future_spend_skew"] = round(float(sps.skew(nz)), 3) if len(nz) > 2 else None
    stats["nonzero_future_spend_mean"] = round(float(nz.mean()), 2) if len(nz) else None
    stats["nonzero_future_spend_median"] = round(float(nz.median()), 2) if len(nz) else None

    # ----- products / categories -----
    if prod_path is not None:
        prod = pd.read_csv(prod_path)
        prod.columns = [c.strip().upper() for c in prod.columns]
        for cat in ("COMMODITY_DESC", "SUB_COMMODITY_DESC", "DEPARTMENT"):
            if cat in prod.columns:
                stats[f"n_{cat.lower()}"] = int(prod[cat].nunique())
        stats["product_table_rows"] = int(len(prod))

    # ----- demographics coverage -----
    if demo_path is not None:
        demo = pd.read_csv(demo_path)
        demo.columns = [c.strip().upper() for c in demo.columns]
        hh_demo = set(demo[HH]) if HH in demo.columns else set()
        cov = np.mean([h in hh_demo for h in tx[HH].unique()])
        stats["demographics_coverage"] = round(float(cov), 4)
        stats["n_hh_with_demographics"] = int(len(hh_demo))

    # ----- figure -----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(["zero future\nspend", "non-zero"],
                [stats["zero_future_spend_rate"], 1 - stats["zero_future_spend_rate"]],
                color=["#d95f0e", "#31a354"])
    axes[0].set_title(f"Future-spend zero rate (obs<=w{obs_end})")
    axes[0].set_ylabel("fraction of households")
    if len(nz):
        axes[1].hist(np.log1p(nz), bins=40, color="#3182bd")
        axes[1].set_title(f"log1p(non-zero future spend)  skew={stats['nonzero_future_spend_skew']}")
        axes[1].set_xlabel("log1p(SALES_VALUE)")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "audit_dunnhumby_zero_skew.png", dpi=140)

    # ----- feasibility -----
    reasons = []
    ok = True
    if stats["n_households"] < 1000:
        ok = False; reasons.append("too few households")
    if stats["pct_hh_ge_2_baskets"] < 0.5:
        reasons.append("WARNING: many one-basket households")
    if not (0.05 <= stats["zero_future_spend_rate"] <= 0.95):
        reasons.append("WARNING: degenerate zero-rate for CLV target")
    feas = {"dataset": "Dunnhumby Complete Journey",
            "decision": "PASS" if ok else "REVIEW", "reasons": reasons, "stats": stats}
    (C.RESULTS / "dunnhumby_feasibility.json").write_text(json.dumps(feas, indent=2))
    pd.DataFrame(sorted(stats.items()), columns=["metric", "value"]).to_csv(
        C.TABLES / "dunnhumby_audit_statistics.csv", index=False)

    for k, v in stats.items():
        print(f"    {k}: {v}")
    print(f"\n  DECISION: {feas['decision']}  reasons={reasons}")
    print(f"  tables -> {C.TABLES / 'dunnhumby_audit_statistics.csv'}")


if __name__ == "__main__":
    main()
