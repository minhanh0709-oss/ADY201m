"""Phase 2 — Week 4 — CLV model benchmark + feature ablation (RQ1, RQ2).

Runs the model suite and a Hurdle feature-group ablation on one CLV context, using the
feature table from p2_02_build_features.py (+ semantic embeddings from the npz). Stratified
80/20 split on is_vip_top10 (seed 42), matching Phase 1 protocol.

Outputs (per dataset):
  results/{ds}_clv_benchmark.csv      model comparison (NG, RC@10, Precision@10, MAE...)
  results/{ds}_clv_ablation.csv       Hurdle over {rfm, +behavioral, +sequence, +semantic, all}
  figures/{ds}_revenue_capture.png    capture curve: best vs Monetary vs Random
  figures/{ds}_decile_calibration.png decile calibration of best model

Run:  python phase2/experiments/p2_04_clv_benchmark.py --dataset online_retail
      python phase2/experiments/p2_04_clv_benchmark.py --dataset dunnhumby
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config as C  # noqa: E402
import utils_metrics as M  # noqa: E402
from models_clv import MODELS, m_monetary  # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RFM = ["recency_days", "frequency", "monetary", "log_monetary", "log_frequency"]
BEHAVIORAL = ["n_baskets", "tenure_days", "active_days", "avg_order_value", "avg_basket_size",
              "avg_days_between_orders", "product_diversity", "category_diversity",
              "recency_to_snapshot", "regularity", "spend_per_active_day"]
SEQUENCE = ["seq_spend_last_period", "seq_spend_recent_half", "seq_recent_vs_older_ratio",
            "seq_active_periods", "seq_spend_slope", "seq_orders_slope"]
SEM_SCALAR = ["sem_taste_drift", "sem_taste_diversity"]


def load(ds):
    df = pd.read_parquet(C.PROCESSED / f"{ds}_features.parquet")
    # merge semantic embeddings (sem_emb_0..k)
    npz = C.PROCESSED / f"{ds}_semantic.npz"
    sem_cols = []
    if npz.exists():
        z = np.load(npz, allow_pickle=True)
        emb = pd.DataFrame(z["embeddings"],
                           columns=[f"sem_emb_{i}" for i in range(z["embeddings"].shape[1])])
        emb["customer_id"] = z["customers"]
        df = df.merge(emb, on="customer_id", how="left")
        sem_cols = [c for c in df.columns if c.startswith("sem_emb_")]
    return df, sem_cols


def evaluate_suite(Xtr, ytr, Xte, yte):
    rows = []
    preds = {}
    for name, fn in MODELS.items():
        try:
            yp = fn(Xtr, ytr, Xte)
            preds[name] = yp
            rows.append({"model": name, **M.comprehensive_metrics(yte, yp)})
        except Exception as e:
            rows.append({"model": name, "error": str(e)[:80]})
    return pd.DataFrame(rows), preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["online_retail", "dunnhumby"], required=True)
    args = ap.parse_args()
    ds = args.dataset
    df, sem_emb = load(ds)
    all_feats = RFM + BEHAVIORAL + SEQUENCE + SEM_SCALAR + sem_emb
    all_feats = [c for c in all_feats if c in df.columns]
    y = df["actual_clv"].values
    strat = df["is_vip_top10"].values

    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.2,
                                      stratify=strat, random_state=42)
    print(f"=== CLV benchmark: {ds} ===")
    print(f"  n={len(df)}  features={len(all_feats)}  train={len(idx_tr)} test={len(idx_te)}  "
          f"vip_test={int(strat[idx_te].sum())}")

    X = df[all_feats]
    Xtr, Xte = X.iloc[idx_tr], X.iloc[idx_te]
    ytr, yte = y[idx_tr], y[idx_te]

    # ---- model suite ----
    bench, preds = evaluate_suite(Xtr, ytr, Xte, yte)
    bench = bench.sort_values("Norm_Gini", ascending=False, na_position="last")
    bench.to_csv(C.RESULTS / f"{ds}_clv_benchmark.csv", index=False)
    print("\n  -- model suite (sorted by Norm_Gini) --")
    print(bench.to_string(index=False))

    # ---- ablation (Hurdle over feature groups) ----
    groups = {
        "RFM": RFM,
        "RFM+behavioral": RFM + BEHAVIORAL,
        "RFM+sequence": RFM + SEQUENCE,
        "RFM+semantic": RFM + SEM_SCALAR + sem_emb,
        "ALL": all_feats,
    }
    abl_rows = []
    for gname, cols in groups.items():
        cols = [c for c in cols if c in df.columns]
        yp = MODELS["Hurdle"](X[cols].iloc[idx_tr], ytr, X[cols].iloc[idx_te])
        abl_rows.append({"variant": gname, "n_features": len(cols),
                         **M.comprehensive_metrics(yte, yp)})
    abl = pd.DataFrame(abl_rows)
    abl.to_csv(C.RESULTS / f"{ds}_clv_ablation.csv", index=False)
    print("\n  -- Hurdle ablation --")
    print(abl[["variant", "n_features", "Norm_Gini", "RevCapture_10", "Precision_10", "MAE"]].to_string(index=False))

    # ---- figures ----
    best = bench.iloc[0]["model"]
    yp_best = preds[best]
    ks = np.linspace(0.02, 0.5, 25)
    cap_best = [M.revenue_capture_at_k(yte, yp_best, k) for k in ks]
    cap_mon = [M.revenue_capture_at_k(yte, m_monetary(Xtr, ytr, Xte), k) for k in ks]
    cap_rand = [100 * k for k in ks]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(ks * 100, cap_best, "-o", ms=3, label=f"{best} (best)")
    ax.plot(ks * 100, cap_mon, "-s", ms=3, label="Monetary baseline")
    ax.plot(ks * 100, cap_rand, "--", color="grey", label="Random")
    ax.set_xlabel("Top K% targeted"); ax.set_ylabel("Revenue captured (%)")
    ax.set_title(f"Revenue capture curve — {ds}"); ax.legend()
    fig.tight_layout(); fig.savefig(C.FIGURES / f"{ds}_revenue_capture.png", dpi=140)

    g, dmape = M.decile_calibration(yte, yp_best)
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.5))
    ax2.plot(g["decile"], g["mean_actual"], "-o", label="actual")
    ax2.plot(g["decile"], g["mean_pred"], "-s", label="predicted")
    ax2.set_xlabel("Predicted decile"); ax2.set_ylabel("Mean CLV ($)")
    ax2.set_title(f"Decile calibration — {ds} ({best}, MAPE={dmape:.2f})"); ax2.legend()
    fig2.tight_layout(); fig2.savefig(C.FIGURES / f"{ds}_decile_calibration.png", dpi=140)

    print(f"\n  best model: {best}  ->  results/{ds}_clv_benchmark.csv, {ds}_clv_ablation.csv")


if __name__ == "__main__":
    main()
