"""Phase 2 — Week 8 — Targeting policy comparison + value-adjusted uplift (RQ4, central contribution).

On the X5 RCT test set, compares targeting policies by INCREMENTAL effect (valid because
treatment is randomized, so any top-K set contains both treated & control):
  - Random
  - RFM-only          (recency/frequency/monetary composite)
  - Value-only (CLV)  (value proxy = historical monetary)   [NOT actual future CLV]
  - Uplift-only       (best uplift model by Qini)
  - Value-adjusted    (uplift x value proxy)   <- central contribution

Metrics @K: incremental conversions IC@K and incremental revenue IR@K, estimated from the
RCT as (rate|treated,topK − rate|control,topK) x n_targeted. Profit@K = IR − cost·n.

Outputs: results/x5_policy_comparison.csv, figures/x5_policy_profit.png,
         figures/x5_value_adjusted_uplift.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config as C  # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def incremental_at_k(score, treatment, target, value, k):
    """Top-k fraction by score; estimate incremental conversions & revenue from the RCT."""
    n = len(score)
    n_top = max(1, int(n * k))
    top = np.argsort(score)[-n_top:]
    t, y, v = treatment[top], target[top], value[top]
    nt, nc = (t == 1).sum(), (t == 0).sum()
    if nt == 0 or nc == 0:
        return 0.0, 0.0, n_top
    rt, rc = y[t == 1].mean(), y[t == 0].mean()             # conversion rates
    vt = (y * v)[t == 1].mean(); vc = (y * v)[t == 0].mean()  # value-weighted
    ic = (rt - rc) * n_top                                   # incremental conversions
    ir = (vt - vc) * n_top                                   # incremental revenue
    return ic, ir, n_top


def main():
    s = pd.read_parquet(C.PROCESSED / "x5_uplift_test_scores.parquet")
    feats = pd.read_parquet(C.PROCESSED / "x5_uplift_features.parquet")[
        ["client_id", "recency_days", "n_transactions", "monetary"]]
    s = s.merge(feats, on="client_id", how="left")

    t = s["treatment_flg"].values.astype(int)
    y = s["target"].values.astype(int)
    v = s["value_proxy"].values.astype(float)

    # pick best uplift model by Qini from the models table
    mt = pd.read_csv(C.RESULTS / "x5_uplift_models.csv")
    best_up = mt.sort_values("Qini_AUC", ascending=False).iloc[0]["model"]
    uplift = s[f"uplift_{best_up}"].values
    print(f"=== policy comparison ===  best uplift model = {best_up}")

    # RFM composite (z-scored -R +F +M)
    def z(a):
        a = np.asarray(a, float); sd = a.std()
        return (a - a.mean()) / sd if sd > 0 else a * 0
    rfm = -z(s["recency_days"]) + z(s["n_transactions"]) + z(s["monetary"])

    policies = {
        "Random": np.random.RandomState(0).rand(len(s)),
        "RFM-only": rfm,
        "Value-only(CLV proxy)": v,
        f"Uplift-only({best_up})": uplift,
        "Value-adjusted(uplift x value)": uplift * v,
    }

    ks = np.linspace(0.05, 0.5, 19)
    cost_per_contact = 100.0   # assumed marketing cost per targeted customer (rubles)
    curves = {}
    rows = []
    for name, score in policies.items():
        ic_c, ir_c, profit_c = [], [], []
        for k in ks:
            ic, ir, n_top = incremental_at_k(score, t, y, v, k)
            ic_c.append(ic); ir_c.append(ir); profit_c.append(ir - cost_per_contact * n_top)
        curves[name] = (ic_c, ir_c, profit_c)
        for kk in (0.10, 0.20, 0.30):
            ic, ir, n_top = incremental_at_k(score, t, y, v, kk)
            rows.append({"policy": name, "K": f"{int(kk*100)}%", "n_targeted": n_top,
                         "incremental_conversions": round(ic, 1),
                         "incremental_revenue": round(ir, 0),
                         "profit": round(ir - cost_per_contact * n_top, 0)})
    res = pd.DataFrame(rows)
    res.to_csv(C.RESULTS / "x5_policy_comparison.csv", index=False)
    print("\n" + res.to_string(index=False))

    # figure: profit vs K
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (ic_c, ir_c, profit_c) in curves.items():
        ax.plot(ks * 100, np.array(profit_c) / 1e6, "-o", ms=3, label=name)
    ax.axhline(0, color="grey", lw=0.8); ax.set_xlabel("Top K% targeted")
    ax.set_ylabel("Profit (million rubles)"); ax.set_title("X5 targeting profit by policy")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(C.FIGURES / "x5_policy_profit.png", dpi=140)

    # figure: incremental revenue (value-adjusted vs uplift-only vs value-only)
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    for name in ["Value-adjusted(uplift x value)", f"Uplift-only({best_up})",
                 "Value-only(CLV proxy)", "Random"]:
        ax2.plot(ks * 100, np.array(curves[name][1]) / 1e6, "-o", ms=3, label=name)
    ax2.set_xlabel("Top K% targeted"); ax2.set_ylabel("Incremental revenue (million rubles)")
    ax2.set_title("X5 incremental revenue by policy"); ax2.legend(fontsize=8)
    fig2.tight_layout(); fig2.savefig(C.FIGURES / "x5_value_adjusted_uplift.png", dpi=140)
    print(f"\n  -> results/x5_policy_comparison.csv + figures/x5_policy_profit.png")


if __name__ == "__main__":
    main()
