"""Phase 2 — REVISION P1.3 — Profit-simulation sensitivity (RQ4 support).

Profit@K = sum_{i in TopK} (tau_i * V_i * margin - contact_cost), estimated on the RCT test set.
We sweep contact_cost in {50,100,200} rubles and margin in {0.1,0.2,0.3} and report, for each
(cost,margin), the profit@10% of each policy and which policy wins -- to show the ranking is
stable, not an artefact of one cost assumption.

Output: results/x5_profit_sensitivity.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config as C  # noqa: E402
from p2_09_policy_comparison import incremental_at_k  # reuse the estimator


def main():
    s = pd.read_parquet(C.PROCESSED / "x5_uplift_test_scores.parquet")
    feats = pd.read_parquet(C.PROCESSED / "x5_uplift_features.parquet")[
        ["client_id", "recency_days", "n_transactions", "monetary"]]
    s = s.merge(feats, on="client_id", how="left")
    t = s["treatment_flg"].values.astype(int)
    y = s["target"].values.astype(int)
    v = s["value_proxy"].values.astype(float)
    mt = pd.read_csv(C.RESULTS / "x5_uplift_models.csv")
    best = mt.sort_values("Qini_AUC", ascending=False).iloc[0]["model"]
    up = s[f"uplift_{best}"].values

    def z(a):
        a = np.asarray(a, float); sd = a.std(); return (a - a.mean()) / sd if sd else a * 0
    rfm = -z(s["recency_days"]) + z(s["n_transactions"]) + z(s["monetary"])
    policies = {"RFM": rfm, "Value": v, "Uplift": up, "Value-adjusted": up * v}

    K = 0.10
    rows = []
    for cost in (50, 100, 200):
        for margin in (0.1, 0.2, 0.3):
            profits = {}
            for name, sc in policies.items():
                ic, ir, n_top = incremental_at_k(sc, t, y, v, K)
                profits[name] = ir * margin - cost * n_top
            winner = max(profits, key=profits.get)
            rows.append({"contact_cost": cost, "margin": margin, "winner@10%": winner,
                         **{f"profit_{n}": round(p, 0) for n, p in profits.items()}})
    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS / "x5_profit_sensitivity.csv", index=False)
    print("=== profit@10% sensitivity (cost x margin) ===")
    print(out.to_string(index=False))
    wins = out["winner@10%"].value_counts()
    print(f"\n  winner counts across 9 (cost,margin) settings: {dict(wins)}")


if __name__ == "__main__":
    main()
