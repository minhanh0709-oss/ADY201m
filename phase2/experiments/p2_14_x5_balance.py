"""Phase 2 — REVISION P1.2 — X5 audit + covariate balance (RQ3 support).

Shows the campaign is a balanced treatment/control design by reporting, per pre-communication
covariate, the treated/control means and the standardised mean difference (SMD). A well-randomised
design has |SMD| < 0.1 for all covariates. Also re-states the headline audit numbers.

Output: results/x5_balance.csv
"""
import sys, math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config as C  # noqa: E402

COVARS = ["monetary", "n_transactions", "avg_basket_value", "recency_days", "tenure_days",
          "n_product_lines", "points_received", "age"]


def main():
    df = pd.read_parquet(C.PROCESSED / "x5_uplift_features.parquet")
    t = df["treatment_flg"].astype(int)
    y = df["target"].astype(int)
    nt, nc = int((t == 1).sum()), int((t == 0).sum())
    rt, rc = y[t == 1].mean(), y[t == 0].mean()
    raw = rt - rc
    se = math.sqrt(rt * (1 - rt) / nt + rc * (1 - rc) / nc)

    rows = [{"covariate": "(n clients)", "treated_mean": nt, "control_mean": nc, "SMD": ""},
            {"covariate": "(response rate)", "treated_mean": round(rt, 4),
             "control_mean": round(rc, 4),
             "SMD": f"raw uplift={raw:.4f} [{raw-1.96*se:.4f},{raw+1.96*se:.4f}]"}]
    maxabs = 0.0
    for cov in COVARS:
        if cov not in df.columns:
            continue
        a, b = df.loc[t == 1, cov].astype(float), df.loc[t == 0, cov].astype(float)
        pooled = math.sqrt((a.var() + b.var()) / 2)
        smd = (a.mean() - b.mean()) / pooled if pooled > 0 else 0.0
        maxabs = max(maxabs, abs(smd))
        rows.append({"covariate": cov, "treated_mean": round(float(a.mean()), 2),
                     "control_mean": round(float(b.mean()), 2), "SMD": round(float(smd), 4)})
    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS / "x5_balance.csv", index=False)
    print("=== X5 treatment/control balance ===")
    print(out.to_string(index=False))
    print(f"\n  treatment ratio = {nt/(nt+nc):.3f}  max |SMD| over covariates = {maxabs:.4f}")
    print(f"  {'BALANCED (all |SMD|<0.1)' if maxabs < 0.1 else 'check: some |SMD|>=0.1'}")


if __name__ == "__main__":
    main()
