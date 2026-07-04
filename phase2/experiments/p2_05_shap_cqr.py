"""Phase 2 — Week 5 — Explainability (SHAP) + Uncertainty (CQR) for the Hurdle CLV model.

SHAP: TreeExplainer on Stage-1 (P(return)) and Stage-2 (log spend | return) -> per-stage
top feature importances (mean |SHAP|), per context.
CQR: split-conformalized quantile regression (Romano et al. 2019) for CLV prediction
intervals -> empirical coverage vs nominal + mean interval width, per context.

Outputs:
  results/{ds}_shap_stage1.csv, {ds}_shap_stage2.csv
  figures/{ds}_shap_stages.png
  results/{ds}_cqr_coverage.csv

Run:  python phase2/experiments/p2_05_shap_cqr.py --dataset online_retail|dunnhumby
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import shap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config as C  # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from p2_04_clv_benchmark import load, RFM, BEHAVIORAL, SEQUENCE, SEM_SCALAR  # noqa: E402


def shap_importance(model, X, kind):
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(X)
    if isinstance(sv, list):           # classifier -> [class0, class1]
        sv = sv[1]
    imp = np.abs(sv).mean(axis=0)
    imp = imp / imp.sum() if imp.sum() > 0 else imp
    return pd.DataFrame({"feature": X.columns, "mean_abs_shap_norm": imp}) \
        .sort_values("mean_abs_shap_norm", ascending=False).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["online_retail", "dunnhumby"], required=True)
    args = ap.parse_args()
    ds = args.dataset
    df, sem_emb = load(ds)
    feats = [c for c in RFM + BEHAVIORAL + SEQUENCE + SEM_SCALAR + sem_emb if c in df.columns]
    y = df["actual_clv"].values
    strat = df["is_vip_top10"].values
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.2,
                                      stratify=strat, random_state=42)
    X = df[feats]
    Xtr, Xte = X.iloc[idx_tr], X.iloc[idx_te]
    ytr, yte = y[idx_tr], y[idx_te]
    print(f"=== SHAP + CQR: {ds} ===  n={len(df)} feats={len(feats)}")

    # ---------- Hurdle stages for SHAP ----------
    pos = ytr > 0
    clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31,
                             random_state=42, verbose=-1).fit(Xtr, pos.astype(int))
    reg = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31,
                            random_state=42, verbose=-1).fit(Xtr[pos], np.log(ytr[pos]))
    s1 = shap_importance(clf, Xte, "clf")
    s2 = shap_importance(reg, Xte, "reg")
    s1.to_csv(C.RESULTS / f"{ds}_shap_stage1.csv", index=False)
    s2.to_csv(C.RESULTS / f"{ds}_shap_stage2.csv", index=False)
    print("\n  Stage-1 (P return) top5:");  print(s1.head(5).to_string(index=False))
    print("\n  Stage-2 (log spend) top5:"); print(s2.head(5).to_string(index=False))

    # group semantic-embedding importance for readability
    def grp(s):
        s = s.copy()
        s["group"] = np.where(s["feature"].str.startswith("sem_emb_"), "semantic_emb", s["feature"])
        return s.groupby("group")["mean_abs_shap_norm"].sum().sort_values(ascending=False)
    g1, g2 = grp(s1).head(8), grp(s2).head(8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].barh(g1.index[::-1], g1.values[::-1], color="#2c7fb8"); axes[0].set_title(f"{ds} Stage-1 SHAP (P return)")
    axes[1].barh(g2.index[::-1], g2.values[::-1], color="#d95f0e"); axes[1].set_title(f"{ds} Stage-2 SHAP (log spend)")
    fig.tight_layout(); fig.savefig(C.FIGURES / f"{ds}_shap_stages.png", dpi=140)

    # ---------- CQR (split-conformal quantile regression) ----------
    # proper-train / calibration split inside training fold
    i_pt, i_cal = train_test_split(np.arange(len(idx_tr)), test_size=0.5, random_state=42)
    Xpt, Xcal = Xtr.iloc[i_pt], Xtr.iloc[i_cal]
    ypt, ycal = ytr[i_pt], ytr[i_cal]
    rows = []
    for alpha in (0.10, 0.20):
        lo = lgb.LGBMRegressor(objective="quantile", alpha=alpha / 2, n_estimators=400,
                               learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1).fit(Xpt, ypt)
        hi = lgb.LGBMRegressor(objective="quantile", alpha=1 - alpha / 2, n_estimators=400,
                               learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1).fit(Xpt, ypt)
        q_lo_cal, q_hi_cal = lo.predict(Xcal), hi.predict(Xcal)
        E = np.maximum(q_lo_cal - ycal, ycal - q_hi_cal)        # CQR nonconformity
        q_hat = np.quantile(E, (1 - alpha) * (1 + 1 / len(ycal)))
        lo_te = lo.predict(Xte) - q_hat
        hi_te = hi.predict(Xte) + q_hat
        cov = float(np.mean((yte >= lo_te) & (yte <= hi_te)))
        width = float(np.mean(hi_te - lo_te))
        rows.append({"nominal_coverage": round(1 - alpha, 2), "empirical_coverage": round(cov, 4),
                     "mean_interval_width": round(width, 2), "q_hat": round(float(q_hat), 2)})
        print(f"  CQR nominal={1-alpha:.0%}  empirical={cov:.3f}  mean_width=${width:,.0f}")
    pd.DataFrame(rows).to_csv(C.RESULTS / f"{ds}_cqr_coverage.csv", index=False)
    print(f"\n  -> results/{ds}_shap_stage1/2.csv, {ds}_cqr_coverage.csv, figures/{ds}_shap_stages.png")


if __name__ == "__main__":
    main()
