"""Phase 2 — Week 6-7 — X5 uplift models (RQ3).

Trains uplift learners on X5 (treatment_flg/target) with pre-communication features and
compares ranking by Qini AUC / AUUC / uplift@K:
  - Random            (baseline)
  - ResponseModel     (P(target), ignores treatment -> ranks responders, not incremental)
  - S-Learner         (sklift SoloModel)
  - T-Learner         (sklift TwoModels, vanilla)
  - ClassTransform    (sklift ClassTransformation)
  - X-Learner         (manual meta-learner)

Outputs: results/x5_uplift_models.csv, figures/x5_qini_curves.png, and the test-set
uplift scores for the best models (-> used by the policy comparison, Week 8).

Run:  python phase2/experiments/p2_08_x5_uplift_models.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklift.models import SoloModel, TwoModels, ClassTransformation
from sklift.metrics import qini_auc_score, uplift_auc_score, uplift_at_k, qini_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config as C  # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FEATS = ["n_transactions", "monetary", "avg_basket_value", "total_quantity", "n_product_lines",
         "recency_days", "tenure_days", "points_received", "points_spent",
         "avg_days_between_txn", "log_monetary", "log_frequency",
         "age", "gender_code", "card_tenure_days", "has_redeemed", "redeem_latency_days"]


def base():
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                              subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)


def x_learner(Xtr, ytr, ttr, Xte):
    t = ttr.values if hasattr(ttr, "values") else ttr
    m0 = base().fit(Xtr[t == 0], ytr[t == 0])
    m1 = base().fit(Xtr[t == 1], ytr[t == 1])
    d1 = ytr[t == 1] - m0.predict_proba(Xtr[t == 1])[:, 1]
    d0 = m1.predict_proba(Xtr[t == 0])[:, 1] - ytr[t == 0]
    reg = lambda: lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                    random_state=42, verbose=-1)
    tau1 = reg().fit(Xtr[t == 1], d1)
    tau0 = reg().fit(Xtr[t == 0], d0)
    return 0.5 * (tau0.predict(Xte) + tau1.predict(Xte))   # propensity ~0.5 (balanced RCT)


def main():
    df = pd.read_parquet(C.PROCESSED / "x5_uplift_features.parquet")
    X = df[FEATS].astype("float32")
    y = df["target"].astype(int).values
    t = df["treatment_flg"].astype(int).values
    strat = t * 2 + y
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.3,
                                      stratify=strat, random_state=42)
    Xtr, Xte = X.iloc[idx_tr], X.iloc[idx_te]
    ytr, yte = y[idx_tr], y[idx_te]
    ttr, tte = t[idx_tr], t[idx_te]
    print(f"=== X5 uplift models ===  n={len(df)}  train={len(idx_tr)} test={len(idx_te)}")
    print(f"  test treat_rate={tte.mean():.3f}  test target_rate={yte.mean():.3f}")

    uplifts = {}
    rng = np.random.RandomState(42)
    uplifts["Random"] = rng.rand(len(idx_te))
    # Response model: P(target) ignoring treatment
    rm = base().fit(Xtr, ytr)
    uplifts["ResponseModel"] = rm.predict_proba(Xte)[:, 1]
    # S-Learner
    uplifts["S-Learner"] = SoloModel(base()).fit(Xtr, ytr, ttr).predict(Xte)
    # T-Learner
    uplifts["T-Learner"] = TwoModels(base(), base(), method="vanilla").fit(Xtr, ytr, ttr).predict(Xte)
    # Class Transformation
    uplifts["ClassTransform"] = ClassTransformation(base()).fit(Xtr, ytr, ttr).predict(Xte)
    # X-Learner
    uplifts["X-Learner"] = x_learner(Xtr, ytr, ttr, Xte)

    rows = []
    for name, up in uplifts.items():
        rows.append({
            "model": name,
            "Qini_AUC": round(qini_auc_score(yte, up, tte), 4),
            "AUUC": round(uplift_auc_score(yte, up, tte), 4),
            "uplift@10": round(uplift_at_k(yte, up, tte, strategy="overall", k=0.1), 4),
            "uplift@20": round(uplift_at_k(yte, up, tte, strategy="overall", k=0.2), 4),
            "uplift@30": round(uplift_at_k(yte, up, tte, strategy="overall", k=0.3), 4),
        })
    res = pd.DataFrame(rows).sort_values("Qini_AUC", ascending=False)
    res.to_csv(C.RESULTS / "x5_uplift_models.csv", index=False)
    print("\n" + res.to_string(index=False))

    # save test-set uplift scores + value proxy for the policy comparison (Week 8)
    scores = pd.DataFrame({"client_id": df.iloc[idx_te]["client_id"].values,
                           "treatment_flg": tte, "target": yte,
                           "value_proxy": df.iloc[idx_te]["monetary"].values})
    for name, up in uplifts.items():
        scores[f"uplift_{name}"] = up
    scores.to_parquet(C.assert_phase2_path(C.PROCESSED / "x5_uplift_test_scores.parquet"), index=False)

    # Qini curves
    fig, ax = plt.subplots(figsize=(6.8, 5))
    for name in ["X-Learner", "T-Learner", "S-Learner", "ClassTransform", "ResponseModel", "Random"]:
        x_q, y_q = qini_curve(yte, uplifts[name], tte)
        ax.plot(x_q, y_q, label=f"{name} (Qini={res.set_index('model').loc[name,'Qini_AUC']})")
    ax.set_xlabel("n targeted"); ax.set_ylabel("cumulative incremental conversions")
    ax.set_title("X5 Qini curves"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(C.FIGURES / "x5_qini_curves.png", dpi=140)
    print(f"\n  best (Qini): {res.iloc[0]['model']}  -> results/x5_uplift_models.csv")


if __name__ == "__main__":
    main()
