"""Phase 2 — Week 9 — Robustness: bootstrap CIs + paired significance for the key claims.

CLV (per context): bootstrap test set (B reps) for Hurdle NG & RC@10%; paired bootstrap
  differences for (Hurdle ALL − Hurdle RFM)  [does extra structure help?]
  and (Hurdle − Monetary baseline). Reports mean, 95% CI, and bootstrap p (P(diff<=0)).
Uplift (X5): bootstrap Qini per model; paired (S-Learner − ResponseModel).
Policy (X5): bootstrap profit@10/20% for value-adjusted vs uplift-only vs value-only.

Outputs: results/robustness_clv.csv, robustness_uplift.csv, robustness_policy.csv,
         results/robustness_summary.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config as C  # noqa: E402
import utils_metrics as M  # noqa: E402
from models_clv import MODELS  # noqa: E402
from p2_04_clv_benchmark import load, RFM, BEHAVIORAL, SEQUENCE, SEM_SCALAR  # noqa: E402
from sklift.metrics import qini_auc_score  # noqa: E402

B = 1000
RNG = np.random.RandomState(42)


def ci(arr):
    a = np.asarray(arr)
    return round(float(a.mean()), 4), round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)


def boot_idx(n, b):
    return RNG.randint(0, n, size=(b, n))


# ---------------- CLV robustness ----------------
def clv_robustness(ds):
    df, sem_emb = load(ds)
    allf = [c for c in RFM + BEHAVIORAL + SEQUENCE + SEM_SCALAR + sem_emb if c in df.columns]
    y = df["actual_clv"].values
    strat = df["is_vip_top10"].values
    itr, ite = train_test_split(np.arange(len(df)), test_size=0.2, stratify=strat, random_state=42)
    X = df[allf]
    yte = y[ite]
    # predictions (train once each)
    preds = {
        "Hurdle_ALL": MODELS["Hurdle"](X.iloc[itr], y[itr], X.iloc[ite]),
        "Hurdle_RFM": MODELS["Hurdle"](X[RFM].iloc[itr], y[itr], X[RFM].iloc[ite]),
        "Monetary": MODELS["Monetary"](X.iloc[itr], y[itr], X.iloc[ite]),
    }
    idxs = boot_idx(len(ite), B)
    ng = {k: [] for k in preds}; rc = {k: [] for k in preds}
    d_all_rfm, d_hur_mon = [], []
    for bi in idxs:
        yb = yte[bi]
        m = {}
        for k, p in preds.items():
            pb = p[bi]
            m[k] = (M.normalized_gini(yb, pb), M.revenue_capture_at_k(yb, pb, 0.10))
            ng[k].append(m[k][0]); rc[k].append(m[k][1])
        d_all_rfm.append(m["Hurdle_ALL"][0] - m["Hurdle_RFM"][0])
        d_hur_mon.append(m["Hurdle_ALL"][0] - m["Monetary"][0])
    rows = []
    for k in preds:
        ng_m, ng_lo, ng_hi = ci(ng[k]); rc_m, rc_lo, rc_hi = ci(rc[k])
        rows.append({"context": ds, "model": k, "NG_mean": ng_m, "NG_CI": f"[{ng_lo},{ng_hi}]",
                     "RC10_mean": rc_m, "RC10_CI": f"[{rc_lo},{rc_hi}]"})
    p_all_rfm = float(np.mean(np.asarray(d_all_rfm) <= 0))
    p_hur_mon = float(np.mean(np.asarray(d_hur_mon) <= 0))
    rows.append({"context": ds, "model": "DIFF ALL-RFM (NG)", "NG_mean": round(np.mean(d_all_rfm), 4),
                 "NG_CI": f"p(diff<=0)={p_all_rfm:.3f}", "RC10_mean": "", "RC10_CI": ""})
    rows.append({"context": ds, "model": "DIFF Hurdle-Monetary (NG)", "NG_mean": round(np.mean(d_hur_mon), 4),
                 "NG_CI": f"p(diff<=0)={p_hur_mon:.3f}", "RC10_mean": "", "RC10_CI": ""})
    return pd.DataFrame(rows), {"all_rfm_p": p_all_rfm, "hur_mon_p": p_hur_mon}


# ---------------- Uplift + policy robustness ----------------
def uplift_robustness():
    s = pd.read_parquet(C.PROCESSED / "x5_uplift_test_scores.parquet")
    t = s["treatment_flg"].values.astype(int); y = s["target"].values.astype(int)
    v = s["value_proxy"].values.astype(float)
    models = [c[len("uplift_"):] for c in s.columns if c.startswith("uplift_")]
    n = len(s)
    idxs = boot_idx(n, 400)   # fewer reps: Qini is O(n log n)
    qini = {m: [] for m in models}
    d_s_resp = []
    for bi in idxs:
        tb, yb = t[bi], y[bi]
        if tb.sum() == 0 or (1 - tb).sum() == 0:
            continue
        mm = {}
        for m in models:
            ub = s[f"uplift_{m}"].values[bi]
            mm[m] = qini_auc_score(yb, ub, tb)
            qini[m].append(mm[m])
        if "S-Learner" in mm and "ResponseModel" in mm:
            d_s_resp.append(mm["S-Learner"] - mm["ResponseModel"])
    urows = []
    for m in models:
        mn, lo, hi = ci(qini[m])
        urows.append({"model": m, "Qini_mean": mn, "Qini_CI": f"[{lo},{hi}]"})
    p_s_resp = float(np.mean(np.asarray(d_s_resp) <= 0))
    urows.append({"model": "DIFF S-Learner - Response", "Qini_mean": round(np.mean(d_s_resp), 4),
                  "Qini_CI": f"p(diff<=0)={p_s_resp:.3f}"})
    udf = pd.DataFrame(urows)

    # policy profit bootstrap
    from p2_09_policy_comparison import incremental_at_k
    mt = pd.read_csv(C.RESULTS / "x5_uplift_models.csv")
    best = mt.sort_values("Qini_AUC", ascending=False).iloc[0]["model"]
    up = s[f"uplift_{best}"].values
    pol = {"Value-only": v, "Uplift-only": up, "Value-adjusted": up * v}
    cost = 100.0
    prows = []
    for name, score in pol.items():
        for k in (0.10, 0.20):
            profits = []
            for bi in idxs:
                ic, ir, nt = incremental_at_k(score[bi], t[bi], y[bi], v[bi], k)
                profits.append(ir - cost * nt)
            mn, lo, hi = ci(profits)
            prows.append({"policy": name, "K": f"{int(k*100)}%", "profit_mean": round(mn, 0),
                          "profit_CI": f"[{lo:,.0f},{hi:,.0f}]"})
    pdf = pd.DataFrame(prows)
    return udf, pdf, {"s_resp_p": p_s_resp, "best": best}


def main():
    print("=== robustness (bootstrap CI) ===")
    clv_all = []
    sig = {}
    for ds in ["online_retail", "dunnhumby"]:
        r, s = clv_robustness(ds)
        clv_all.append(r); sig[ds] = s
        print(f"\n-- CLV {ds} --"); print(r.to_string(index=False))
    clv = pd.concat(clv_all, ignore_index=True)
    clv.to_csv(C.RESULTS / "robustness_clv.csv", index=False)

    udf, pdf, usig = uplift_robustness()
    udf.to_csv(C.RESULTS / "robustness_uplift.csv", index=False)
    pdf.to_csv(C.RESULTS / "robustness_policy.csv", index=False)
    print("\n-- Uplift Qini CIs --"); print(udf.to_string(index=False))
    print("\n-- Policy profit CIs --"); print(pdf.to_string(index=False))

    summary = pd.DataFrame([
        {"claim": "Hurdle>Monetary (e-com)", "evidence": f"p(diff<=0)={sig['online_retail']['hur_mon_p']:.3f}"},
        {"claim": "extra features>RFM (e-com)", "evidence": f"p(diff<=0)={sig['online_retail']['all_rfm_p']:.3f}"},
        {"claim": "extra features>RFM (grocery)", "evidence": f"p(diff<=0)={sig['dunnhumby']['all_rfm_p']:.3f}"},
        {"claim": "Hurdle>Monetary (grocery)", "evidence": f"p(diff<=0)={sig['dunnhumby']['hur_mon_p']:.3f}"},
        {"claim": "S-Learner>ResponseModel (uplift Qini)", "evidence": f"p(diff<=0)={usig['s_resp_p']:.3f}"},
    ])
    summary.to_csv(C.RESULTS / "robustness_summary.csv", index=False)
    print("\n-- significance summary --"); print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
