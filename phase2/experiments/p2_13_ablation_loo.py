"""Phase 2 — REVISION P1.1 — Leave-one-group-out feature ablation (walk-forward), RQ2.

Reviewers asked whether semantic features add value *given* sequence/behavioural features. We
therefore drop each group from the full set and measure the change, under the walk-forward protocol.

  variants: ALL, ALL-semantic, ALL-sequence (Dunnhumby), ALL-behavioural, RFM-only
  model:    Hurdle (proposed)
  report:   per-window NG, mean +/- std, and paired t-test of ALL vs each variant across windows.

  online_retail : Phase 1 windows W1..W3 + precomputed semantic_v2 npz (RFM/behavioural/semantic).
  dunnhumby     : windows D1..D4, features (RFM/behavioural/sequence/semantic) rebuilt per window.

Output: results/{ds}_ablation_loo.csv, results/ablation_loo_summary.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config as C  # noqa: E402
import utils_metrics as M  # noqa: E402
from models_clv import MODELS  # noqa: E402
from features import schema as S  # noqa: E402
from features.rfm import rfm_features  # noqa: E402
from features.behavioral import behavioral_features  # noqa: E402
from features.sequence import trend_features  # noqa: E402
from features.semantic_graph import build_product_embeddings, customer_semantic  # noqa: E402


def hurdle_ng(X, y, strat, cols):
    itr, ite = train_test_split(np.arange(len(X)), test_size=0.2, stratify=strat, random_state=42)
    yp = MODELS["Hurdle"](X[cols].iloc[itr], y[itr], X[cols].iloc[ite])
    return M.normalized_gini(y[ite], yp)


def or2_window(w):
    P1 = C.PROJECT_ROOT / "data" / "processed"
    df = pd.read_csv(P1 / f"window_{w}_features.csv").rename(
        columns={"Recency": "recency_days", "Frequency": "frequency", "Monetary": "monetary"})
    df["log_monetary"] = np.log1p(df["monetary"].clip(lower=0))
    df["log_frequency"] = np.log1p(df["frequency"])
    z = np.load(P1 / f"semantic_v2_window_{w}.npz", allow_pickle=True)
    sem = pd.DataFrame(z["sem_full"], columns=[f"sem_{i}" for i in range(z["sem_full"].shape[1])])
    sem["sem_drift"] = z["sem_drift"]
    sem["CustomerID"] = z["customer_ids"]
    df = df.merge(sem, on="CustomerID", how="left").fillna(0.0)
    rfm = ["recency_days", "frequency", "monetary", "log_monetary", "log_frequency"]
    beh = ["Tenure", "ActiveMonths", "ProductDiversity", "AvgOrderValue",
           "AvgDaysBetweenOrders", "Regularity", "IsUK", "T_BGNBD", "Recency_BGNBD"]
    semc = [c for c in df.columns if c.startswith("sem_")]
    groups = {"RFM": rfm, "behavioural": beh, "semantic": semc}
    return df, df["ActualCLV"].values, df["IsVIP"].values, groups


def dh_window(tx, obs_w, pred_w):
    obs = tx[tx["WEEK_NO"] <= obs_w]
    pred = tx[(tx["WEEK_NO"] > obs_w) & (tx["WEEK_NO"] <= obs_w + pred_w)]
    customers = np.sort(obs["customer_id"].unique())
    snap = obs["date"].max()
    rfm = rfm_features(obs, snap)
    beh = behavioral_features(obs, snap)
    trd = trend_features(obs, snap, n_periods=12, freq="M")
    p2i, emb = build_product_embeddings(obs, n_components=32, min_occur=10)
    sc, smat, stab = customer_semantic(obs, snap, p2i, emb)
    sem = pd.DataFrame(smat, columns=[f"sem_{i}" for i in range(smat.shape[1])])
    sem["customer_id"] = sc
    feats = (rfm.join(beh, how="outer").join(trd, how="left").join(stab, how="left")
             .reset_index().merge(sem, on="customer_id", how="left"))
    label = S.actual_clv(pred, customers).reset_index()
    feats = feats.merge(label, on="customer_id", how="left")
    feats = feats.fillna(0.0)
    y = feats["actual_clv"].values
    vip = (y >= np.quantile(y, 0.90)).astype(int)
    rfmc = ["recency_days", "frequency", "monetary", "log_monetary", "log_frequency"]
    behc = ["n_baskets", "tenure_days", "active_days", "avg_order_value", "avg_basket_size",
            "avg_days_between_orders", "product_diversity", "category_diversity",
            "recency_to_snapshot", "regularity", "spend_per_active_day"]
    seqc = [c for c in feats.columns if c.startswith("seq_")]
    semc = [c for c in feats.columns if c.startswith("sem_")]
    groups = {"RFM": rfmc, "behavioural": behc, "sequence": seqc, "semantic": semc}
    groups = {k: [c for c in v if c in feats.columns] for k, v in groups.items()}
    return feats, y, vip, groups


def variants_for(groups):
    allg = sum(groups.values(), [])
    out = {"ALL": allg, "RFM-only": groups["RFM"]}
    for g in groups:
        if g != "RFM":
            out[f"ALL-{g}"] = [c for c in allg if c not in groups[g]]
    return out


def run_context(name, windows_iter):
    per = []   # rows: window, variant, NG
    for win, (df, y, strat, groups) in windows_iter:
        V = variants_for(groups)
        for vn, cols in V.items():
            ng = hurdle_ng(df, y, strat, cols)
            per.append({"context": name, "window": win, "variant": vn, "NG": round(ng, 4)})
        print(f"  {name} {win}: variants done")
    return pd.DataFrame(per)


def summarise(per):
    rows = []
    for ctx in per["context"].unique():
        sub = per[per.context == ctx]
        piv = sub.pivot(index="window", columns="variant", values="NG")
        for v in piv.columns:
            d = piv["ALL"] - piv[v]
            mean_ng, std_ng = piv[v].mean(), piv[v].std()
            if v == "ALL":
                p = np.nan
            else:
                p = ttest_rel(piv["ALL"], piv[v]).pvalue if len(piv) > 1 else np.nan
            rows.append({"context": ctx, "variant": v, "NG_mean": round(mean_ng, 4),
                         "NG_std": round(std_ng, 4), "dNG_vs_ALL": round(d.mean(), 4),
                         "paired_p": round(float(p), 4) if p == p else ""})
    return pd.DataFrame(rows)


def main():
    print("=== LOO ablation (walk-forward) ===")
    or2 = run_context("Online Retail II",
                      ((f"W{w}", or2_window(w)) for w in (1, 2, 3)))
    # Dunnhumby
    tx_path = next(C.DUNN_RAW.rglob("transaction_data.csv"))
    raw = pd.read_csv(tx_path); raw.columns = [c.upper() for c in raw.columns]
    tx = S.normalize_dunnhumby(raw)
    tx["WEEK_NO"] = pd.to_numeric(raw["WEEK_NO"], errors="coerce").values
    dh = run_context("Dunnhumby",
                     ((tag, dh_window(tx, ow, pw)) for tag, ow, pw in
                      [("D1", 26, 8), ("D2", 39, 13), ("D3", 52, 13), ("D4", 78, 13)]))

    per = pd.concat([or2, dh], ignore_index=True)
    per.to_csv(C.RESULTS / "ablation_loo_per_window.csv", index=False)
    summ = summarise(per)
    summ.to_csv(C.RESULTS / "ablation_loo_summary.csv", index=False)
    print("\n-- LOO ablation summary (Hurdle NG; dNG_vs_ALL = ALL minus variant) --")
    print(summ.to_string(index=False))
    print("\n  Interpretation: dNG_vs_ALL>0 means dropping that group HURTS (group adds value).")


if __name__ == "__main__":
    main()
