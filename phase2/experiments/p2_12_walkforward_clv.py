"""Phase 2 — REVISION P0.1 — Walk-forward (temporal) CLV evaluation.

Replaces the single random 80/20 split with Phase 1's walk-forward protocol so the Online
Retail II numbers are temporally valid and ALIGNED with Phase 1 (NG approx 0.834 / RC@10 approx 61).

  - online_retail : REUSE Phase 1 windows W1..W5 (data/processed/window_{w}_features.csv),
                    RFM+behavioural features + ActualCLV + IsVIP (read-only). This is exactly the
                    Phase 1 Hurdle feature set, so results must reproduce Phase 1 within noise.
  - dunnhumby     : build week-level windows D1..D4 (obs->pred): 26->8, 39->13, 52->13, 78->13,
                    recomputing RFM+behavioural+sequence features per window (no leakage).

Stratified 80/20 within each window on IsVIP (seed 42). Reports per-window metrics and the
across-window mean +/- std for every model.

Outputs:
  results/online_retail_clv_walkforward.csv
  results/dunnhumby_clv_walkforward.csv
  results/clv_walkforward_summary.csv
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
from features import schema as S  # noqa: E402
from features.rfm import rfm_features  # noqa: E402
from features.behavioral import behavioral_features  # noqa: E402
from features.sequence import trend_features  # noqa: E402

METRICS = ["Norm_Gini", "RevCapture_10", "Precision_10", "Spearman", "MAE"]


def eval_window(X, y, strat, win):
    itr, ite = train_test_split(np.arange(len(X)), test_size=0.2, stratify=strat, random_state=42)
    rows = []
    for name, fn in MODELS.items():
        try:
            yp = fn(X.iloc[itr], y[itr], X.iloc[ite])
            m = M.comprehensive_metrics(y[ite], yp)
            rows.append({"window": win, "model": name, **{k: m[k] for k in METRICS}})
        except Exception as e:
            rows.append({"window": win, "model": name, "Norm_Gini": np.nan, "error": str(e)[:60]})
    return rows


# ---------------- Online Retail II: reuse Phase 1 windows ----------------
def run_or2():
    P1 = C.PROJECT_ROOT / "data" / "processed"
    rows = []
    for w in range(1, 6):
        df = pd.read_csv(P1 / f"window_{w}_features.csv")
        df = df.rename(columns={"Recency": "recency_days", "Frequency": "frequency",
                                "Monetary": "monetary"})
        df["log_monetary"] = np.log1p(df["monetary"].clip(lower=0))
        df["log_frequency"] = np.log1p(df["frequency"])
        feat_cols = [c for c in df.columns if c not in
                     ("CustomerID", "ActualCLV", "IsVIP")]
        X = df[feat_cols]
        y = df["ActualCLV"].values
        rows += eval_window(X, y, df["IsVIP"].values, f"W{w}")
        print(f"  OR-II W{w}: n={len(df)} feats={len(feat_cols)} done")
    return pd.DataFrame(rows)


# ---------------- Dunnhumby: week-level walk-forward ----------------
def run_dunnhumby():
    tx_path = next(C.DUNN_RAW.rglob("transaction_data.csv"))
    raw = pd.read_csv(tx_path)
    raw.columns = [c.upper() for c in raw.columns]
    tx = S.normalize_dunnhumby(raw)
    tx["WEEK_NO"] = pd.to_numeric(raw["WEEK_NO"], errors="coerce").values
    windows = [("D1", 26, 8), ("D2", 39, 13), ("D3", 52, 13), ("D4", 78, 13)]
    rows = []
    for tag, obs_w, pred_w in windows:
        obs = tx[tx["WEEK_NO"] <= obs_w]
        pred = tx[(tx["WEEK_NO"] > obs_w) & (tx["WEEK_NO"] <= obs_w + pred_w)]
        customers = obs["customer_id"].unique()
        snap = obs["date"].max()
        rfm = rfm_features(obs, snap)
        beh = behavioral_features(obs, snap)
        trd = trend_features(obs, snap, n_periods=12, freq="M")
        label = S.actual_clv(pred, customers)
        feats = rfm.join(beh, how="outer").join(trd, how="left").join(label, how="left")
        feats["actual_clv"] = feats["actual_clv"].fillna(0.0)
        feats["is_vip"] = (feats["actual_clv"] >= feats["actual_clv"].quantile(0.90)).astype(int)
        feat_cols = [c for c in feats.columns if c not in ("actual_clv", "is_vip")]
        X = feats[feat_cols].reset_index(drop=True)
        y = feats["actual_clv"].values
        rows += eval_window(X, y, feats["is_vip"].values, tag)
        zr = (y <= 0).mean()
        print(f"  Dunnhumby {tag}: obs<=w{obs_w} pred {pred_w}w  n={len(feats)} zero={zr:.3f} done")
    return pd.DataFrame(rows)


def summarise(df, context):
    g = df.dropna(subset=["Norm_Gini"]).groupby("model")[METRICS]
    s = g.agg(["mean", "std"])
    out = pd.DataFrame({"context": context, "model": s.index})
    for m in METRICS:
        out[f"{m}_mean"] = s[(m, "mean")].values.round(4)
        out[f"{m}_std"] = s[(m, "std")].values.round(4)
    return out.sort_values("Norm_Gini_mean", ascending=False)


def main():
    print("=== walk-forward CLV (P0.1) ===")
    or2 = run_or2()
    or2.to_csv(C.RESULTS / "online_retail_clv_walkforward.csv", index=False)
    dh = run_dunnhumby()
    dh.to_csv(C.RESULTS / "dunnhumby_clv_walkforward.csv", index=False)

    s_or2 = summarise(or2, "Online Retail II")
    s_dh = summarise(dh, "Dunnhumby")
    summary = pd.concat([s_or2, s_dh], ignore_index=True)
    summary.to_csv(C.RESULTS / "clv_walkforward_summary.csv", index=False)

    print("\n-- Online Retail II (walk-forward mean +/- std) --")
    print(s_or2[["model", "Norm_Gini_mean", "Norm_Gini_std", "RevCapture_10_mean",
                 "RevCapture_10_std", "Precision_10_mean"]].to_string(index=False))
    hur = s_or2[s_or2["model"] == "Hurdle"].iloc[0]
    print(f"\n  >>> ALIGNMENT CHECK: OR-II Hurdle NG={hur['Norm_Gini_mean']:.4f}"
          f"+/-{hur['Norm_Gini_std']:.4f}, RC@10={hur['RevCapture_10_mean']:.2f}")
    print(f"      Phase 1 reference: NG=0.834+/-0.057, RC@10=60.96+/-9.26")
    print("\n-- Dunnhumby (walk-forward mean +/- std) --")
    print(s_dh[["model", "Norm_Gini_mean", "Norm_Gini_std", "RevCapture_10_mean",
                "Precision_10_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
