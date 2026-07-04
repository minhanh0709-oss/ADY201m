"""CLV evaluation metrics for Phase 2. Definitions mirror Phase 1 src/utils_metrics.py
so numbers are directly comparable across phases."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def gini_coefficient(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    arr = np.array([y_true, y_pred]).T
    arr = arr[arr[:, 1].argsort()]
    cum_actual = np.cumsum(arr[:, 0])
    sum_actual = arr[:, 0].sum()
    if sum_actual == 0:
        return 0.0
    lorenz = cum_actual / sum_actual
    return (np.sum(lorenz) - len(arr) / 2) / len(arr)


def normalized_gini(y_true, y_pred):
    a = gini_coefficient(y_true, y_pred)
    p = gini_coefficient(y_true, y_true)
    return 0.0 if p == 0 else a / p


def revenue_capture_at_k(y_true, y_pred, k_pct=0.10):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n_top = max(1, int(len(y_pred) * k_pct))
    top = np.argsort(y_pred)[-n_top:]
    tot = y_true.sum()
    return 0.0 if tot == 0 else 100 * y_true[top].sum() / tot


def lift_at_k(y_true, y_pred, k_pct=0.10):
    return revenue_capture_at_k(y_true, y_pred, k_pct) / (k_pct * 100)


def precision_at_k(y_true, y_pred, k_pct=0.10):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n_top = max(1, int(len(y_pred) * k_pct))
    tp = set(np.argsort(y_pred)[-n_top:])
    ta = set(np.argsort(y_true)[-n_top:])
    return len(tp & ta) / n_top


def spearman(y_true, y_pred):
    if np.std(y_pred) == 0:
        return 0.0
    r, _ = spearmanr(y_true, y_pred)
    return float(r) if np.isfinite(r) else 0.0


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return mae, rmse, r2


def decile_calibration(y_true, y_pred, n=10):
    """Return per-decile (by predicted rank) mean predicted vs mean actual + decile MAPE."""
    df = pd.DataFrame({"pred": np.asarray(y_pred, float), "actual": np.asarray(y_true, float)})
    df["decile"] = pd.qcut(df["pred"].rank(method="first"), n, labels=False)
    g = df.groupby("decile").agg(mean_pred=("pred", "mean"), mean_actual=("actual", "mean"),
                                 sum_actual=("actual", "sum"), count=("actual", "count"))
    mape = float(np.mean(np.abs(g["mean_actual"] - g["mean_pred"]) /
                         np.maximum(g["mean_actual"], 1)))
    return g.reset_index(), mape


def comprehensive_metrics(y_true, y_pred):
    mae, rmse, r2 = regression_metrics(y_true, y_pred)
    _, dmape = decile_calibration(y_true, y_pred)
    return {
        "Norm_Gini": round(normalized_gini(y_true, y_pred), 4),
        "RevCapture_10": round(revenue_capture_at_k(y_true, y_pred, 0.10), 2),
        "RevCapture_20": round(revenue_capture_at_k(y_true, y_pred, 0.20), 2),
        "Lift_10": round(lift_at_k(y_true, y_pred, 0.10), 3),
        "Precision_10": round(precision_at_k(y_true, y_pred, 0.10), 4),
        "Spearman": round(spearman(y_true, y_pred), 4),
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "Decile_MAPE": round(dmape, 4),
    }
