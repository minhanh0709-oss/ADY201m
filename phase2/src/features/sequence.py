"""Temporal sequence features from a normalized observation-window transaction frame.

Two outputs:
  - sequence_matrices(): per-customer monthly spend / order-count / active-flag matrices
    (for DL models later), aligned to a fixed number of trailing periods.
  - trend_features(): compact tabular recent-trend features for tree/linear models
    (last-period spend, recent vs older ratio, OLS slope over monthly spend).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _monthly_panel(obs: pd.DataFrame, snapshot, n_periods: int = 12, freq: str = "M"):
    """Return (customers, periods, spend[c,p], orders[c,p]) trailing `n_periods` buckets
    ending at snapshot."""
    snap = pd.Timestamp(snapshot)
    o = obs[obs["date"] <= snap].copy()
    # period index counting backwards from snapshot month
    o["period"] = o["date"].dt.to_period(freq)
    last_period = snap.to_period(freq)
    periods = pd.period_range(end=last_period, periods=n_periods, freq=freq)
    pidx = {p: i for i, p in enumerate(periods)}
    o = o[o["period"].isin(pidx)]
    customers = np.sort(obs["customer_id"].unique())
    cidx = {c: i for i, c in enumerate(customers)}

    spend = np.zeros((len(customers), n_periods), dtype="float32")
    orders = np.zeros((len(customers), n_periods), dtype="float32")
    grp_rev = o.groupby(["customer_id", "period"])["revenue"].sum()
    grp_ord = o.groupby(["customer_id", "period"])["basket_id"].nunique()
    for (c, p), v in grp_rev.items():
        spend[cidx[c], pidx[p]] = v
    for (c, p), v in grp_ord.items():
        orders[cidx[c], pidx[p]] = v
    return customers, periods, spend, orders


def sequence_matrices(obs, snapshot, n_periods: int = 12, freq: str = "M"):
    customers, periods, spend, orders = _monthly_panel(obs, snapshot, n_periods, freq)
    active = (orders > 0).astype("float32")
    return {"customers": customers, "periods": [str(p) for p in periods],
            "spend": spend, "orders": orders, "active": active}


def trend_features(obs, snapshot, n_periods: int = 12, freq: str = "M") -> pd.DataFrame:
    customers, periods, spend, orders = _monthly_panel(obs, snapshot, n_periods, freq)
    x = np.arange(n_periods, dtype="float32")
    x_c = x - x.mean()
    denom = (x_c ** 2).sum()

    def slope(row):
        return float((x_c * (row - row.mean())).sum() / denom) if denom > 0 else 0.0

    half = n_periods // 2
    recent = spend[:, half:].sum(axis=1)
    older = spend[:, :half].sum(axis=1)
    feats = pd.DataFrame({
        "seq_spend_last_period": spend[:, -1],
        "seq_spend_recent_half": recent,
        "seq_recent_vs_older_ratio": (recent + 1.0) / (older + 1.0),
        "seq_active_periods": (orders > 0).sum(axis=1).astype("float32"),
        "seq_spend_slope": np.apply_along_axis(slope, 1, spend),
        "seq_orders_slope": np.apply_along_axis(slope, 1, orders),
    }, index=customers)
    feats.index.name = "customer_id"
    return feats
