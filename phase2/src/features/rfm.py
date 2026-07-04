"""RFM features from a normalized observation-window transaction frame.

Operates on the schema from features/schema.py. Snapshot-relative recency, so it is
leakage-safe by construction (only uses tx with date <= snapshot)."""
from __future__ import annotations
import numpy as np
import pandas as pd


def rfm_features(obs: pd.DataFrame, snapshot) -> pd.DataFrame:
    """Return per-customer RFM frame indexed by customer_id.

    Recency  = days from last purchase to snapshot (smaller = more recent)
    Frequency = number of distinct baskets
    Monetary = total revenue in observation window
    """
    snap = pd.Timestamp(snapshot)
    g = obs.groupby("customer_id")
    last = g["date"].max()
    rfm = pd.DataFrame({
        "recency_days": (snap - last).dt.days.astype("float"),
        "frequency": g["basket_id"].nunique().astype("float"),
        "monetary": g["revenue"].sum().astype("float"),
    })
    # robust log transforms (monetary is heavy-tailed / can be slightly negative on returns)
    rfm["log_monetary"] = np.log1p(rfm["monetary"].clip(lower=0))
    rfm["log_frequency"] = np.log1p(rfm["frequency"])
    return rfm
