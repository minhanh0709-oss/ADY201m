"""Normalize heterogeneous retail transaction tables to one schema, so the same
feature pipeline serves Online Retail II (Phase 1) and Dunnhumby.

Normalized transaction-line frame columns:
    customer_id : object/int   customer or household id
    basket_id   : object       order / invoice / basket id
    date        : datetime64    transaction timestamp (day resolution is fine)
    product_id  : object        product / stockcode id
    quantity    : float         units
    revenue     : float         line revenue (price*qty / sales_value)

Each loader returns this frame; downstream feature code is dataset-agnostic.
"""
from __future__ import annotations
import pandas as pd

NORM_COLS = ["customer_id", "basket_id", "date", "product_id", "quantity", "revenue"]


def normalize_online_retail(df: pd.DataFrame) -> pd.DataFrame:
    """Online Retail II cleaned schema -> normalized frame."""
    out = pd.DataFrame({
        "customer_id": df["CustomerID"],
        "basket_id": df["Invoice"].astype(str),
        "date": pd.to_datetime(df["InvoiceDate"]),
        "product_id": df["StockCode"].astype(str),
        "quantity": pd.to_numeric(df["Quantity"], errors="coerce"),
        "revenue": pd.to_numeric(
            df["TotalPrice"] if "TotalPrice" in df.columns else df["Quantity"] * df["Price"],
            errors="coerce"),
    })
    return out.dropna(subset=["customer_id", "date"])


def normalize_dunnhumby(df: pd.DataFrame, day_origin: str = "2000-01-01") -> pd.DataFrame:
    """Dunnhumby transaction_data schema -> normalized frame.

    DAY is an integer 1..711 with no calendar anchor; we map it to a synthetic date
    (origin + DAY days). Absolute dates are irrelevant — only intervals/recency matter.
    """
    df = df.rename(columns={c: c.upper() for c in df.columns})
    origin = pd.Timestamp(day_origin)
    out = pd.DataFrame({
        "customer_id": df["HOUSEHOLD_KEY"],
        "basket_id": df["BASKET_ID"].astype(str),
        "date": origin + pd.to_timedelta(pd.to_numeric(df["DAY"], errors="coerce"), unit="D"),
        "product_id": df["PRODUCT_ID"].astype(str),
        "quantity": pd.to_numeric(df["QUANTITY"], errors="coerce"),
        "revenue": pd.to_numeric(df["SALES_VALUE"], errors="coerce"),
    })
    return out.dropna(subset=["customer_id", "date"])


def split_observation_prediction(tx: pd.DataFrame, snapshot, horizon_days: int):
    """Split normalized tx into observation (<= snapshot) and prediction window
    (snapshot, snapshot+horizon]. Returns (obs, pred, snapshot_ts)."""
    snap = pd.Timestamp(snapshot)
    end = snap + pd.Timedelta(days=horizon_days)
    obs = tx[tx["date"] <= snap]
    pred = tx[(tx["date"] > snap) & (tx["date"] <= end)]
    return obs, pred, snap


def actual_clv(pred: pd.DataFrame, customers) -> pd.Series:
    """Future revenue per customer over the prediction window (0 if no purchase)."""
    s = pred.groupby("customer_id")["revenue"].sum()
    return s.reindex(customers).fillna(0.0).rename("actual_clv")
