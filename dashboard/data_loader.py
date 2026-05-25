"""Cached data loaders for the CLV dashboard.

All CSV / NPY artefacts are loaded ONCE at import time; subsequent
dashboard callbacks read from in-memory DataFrames.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import pandas as pd
import numpy as np

ROOT       = Path(__file__).resolve().parents[1]
SQL_OUTPUT = ROOT / "notebooks" / "outputs"
RESULTS    = ROOT / "results"
DASH_DATA  = Path(__file__).parent / "data"


# ---------- helpers ----------
def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file missing: {path}\n"
            "Run notebooks/02_sql_analysis.ipynb and dashboard/_make_predictions.py first."
        )
    return pd.read_csv(path)


# ---------- SQL outputs (from notebooks/02_sql_analysis.ipynb) ----------
@lru_cache(maxsize=1)
def rfm() -> pd.DataFrame:
    return _read(SQL_OUTPUT / "q1_rfm.csv")


@lru_cache(maxsize=1)
def cohort() -> pd.DataFrame:
    df = _read(SQL_OUTPUT / "q2_cohort.csv")
    df["cohort_month"] = pd.to_datetime(df["cohort_month"])
    df["obs_month"]    = pd.to_datetime(df["obs_month"])
    return df


@lru_cache(maxsize=1)
def top_vip() -> pd.DataFrame:
    return _read(SQL_OUTPUT / "q3_top_vip.csv")


@lru_cache(maxsize=1)
def country() -> pd.DataFrame:
    return _read(SQL_OUTPUT / "q4_country.csv")


@lru_cache(maxsize=1)
def churn() -> pd.DataFrame:
    return _read(SQL_OUTPUT / "q5_churn.csv")


@lru_cache(maxsize=1)
def mac() -> pd.DataFrame:
    df = _read(SQL_OUTPUT / "q6_mac.csv")
    df["year_month"] = pd.to_datetime(df["year_month"])
    return df


@lru_cache(maxsize=1)
def rfm_segments() -> pd.DataFrame:
    """Per-customer RFM quintile scores and named segment (from Q7 SQL)."""
    return _read(SQL_OUTPUT / "q7_rfm_segments.csv")


# ---------- model outputs (results/) ----------
@lru_cache(maxsize=1)
def predictions() -> pd.DataFrame:
    """Per-customer Hurdle + CQR predictions on Window 3 test fold."""
    return _read(DASH_DATA / "customer_predictions_w3.csv")


@lru_cache(maxsize=1)
def revenue_curve() -> pd.DataFrame:
    return _read(RESULTS / "revenue_capture_curve.csv")


@lru_cache(maxsize=1)
def profit_sim() -> pd.DataFrame:
    return _read(RESULTS / "profit_simulation.csv")


@lru_cache(maxsize=1)
def decile() -> pd.DataFrame:
    return _read(RESULTS / "decile_analysis.csv")


@lru_cache(maxsize=1)
def sem_cluster_stats() -> pd.DataFrame:
    return _read(RESULTS / "semantic_cluster_stats.csv")


@lru_cache(maxsize=1)
def sem_cluster_products() -> pd.DataFrame:
    return _read(RESULTS / "semantic_cluster_products.csv")


# ---------- KPI helpers ----------
def kpi_summary() -> dict:
    rfm_df = rfm()
    mac_df = mac()
    return {
        "total_revenue":  float(rfm_df["Monetary"].sum()),
        "total_customers": int(len(rfm_df)),
        "total_invoices":  int(mac_df["n_invoices"].sum()),
        "avg_order_value": float(rfm_df["AvgOrderValue"].mean()),
        "retention_90d":   float(100 * (rfm_df["Recency"] <= 90).mean()),
        "month_min":  mac_df["year_month"].min(),
        "month_max":  mac_df["year_month"].max(),
    }


# ---------- RFM 5x5 matrix (for page 2) ----------
def rfm_matrix() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pre-aggregated 5x5 RFM grid (Recency quintile x Frequency quintile)."""
    df = rfm_segments().copy()
    df["R_q"] = df["R_score"].astype(int)
    df["F_q"] = df["F_score"].astype(int)
    df["M_q"] = df["M_score"].astype(int)
    grid = (df.groupby(["R_q", "F_q"])
              .agg(n=("CustomerID", "count"),
                   mean_M=("Monetary", "mean"))
              .reset_index())
    return grid, df
