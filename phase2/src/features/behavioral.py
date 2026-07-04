"""Behavioural features from a normalized observation-window transaction frame.

Dataset-agnostic; relies only on features/schema.py columns. An optional product->category
map enables category diversity (Dunnhumby has COMMODITY_DESC; Online Retail II does not, so
category diversity falls back to product diversity)."""
from __future__ import annotations
import numpy as np
import pandas as pd


def behavioral_features(obs: pd.DataFrame, snapshot, product_category: dict | None = None) -> pd.DataFrame:
    snap = pd.Timestamp(snapshot)
    obs = obs.copy()
    g = obs.groupby("customer_id")

    n_baskets = g["basket_id"].nunique()
    total_rev = g["revenue"].sum()
    first = g["date"].min()
    last = g["date"].max()
    tenure_days = (last - first).dt.days.astype("float")
    active_days = g["date"].apply(lambda s: s.dt.normalize().nunique()).astype("float")

    # basket-level value
    basket_rev = obs.groupby(["customer_id", "basket_id"])["revenue"].sum()
    aov = basket_rev.groupby(level=0).mean()
    basket_size = obs.groupby(["customer_id", "basket_id"])["quantity"].sum().groupby(level=0).mean()

    # inter-order gaps
    def avg_gap(dates):
        d = np.sort(dates.dt.normalize().unique())
        return np.diff(d).astype("timedelta64[D]").astype(float).mean() if len(d) > 1 else np.nan
    avg_days_between = g["date"].apply(avg_gap)

    # diversity
    prod_div = g["product_id"].nunique().astype("float")
    if product_category:
        obs["category"] = obs["product_id"].map(product_category)
        cat_div = obs.groupby("customer_id")["category"].nunique().astype("float")
    else:
        cat_div = prod_div

    feats = pd.DataFrame({
        "n_baskets": n_baskets.astype("float"),
        "tenure_days": tenure_days,
        "active_days": active_days,
        "avg_order_value": aov,
        "avg_basket_size": basket_size,
        "avg_days_between_orders": avg_days_between,
        "product_diversity": prod_div,
        "category_diversity": cat_div,
        "recency_to_snapshot": (snap - last).dt.days.astype("float"),
    })
    # regularity: 1 / (1 + CV of inter-order gaps); high = regular shopper
    def regularity(dates):
        d = np.sort(dates.dt.normalize().unique())
        if len(d) < 3:
            return np.nan
        gaps = np.diff(d).astype("timedelta64[D]").astype(float)
        m = gaps.mean()
        return 1.0 / (1.0 + (gaps.std() / m)) if m > 0 else np.nan
    feats["regularity"] = g["date"].apply(regularity)
    feats["spend_per_active_day"] = total_rev / active_days.replace(0, np.nan)
    return feats
