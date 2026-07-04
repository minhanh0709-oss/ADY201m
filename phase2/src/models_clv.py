"""CLV model suite for Phase 2 cross-context benchmark.

Each model exposes fit_predict(X_tr, y_tr, X_te) -> y_pred (original $ scale).
Tree models (LightGBM/XGBoost) consume NaN natively; linear/RF get median-imputed.
The Hurdle model is the proposed method: P(y>0) x E[y|y>0] with lognormal correction.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import lightgbm as lgb

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False


def _impute(X_tr, X_te):
    imp = SimpleImputer(strategy="median")
    return imp.fit_transform(X_tr), imp.transform(X_te)


# ---------------- baselines ----------------
def m_mean(X_tr, y_tr, X_te):
    return np.full(len(X_te), float(np.mean(y_tr)))


def m_monetary(X_tr, y_tr, X_te, monetary_col="monetary"):
    return np.asarray(X_te[monetary_col].fillna(0).values, dtype=float)


def m_rfm_score(X_tr, y_tr, X_te):
    """Simple standardized R+F+M composite as a ranking baseline."""
    def z(s):
        s = s.fillna(s.median()); sd = s.std()
        return (s - s.mean()) / sd if sd > 0 else s * 0
    score = (-z(X_te.get("recency_days", pd.Series(0, index=X_te.index)))
             + z(X_te.get("frequency", pd.Series(0, index=X_te.index)))
             + z(X_te.get("monetary", pd.Series(0, index=X_te.index))))
    return np.asarray(score.values, dtype=float)


# ---------------- linear / RF ----------------
def m_ridge(X_tr, y_tr, X_te):
    Xtr, Xte = _impute(X_tr, X_te)
    model = Ridge(alpha=1.0)
    model.fit(Xtr, np.log1p(np.clip(y_tr, 0, None)))
    return np.expm1(model.predict(Xte)).clip(min=0)


def m_rf(X_tr, y_tr, X_te):
    Xtr, Xte = _impute(X_tr, X_te)
    model = RandomForestRegressor(n_estimators=300, max_depth=12, n_jobs=-1, random_state=42)
    model.fit(Xtr, np.log1p(np.clip(y_tr, 0, None)))
    return np.expm1(model.predict(Xte)).clip(min=0)


# ---------------- single GBM ----------------
def m_lgbm(X_tr, y_tr, X_te):
    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31,
                              subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    model.fit(X_tr, np.log1p(np.clip(y_tr, 0, None)))
    return np.expm1(model.predict(X_te)).clip(min=0)


def m_xgb(X_tr, y_tr, X_te):
    if not HAS_XGB:
        return m_lgbm(X_tr, y_tr, X_te)
    model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=6,
                             subsample=0.8, colsample_bytree=0.8, random_state=42,
                             tree_method="hist")
    model.fit(X_tr, np.log1p(np.clip(y_tr, 0, None)))
    return np.expm1(model.predict(X_te)).clip(min=0)


# ---------------- Hurdle (proposed) ----------------
def m_hurdle(X_tr, y_tr, X_te):
    y_tr = np.asarray(y_tr, dtype=float)
    pos = y_tr > 0
    # Stage 1: P(y>0)
    clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31,
                             subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    clf.fit(X_tr, pos.astype(int))
    p = clf.predict_proba(X_te)[:, 1]
    # Stage 2: E[log y | y>0]
    reg = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31,
                            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    logy = np.log(y_tr[pos])
    reg.fit(X_tr[pos], logy)
    mu = reg.predict(X_te)
    # lognormal bias correction E[y|y>0] = exp(mu + sigma^2/2)
    resid = logy - reg.predict(X_tr[pos])
    sigma2 = float(np.var(resid))
    cond_mean = np.exp(mu + sigma2 / 2.0)
    return (p * cond_mean).clip(min=0)


MODELS = {
    "Mean": m_mean,
    "Monetary": m_monetary,
    "RFM_score": m_rfm_score,
    "Ridge": m_ridge,
    "RandomForest": m_rf,
    "XGBoost": m_xgb,
    "LightGBM": m_lgbm,
    "Hurdle": m_hurdle,
}
