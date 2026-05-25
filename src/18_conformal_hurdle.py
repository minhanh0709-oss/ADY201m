"""
18_conformal_hurdle.py
Conformalized Quantile Regression (CQR) for Hurdle Model

Provides distribution-free prediction intervals with guaranteed
marginal coverage, directly addressing the paper's identified limitation:
  "MCD-ZILN 95% intervals cover only 32% of actual values"

Algorithm (Romano et al., NeurIPS 2019):
  1. Split train set: 70% proper-train, 15% calibration, 15% val (for early stopping)
  2. Fit two quantile LightGBM regressors on proper-train:
       q_lo: 5th-percentile quantile regressor
       q_hi: 95th-percentile quantile regressor
  3. Compute non-conformity scores on calibration set:
       s_i = max(q_lo_i - y_i,  y_i - q_hi_i)
  4. Set q_hat = (1 - alpha)-quantile of {s_i}  with alpha = 0.10
  5. Adjusted interval: [q_lo - q_hat,  q_hi + q_hat]
  6. Empirical coverage = fraction of test y_i in adjusted interval (should >= 90%)

Also computes:
  - Mean prediction interval width
  - Coverage for MCD-ZILN (from existing results, for comparison)
  - Risk-aware VIP ranking: rank by lower bound instead of expected value
"""

import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
exec(open(Path(__file__).parent / '21_utils_cv_metrics.py').read())

import lightgbm as lgb
from sklearn.model_selection import train_test_split

DATA_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR        = Path(__file__).parent.parent / "results"

ALPHA      = 0.10   # target miscoverage = 10% → target coverage = 90%
N_TRIALS   = 15     # Optuna trials per quantile regressor


def load_windows():
    p5 = DATA_PROCESSED_DIR / 'walk_forward_windows_5fold.pkl'
    p3 = DATA_PROCESSED_DIR / 'walk_forward_windows.pkl'
    pkl = p5 if p5.exists() else p3
    with open(pkl, 'rb') as f:
        return pickle.load(f)


def rfm_seq_features(feats_df, win):
    """Same feature set as Hurdle-Seq (existing best model)."""
    cols = ['Recency', 'Frequency', 'Monetary', 'Tenure',
            'ActiveMonths', 'ProductDiversity', 'AvgOrderValue',
            'AvgDaysBetweenOrders', 'Regularity', 'IsUK']
    X = feats_df[cols].fillna(0).copy()
    X['log_Monetary']      = np.log1p(X['Monetary'])
    X['log_Frequency']     = np.log1p(X['Frequency'])
    X['log_AvgOrderValue'] = np.log1p(X['AvgOrderValue'])
    X['M_per_F']           = X['Monetary'] / np.maximum(X['Frequency'], 1)
    X['M_per_T']           = X['Monetary'] / np.maximum(X['Tenure'],    1)
    X['Active_ratio']      = X['ActiveMonths'] / np.maximum(X['Tenure'] / 30, 1)

    seq = win['revenue_seq']
    X['seq_mean']          = seq.mean(axis=1)
    X['seq_max']           = seq.max(axis=1)
    X['seq_recent3_mean']  = seq[:, -3:].mean(axis=1)
    X['seq_recent3_max']   = seq[:, -3:].max(axis=1)
    X['seq_active_months'] = (seq > 0).sum(axis=1).astype(float)
    X['seq_std']           = seq.std(axis=1)
    X['seq_trend']         = (seq[:, seq.shape[1]//2:].mean(axis=1)
                               - seq[:, :seq.shape[1]//2].mean(axis=1))
    return X.values


def fit_quantile_lgbm(X_tr, y_tr, X_val, y_val, alpha_q):
    """Fit LightGBM quantile regressor at quantile alpha_q."""
    params = {
        'objective': 'quantile',
        'alpha':     alpha_q,
        'metric':    'quantile',
        'verbosity': -1,
        'num_leaves': 40,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'min_data_in_leaf': 10,
        'random_state': 42,
    }
    model = lgb.LGBMRegressor(**params, n_estimators=500)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(30, verbose=False)])
    return model


def cqr_on_window(win):
    """Run CQR for one walk-forward window.  Returns metrics dict."""
    feats = win['features']
    y     = feats['ActualCLV'].values
    X     = rfm_seq_features(feats, win)

    # 3-way split: train | calibration | test
    idx          = np.arange(len(feats))
    tr_cal, te   = train_test_split(idx, test_size=0.20, random_state=42,
                                     stratify=feats['IsVIP'])
    tr, cal      = train_test_split(tr_cal, test_size=0.1875, random_state=42)
    # 0.1875 of 80% ≈ 15% overall → ~65% train / 15% cal / 20% test

    X_tr,  y_tr  = X[tr],  y[tr]
    X_cal, y_cal = X[cal], y[cal]
    X_te,  y_te  = X[te],  y[te]

    # Further split train into proper-train / val for early stopping
    tr2, val = train_test_split(np.arange(len(tr)), test_size=0.15, random_state=42)
    X_tr2,  y_tr2  = X_tr[tr2],  y_tr[tr2]
    X_val_s, y_val_s = X_tr[val], y_tr[val]

    # Fit on LOG-scale positives for better numerical behaviour
    # (both quantile models predict log(1+y), then expm1)
    y_tr2_log  = np.log1p(y_tr2)
    y_val_log  = np.log1p(y_val_s)

    print(f"    Train={len(tr2):,}  Cal={len(cal):,}  Test={len(te):,}")

    q_lo_model = fit_quantile_lgbm(X_tr2, y_tr2_log, X_val_s, y_val_log, ALPHA / 2)
    q_hi_model = fit_quantile_lgbm(X_tr2, y_tr2_log, X_val_s, y_val_log, 1 - ALPHA / 2)

    # Calibration non-conformity scores
    q_lo_cal = np.expm1(q_lo_model.predict(X_cal))
    q_hi_cal = np.expm1(q_hi_model.predict(X_cal))
    scores   = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)

    # q_hat: (ceil((n+1)*(1-alpha))/n)-th quantile of scores
    n        = len(scores)
    level    = np.ceil((n + 1) * (1 - ALPHA)) / n
    level    = min(level, 1.0)
    q_hat    = np.quantile(scores, level)

    # Test prediction intervals
    q_lo_te  = np.expm1(q_lo_model.predict(X_te)) - q_hat
    q_hi_te  = np.expm1(q_hi_model.predict(X_te)) + q_hat
    q_lo_te  = np.maximum(q_lo_te, 0)

    # Coverage and width
    covered   = ((y_te >= q_lo_te) & (y_te <= q_hi_te))
    coverage  = covered.mean() * 100
    mean_width = (q_hi_te - q_lo_te).mean()
    med_width  = np.median(q_hi_te - q_lo_te)

    # Risk-aware VIP ranking: sort by lower bound
    ranking_lb  = comprehensive_metrics(y_te, q_lo_te)
    ranking_mid = comprehensive_metrics(y_te, (q_lo_te + q_hi_te) / 2)

    print(f"    CQR Coverage (target 90%): {coverage:.1f}%")
    print(f"    Mean interval width: ${mean_width:,.0f}")
    print(f"    NG (lower-bound ranking): {ranking_lb['Norm_Gini']:.4f}")
    print(f"    NG (midpoint ranking):    {ranking_mid['Norm_Gini']:.4f}")

    return {
        'Window':          win['window_id'],
        'CQR_Coverage':    coverage,
        'Mean_Width':      mean_width,
        'Median_Width':    med_width,
        'q_hat':           q_hat,
        'NG_LowerBound':   ranking_lb['Norm_Gini'],
        'RC10_LowerBound': ranking_lb['Revenue_Capture_10'],
        'NG_Midpoint':     ranking_mid['Norm_Gini'],
        'RC10_Midpoint':   ranking_mid['Revenue_Capture_10'],
    }


def main():
    print("\n" + "="*70)
    print("[Phase D] Conformalized Quantile Regression (CQR) for Hurdle Model")
    print(f"          Target coverage = {int((1-ALPHA)*100)}%  (alpha={ALPHA})")
    print("="*70)

    windows = load_windows()
    results = []

    for win in windows:
        wid = win['window_id']
        print(f"\nWindow {wid}: {win['pred_start']} -> {win['pred_end']}")
        res = cqr_on_window(win)
        results.append(res)

    df = pd.DataFrame(results)
    out = RESULTS_DIR / 'conformal_prediction.csv'
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}")

    print("\n" + "="*70)
    print("[SUMMARY] CQR vs MCD-ZILN Coverage Comparison")
    print("="*70)
    print(f"{'Metric':<30} {'CQR-Hurdle':>12} {'MCD-ZILN (reported)':>20}")
    print("-"*65)
    print(f"{'Target coverage':<30} {'90%':>12} {'95%':>20}")
    print(f"{'Empirical coverage':<30} {df['CQR_Coverage'].mean():.1f}%{'':<6} {'32% (paper §5.6)':>20}")
    print(f"{'NG (midpoint ranking)':<30} {df['NG_Midpoint'].mean():.4f}{'':<6} {'0.834 (MASTER_TABLE)':>20}")
    print(f"{'RC@10% (midpoint)':<30} {df['RC10_Midpoint'].mean():.2f}%{'':<6} {'60.68%':>20}")
    print(f"{'Mean PI width ($)':<30} ${df['Mean_Width'].mean():,.0f}{'':<7} {'N/A':>20}")
    print("\nNote: CQR provides guaranteed marginal coverage >= 90%;")
    print("      MCD-ZILN's empirical 95% intervals cover only 32% of actual values.")
    print("="*70)


if __name__ == "__main__":
    main()
