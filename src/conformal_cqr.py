"""
18b_conformal_hurdle_alpha05.py
Conformalized Quantile Regression (CQR) for Hurdle Model -- ALPHA = 0.05

Re-runs CQR targeting 95% coverage (alpha=0.05) instead of 90% (alpha=0.10).
Motivation:
  In 18_conformal_hurdle.py, q_hat ended up 0 in 4/5 windows, meaning the
  conformal calibration adjustment was trivial.  By targeting 95% coverage,
  the raw 2.5th-97.5th percentile regressors will under-cover the calibration
  set, forcing q_hat > 0 and demonstrating real CQR mechanism.

Algorithm:
  alpha    = 0.05  (target coverage = 95%)
  q_lo     = quantile regressor at 0.025
  q_hi     = quantile regressor at 0.975
  q_hat    = ceil((n+1)*(1-alpha))/n -th quantile of nonconformity scores
  interval = [q_lo - q_hat, q_hi + q_hat]
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

ALPHA      = 0.05   # target coverage = 95%


def load_windows():
    p5 = DATA_PROCESSED_DIR / 'walk_forward_windows_5fold.pkl'
    p3 = DATA_PROCESSED_DIR / 'walk_forward_windows.pkl'
    pkl = p5 if p5.exists() else p3
    with open(pkl, 'rb') as f:
        return pickle.load(f)


def rfm_seq_features(feats_df, win):
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


def cqr_on_window(win, alpha):
    feats = win['features']
    y     = feats['ActualCLV'].values
    X     = rfm_seq_features(feats, win)

    idx        = np.arange(len(feats))
    tr_cal, te = train_test_split(idx, test_size=0.20, random_state=42,
                                  stratify=feats['IsVIP'])
    tr, cal    = train_test_split(tr_cal, test_size=0.1875, random_state=42)

    X_tr,  y_tr  = X[tr],  y[tr]
    X_cal, y_cal = X[cal], y[cal]
    X_te,  y_te  = X[te],  y[te]

    tr2, val = train_test_split(np.arange(len(tr)), test_size=0.15, random_state=42)
    X_tr2,  y_tr2  = X_tr[tr2],  y_tr[tr2]
    X_val_s, y_val_s = X_tr[val], y_tr[val]

    y_tr2_log = np.log1p(y_tr2)
    y_val_log = np.log1p(y_val_s)

    print(f"    Train={len(tr2):,}  Cal={len(cal):,}  Test={len(te):,}")

    q_lo_model = fit_quantile_lgbm(X_tr2, y_tr2_log, X_val_s, y_val_log, alpha / 2)
    q_hi_model = fit_quantile_lgbm(X_tr2, y_tr2_log, X_val_s, y_val_log, 1 - alpha / 2)

    # Raw quantile predictions on calibration (no conformal adjustment)
    q_lo_cal_raw = np.expm1(q_lo_model.predict(X_cal))
    q_hi_cal_raw = np.expm1(q_hi_model.predict(X_cal))
    raw_cov_cal  = ((y_cal >= q_lo_cal_raw) & (y_cal <= q_hi_cal_raw)).mean() * 100

    # Conformal adjustment
    scores = np.maximum(q_lo_cal_raw - y_cal, y_cal - q_hi_cal_raw)
    n      = len(scores)
    level  = np.ceil((n + 1) * (1 - alpha)) / n
    level  = min(level, 1.0)
    q_hat  = np.quantile(scores, level)

    # Raw quantile coverage on TEST (without conformal)
    q_lo_te_raw = np.expm1(q_lo_model.predict(X_te))
    q_hi_te_raw = np.expm1(q_hi_model.predict(X_te))
    q_lo_te_raw_clip = np.maximum(q_lo_te_raw, 0)
    raw_cov_test = ((y_te >= q_lo_te_raw_clip) & (y_te <= q_hi_te_raw)).mean() * 100
    raw_width    = (q_hi_te_raw - q_lo_te_raw_clip).mean()

    # Conformal adjusted test intervals
    q_lo_te = q_lo_te_raw - q_hat
    q_hi_te = q_hi_te_raw + q_hat
    q_lo_te = np.maximum(q_lo_te, 0)

    covered    = ((y_te >= q_lo_te) & (y_te <= q_hi_te))
    coverage   = covered.mean() * 100
    mean_width = (q_hi_te - q_lo_te).mean()
    med_width  = np.median(q_hi_te - q_lo_te)

    # Ranking via midpoint and lower bound
    rank_lb  = comprehensive_metrics(y_te, q_lo_te)
    rank_mid = comprehensive_metrics(y_te, (q_lo_te + q_hi_te) / 2)

    print(f"    Raw cov (calibration): {raw_cov_cal:.1f}% (target {(1-alpha)*100:.0f}%)")
    print(f"    Raw cov (test):        {raw_cov_test:.1f}%")
    print(f"    q_hat (log-scale):     {q_hat:.4f}")
    print(f"    CQR cov (test):        {coverage:.1f}%")
    print(f"    Raw width:             ${raw_width:,.0f}")
    print(f"    CQR width:             ${mean_width:,.0f}")
    print(f"    NG (midpoint):         {rank_mid['Norm_Gini']:.4f}")

    return {
        'Window':            win['window_id'],
        'Raw_Cov_Cal':       raw_cov_cal,
        'Raw_Cov_Test':      raw_cov_test,
        'Raw_Width':         raw_width,
        'CQR_Coverage':      coverage,
        'Mean_Width':        mean_width,
        'Median_Width':      med_width,
        'q_hat':             q_hat,
        'NG_LowerBound':     rank_lb['Norm_Gini'],
        'RC10_LowerBound':   rank_lb['Revenue_Capture_10'],
        'NG_Midpoint':       rank_mid['Norm_Gini'],
        'RC10_Midpoint':     rank_mid['Revenue_Capture_10'],
    }


def main():
    print("\n" + "="*70)
    print(f"[Phase D-2] CQR for Hurdle Model  (alpha={ALPHA} -> target {int((1-ALPHA)*100)}%)")
    print("="*70)

    windows = load_windows()
    results = []

    for win in windows:
        wid = win['window_id']
        print(f"\nWindow {wid}: {win['pred_start']} -> {win['pred_end']}")
        res = cqr_on_window(win, ALPHA)
        results.append(res)

    df = pd.DataFrame(results)
    out = RESULTS_DIR / 'conformal_prediction_alpha05.csv'
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}")

    print("\n" + "="*70)
    print(f"[SUMMARY] CQR (alpha=0.05) vs Raw Quantile vs MCD-ZILN")
    print("="*70)
    print(f"{'Metric':<35} {'Raw Quantile':>14} {'CQR':>10} {'MCD-ZILN':>12}")
    print("-"*72)
    print(f"{'Target coverage':<35} {'95%':>14} {'95%':>10} {'95%':>12}")
    print(f"{'Empirical cov (test)':<35} "
          f"{df['Raw_Cov_Test'].mean():>13.1f}% "
          f"{df['CQR_Coverage'].mean():>9.1f}% "
          f"{'32%':>12}")
    print(f"{'q_hat > 0 in N windows':<35} {'-':>14} "
          f"{(df['q_hat']>0).sum():>9d}/{len(df):d}{'':<2} {'-':>12}")
    print(f"{'Mean PI width ($)':<35} "
          f"${df['Raw_Width'].mean():>12,.0f} "
          f"${df['Mean_Width'].mean():>8,.0f} {'N/A':>12}")
    print("="*70)
    print(f"CQR closes the gap from raw-quantile {df['Raw_Cov_Test'].mean():.1f}% to "
          f"{df['CQR_Coverage'].mean():.1f}% (target 95%),")
    print(f"and from MCD-ZILN 32% by a factor of {df['CQR_Coverage'].mean()/32:.2f}x.")


if __name__ == "__main__":
    main()
