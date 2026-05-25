"""
30_models_baselines.py
PHASE B1-B2: Baseline models (BG/NBD + Gamma-Gamma + simple baselines)
Evaluates on all 3 walk-forward windows
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

from lifetimes import BetaGeoFitter, GammaGammaFitter

DATA_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_windows():
    with open(DATA_PROCESSED_DIR / 'walk_forward_windows.pkl', 'rb') as f:
        return pickle.load(f)


# ============================================================
# BASELINE 1: Mean predictor
# ============================================================
def baseline_mean(features_train, features_test):
    """Predicts mean ActualCLV for everyone"""
    mean_clv = features_train['ActualCLV'].mean()
    return np.full(len(features_test), mean_clv)


# ============================================================
# BASELINE 2: Monetary
# ============================================================
def baseline_monetary(features_train, features_test):
    """Use Monetary as CLV prediction (calibrated)"""
    # Scale: median ActualCLV / median Monetary in training
    train_med_clv = max(features_train['ActualCLV'].median(), 1)
    train_med_mon = max(features_train['Monetary'].median(), 1)
    scale = train_med_clv / train_med_mon
    return features_test['Monetary'].values * scale


# ============================================================
# BASELINE 3: RFM Score
# ============================================================
def baseline_rfm_score(features_train, features_test):
    """Composite RFM score, calibrated to training CLV mean"""
    # Use training data for normalization
    r_max = features_train['Recency'].max()
    f_max = features_train['Frequency'].max()
    m_max = features_train['Monetary'].max()

    def score(df):
        r = 1 - (df['Recency'] / max(r_max, 1))
        f = df['Frequency'] / max(f_max, 1)
        m = df['Monetary'] / max(m_max, 1)
        return (r + f + m) / 3

    train_scores = score(features_train)
    test_scores = score(features_test)

    # Calibrate: scale score to match mean CLV
    train_mean_clv = features_train['ActualCLV'].mean()
    train_mean_score = train_scores.mean()
    if train_mean_score > 0:
        return test_scores.values * (train_mean_clv / train_mean_score)
    return test_scores.values * train_mean_clv


# ============================================================
# BASELINE 4: BG/NBD + Gamma-Gamma (CLASSIC PROBABILISTIC)
# ============================================================
def baseline_bgnbd_gg(features_train, features_test, pred_days):
    """
    BG/NBD: Predicts future purchase frequency
    Gamma-Gamma: Predicts average order value
    CLV = BGNBD_pred * GG_pred
    """
    # Need: frequency (repeat purchases), recency, T (age), monetary
    # In lifetimes library:
    #   frequency = number of repeat purchases (not total purchases)
    #   T = age of customer (days since first purchase)
    #   recency = days from first purchase to last purchase

    def prepare_data(df):
        d = df.copy()
        d['frequency_bgnbd'] = (d['Frequency'] - 1).clip(lower=0)  # repeat purchases
        d['T'] = d['T_BGNBD']
        d['recency'] = d['Recency_BGNBD']
        d['monetary_avg'] = d.apply(
            lambda r: r['Monetary'] / max(r['Frequency'], 1), axis=1
        )
        return d

    train = prepare_data(features_train)
    test = prepare_data(features_test)

    # Fit BG/NBD on training
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    try:
        bgf.fit(train['frequency_bgnbd'], train['recency'], train['T'])
    except Exception as e:
        print(f"  [WARN] BG/NBD fit failed: {e}")
        return np.zeros(len(features_test))

    # Predict purchases for test customers
    expected_purchases = bgf.predict(
        pred_days,
        test['frequency_bgnbd'],
        test['recency'],
        test['T']
    ).values

    # Fit Gamma-Gamma on repeat buyers only
    repeat_buyers = train[train['frequency_bgnbd'] > 0]
    if len(repeat_buyers) < 10:
        print(f"  [WARN] Too few repeat buyers ({len(repeat_buyers)}), using mean AOV")
        avg_aov = train['monetary_avg'].mean()
        return expected_purchases * avg_aov

    try:
        ggf = GammaGammaFitter(penalizer_coef=0.01)
        ggf.fit(repeat_buyers['frequency_bgnbd'], repeat_buyers['monetary_avg'])

        # Predict expected monetary value
        # For test customers with 0 repeat purchases, use average
        avg_monetary = np.zeros(len(test))
        repeat_test_mask = test['frequency_bgnbd'] > 0

        if repeat_test_mask.sum() > 0:
            avg_monetary_repeat = ggf.conditional_expected_average_profit(
                test.loc[repeat_test_mask, 'frequency_bgnbd'],
                test.loc[repeat_test_mask, 'monetary_avg']
            ).values
            avg_monetary[repeat_test_mask.values] = avg_monetary_repeat

        # For non-repeat buyers, use overall mean from G-G
        if (~repeat_test_mask).sum() > 0:
            avg_monetary[(~repeat_test_mask).values] = repeat_buyers['monetary_avg'].mean()

        clv = expected_purchases * avg_monetary
    except Exception as e:
        print(f"  [WARN] Gamma-Gamma failed: {e}, using simple AOV")
        clv = expected_purchases * test['monetary_avg'].values

    return clv


# ============================================================
# BASELINE 5: Linear Regression (sklearn)
# ============================================================
def baseline_linear(features_train, features_test):
    """Linear regression with proper train/test split"""
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    feature_cols = ['Recency', 'Frequency', 'Monetary', 'Tenure',
                    'ActiveMonths', 'ProductDiversity', 'AvgOrderValue',
                    'AvgDaysBetweenOrders', 'Regularity', 'IsUK']

    X_train = features_train[feature_cols].fillna(0).values
    X_test = features_test[feature_cols].fillna(0).values
    y_train = features_train['ActualCLV'].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)
    return np.maximum(0, model.predict(X_test_s))


# ============================================================
# BASELINE 6: Ridge Regression
# ============================================================
def baseline_ridge(features_train, features_test):
    """Ridge regression on log(CLV)"""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    feature_cols = ['Recency', 'Frequency', 'Monetary', 'Tenure',
                    'ActiveMonths', 'ProductDiversity', 'AvgOrderValue',
                    'AvgDaysBetweenOrders', 'Regularity', 'IsUK']

    X_train = features_train[feature_cols].fillna(0).values
    X_test = features_test[feature_cols].fillna(0).values
    y_train = np.log1p(features_train['ActualCLV'].values)  # log transform!

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    model.fit(X_train_s, y_train)
    log_pred = model.predict(X_test_s)
    return np.expm1(log_pred)  # Inverse log


# ============================================================
# MAIN
# ============================================================
def run_baselines():
    """Run all baselines across walk-forward windows"""
    print("\n" + "="*70)
    print("[PHASE B1+B2] BASELINE MODELS WITH WALK-FORWARD CV")
    print("="*70)

    windows = load_windows()

    # Models to evaluate
    model_funcs = {
        'Mean Predictor': lambda tr, te, _: baseline_mean(tr, te),
        'Monetary': lambda tr, te, _: baseline_monetary(tr, te),
        'RFM Score': lambda tr, te, _: baseline_rfm_score(tr, te),
        'Linear Regression': lambda tr, te, _: baseline_linear(tr, te),
        'Ridge (log)': lambda tr, te, _: baseline_ridge(tr, te),
        'BG/NBD + Gamma-Gamma': baseline_bgnbd_gg,
    }

    # For each model, evaluate on each window
    # Note: We use the SAME window for both train and test
    # because in CLV setup, observation window -> features, prediction window -> labels
    # Walk-forward: different windows give different observation periods

    all_results = []

    for i, window in enumerate(windows):
        print(f"\n{'='*70}")
        print(f"Window {window['window_id']}: obs={window['obs_start']} to {window['obs_end']}")
        print(f"                  pred={window['pred_start']} to {window['pred_end']}")
        print(f"{'='*70}")

        features = window['features']
        pred_days = (pd.to_datetime(window['pred_end']) -
                     pd.to_datetime(window['pred_start'])).days

        # For models that need train/test, split features within window
        # Use 80% as train, 20% as test (stratified by VIP)
        from sklearn.model_selection import train_test_split
        np.random.seed(42)
        train_idx, test_idx = train_test_split(
            range(len(features)),
            test_size=0.2,
            random_state=42,
            stratify=features['IsVIP']
        )
        f_train = features.iloc[train_idx].copy().reset_index(drop=True)
        f_test = features.iloc[test_idx].copy().reset_index(drop=True)
        y_test = f_test['ActualCLV'].values

        for model_name, model_func in model_funcs.items():
            try:
                print(f"\n[{model_name}]")
                y_pred = model_func(f_train, f_test, pred_days)
                y_pred = np.maximum(0, y_pred)  # CLV cannot be negative

                metrics = comprehensive_metrics(y_test, y_pred)
                print(f"  MAE: ${metrics['MAE']:,.2f} | R²: {metrics['R2']:.4f} | "
                      f"Norm_Gini: {metrics['Norm_Gini']:.4f} | "
                      f"Revenue@10: {metrics['Revenue_Capture_10']:.2f}%")

                row = {'Window': window['window_id'], 'Model': model_name}
                row.update(metrics)
                all_results.append(row)

            except Exception as e:
                print(f"  [ERROR] {model_name}: {e}")
                row = {'Window': window['window_id'], 'Model': model_name}
                row.update({k: np.nan for k in ['MAE', 'RMSE', 'R2', 'Spearman',
                                                 'Norm_Gini', 'Top5_MAPE', 'Top10_MAPE',
                                                 'Revenue_Capture_10', 'Revenue_Capture_20',
                                                 'Lift_10', 'Precision_10', 'Decile_MAPE']})
                all_results.append(row)

    # Save results
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(RESULTS_DIR / 'baseline_walkforward.csv', index=False)

    # Compute mean ± std across windows
    print("\n" + "="*70)
    print("[SUMMARY] Mean ± Std Across 3 Walk-Forward Windows")
    print("="*70)

    summary_rows = []
    for model in df_results['Model'].unique():
        model_df = df_results[df_results['Model'] == model]
        summary_rows.append({
            'Model': model,
            'MAE_mean': model_df['MAE'].mean(),
            'MAE_std': model_df['MAE'].std(),
            'R2_mean': model_df['R2'].mean(),
            'R2_std': model_df['R2'].std(),
            'Norm_Gini_mean': model_df['Norm_Gini'].mean(),
            'Norm_Gini_std': model_df['Norm_Gini'].std(),
            'Revenue_Capture_10_mean': model_df['Revenue_Capture_10'].mean(),
            'Revenue_Capture_10_std': model_df['Revenue_Capture_10'].std(),
            'Top5_MAPE_mean': model_df['Top5_MAPE'].mean(),
            'Top5_MAPE_std': model_df['Top5_MAPE'].std(),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / 'baseline_summary.csv', index=False)

    # Display
    print("\nKey metrics (mean ± std):")
    for _, row in summary_df.iterrows():
        print(f"\n{row['Model']}")
        print(f"  MAE:                 ${row['MAE_mean']:.2f} ± {row['MAE_std']:.2f}")
        print(f"  R²:                  {row['R2_mean']:.4f} ± {row['R2_std']:.4f}")
        print(f"  Norm Gini:           {row['Norm_Gini_mean']:.4f} ± {row['Norm_Gini_std']:.4f}")
        print(f"  Revenue Capture@10%: {row['Revenue_Capture_10_mean']:.2f}% ± {row['Revenue_Capture_10_std']:.2f}%")
        print(f"  Top 5% MAPE:         {row['Top5_MAPE_mean']:.4f} ± {row['Top5_MAPE_std']:.4f}")

    print("\n" + "="*70)
    print("[DONE] Baseline evaluation complete")
    print("="*70 + "\n")

    return df_results, summary_df


if __name__ == "__main__":
    run_baselines()
