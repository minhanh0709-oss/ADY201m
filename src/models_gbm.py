"""
31_models_gbm.py
PHASE B3: Tuned LightGBM & XGBoost with Optuna
Compare with baselines on walk-forward CV
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
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import KFold, train_test_split

DATA_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_windows():
    with open(DATA_PROCESSED_DIR / 'walk_forward_windows.pkl', 'rb') as f:
        return pickle.load(f)


# ============================================================
# Feature extraction
# ============================================================
def get_features(features_df, include_sequence=False, sequence_data=None):
    """Get feature matrix"""
    feature_cols = ['Recency', 'Frequency', 'Monetary', 'Tenure',
                    'ActiveMonths', 'ProductDiversity', 'AvgOrderValue',
                    'AvgDaysBetweenOrders', 'Regularity', 'IsUK']

    X = features_df[feature_cols].fillna(0).copy()

    # Add log features
    X['log_Monetary'] = np.log1p(X['Monetary'])
    X['log_Frequency'] = np.log1p(X['Frequency'])
    X['log_AvgOrderValue'] = np.log1p(X['AvgOrderValue'])

    # Add interaction features
    X['M_per_F'] = X['Monetary'] / np.maximum(X['Frequency'], 1)
    X['M_per_T'] = X['Monetary'] / np.maximum(X['Tenure'], 1)
    X['Active_ratio'] = X['ActiveMonths'] / np.maximum(X['Tenure'] / 30, 1)

    if include_sequence and sequence_data is not None:
        # Add sequence summary statistics
        seq = sequence_data['revenue_seq']
        X['seq_mean'] = seq.mean(axis=1)
        X['seq_std'] = seq.std(axis=1)
        X['seq_max'] = seq.max(axis=1)
        X['seq_recent3_mean'] = seq[:, -3:].mean(axis=1)
        X['seq_recent3_max'] = seq[:, -3:].max(axis=1)
        X['seq_active_months'] = (seq > 0).sum(axis=1)
        # Trend (recent vs early)
        n = seq.shape[1]
        if n >= 4:
            X['seq_trend'] = seq[:, -n//2:].mean(axis=1) - seq[:, :n//2].mean(axis=1)

    return X.values, list(X.columns)


# ============================================================
# OPTUNA Tuning
# ============================================================
def tune_lightgbm(X_train, y_train, n_trials=30, random_state=42):
    """Tune LightGBM hyperparameters with Optuna"""
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'verbosity': -1,
            'num_leaves': trial.suggest_int('num_leaves', 15, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 50),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'lambda_l1': trial.suggest_float('lambda_l1', 0, 5),
            'lambda_l2': trial.suggest_float('lambda_l2', 0, 5),
            'random_state': random_state,
        }

        # Internal CV
        kf = KFold(n_splits=3, shuffle=True, random_state=random_state)
        scores = []
        for tr_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            model = lgb.LGBMRegressor(**params, n_estimators=500)
            model.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])
            pred = model.predict(X_val)
            mae = np.mean(np.abs(y_val - pred))
            scores.append(mae)

        return np.mean(scores)

    study = optuna.create_study(direction='minimize',
                                 sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_xgboost(X_train, y_train, n_trials=30, random_state=42):
    """Tune XGBoost hyperparameters with Optuna"""
    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'verbosity': 0,
            'n_estimators': 500,
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
            'random_state': random_state,
            'tree_method': 'hist',
        }

        kf = KFold(n_splits=3, shuffle=True, random_state=random_state)
        scores = []
        for tr_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            model = xgb.XGBRegressor(**params, early_stopping_rounds=30)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            pred = model.predict(X_val)
            mae = np.mean(np.abs(y_val - pred))
            scores.append(mae)

        return np.mean(scores)

    study = optuna.create_study(direction='minimize',
                                 sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ============================================================
# MODEL TRAINING
# ============================================================
def train_lightgbm(X_train, y_train, X_test, y_test, params, log_target=False):
    """Train LightGBM with given params"""
    if log_target:
        y_train_t = np.log1p(y_train)
    else:
        y_train_t = y_train

    # Final train/val split for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train_t, test_size=0.15, random_state=42
    )

    model = lgb.LGBMRegressor(**params, n_estimators=1000, verbosity=-1)
    model.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])

    pred = model.predict(X_test)
    if log_target:
        pred = np.expm1(pred)
    pred = np.maximum(0, pred)

    return pred, model


def train_xgboost(X_train, y_train, X_test, y_test, params, log_target=False):
    """Train XGBoost with given params"""
    if log_target:
        y_train_t = np.log1p(y_train)
    else:
        y_train_t = y_train

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train_t, test_size=0.15, random_state=42
    )

    model = xgb.XGBRegressor(**params, n_estimators=1000,
                              early_stopping_rounds=50, verbosity=0)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    pred = model.predict(X_test)
    if log_target:
        pred = np.expm1(pred)
    pred = np.maximum(0, pred)

    return pred, model


# ============================================================
# MAIN
# ============================================================
def run_gbm():
    """Run tuned LightGBM and XGBoost across walk-forward windows"""
    print("\n" + "="*70)
    print("[PHASE B3] TUNED LIGHTGBM & XGBOOST")
    print("="*70)

    windows = load_windows()
    all_results = []

    for window in windows:
        print(f"\n{'='*70}")
        print(f"Window {window['window_id']}: pred={window['pred_start']} to {window['pred_end']}")
        print(f"{'='*70}")

        features = window['features']

        # Get feature matrix (without sequence for now)
        X_full, feature_names = get_features(features, include_sequence=False)
        y_full = features['ActualCLV'].values

        # Train/test split (stratified by VIP)
        train_idx, test_idx = train_test_split(
            range(len(features)),
            test_size=0.2,
            random_state=42,
            stratify=features['IsVIP']
        )
        X_train, X_test = X_full[train_idx], X_full[test_idx]
        y_train, y_test = y_full[train_idx], y_full[test_idx]

        print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
        print(f"  Features: {X_full.shape[1]}")

        # ---- LightGBM (raw) ----
        print(f"\n[LightGBM] Tuning hyperparameters...")
        best_params = tune_lightgbm(X_train, y_train, n_trials=20)
        print(f"  Best params: {best_params}")

        y_pred, model_lgb = train_lightgbm(X_train, y_train, X_test, y_test, best_params)
        metrics = comprehensive_metrics(y_test, y_pred)
        print(f"  MAE: ${metrics['MAE']:,.2f} | R²: {metrics['R2']:.4f} | "
              f"Norm_Gini: {metrics['Norm_Gini']:.4f} | "
              f"Revenue@10: {metrics['Revenue_Capture_10']:.2f}%")
        row = {'Window': window['window_id'], 'Model': 'LightGBM (raw)'}
        row.update(metrics)
        all_results.append(row)

        # ---- LightGBM (log target) ----
        print(f"\n[LightGBM Log] Tuning on log target...")
        best_params_log = tune_lightgbm(X_train, np.log1p(y_train), n_trials=20)
        y_pred_log, _ = train_lightgbm(X_train, y_train, X_test, y_test, best_params_log, log_target=True)
        metrics = comprehensive_metrics(y_test, y_pred_log)
        print(f"  MAE: ${metrics['MAE']:,.2f} | R²: {metrics['R2']:.4f} | "
              f"Norm_Gini: {metrics['Norm_Gini']:.4f} | "
              f"Revenue@10: {metrics['Revenue_Capture_10']:.2f}%")
        row = {'Window': window['window_id'], 'Model': 'LightGBM (log)'}
        row.update(metrics)
        all_results.append(row)

        # ---- XGBoost (raw) ----
        print(f"\n[XGBoost] Tuning hyperparameters...")
        best_params_xgb = tune_xgboost(X_train, y_train, n_trials=20)
        y_pred_xgb, model_xgb = train_xgboost(X_train, y_train, X_test, y_test, best_params_xgb)
        metrics = comprehensive_metrics(y_test, y_pred_xgb)
        print(f"  MAE: ${metrics['MAE']:,.2f} | R²: {metrics['R2']:.4f} | "
              f"Norm_Gini: {metrics['Norm_Gini']:.4f} | "
              f"Revenue@10: {metrics['Revenue_Capture_10']:.2f}%")
        row = {'Window': window['window_id'], 'Model': 'XGBoost (raw)'}
        row.update(metrics)
        all_results.append(row)

        # ---- XGBoost (log) ----
        print(f"\n[XGBoost Log] Tuning on log target...")
        best_params_xgb_log = tune_xgboost(X_train, np.log1p(y_train), n_trials=20)
        y_pred_xgb_log, _ = train_xgboost(X_train, y_train, X_test, y_test, best_params_xgb_log, log_target=True)
        metrics = comprehensive_metrics(y_test, y_pred_xgb_log)
        print(f"  MAE: ${metrics['MAE']:,.2f} | R²: {metrics['R2']:.4f} | "
              f"Norm_Gini: {metrics['Norm_Gini']:.4f} | "
              f"Revenue@10: {metrics['Revenue_Capture_10']:.2f}%")
        row = {'Window': window['window_id'], 'Model': 'XGBoost (log)'}
        row.update(metrics)
        all_results.append(row)

        # ---- LightGBM with SEQUENCE features ----
        print(f"\n[LightGBM + Sequence] Adding sequence features...")
        seq_data = {
            'revenue_seq': window['revenue_seq'],
            'frequency_seq': window['frequency_seq']
        }
        X_seq, _ = get_features(features, include_sequence=True, sequence_data=seq_data)
        X_train_seq = X_seq[train_idx]
        X_test_seq = X_seq[test_idx]

        best_params_seq = tune_lightgbm(X_train_seq, np.log1p(y_train), n_trials=20)
        y_pred_seq, _ = train_lightgbm(X_train_seq, y_train, X_test_seq, y_test,
                                        best_params_seq, log_target=True)
        metrics = comprehensive_metrics(y_test, y_pred_seq)
        print(f"  Features: {X_seq.shape[1]} (with sequence)")
        print(f"  MAE: ${metrics['MAE']:,.2f} | R²: {metrics['R2']:.4f} | "
              f"Norm_Gini: {metrics['Norm_Gini']:.4f} | "
              f"Revenue@10: {metrics['Revenue_Capture_10']:.2f}%")
        row = {'Window': window['window_id'], 'Model': 'LightGBM + Sequence (log)'}
        row.update(metrics)
        all_results.append(row)

    # Save results
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(RESULTS_DIR / 'gbm_walkforward.csv', index=False)

    # Summary
    print("\n" + "="*70)
    print("[SUMMARY] LightGBM & XGBoost Results")
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
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / 'gbm_summary.csv', index=False)

    for _, row in summary_df.iterrows():
        print(f"\n{row['Model']}")
        print(f"  MAE:                 ${row['MAE_mean']:.2f} ± {row['MAE_std']:.2f}")
        print(f"  R²:                  {row['R2_mean']:.4f} ± {row['R2_std']:.4f}")
        print(f"  Norm Gini:           {row['Norm_Gini_mean']:.4f} ± {row['Norm_Gini_std']:.4f}")
        print(f"  Revenue Capture@10%: {row['Revenue_Capture_10_mean']:.2f}% ± {row['Revenue_Capture_10_std']:.2f}%")
        print(f"  Top 5% MAPE:         {row['Top5_MAPE_mean']:.4f}")

    print("\n" + "="*70)
    print("[DONE] Tuned GBM evaluation complete")
    print("="*70 + "\n")

    return df_results, summary_df


if __name__ == "__main__":
    run_gbm()
