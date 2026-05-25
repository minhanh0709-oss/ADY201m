"""
33_models_hurdle.py
PHASE B4: Hurdle Model (Two-Stage Approach)
Stage 1: Will customer buy in prediction window? (Classification)
Stage 2: If yes, how much will they spend? (Regression on positives only)

Simpler alternative to ZILN - both fix zero-inflation.
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
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_windows():
    with open(DATA_PROCESSED_DIR / 'walk_forward_windows.pkl', 'rb') as f:
        return pickle.load(f)


def get_features(features_df, include_sequence=False, sequence_data=None):
    """Same feature extraction as GBM script"""
    feature_cols = ['Recency', 'Frequency', 'Monetary', 'Tenure',
                    'ActiveMonths', 'ProductDiversity', 'AvgOrderValue',
                    'AvgDaysBetweenOrders', 'Regularity', 'IsUK']
    X = features_df[feature_cols].fillna(0).copy()
    X['log_Monetary'] = np.log1p(X['Monetary'])
    X['log_Frequency'] = np.log1p(X['Frequency'])
    X['log_AvgOrderValue'] = np.log1p(X['AvgOrderValue'])
    X['M_per_F'] = X['Monetary'] / np.maximum(X['Frequency'], 1)
    X['M_per_T'] = X['Monetary'] / np.maximum(X['Tenure'], 1)
    X['Active_ratio'] = X['ActiveMonths'] / np.maximum(X['Tenure'] / 30, 1)

    if include_sequence and sequence_data is not None:
        seq = sequence_data['revenue_seq']
        X['seq_mean'] = seq.mean(axis=1)
        X['seq_max'] = seq.max(axis=1)
        X['seq_recent3_mean'] = seq[:, -3:].mean(axis=1)
        X['seq_recent3_max'] = seq[:, -3:].max(axis=1)
        X['seq_active_months'] = (seq > 0).sum(axis=1)

    return X.values, list(X.columns)


def tune_classifier(X_train, y_binary, n_trials=15):
    """Tune LightGBM classifier"""
    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'num_leaves': trial.suggest_int('num_leaves', 15, 80),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 50),
            'random_state': 42,
        }
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for tr_idx, val_idx in kf.split(X_train):
            model = lgb.LGBMClassifier(**params, n_estimators=300)
            model.fit(X_train[tr_idx], y_binary[tr_idx],
                      eval_set=[(X_train[val_idx], y_binary[val_idx])],
                      callbacks=[lgb.early_stopping(20, verbose=False)])
            pred = model.predict_proba(X_train[val_idx])[:, 1]
            from sklearn.metrics import roc_auc_score
            scores.append(roc_auc_score(y_binary[val_idx], pred))
        return -np.mean(scores)  # Minimize negative AUC

    study = optuna.create_study(direction='minimize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_regressor(X_train, y_log, n_trials=15):
    """Tune LightGBM regressor on log target"""
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'verbosity': -1,
            'num_leaves': trial.suggest_int('num_leaves', 15, 80),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 30),
            'random_state': 42,
        }
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for tr_idx, val_idx in kf.split(X_train):
            model = lgb.LGBMRegressor(**params, n_estimators=300)
            model.fit(X_train[tr_idx], y_log[tr_idx],
                      eval_set=[(X_train[val_idx], y_log[val_idx])],
                      callbacks=[lgb.early_stopping(20, verbose=False)])
            pred = model.predict(X_train[val_idx])
            mae = np.mean(np.abs(y_log[val_idx] - pred))
            scores.append(mae)
        return np.mean(scores)

    study = optuna.create_study(direction='minimize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def train_hurdle(X_train, y_train, X_test, n_trials=15):
    """
    Train hurdle model:
    Stage 1: P(y > 0) using classification
    Stage 2: E[log(y) | y > 0] using regression on positives
    Final: E[y] = P(y > 0) * exp(E[log(y) | y > 0] + variance/2)
    """
    # Stage 1: Classification
    y_binary = (y_train > 0).astype(int)

    print(f"    Stage 1: Training classifier (positive rate: {y_binary.mean():.2%})")
    clf_params = tune_classifier(X_train, y_binary, n_trials=n_trials)
    clf = lgb.LGBMClassifier(**clf_params, n_estimators=500, verbosity=-1)

    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_binary, test_size=0.15, random_state=42)
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False)])
    prob_positive = clf.predict_proba(X_test)[:, 1]

    # Stage 2: Regression on positives only
    pos_mask = y_train > 0
    X_pos = X_train[pos_mask]
    y_log_pos = np.log1p(y_train[pos_mask])

    print(f"    Stage 2: Training regressor on {pos_mask.sum():,} positive samples")
    reg_params = tune_regressor(X_pos, y_log_pos, n_trials=n_trials)
    reg = lgb.LGBMRegressor(**reg_params, n_estimators=500, verbosity=-1)

    X_tr, X_val, y_tr, y_val = train_test_split(X_pos, y_log_pos, test_size=0.15, random_state=42)
    reg.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False)])

    log_pred = reg.predict(X_test)

    # Compute residual variance for log-normal correction
    val_pred = reg.predict(X_val)
    residual_var = np.var(y_val - val_pred)
    correction = np.exp(residual_var / 2)

    # Combined prediction
    pred_positive = np.expm1(log_pred) * correction
    pred = prob_positive * pred_positive
    pred = np.maximum(0, pred)

    return pred, clf, reg


def run_hurdle():
    """Run Hurdle model on walk-forward windows"""
    print("\n" + "="*70)
    print("[PHASE B4] HURDLE MODEL (Two-Stage)")
    print("="*70)

    windows = load_windows()
    all_results = []

    for window in windows:
        print(f"\n{'='*70}")
        print(f"Window {window['window_id']}: pred={window['pred_start']} to {window['pred_end']}")
        print(f"{'='*70}")

        features = window['features']

        # Get features (without sequence first)
        X, _ = get_features(features, include_sequence=False)
        y = features['ActualCLV'].values

        train_idx, test_idx = train_test_split(
            range(len(features)), test_size=0.2, random_state=42,
            stratify=features['IsVIP']
        )
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Hurdle without sequence
        print(f"\n[Hurdle Model]")
        y_pred, _, _ = train_hurdle(X_train, y_train, X_test, n_trials=15)

        metrics = comprehensive_metrics(y_test, y_pred)
        print(f"  MAE: ${metrics['MAE']:,.2f} | R²: {metrics['R2']:.4f} | "
              f"Norm_Gini: {metrics['Norm_Gini']:.4f} | "
              f"Revenue@10: {metrics['Revenue_Capture_10']:.2f}%")
        row = {'Window': window['window_id'], 'Model': 'Hurdle Model'}
        row.update(metrics)
        all_results.append(row)

        # Hurdle with sequence
        print(f"\n[Hurdle + Sequence]")
        seq_data = {
            'revenue_seq': window['revenue_seq'],
            'frequency_seq': window['frequency_seq']
        }
        X_seq, _ = get_features(features, include_sequence=True, sequence_data=seq_data)
        X_train_seq, X_test_seq = X_seq[train_idx], X_seq[test_idx]

        y_pred_seq, _, _ = train_hurdle(X_train_seq, y_train, X_test_seq, n_trials=15)

        metrics = comprehensive_metrics(y_test, y_pred_seq)
        print(f"  MAE: ${metrics['MAE']:,.2f} | R²: {metrics['R2']:.4f} | "
              f"Norm_Gini: {metrics['Norm_Gini']:.4f} | "
              f"Revenue@10: {metrics['Revenue_Capture_10']:.2f}%")
        row = {'Window': window['window_id'], 'Model': 'Hurdle + Sequence'}
        row.update(metrics)
        all_results.append(row)

    # Save results
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(RESULTS_DIR / 'hurdle_walkforward.csv', index=False)

    # Summary
    print("\n" + "="*70)
    print("[SUMMARY] Hurdle Model Results")
    print("="*70)

    for model in df_results['Model'].unique():
        model_df = df_results[df_results['Model'] == model]
        print(f"\n{model}")
        print(f"  MAE:                 ${model_df['MAE'].mean():.2f} ± {model_df['MAE'].std():.2f}")
        print(f"  R²:                  {model_df['R2'].mean():.4f} ± {model_df['R2'].std():.4f}")
        print(f"  Norm Gini:           {model_df['Norm_Gini'].mean():.4f} ± {model_df['Norm_Gini'].std():.4f}")
        print(f"  Revenue Capture@10%: {model_df['Revenue_Capture_10'].mean():.2f}% ± {model_df['Revenue_Capture_10'].std():.2f}%")
        print(f"  Top 5% MAPE:         {model_df['Top5_MAPE'].mean():.4f}")

    print("\n" + "="*70)
    print("[DONE] Hurdle model evaluation complete")
    print("="*70 + "\n")

    return df_results


if __name__ == "__main__":
    run_hurdle()
