"""
34_ablation_study.py
PHASE C1: Ablation Study
Tests contribution of each feature group:
- RFM only (3 features)
- RFM + Behavioral (10 features)
- RFM + Behavioral + Interactions (16 features)
- RFM + Behavioral + Interactions + Sequence (23 features)

Uses best model architecture (Hurdle + LightGBM) for fair comparison.
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
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_windows():
    with open(DATA_PROCESSED_DIR / 'walk_forward_windows.pkl', 'rb') as f:
        return pickle.load(f)


def get_features_by_group(features_df, group, sequence_data=None):
    """
    Get features based on group:
    - rfm: only Recency, Frequency, Monetary
    - rfm_behavior: + Tenure, ActiveMonths, ProductDiversity, AvgOrderValue,
                     AvgDaysBetweenOrders, Regularity, IsUK
    - rfm_behavior_interaction: + log features and interactions
    - full: + sequence summary features
    """
    if group == 'rfm':
        cols = ['Recency', 'Frequency', 'Monetary']
        X = features_df[cols].fillna(0).copy()

    elif group == 'rfm_behavior':
        cols = ['Recency', 'Frequency', 'Monetary', 'Tenure',
                'ActiveMonths', 'ProductDiversity', 'AvgOrderValue',
                'AvgDaysBetweenOrders', 'Regularity', 'IsUK']
        X = features_df[cols].fillna(0).copy()

    elif group == 'rfm_behavior_interaction':
        cols = ['Recency', 'Frequency', 'Monetary', 'Tenure',
                'ActiveMonths', 'ProductDiversity', 'AvgOrderValue',
                'AvgDaysBetweenOrders', 'Regularity', 'IsUK']
        X = features_df[cols].fillna(0).copy()
        X['log_Monetary'] = np.log1p(X['Monetary'])
        X['log_Frequency'] = np.log1p(X['Frequency'])
        X['log_AvgOrderValue'] = np.log1p(X['AvgOrderValue'])
        X['M_per_F'] = X['Monetary'] / np.maximum(X['Frequency'], 1)
        X['M_per_T'] = X['Monetary'] / np.maximum(X['Tenure'], 1)
        X['Active_ratio'] = X['ActiveMonths'] / np.maximum(X['Tenure'] / 30, 1)

    elif group == 'full':
        cols = ['Recency', 'Frequency', 'Monetary', 'Tenure',
                'ActiveMonths', 'ProductDiversity', 'AvgOrderValue',
                'AvgDaysBetweenOrders', 'Regularity', 'IsUK']
        X = features_df[cols].fillna(0).copy()
        X['log_Monetary'] = np.log1p(X['Monetary'])
        X['log_Frequency'] = np.log1p(X['Frequency'])
        X['log_AvgOrderValue'] = np.log1p(X['AvgOrderValue'])
        X['M_per_F'] = X['Monetary'] / np.maximum(X['Frequency'], 1)
        X['M_per_T'] = X['Monetary'] / np.maximum(X['Tenure'], 1)
        X['Active_ratio'] = X['ActiveMonths'] / np.maximum(X['Tenure'] / 30, 1)

        if sequence_data is not None:
            seq = sequence_data['revenue_seq']
            X['seq_mean'] = seq.mean(axis=1)
            X['seq_std'] = seq.std(axis=1)
            X['seq_max'] = seq.max(axis=1)
            X['seq_recent3_mean'] = seq[:, -3:].mean(axis=1)
            X['seq_recent3_max'] = seq[:, -3:].max(axis=1)
            X['seq_active_months'] = (seq > 0).sum(axis=1)
            n = seq.shape[1]
            if n >= 4:
                X['seq_trend'] = seq[:, -n//2:].mean(axis=1) - seq[:, :n//2].mean(axis=1)

    return X.values, list(X.columns)


def train_lgb_hurdle(X_train, y_train, X_test):
    """Train Hurdle LightGBM (best architecture from experiments)"""
    # Stage 1: Classification
    y_binary = (y_train > 0).astype(int)

    clf = lgb.LGBMClassifier(
        n_estimators=300, num_leaves=31, learning_rate=0.05,
        feature_fraction=0.8, bagging_fraction=0.8,
        min_data_in_leaf=20, random_state=42, verbosity=-1
    )
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_binary, test_size=0.15, random_state=42)
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(20, verbose=False)])
    prob_positive = clf.predict_proba(X_test)[:, 1]

    # Stage 2: Regression on positives only
    pos_mask = y_train > 0
    X_pos = X_train[pos_mask]
    y_log_pos = np.log1p(y_train[pos_mask])

    reg = lgb.LGBMRegressor(
        n_estimators=300, num_leaves=31, learning_rate=0.05,
        feature_fraction=0.8, bagging_fraction=0.8,
        min_data_in_leaf=10, random_state=42, verbosity=-1
    )
    X_tr, X_val, y_tr, y_val = train_test_split(X_pos, y_log_pos, test_size=0.15, random_state=42)
    reg.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(20, verbose=False)])

    log_pred = reg.predict(X_test)

    # Bias correction with residual variance
    val_pred = reg.predict(X_val)
    residual_var = np.var(y_val - val_pred)
    correction = np.exp(residual_var / 2)

    pred = prob_positive * np.expm1(log_pred) * correction
    return np.maximum(0, pred)


def run_ablation():
    """Run ablation study across walk-forward windows"""
    print("\n" + "="*70)
    print("[PHASE C1] ABLATION STUDY")
    print("="*70)

    windows = load_windows()

    feature_groups = [
        ('RFM only', 'rfm'),
        ('+ Behavioral', 'rfm_behavior'),
        ('+ Interactions', 'rfm_behavior_interaction'),
        ('+ Sequence (Full)', 'full'),
    ]

    all_results = []

    for window in windows:
        print(f"\n{'='*70}")
        print(f"Window {window['window_id']}: pred={window['pred_start']} to {window['pred_end']}")
        print(f"{'='*70}")

        features = window['features']
        seq_data = {
            'revenue_seq': window['revenue_seq'],
            'frequency_seq': window['frequency_seq']
        }
        y = features['ActualCLV'].values

        train_idx, test_idx = train_test_split(
            range(len(features)), test_size=0.2, random_state=42,
            stratify=features['IsVIP']
        )
        y_train, y_test = y[train_idx], y[test_idx]

        for group_name, group_id in feature_groups:
            X, feature_names = get_features_by_group(features, group_id, seq_data)
            X_train, X_test = X[train_idx], X[test_idx]

            print(f"\n[{group_name}] {len(feature_names)} features")
            y_pred = train_lgb_hurdle(X_train, y_train, X_test)

            metrics = comprehensive_metrics(y_test, y_pred)
            print(f"  MAE: ${metrics['MAE']:,.2f} | R²: {metrics['R2']:.4f} | "
                  f"Norm_Gini: {metrics['Norm_Gini']:.4f} | "
                  f"Revenue@10: {metrics['Revenue_Capture_10']:.2f}%")

            row = {
                'Window': window['window_id'],
                'Feature Group': group_name,
                'N_Features': len(feature_names)
            }
            row.update(metrics)
            all_results.append(row)

    # Save
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / 'ablation_walkforward.csv', index=False)

    # Summary
    print("\n" + "="*70)
    print("[ABLATION SUMMARY] Mean ± Std Across 3 Windows")
    print("="*70)

    summary_rows = []
    for group, _ in feature_groups:
        df_g = df[df['Feature Group'] == group]
        n_feat = df_g['N_Features'].iloc[0]
        print(f"\n{group} ({n_feat} features)")
        print(f"  MAE:                 ${df_g['MAE'].mean():.2f} ± {df_g['MAE'].std():.2f}")
        print(f"  R²:                  {df_g['R2'].mean():.4f} ± {df_g['R2'].std():.4f}")
        print(f"  Norm Gini:           {df_g['Norm_Gini'].mean():.4f} ± {df_g['Norm_Gini'].std():.4f}")
        print(f"  Revenue Capture@10%: {df_g['Revenue_Capture_10'].mean():.2f}% ± {df_g['Revenue_Capture_10'].std():.2f}%")
        print(f"  Top 5% MAPE:         {df_g['Top5_MAPE'].mean():.4f}")

        summary_rows.append({
            'Feature_Group': group,
            'N_Features': n_feat,
            'MAE_mean': df_g['MAE'].mean(),
            'MAE_std': df_g['MAE'].std(),
            'R2_mean': df_g['R2'].mean(),
            'R2_std': df_g['R2'].std(),
            'Norm_Gini_mean': df_g['Norm_Gini'].mean(),
            'Norm_Gini_std': df_g['Norm_Gini'].std(),
            'Revenue_Capture_10_mean': df_g['Revenue_Capture_10'].mean(),
            'Revenue_Capture_10_std': df_g['Revenue_Capture_10'].std(),
            'Top5_MAPE_mean': df_g['Top5_MAPE'].mean(),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / 'ablation_summary.csv', index=False)

    # Compute deltas
    print("\n[DELTA] Improvement vs RFM only:")
    base_gini = summary_df[summary_df['Feature_Group'] == 'RFM only']['Norm_Gini_mean'].iloc[0]
    base_rev = summary_df[summary_df['Feature_Group'] == 'RFM only']['Revenue_Capture_10_mean'].iloc[0]
    base_mae = summary_df[summary_df['Feature_Group'] == 'RFM only']['MAE_mean'].iloc[0]

    for _, row in summary_df.iterrows():
        if row['Feature_Group'] != 'RFM only':
            d_gini = row['Norm_Gini_mean'] - base_gini
            d_rev = row['Revenue_Capture_10_mean'] - base_rev
            d_mae = row['MAE_mean'] - base_mae
            print(f"  {row['Feature_Group']:30s} | dGini: {d_gini:+.4f} | "
                  f"dRevenue: {d_rev:+.2f}pp | dMAE: ${d_mae:+.2f}")

    print("\n" + "="*70)
    print("[DONE] Ablation study complete")
    print("="*70 + "\n")

    return df, summary_df


if __name__ == "__main__":
    run_ablation()
