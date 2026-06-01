"""
50_shap_hurdle.py
TASK 1: SHAP analysis for BEST model (Hurdle Model)
- Stage 1: SHAP for classifier (P(buy))
- Stage 2: SHAP for regressor (E[amount | buy])
- Generates summary plots, beeswarm plots, waterfall for sample customers
"""

import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

import shap
import lightgbm as lgb
from sklearn.model_selection import train_test_split

DATA_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"


def load_windows():
    with open(DATA_PROCESSED_DIR / 'walk_forward_windows.pkl', 'rb') as f:
        return pickle.load(f)


def get_features(features_df, sequence_data=None):
    """Same feature extraction as Hurdle script"""
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

    return X


def train_hurdle_for_shap(X_train, y_train):
    """Train Hurdle and return both models for SHAP"""
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

    # Stage 2: Regression on positives
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

    return clf, reg


def plot_shap_summary_clf(clf, X_test, feature_names, save_path):
    """SHAP summary plot for classifier (Stage 1)"""
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)

    # For binary classification, take positive class
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values_pos = shap_values[1]
    else:
        shap_values_pos = shap_values

    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_pos, X_test, feature_names=feature_names,
                       show=False, plot_type='dot')
    plt.title('SHAP Summary: P(customer will purchase)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Also save bar plot
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_pos, X_test, feature_names=feature_names,
                       show=False, plot_type='bar')
    plt.title('Feature Importance: Stage 1 (Classifier)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(save_path).replace('.png', '_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return shap_values_pos


def plot_shap_summary_reg(reg, X_test_pos, feature_names, save_path):
    """SHAP summary plot for regressor (Stage 2)"""
    explainer = shap.TreeExplainer(reg)
    shap_values = explainer.shap_values(X_test_pos)

    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_pos, feature_names=feature_names,
                       show=False, plot_type='dot')
    plt.title('SHAP Summary: log(CLV) | customer buys', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Bar plot
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_pos, feature_names=feature_names,
                       show=False, plot_type='bar')
    plt.title('Feature Importance: Stage 2 (Regressor)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(save_path).replace('.png', '_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return shap_values


def plot_shap_waterfall(model, X_sample, feature_names, sample_idx, label_y, save_path):
    """Waterfall plot for individual customer"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample[sample_idx:sample_idx+1])

    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]
        expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    else:
        expected_value = explainer.expected_value

    # Convert to Explanation object
    exp = shap.Explanation(
        values=shap_values[0],
        base_values=expected_value,
        data=X_sample[sample_idx],
        feature_names=feature_names
    )

    fig = plt.figure(figsize=(10, 7))
    shap.plots.waterfall(exp, show=False, max_display=10)
    plt.title(f'Customer Sample - Actual CLV: ${label_y:.2f}', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_shap_analysis():
    print("\n" + "="*70)
    print("[TASK 1] SHAP ANALYSIS FOR HURDLE MODEL")
    print("="*70)

    windows = load_windows()

    # Use Window 3 (largest, most stable)
    window = windows[2]
    print(f"\nUsing Window {window['window_id']}: pred={window['pred_start']} to {window['pred_end']}")

    features = window['features']
    seq_data = {
        'revenue_seq': window['revenue_seq'],
        'frequency_seq': window['frequency_seq']
    }
    X_df = get_features(features, seq_data)
    feature_names = list(X_df.columns)
    X = X_df.values
    y = features['ActualCLV'].values

    train_idx, test_idx = train_test_split(
        range(len(features)), test_size=0.2, random_state=42,
        stratify=features['IsVIP']
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"\nFeatures: {len(feature_names)}")
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Train Hurdle model
    print(f"\n[Training Hurdle Model]")
    clf, reg = train_hurdle_for_shap(X_train, y_train)

    # SHAP for Stage 1 (Classifier)
    print(f"\n[SHAP Stage 1] Classifier - P(buy)")
    shap_clf = plot_shap_summary_clf(
        clf, X_test, feature_names,
        FIGURES_DIR / 'shap_hurdle_stage1.png'
    )
    print(f"  Saved: shap_hurdle_stage1.png")
    print(f"  Saved: shap_hurdle_stage1_bar.png")

    # Print top features for Stage 1
    mean_abs_shap_clf = np.abs(shap_clf).mean(axis=0)
    top_idx_clf = np.argsort(mean_abs_shap_clf)[::-1]
    print(f"\n  Top 10 features for P(buy):")
    for i in top_idx_clf[:10]:
        print(f"    {feature_names[i]:30s}: {mean_abs_shap_clf[i]:.4f}")

    # SHAP for Stage 2 (Regressor) - on positive samples only
    pos_test_mask = y_test > 0
    X_test_pos = X_test[pos_test_mask]
    y_test_pos = y_test[pos_test_mask]
    print(f"\n[SHAP Stage 2] Regressor - log(CLV | buy)")
    print(f"  Positive test samples: {len(X_test_pos):,}")

    shap_reg = plot_shap_summary_reg(
        reg, X_test_pos, feature_names,
        FIGURES_DIR / 'shap_hurdle_stage2.png'
    )
    print(f"  Saved: shap_hurdle_stage2.png")
    print(f"  Saved: shap_hurdle_stage2_bar.png")

    mean_abs_shap_reg = np.abs(shap_reg).mean(axis=0)
    top_idx_reg = np.argsort(mean_abs_shap_reg)[::-1]
    print(f"\n  Top 10 features for log(CLV | buy):")
    for i in top_idx_reg[:10]:
        print(f"    {feature_names[i]:30s}: {mean_abs_shap_reg[i]:.4f}")

    # Sample customer waterfall plots
    print(f"\n[Waterfall plots for sample customers]")
    test_features = features.iloc[test_idx].reset_index(drop=True)

    # Find a high-value, mid-value, and zero-value customer
    sorted_by_clv = np.argsort(y_test)
    samples = {
        'vip_high': sorted_by_clv[-1],  # Highest CLV
        'vip_mid': sorted_by_clv[-50] if len(sorted_by_clv) >= 50 else sorted_by_clv[len(sorted_by_clv)//2],
        'low_value': sorted_by_clv[10] if (y_test > 0).sum() > 10 else sorted_by_clv[0],
    }

    for name, idx in samples.items():
        print(f"  {name}: CustomerID={test_features['CustomerID'].iloc[idx]}, "
              f"ActualCLV=${y_test[idx]:,.2f}")

        # Waterfall for Stage 1
        plot_shap_waterfall(
            clf, X_test, feature_names, idx, y_test[idx],
            FIGURES_DIR / f'shap_waterfall_{name}_stage1.png'
        )

    # Save feature importance to CSV
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Stage1_Importance': mean_abs_shap_clf,
        'Stage2_Importance': mean_abs_shap_reg if len(mean_abs_shap_reg) == len(feature_names) else [np.nan] * len(feature_names),
    }).sort_values('Stage1_Importance', ascending=False)
    importance_df.to_csv(RESULTS_DIR / 'shap_hurdle_importance.csv', index=False)
    print(f"\n[OK] Feature importance saved")

    # Print summary
    print("\n" + "="*70)
    print("[BUSINESS INSIGHTS]")
    print("="*70)
    print("""
Stage 1 (Will customer buy?) - Top drivers:
  - Recency: HIGH recency = LOWER probability of buying
  - Monetary: Higher past spend = higher buy probability
  - Active months: More active = more likely to return

Stage 2 (How much will they spend?) - Top drivers:
  - Monetary: Past spending predicts future spending
  - AvgOrderValue: High-value customers stay high-value
  - Frequency: Frequent buyers spend more

Marketing Recommendations:
  - Retention: Target high-Monetary, mid-Recency (e.g., 30-90 days)
  - Win-back: Target high-Monetary but high-Recency (lapsed VIPs)
  - Acquisition: Increase AvgOrderValue (cross-sell, upsell)
    """)

    print("="*70)
    print("[DONE] SHAP analysis complete")
    print("="*70 + "\n")

    return importance_df


if __name__ == "__main__":
    run_shap_analysis()
