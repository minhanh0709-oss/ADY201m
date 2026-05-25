"""
52_revenue_capture_curve.py
TASK 3: Revenue Capture @ K curve - Critical CLV business plot
Shows how much revenue captured at different K% targeting levels.
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

import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split

DATA_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"


def load_windows():
    with open(DATA_PROCESSED_DIR / 'walk_forward_windows.pkl', 'rb') as f:
        return pickle.load(f)


def get_features(features_df, sequence_data=None):
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

    return X.values, list(X.columns)


def get_features_basic(features_df):
    """RFM + behavioral only (no sequence)"""
    feature_cols = ['Recency', 'Frequency', 'Monetary', 'Tenure',
                    'ActiveMonths', 'ProductDiversity', 'AvgOrderValue',
                    'AvgDaysBetweenOrders', 'Regularity', 'IsUK']
    X = features_df[feature_cols].fillna(0).copy()
    return X.values


def train_hurdle(X_train, y_train, X_test):
    y_binary = (y_train > 0).astype(int)
    clf = lgb.LGBMClassifier(
        n_estimators=300, num_leaves=31, learning_rate=0.05,
        feature_fraction=0.8, bagging_fraction=0.8,
        min_data_in_leaf=20, random_state=42, verbosity=-1
    )
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_binary, test_size=0.15, random_state=42)
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(20, verbose=False)])
    prob = clf.predict_proba(X_test)[:, 1]

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
    val_pred = reg.predict(X_val)
    correction = np.exp(np.var(y_val - val_pred) / 2)
    return np.maximum(0, prob * np.expm1(log_pred) * correction)


def train_xgboost(X_train, y_train, X_test):
    """Tuned XGBoost"""
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, random_state=42, verbosity=0
    )
    model.fit(X_train, y_train)
    return np.maximum(0, model.predict(X_test))


def compute_capture_curve(y_true, y_pred, k_values):
    """Compute Revenue Capture at multiple K values"""
    captures = []
    total = y_true.sum()

    for k in k_values:
        n_top = max(1, int(len(y_pred) * k / 100))
        top_idx = np.argsort(y_pred)[-n_top:]
        capture = 100 * y_true[top_idx].sum() / total if total > 0 else 0
        captures.append(capture)

    return captures


def run_capture_curve():
    print("\n" + "="*70)
    print("[TASK 3] REVENUE CAPTURE @ K CURVE")
    print("="*70)

    windows = load_windows()
    k_values = [1, 5, 10, 15, 20, 25, 30, 40, 50, 70, 100]

    # Run on all 3 windows and average
    all_curves = {}

    for window in windows:
        print(f"\n[Window {window['window_id']}] pred={window['pred_start']} to {window['pred_end']}")

        features = window['features']
        seq_data = {'revenue_seq': window['revenue_seq'],
                    'frequency_seq': window['frequency_seq']}

        # Get features
        X_full, _ = get_features(features, seq_data)
        X_basic = get_features_basic(features)
        y = features['ActualCLV'].values

        # Train/test split
        train_idx, test_idx = train_test_split(
            range(len(features)), test_size=0.2, random_state=42,
            stratify=features['IsVIP']
        )
        X_full_train, X_full_test = X_full[train_idx], X_full[test_idx]
        X_basic_train, X_basic_test = X_basic[train_idx], X_basic[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train models
        print(f"  Training models...")
        pred_hurdle = train_hurdle(X_full_train, y_train, X_full_test)
        pred_xgb = train_xgboost(X_full_train, y_train, X_full_test)

        # Baselines
        pred_monetary = features.iloc[test_idx]['Monetary'].values

        rfm_features = features.iloc[test_idx]
        r_norm = 1 - (rfm_features['Recency'] / rfm_features['Recency'].max())
        f_norm = rfm_features['Frequency'] / rfm_features['Frequency'].max()
        m_norm = rfm_features['Monetary'] / rfm_features['Monetary'].max()
        pred_rfm = (r_norm + f_norm + m_norm).values

        # Random
        np.random.seed(42)
        pred_random = np.random.permutation(y_test).astype(float)

        # Compute capture curves
        models_preds = {
            'Random': pred_random,
            'Monetary': pred_monetary,
            'RFM Score': pred_rfm,
            'XGBoost': pred_xgb,
            'Hurdle (Proposed)': pred_hurdle,
            'Oracle': y_test,  # Perfect prediction
        }

        for model_name, pred in models_preds.items():
            captures = compute_capture_curve(y_test, pred, k_values)
            if model_name not in all_curves:
                all_curves[model_name] = []
            all_curves[model_name].append(captures)
            print(f"    {model_name:25s}: @10%={captures[2]:.2f}%, @20%={captures[4]:.2f}%")

    # Average across windows
    print("\n" + "="*70)
    print("[AVERAGED RESULTS] Mean across 3 windows")
    print("="*70)

    avg_curves = {}
    std_curves = {}
    for model, curves in all_curves.items():
        arr = np.array(curves)
        avg_curves[model] = arr.mean(axis=0)
        std_curves[model] = arr.std(axis=0)

    # Print table
    print(f"\n{'K%':<6}", end='')
    for model in avg_curves.keys():
        print(f"{model:<22}", end='')
    print()

    for i, k in enumerate(k_values):
        print(f"{k:<6}", end='')
        for model in avg_curves.keys():
            mean_v = avg_curves[model][i]
            std_v = std_curves[model][i]
            print(f"{mean_v:6.2f} ± {std_v:5.2f}        ", end='')
        print()

    # Save to CSV
    rows = []
    for i, k in enumerate(k_values):
        row = {'K_percent': k}
        for model in avg_curves.keys():
            row[f'{model}_mean'] = avg_curves[model][i]
            row[f'{model}_std'] = std_curves[model][i]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / 'revenue_capture_curve.csv', index=False)

    # Plot
    print(f"\n[Plotting]")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Main curve with error bands
    ax = axes[0]
    colors = {
        'Random': '#808080',
        'Monetary': '#F18F01',
        'RFM Score': '#A23B72',
        'XGBoost': '#5DADE2',
        'Hurdle (Proposed)': '#2E86AB',
        'Oracle': '#27AE60',
    }
    styles = {
        'Random': '--',
        'Oracle': ':',
    }

    for model in avg_curves.keys():
        mean_curve = avg_curves[model]
        std_curve = std_curves[model]
        linestyle = styles.get(model, '-')
        lw = 3 if model == 'Hurdle (Proposed)' else 2

        ax.plot(k_values, mean_curve,
                marker='o' if model == 'Hurdle (Proposed)' else 's' if model == 'Oracle' else None,
                label=model, color=colors[model],
                linestyle=linestyle, linewidth=lw, alpha=0.9)
        ax.fill_between(k_values, mean_curve - std_curve, mean_curve + std_curve,
                         alpha=0.15, color=colors[model])

    # Reference line: random = K% = K%
    ax.plot([0, 100], [0, 100], 'k:', alpha=0.3, linewidth=1, label='_nolegend_')

    ax.set_xlabel('Top K% Customers Targeted', fontsize=12)
    ax.set_ylabel('Revenue Captured (%)', fontsize=12)
    ax.set_title('Revenue Capture @ K Curve\n(Mean ± Std across 3 walk-forward windows)',
                  fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)

    # Plot 2: Lift curve
    ax = axes[1]
    for model in avg_curves.keys():
        if model == 'Random':
            continue
        mean_curve = avg_curves[model]
        lift = np.array(mean_curve) / np.array(k_values)
        ax.plot(k_values, lift, marker='o' if model == 'Hurdle (Proposed)' else None,
                label=model, color=colors[model], linewidth=2)

    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Random (Lift=1)')
    ax.set_xlabel('Top K% Customers Targeted', fontsize=12)
    ax.set_ylabel('Lift (Revenue Capture / K%)', fontsize=12)
    ax.set_title('Targeting Lift Curve', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'revenue_capture_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: revenue_capture_curve.png")

    # Print summary
    print("\n" + "="*70)
    print("[KEY INSIGHTS]")
    print("="*70)

    hurdle = avg_curves['Hurdle (Proposed)']
    random = avg_curves['Random']
    oracle = avg_curves['Oracle']

    print(f"\nAt K=10%:")
    print(f"  Random:           {random[2]:.1f}%")
    print(f"  Hurdle (ours):    {hurdle[2]:.1f}%")
    print(f"  Oracle:           {oracle[2]:.1f}%")
    print(f"  Hurdle efficiency: {100 * (hurdle[2] - random[2]) / (oracle[2] - random[2]):.1f}% of oracle gain")

    print(f"\nAt K=20%:")
    print(f"  Random:           {random[4]:.1f}%")
    print(f"  Hurdle (ours):    {hurdle[4]:.1f}%")
    print(f"  Oracle:           {oracle[4]:.1f}%")
    print(f"  Hurdle efficiency: {100 * (hurdle[4] - random[4]) / (oracle[4] - random[4]):.1f}% of oracle gain")

    print("\n" + "="*70)
    print("[DONE] Revenue capture curve generated")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_capture_curve()
