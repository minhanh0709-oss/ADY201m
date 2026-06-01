"""
51_calibration_decile.py
TASK 2: Decile chart + Calibration plot
Google's recommendation for CLV evaluation (Wang et al. 2019).
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


def train_hurdle(X_train, y_train, X_test):
    """Train Hurdle and predict"""
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
    residual_var = np.var(y_val - val_pred)
    correction = np.exp(residual_var / 2)
    pred = prob_positive * np.expm1(log_pred) * correction
    return np.maximum(0, pred)


def decile_analysis(y_true, y_pred, n_deciles=10):
    """Compute decile statistics"""
    df = pd.DataFrame({'pred': y_pred, 'actual': y_true})
    df['decile'] = pd.qcut(df['pred'].rank(method='first'),
                            n_deciles, labels=list(range(1, n_deciles+1)))

    decile_stats = df.groupby('decile').agg({
        'pred': ['mean', 'min', 'max'],
        'actual': ['mean', 'sum', 'count']
    })
    decile_stats.columns = ['pred_mean', 'pred_min', 'pred_max',
                             'actual_mean', 'actual_sum', 'count']
    decile_stats = decile_stats.reset_index()
    decile_stats['cumulative_actual'] = decile_stats.sort_values('decile', ascending=False)['actual_sum'].cumsum()[::-1]
    decile_stats['cumulative_pct'] = 100 * decile_stats['cumulative_actual'] / df['actual'].sum()

    return decile_stats


def plot_decile_chart(decile_stats, save_path):
    """Decile chart: bar chart of mean per decile"""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(decile_stats))
    width = 0.35

    bars1 = ax.bar(x - width/2, decile_stats['pred_mean'], width,
                    label='Predicted', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, decile_stats['actual_mean'], width,
                    label='Actual', color='#A23B72', alpha=0.8)

    ax.set_xlabel('Decile (1=lowest predicted, 10=highest predicted)', fontsize=11)
    ax.set_ylabel('Mean CLV ($)', fontsize=11)
    ax.set_title('Decile Chart: Calibration Quality (Hurdle Model, Window 3)',
                  fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(decile_stats['decile'])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (p, a) in enumerate(zip(decile_stats['pred_mean'], decile_stats['actual_mean'])):
        if p > 50 or a > 50:
            ax.text(i - width/2, p, f'${p:.0f}', ha='center', va='bottom', fontsize=7)
            ax.text(i + width/2, a, f'${a:.0f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_calibration_curve(decile_stats, save_path):
    """Calibration curve: predicted vs actual per decile"""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot predictions vs actuals
    ax.scatter(decile_stats['pred_mean'], decile_stats['actual_mean'],
                s=200, c=decile_stats['decile'], cmap='viridis',
                edgecolor='black', linewidth=1.5, zorder=3)

    # Perfect calibration line
    max_val = max(decile_stats['pred_mean'].max(), decile_stats['actual_mean'].max())
    ax.plot([0, max_val * 1.1], [0, max_val * 1.1], 'r--', linewidth=2,
            label='Perfect Calibration', zorder=1)

    # Annotate deciles
    for _, row in decile_stats.iterrows():
        ax.annotate(f"D{row['decile']}",
                    (row['pred_mean'], row['actual_mean']),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=9, fontweight='bold')

    ax.set_xlabel('Predicted Mean CLV per Decile ($)', fontsize=11)
    ax.set_ylabel('Actual Mean CLV per Decile ($)', fontsize=11)
    ax.set_title('Calibration Plot: Hurdle Model (Window 3)',
                  fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_val * 1.15)
    ax.set_ylim(0, max_val * 1.15)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_predicted_vs_actual(y_true, y_pred, save_path):
    """Scatter plot of predictions vs actuals (sampled)"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Log scale (for skewed data)
    ax = axes[0]
    mask = (y_true > 0) & (y_pred > 0)
    ax.scatter(np.log1p(y_true[mask]), np.log1p(y_pred[mask]),
                alpha=0.4, s=15, color='#2E86AB')

    # Perfect line
    max_val = max(np.log1p(y_true[mask]).max(), np.log1p(y_pred[mask]).max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect')

    ax.set_xlabel('log(1 + Actual CLV)', fontsize=11)
    ax.set_ylabel('log(1 + Predicted CLV)', fontsize=11)
    ax.set_title(f'Predicted vs Actual (log scale, n={mask.sum()})',
                  fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Linear scale (capped for visibility)
    ax = axes[1]
    cap = 10000  # Cap for visualization
    y_true_cap = np.clip(y_true, 0, cap)
    y_pred_cap = np.clip(y_pred, 0, cap)

    ax.scatter(y_true_cap, y_pred_cap, alpha=0.4, s=15, color='#A23B72')
    ax.plot([0, cap], [0, cap], 'r--', linewidth=2, label='Perfect')
    ax.set_xlabel('Actual CLV ($, capped at $10K)', fontsize=11)
    ax.set_ylabel('Predicted CLV ($, capped at $10K)', fontsize=11)
    ax.set_title(f'Predicted vs Actual (linear scale, capped)',
                  fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, cap * 1.1)
    ax.set_ylim(0, cap * 1.1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_calibration():
    print("\n" + "="*70)
    print("[TASK 2] CALIBRATION + DECILE ANALYSIS")
    print("="*70)

    windows = load_windows()
    window = windows[2]  # Window 3 (largest, most data)
    print(f"\nUsing Window {window['window_id']}: pred={window['pred_start']} to {window['pred_end']}")

    features = window['features']
    seq_data = {
        'revenue_seq': window['revenue_seq'],
        'frequency_seq': window['frequency_seq']
    }
    X, _ = get_features(features, seq_data)
    y = features['ActualCLV'].values

    train_idx, test_idx = train_test_split(
        range(len(features)), test_size=0.2, random_state=42,
        stratify=features['IsVIP']
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train and predict
    print(f"\n[Training Hurdle Model]")
    y_pred = train_hurdle(X_train, y_train, X_test)

    # Decile analysis
    print(f"\n[Decile Analysis]")
    decile_stats = decile_analysis(y_test, y_pred)
    print(decile_stats.to_string(index=False))

    decile_stats.to_csv(RESULTS_DIR / 'decile_analysis.csv', index=False)

    # Plot decile chart
    print(f"\n[Plotting decile chart]")
    plot_decile_chart(decile_stats, FIGURES_DIR / 'decile_chart.png')
    print(f"  Saved: decile_chart.png")

    # Calibration curve
    print(f"\n[Plotting calibration curve]")
    plot_calibration_curve(decile_stats, FIGURES_DIR / 'calibration_curve.png')
    print(f"  Saved: calibration_curve.png")

    # Predicted vs Actual
    print(f"\n[Plotting predicted vs actual]")
    plot_predicted_vs_actual(y_test, y_pred, FIGURES_DIR / 'predicted_vs_actual.png')
    print(f"  Saved: predicted_vs_actual.png")

    # Compute calibration metrics
    print(f"\n[Calibration metrics]")
    # Decile MAPE
    decile_mape_vals = np.abs(decile_stats['pred_mean'] - decile_stats['actual_mean']) / np.maximum(decile_stats['actual_mean'], 1)
    decile_mape = decile_mape_vals.mean()
    print(f"  Decile MAPE: {decile_mape:.4f}")

    # Top decile capture
    top_decile_actual = decile_stats[decile_stats['decile'] == 10]['actual_sum'].iloc[0]
    total_actual = decile_stats['actual_sum'].sum()
    top_decile_capture = 100 * top_decile_actual / total_actual
    print(f"  Top decile captures {top_decile_capture:.2f}% of total revenue")

    # Bottom decile
    bot_decile_actual = decile_stats[decile_stats['decile'] == 1]['actual_sum'].iloc[0]
    bot_decile_capture = 100 * bot_decile_actual / total_actual
    print(f"  Bottom decile captures {bot_decile_capture:.2f}% of total revenue")

    print(f"  Ratio top/bottom: {top_decile_capture/max(bot_decile_capture, 0.01):.1f}x")

    print("\n" + "="*70)
    print("[DONE] Calibration analysis complete")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_calibration()
