"""
53_bootstrap_importance.py
TASK 4: Bootstrap Confidence Intervals + Permutation Feature Importance
- Bootstrap resampling for stable metric estimates
- Permutation importance (more reliable than gain-based for tree models)
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
exec(open(Path(__file__).parent / '21_utils_cv_metrics.py').read())

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
    pred = np.maximum(0, prob * np.expm1(log_pred) * correction)
    return pred, clf, reg


def bootstrap_metric(y_true, y_pred, metric_fn, n_bootstrap=1000, seed=42):
    """Bootstrap confidence interval for a metric"""
    np.random.seed(seed)
    n = len(y_true)
    metrics_bootstrap = []

    for _ in range(n_bootstrap):
        # Sample with replacement
        idx = np.random.choice(n, n, replace=True)
        y_t = y_true[idx]
        y_p = y_pred[idx]
        try:
            m = metric_fn(y_t, y_p)
            if not np.isnan(m) and not np.isinf(m):
                metrics_bootstrap.append(m)
        except:
            pass

    if len(metrics_bootstrap) < 100:
        return None, None, None

    metrics_arr = np.array(metrics_bootstrap)
    mean = metrics_arr.mean()
    ci_low = np.percentile(metrics_arr, 2.5)
    ci_high = np.percentile(metrics_arr, 97.5)

    return mean, ci_low, ci_high


def permutation_importance(model_predict_fn, X, y, feature_names,
                            metric_fn=None, n_repeats=10, seed=42):
    """Compute permutation feature importance"""
    if metric_fn is None:
        metric_fn = lambda y_true, y_pred: normalized_gini(y_true, y_pred)

    np.random.seed(seed)
    baseline_pred = model_predict_fn(X)
    baseline = metric_fn(y, baseline_pred)

    importances = []
    for i, feat_name in enumerate(feature_names):
        scores = []
        for rep in range(n_repeats):
            X_perm = X.copy()
            np.random.seed(seed + rep)
            np.random.shuffle(X_perm[:, i])
            pred_perm = model_predict_fn(X_perm)
            score = metric_fn(y, pred_perm)
            scores.append(baseline - score)  # Higher = more important
        importances.append({
            'feature': feat_name,
            'importance_mean': np.mean(scores),
            'importance_std': np.std(scores),
        })

    return pd.DataFrame(importances).sort_values('importance_mean', ascending=False)


def run_bootstrap():
    print("\n" + "="*70)
    print("[TASK 4] BOOTSTRAP CI + FEATURE IMPORTANCE")
    print("="*70)

    windows = load_windows()

    # ====== PART 1: Bootstrap confidence intervals ======
    print("\n" + "="*70)
    print("[PART 1] Bootstrap Confidence Intervals (n=1000)")
    print("="*70)

    all_results = []

    for window in windows:
        print(f"\n[Window {window['window_id']}]")

        features = window['features']
        seq_data = {'revenue_seq': window['revenue_seq'],
                    'frequency_seq': window['frequency_seq']}
        X, feature_names = get_features(features, seq_data)
        y = features['ActualCLV'].values

        train_idx, test_idx = train_test_split(
            range(len(features)), test_size=0.2, random_state=42,
            stratify=features['IsVIP']
        )
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train Hurdle
        y_pred, clf, reg = train_hurdle(X_train, y_train, X_test)

        # Bootstrap metrics
        metrics_to_test = {
            'Norm_Gini': normalized_gini,
            'Revenue_Capture_10': lambda yt, yp: revenue_capture_at_k(yt, yp, 0.10),
            'Revenue_Capture_20': lambda yt, yp: revenue_capture_at_k(yt, yp, 0.20),
            'Lift_10': lambda yt, yp: lift_at_k(yt, yp, 0.10),
            'Top5_MAPE': lambda yt, yp: top_k_mape(yt, yp, 0.05),
        }

        for metric_name, metric_fn in metrics_to_test.items():
            mean_v, ci_low, ci_high = bootstrap_metric(y_test, y_pred, metric_fn, n_bootstrap=1000)
            if mean_v is not None:
                print(f"  {metric_name:20s}: {mean_v:.4f} [{ci_low:.4f}, {ci_high:.4f}]")
                all_results.append({
                    'Window': window['window_id'],
                    'Model': 'Hurdle',
                    'Metric': metric_name,
                    'Mean': mean_v,
                    'CI_Low_95': ci_low,
                    'CI_High_95': ci_high,
                })

    # Save
    df_bs = pd.DataFrame(all_results)
    df_bs.to_csv(RESULTS_DIR / 'bootstrap_ci.csv', index=False)

    # ====== PART 2: Permutation Feature Importance ======
    print("\n" + "="*70)
    print("[PART 2] Permutation Feature Importance")
    print("="*70)

    # Use Window 3 (largest data)
    window = windows[2]
    features = window['features']
    seq_data = {'revenue_seq': window['revenue_seq'],
                'frequency_seq': window['frequency_seq']}
    X, feature_names = get_features(features, seq_data)
    y = features['ActualCLV'].values

    train_idx, test_idx = train_test_split(
        range(len(features)), test_size=0.2, random_state=42,
        stratify=features['IsVIP']
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train Hurdle
    print(f"\n[Training Hurdle on Window 3]")
    y_pred, clf, reg = train_hurdle(X_train, y_train, X_test)

    # Define prediction function for permutation
    def hurdle_predict(X):
        prob = clf.predict_proba(X)[:, 1]
        log_pred = reg.predict(X)
        return np.maximum(0, prob * np.expm1(log_pred))

    # Compute permutation importance using Norm Gini
    print(f"\n[Computing permutation importance for Norm Gini]")
    print(f"  (n_repeats=10, may take a few minutes...)")
    imp_gini = permutation_importance(
        hurdle_predict, X_test, y_test, feature_names,
        metric_fn=normalized_gini, n_repeats=10
    )

    print(f"\nTop 15 features by permutation importance (Norm Gini drop):")
    for _, row in imp_gini.head(15).iterrows():
        print(f"  {row['feature']:30s}: {row['importance_mean']:.4f} ± {row['importance_std']:.4f}")

    imp_gini.to_csv(RESULTS_DIR / 'permutation_importance.csv', index=False)

    # ====== PART 3: Visualization ======
    print("\n" + "="*70)
    print("[PART 3] Visualizations")
    print("="*70)

    # Plot 1: Bootstrap CI plot
    fig, ax = plt.subplots(figsize=(12, 6))
    metric_show = ['Norm_Gini', 'Revenue_Capture_10', 'Revenue_Capture_20', 'Lift_10']

    plot_data = []
    for metric in metric_show:
        for window_id in [1, 2, 3]:
            row = df_bs[(df_bs['Metric'] == metric) & (df_bs['Window'] == window_id)]
            if len(row) > 0:
                plot_data.append({
                    'Metric': metric,
                    'Window': f'W{window_id}',
                    'Mean': row.iloc[0]['Mean'],
                    'CI_Low': row.iloc[0]['CI_Low_95'],
                    'CI_High': row.iloc[0]['CI_High_95'],
                })

    plot_df = pd.DataFrame(plot_data)

    # Group by metric
    metrics_unique = plot_df['Metric'].unique()
    n_windows = 3
    x_pos = np.arange(len(metrics_unique))
    width = 0.25
    colors_w = ['#2E86AB', '#A23B72', '#F18F01']

    for i, w in enumerate(['W1', 'W2', 'W3']):
        sub = plot_df[plot_df['Window'] == w]
        means = sub['Mean'].values
        ci_low = sub['CI_Low'].values
        ci_high = sub['CI_High'].values
        err_low = means - ci_low
        err_high = ci_high - means

        ax.bar(x_pos + (i - 1) * width, means, width, label=w, color=colors_w[i],
                yerr=[err_low, err_high], capsize=4, alpha=0.85)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics_unique, rotation=15)
    ax.set_ylabel('Metric Value', fontsize=11)
    ax.set_title('Hurdle Model: Bootstrap 95% Confidence Intervals (1000 resamples)',
                  fontsize=12, fontweight='bold')
    ax.legend(title='Window', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'bootstrap_ci.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: bootstrap_ci.png")

    # Plot 2: Permutation importance
    fig, ax = plt.subplots(figsize=(10, 8))
    top_features = imp_gini.head(15)
    colors_imp = ['#2E86AB' if 'seq_' in f or 'log_' in f or '_per_' in f or '_ratio' in f
                   else '#A23B72'
                   for f in top_features['feature']]

    ax.barh(top_features['feature'][::-1], top_features['importance_mean'][::-1],
            xerr=top_features['importance_std'][::-1],
            color=colors_imp[::-1], alpha=0.85,
            error_kw={'ecolor': 'black', 'capsize': 3})

    ax.set_xlabel('Permutation Importance (Norm Gini drop)', fontsize=11)
    ax.set_title('Top 15 Features by Permutation Importance\n(Hurdle Model, Window 3)',
                  fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Legend
    import matplotlib.patches as mpatches
    raw_patch = mpatches.Patch(color='#A23B72', label='Raw RFM/Behavioral')
    eng_patch = mpatches.Patch(color='#2E86AB', label='Engineered (log/seq/interactions)')
    ax.legend(handles=[raw_patch, eng_patch], loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'permutation_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: permutation_importance.png")

    # ====== Summary ======
    print("\n" + "="*70)
    print("[SUMMARY]")
    print("="*70)

    # Average CIs across windows
    print("\nAverage 95% CI across 3 windows (Hurdle Model):")
    for metric in metric_show:
        sub = df_bs[df_bs['Metric'] == metric]
        if len(sub) > 0:
            m = sub['Mean'].mean()
            ci_l = sub['CI_Low_95'].mean()
            ci_h = sub['CI_High_95'].mean()
            print(f"  {metric:20s}: {m:.4f} [95% CI: {ci_l:.4f}, {ci_h:.4f}]")

    # Top engineered features
    seq_features = imp_gini[imp_gini['feature'].str.startswith('seq_')]
    raw_features = imp_gini[~imp_gini['feature'].str.startswith('seq_') &
                             ~imp_gini['feature'].str.startswith('log_') &
                             ~imp_gini['feature'].str.contains('_per_') &
                             ~imp_gini['feature'].str.contains('_ratio')]

    print(f"\nSequence feature importance contribution:")
    print(f"  Top sequence features: {seq_features.head(3)['feature'].tolist()}")
    print(f"  Sum of all sequence feature importances: {seq_features['importance_mean'].sum():.4f}")
    print(f"  Sum of raw feature importances:         {raw_features['importance_mean'].sum():.4f}")

    print("\n" + "="*70)
    print("[DONE] Bootstrap + Permutation Importance complete")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_bootstrap()
