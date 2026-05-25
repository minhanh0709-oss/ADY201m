"""
21_utils_cv_metrics.py
PHASE A2 + A3: Walk-forward CV + SOTA evaluation metrics
- Walk-forward time-series cross-validation
- Normalized Gini coefficient
- Decile MAPE
- Top-K MAPE
- Revenue Capture@K
- Precision@TopK
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from scipy.stats import spearmanr, ttest_rel
import warnings
warnings.filterwarnings('ignore')

DATA_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


# ============================================================
# WALK-FORWARD CV
# ============================================================
def get_walk_forward_splits():
    """
    Generate 3 walk-forward windows for temporal validation.
    Each window: (obs_start, obs_end, pred_start, pred_end)

    Dataset span: 2009-12-01 to 2011-12-09 (~24 months)
    """
    splits = [
        # Window 1: Train 12 months -> Predict 3 months
        ('2009-12-01', '2010-11-30', '2010-12-01', '2011-02-28'),
        # Window 2: Train 15 months -> Predict 3 months
        ('2009-12-01', '2011-02-28', '2011-03-01', '2011-05-31'),
        # Window 3: Train 18 months -> Predict 6 months
        ('2009-12-01', '2011-05-31', '2011-06-01', '2011-12-09'),
    ]
    return splits


# ============================================================
# EVALUATION METRICS
# ============================================================
def regression_metrics(y_true, y_pred):
    """Standard regression metrics"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Handle NaN
    mask = ~(np.isnan(y_pred) | np.isnan(y_true))
    y_true_valid = y_true[mask]
    y_pred_valid = y_pred[mask]

    if len(y_true_valid) == 0:
        return {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'Spearman': np.nan}

    mae = np.mean(np.abs(y_true_valid - y_pred_valid))
    rmse = np.sqrt(np.mean((y_true_valid - y_pred_valid) ** 2))

    ss_res = np.sum((y_true_valid - y_pred_valid) ** 2)
    ss_tot = np.sum((y_true_valid - np.mean(y_true_valid)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    try:
        sp, _ = spearmanr(y_true_valid, y_pred_valid)
    except:
        sp = np.nan

    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Spearman': sp}


def gini_coefficient(y_true, y_pred):
    """Compute Gini coefficient"""
    assert len(y_true) == len(y_pred)
    arr = np.array([y_true, y_pred]).T
    arr = arr[arr[:, 1].argsort()]  # sort by prediction
    cum_actual = np.cumsum(arr[:, 0])
    sum_actual = arr[:, 0].sum()
    if sum_actual == 0:
        return 0
    lorenz = cum_actual / sum_actual
    gini = (np.sum(lorenz) - len(arr) / 2) / len(arr)
    return gini


def normalized_gini(y_true, y_pred):
    """
    Normalized Gini coefficient (Google's recommendation for CLV)
    Range: -1 (worst) to 1 (perfect)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    actual_gini = gini_coefficient(y_true, y_pred)
    perfect_gini = gini_coefficient(y_true, y_true)
    if perfect_gini == 0:
        return 0
    return actual_gini / perfect_gini


def decile_metrics(y_true, y_pred, n_deciles=10):
    """
    Decile chart metrics - calibration check
    Split predictions into deciles, check actual vs predicted within each
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    df = pd.DataFrame({'pred': y_pred, 'actual': y_true})
    df['decile'] = pd.qcut(df['pred'].rank(method='first'), n_deciles, labels=False)

    decile_summary = df.groupby('decile').agg({
        'pred': 'mean',
        'actual': ['mean', 'sum', 'count']
    })

    # Decile MAPE
    decile_means_actual = df.groupby('decile')['actual'].mean()
    decile_means_pred = df.groupby('decile')['pred'].mean()
    decile_mape = np.mean(
        np.abs(decile_means_actual - decile_means_pred) /
        np.maximum(decile_means_actual, 1)
    )

    return decile_summary, decile_mape


def top_k_mape(y_true, y_pred, k_pct=0.05):
    """
    Top K% MAPE - focuses on highest-value customers (VIPs)
    aSMAPE variant: floor of $1 to avoid division by zero
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    n_top = int(len(y_pred) * k_pct)
    top_indices = np.argsort(y_pred)[-n_top:]

    actual_top = y_true[top_indices]
    pred_top = y_pred[top_indices]

    # aSMAPE with floor at $1 to handle zeros
    denominator = np.maximum((np.abs(actual_top) + np.abs(pred_top)) / 2, 1.0)
    mape = np.mean(np.abs(actual_top - pred_top) / denominator)
    return mape


def revenue_capture_at_k(y_true, y_pred, k_pct=0.10):
    """
    Revenue Capture@K - business metric
    What % of total revenue is captured by top K% predicted customers?
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    n_top = max(1, int(len(y_pred) * k_pct))
    top_indices = np.argsort(y_pred)[-n_top:]

    top_revenue = y_true[top_indices].sum()
    total_revenue = y_true.sum()

    if total_revenue == 0:
        return 0
    return 100 * top_revenue / total_revenue


def lift_at_k(y_true, y_pred, k_pct=0.10):
    """Lift vs random targeting"""
    capture = revenue_capture_at_k(y_true, y_pred, k_pct)
    return capture / (k_pct * 100)


def precision_at_k(y_true, y_pred, k_pct=0.10):
    """
    Precision@TopK%
    Of top K% predicted, how many are actually in top K%?
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    n_top = max(1, int(len(y_pred) * k_pct))
    top_pred_idx = set(np.argsort(y_pred)[-n_top:])
    top_actual_idx = set(np.argsort(y_true)[-n_top:])

    intersect = len(top_pred_idx & top_actual_idx)
    return intersect / n_top


def comprehensive_metrics(y_true, y_pred):
    """All metrics in one call"""
    reg = regression_metrics(y_true, y_pred)

    metrics = {
        **reg,
        'Norm_Gini': normalized_gini(y_true, y_pred),
        'Top5_MAPE': top_k_mape(y_true, y_pred, 0.05),
        'Top10_MAPE': top_k_mape(y_true, y_pred, 0.10),
        'Revenue_Capture_10': revenue_capture_at_k(y_true, y_pred, 0.10),
        'Revenue_Capture_20': revenue_capture_at_k(y_true, y_pred, 0.20),
        'Lift_10': lift_at_k(y_true, y_pred, 0.10),
        'Precision_10': precision_at_k(y_true, y_pred, 0.10),
    }

    _, decile_mape = decile_metrics(y_true, y_pred)
    metrics['Decile_MAPE'] = decile_mape

    return metrics


# ============================================================
# STATISTICAL TESTING
# ============================================================
def paired_ttest(scores_a, scores_b, alpha=0.05):
    """
    Paired t-test for comparing models across CV folds
    Returns: (statistic, p-value, significant)
    """
    if len(scores_a) < 2 or len(scores_b) < 2:
        return None, None, False

    stat, p_value = ttest_rel(scores_a, scores_b)
    significant = p_value < alpha
    return stat, p_value, significant


# ============================================================
# TEST
# ============================================================
def test():
    """Test all metrics with synthetic data"""
    print("\n" + "="*70)
    print("[TEST] Utility Functions")
    print("="*70)

    np.random.seed(42)
    n = 1000

    # Test data: some zeros (zero-inflated), some big spenders (heavy-tail)
    y_true = np.concatenate([
        np.zeros(500),  # 50% zeros
        np.random.lognormal(5, 1.5, 500)  # Heavy-tail positive values
    ])
    np.random.shuffle(y_true)

    # Test 1: Random predictions
    y_pred_random = np.random.uniform(0, 1000, n)
    print("\n[Test 1] Random predictions:")
    metrics = comprehensive_metrics(y_true, y_pred_random)
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.4f}")

    # Test 2: Good predictions (correlated with truth)
    y_pred_good = y_true + np.random.normal(0, 50, n)
    y_pred_good = np.maximum(0, y_pred_good)
    print("\n[Test 2] Good predictions:")
    metrics = comprehensive_metrics(y_true, y_pred_good)
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.4f}")

    # Test 3: Perfect predictions
    print("\n[Test 3] Perfect predictions:")
    metrics = comprehensive_metrics(y_true, y_true)
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.4f}")

    # Test walk-forward
    print("\n[Test 4] Walk-forward splits:")
    for i, (a, b, c, d) in enumerate(get_walk_forward_splits()):
        print(f"  Window {i+1}: Train [{a} -> {b}], Test [{c} -> {d}]")

    print("\n" + "="*70)
    print("[OK] All utility functions working")
    print("="*70 + "\n")


if __name__ == "__main__":
    test()
