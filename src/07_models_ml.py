"""
07_models_ml.py
Machine Learning models using simple ensemble methods
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def spearmanr(a, b):
    """Simple Spearman correlation"""
    rank_a = pd.Series(a).rank()
    rank_b = pd.Series(b).rank()
    corr = np.corrcoef(rank_a, rank_b)[0, 1]
    return corr, None

# Paths
DATA_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
FINAL_FILE = DATA_PROCESSED_DIR / "customers_with_labels.csv"
RESULTS_DIR = Path(__file__).parent.parent / "results"

def load_data():
    df = pd.read_csv(FINAL_FILE)
    return df

def evaluate_model(y_true, y_pred, model_name):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    spearman, _ = spearmanr(y_true, y_pred)
    return {'Model': model_name, 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Spearman': spearman}

def model_ensemble_weighted(df):
    """Weighted ensemble of features"""
    print("\n[Model 1] Weighted Ensemble...")

    # Normalize features
    f_norm = df['Frequency'] / df['Frequency'].max()
    m_norm = df['Monetary'] / df['Monetary'].max()
    t_norm = df['Tenure'] / df['Tenure'].max()
    a_norm = df['AvgOrderValue'] / df['AvgOrderValue'].max()

    # Weighted average (tuned weights based on correlation)
    y_pred = 0.4 * m_norm + 0.3 * f_norm + 0.2 * a_norm + 0.1 * t_norm
    y_pred = y_pred * df['Monetary'].max()

    metrics = evaluate_model(df['ActualCLV'].values, y_pred, 'Weighted Ensemble')
    print(f"  MAE: ${metrics['MAE']:,.2f}, RMSE: ${metrics['RMSE']:,.2f}, R2: {metrics['R2']:.4f}")
    return metrics, y_pred

def model_segmented(df):
    """Segmented model based on customer frequency"""
    print("\n[Model 2] Segmented by Frequency...")

    y_pred = np.zeros(len(df))

    # Segment 1: Low frequency (1-2 purchases)
    low_freq = df['Frequency'] <= 2
    if low_freq.sum() > 0:
        mean_clv_low = df[low_freq]['ActualCLV'].mean()
        y_pred[low_freq] = mean_clv_low

    # Segment 2: Medium frequency (3-10 purchases)
    med_freq = (df['Frequency'] > 2) & (df['Frequency'] <= 10)
    if med_freq.sum() > 0:
        mean_clv_med = df[med_freq]['ActualCLV'].mean()
        y_pred[med_freq] = mean_clv_med

    # Segment 3: High frequency (>10 purchases)
    high_freq = df['Frequency'] > 10
    if high_freq.sum() > 0:
        mean_clv_high = df[high_freq]['ActualCLV'].mean()
        y_pred[high_freq] = mean_clv_high

    metrics = evaluate_model(df['ActualCLV'].values, y_pred, 'Segmented by Frequency')
    print(f"  MAE: ${metrics['MAE']:,.2f}, RMSE: ${metrics['RMSE']:,.2f}, R2: {metrics['R2']:.4f}")
    return metrics, y_pred

def model_linear_combination(df):
    """Simple linear combination of features"""
    print("\n[Model 3] Linear Combination...")

    # Simple linear regression manually
    feature_cols = ['Frequency', 'Monetary', 'Tenure', 'AvgOrderValue']
    X = df[feature_cols].values
    y = df['ActualCLV'].values

    # Add intercept
    X_with_intercept = np.column_stack([np.ones(len(X)), X])

    # Normal equation: beta = (X'X)^-1 X'y
    try:
        beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
        y_pred = X_with_intercept @ beta
    except:
        y_pred = df['Monetary'].values

    metrics = evaluate_model(y, y_pred, 'Linear Combination')
    print(f"  MAE: ${metrics['MAE']:,.2f}, RMSE: ${metrics['RMSE']:,.2f}, R2: {metrics['R2']:.4f}")
    return metrics, y_pred

def model_tree_simple(df):
    """Simple decision-tree-like split"""
    print("\n[Model 4] Simple Tree Split...")

    y_pred = np.zeros(len(df))

    # Split by Monetary value (simulating a tree split)
    for threshold in df['Monetary'].quantile([0.25, 0.5, 0.75]):
        mask = (df['Monetary'] >= threshold) & (df['Monetary'] < (threshold + df['Monetary'].max()/4))
        if mask.sum() > 0:
            y_pred[mask] = df[mask]['ActualCLV'].mean()

    # High spenders
    high_spenders = df['Monetary'] >= df['Monetary'].quantile(0.75)
    if high_spenders.sum() > 0:
        y_pred[high_spenders] = df[high_spenders]['ActualCLV'].mean()

    metrics = evaluate_model(df['ActualCLV'].values, y_pred, 'Simple Tree')
    print(f"  MAE: ${metrics['MAE']:,.2f}, RMSE: ${metrics['RMSE']:,.2f}, R2: {metrics['R2']:.4f}")
    return metrics, y_pred

def model_rfm_enhanced(df):
    """Enhanced RFM with multiplicative scaling"""
    print("\n[Model 5] RFM Enhanced...")

    # Normalize
    r = 1 - (df['Recency'] / df['Recency'].max())
    f = df['Frequency'] / df['Frequency'].max()
    m = df['Monetary'] / df['Monetary'].max()

    # Multiplicative RFM score
    rfm = r * f * m
    y_pred = rfm * df['Monetary'].max() * (1 + df['Tenure'] / df['Tenure'].max())

    metrics = evaluate_model(df['ActualCLV'].values, y_pred, 'RFM Enhanced')
    print(f"  MAE: ${metrics['MAE']:,.2f}, RMSE: ${metrics['RMSE']:,.2f}, R2: {metrics['R2']:.4f}")
    return metrics, y_pred

def main():
    print("\n" + "="*70)
    print("[MACHINE LEARNING MODELS] Ensemble & Statistical Methods")
    print("="*70)

    df = load_data()
    print(f"\nData: {len(df):,} customers")

    all_metrics = []

    metrics1, _ = model_ensemble_weighted(df)
    all_metrics.append(metrics1)

    metrics2, _ = model_segmented(df)
    all_metrics.append(metrics2)

    metrics3, _ = model_linear_combination(df)
    all_metrics.append(metrics3)

    metrics4, _ = model_tree_simple(df)
    all_metrics.append(metrics4)

    metrics5, _ = model_rfm_enhanced(df)
    all_metrics.append(metrics5)

    # Save
    print("\n" + "="*70)
    print("[SAVE RESULTS]")
    print("="*70)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_file = RESULTS_DIR / 'ml_metrics.csv'
    metrics_df.to_csv(metrics_file, index=False)

    print(f"\n[OK] ML metrics saved: {metrics_file}")
    print(metrics_df.to_string(index=False))

    print("\n" + "="*70)
    print("[DONE] ML models trained!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
