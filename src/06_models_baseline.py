"""
06_models_baseline.py
Simplified baseline models using pandas/numpy
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
    """Load data"""
    df = pd.read_csv(FINAL_FILE)
    return df

def evaluate_model(y_true, y_pred, model_name):
    """Evaluate model"""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    spearman, _ = spearmanr(y_true, y_pred)

    return {
        'Model': model_name,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Spearman': spearman
    }

def baseline_bgnbd(df):
    """Simple BG/NBD-like baseline using RFM"""
    print("\n[Model 1] BG/NBD-inspired (RFM-based)...")

    # Use monetary as proxy for CLV (simple baseline)
    y_pred = df['Monetary'].values
    metrics = evaluate_model(df['ActualCLV'].values, y_pred, 'RFM Monetary Baseline')
    print(f"  MAE: ${metrics['MAE']:,.2f}, RMSE: ${metrics['RMSE']:,.2f}, R2: {metrics['R2']:.4f}")

    return metrics, y_pred

def baseline_rfm_score(df):
    """RFM score baseline"""
    print("\n[Model 2] RFM Score Baseline...")

    # Normalize RFM
    r_norm = 1 - (df['Recency'] / df['Recency'].max())
    f_norm = df['Frequency'] / df['Frequency'].max()
    m_norm = df['Monetary'] / df['Monetary'].max()

    # RFM score
    rfm_score = (r_norm + f_norm + m_norm) / 3
    y_pred = rfm_score * df['Monetary'].max()

    metrics = evaluate_model(df['ActualCLV'].values, y_pred, 'RFM Score')
    print(f"  MAE: ${metrics['MAE']:,.2f}, RMSE: ${metrics['RMSE']:,.2f}, R2: {metrics['R2']:.4f}")

    return metrics, y_pred

def baseline_tenure_adjusted(df):
    """Tenure-adjusted monetary baseline"""
    print("\n[Model 3] Tenure-Adjusted Monetary...")

    # Adjust monetary by tenure
    y_pred = df['Monetary'] * (df['Tenure'] / df['Tenure'].max())

    metrics = evaluate_model(df['ActualCLV'].values, y_pred, 'Tenure-Adjusted')
    print(f"  MAE: ${metrics['MAE']:,.2f}, RMSE: ${metrics['RMSE']:,.2f}, R2: {metrics['R2']:.4f}")

    return metrics, y_pred

def baseline_average_churn(df):
    """Average-based baseline"""
    print("\n[Model 4] Average CLV Baseline...")

    # Simple average CLV
    avg_clv = df['ActualCLV'].mean()
    y_pred = np.full(len(df), avg_clv)

    metrics = evaluate_model(df['ActualCLV'].values, y_pred, 'Average CLV')
    print(f"  MAE: ${metrics['MAE']:,.2f}, RMSE: ${metrics['RMSE']:,.2f}, R2: {metrics['R2']:.4f}")

    return metrics, y_pred

def baseline_frequency_adjusted(df):
    """Frequency-adjusted baseline"""
    print("\n[Model 5] Frequency-Adjusted Baseline...")

    # Adjust average CLV by frequency
    clv_by_freq = df.groupby('Frequency')['ActualCLV'].transform('mean')
    y_pred = clv_by_freq.fillna(df['ActualCLV'].mean())

    metrics = evaluate_model(df['ActualCLV'].values, y_pred, 'Frequency-Adjusted')
    print(f"  MAE: ${metrics['MAE']:,.2f}, RMSE: ${metrics['RMSE']:,.2f}, R2: {metrics['R2']:.4f}")

    return metrics, y_pred

def main():
    print("\n" + "="*70)
    print("[BASELINE MODELS] Simple benchmarks")
    print("="*70)

    # Load
    df = load_data()
    print(f"\nData loaded: {len(df):,} customers")

    # Train models
    all_metrics = []

    metrics1, _ = baseline_bgnbd(df)
    all_metrics.append(metrics1)

    metrics2, _ = baseline_rfm_score(df)
    all_metrics.append(metrics2)

    metrics3, _ = baseline_tenure_adjusted(df)
    all_metrics.append(metrics3)

    metrics4, _ = baseline_average_churn(df)
    all_metrics.append(metrics4)

    metrics5, _ = baseline_frequency_adjusted(df)
    all_metrics.append(metrics5)

    # Save metrics
    print("\n" + "="*70)
    print("[STEP 4] Save baseline metrics")
    print("="*70)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(RESULTS_DIR / 'baseline_metrics.csv', index=False)
    print(f"\n[OK] Baseline metrics saved")
    print(metrics_df.to_string(index=False))

    print("\n" + "="*70)
    print("[DONE] Baseline models trained!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
