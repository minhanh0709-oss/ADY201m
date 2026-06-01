"""
61_models_mcd.py
SOTA Implementation: Monte Carlo Dropout (MCD)
- Bayesian uncertainty estimation via dropout at inference
- Provides prediction intervals + point estimates
- Reference: arXiv 2411.15944 (2024)
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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

DATA_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"


def load_windows():
    with open(DATA_PROCESSED_DIR / 'walk_forward_windows.pkl', 'rb') as f:
        return pickle.load(f)


def get_features(features_df):
    feature_cols = ['Recency', 'Frequency', 'Monetary', 'Tenure',
                    'ActiveMonths', 'ProductDiversity', 'AvgOrderValue',
                    'AvgDaysBetweenOrders', 'Regularity', 'IsUK']
    X = features_df[feature_cols].fillna(0).copy()
    X['log_Monetary'] = np.log1p(X['Monetary'])
    X['log_Frequency'] = np.log1p(X['Frequency'])
    X['log_AvgOrderValue'] = np.log1p(X['AvgOrderValue'])
    X['M_per_F'] = X['Monetary'] / np.maximum(X['Frequency'], 1)
    return X.values


# ============================================================
# MCD ZILN Model
# ============================================================
class MCDZILNModel(nn.Module):
    """ZILN with Monte Carlo Dropout for uncertainty"""
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.4):
        super().__init__()
        self.dropout_rate = dropout
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, 3)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    def predict_clv(self, x):
        logits = self.forward(x)
        p = torch.sigmoid(logits[:, 0])
        mu = torch.clamp(logits[:, 1], min=-10, max=15)
        sigma = torch.clamp(F.softplus(logits[:, 2]) + 1e-4, min=0.01, max=3.0)
        expected = p * torch.exp(mu + sigma ** 2 / 2)
        return torch.clamp(expected, min=0, max=200000)

    def predict_with_uncertainty(self, x, n_samples=100):
        """Keep dropout ACTIVE at inference time"""
        self.train()  # Enable dropout
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.predict_clv(x)
                preds.append(pred.cpu().numpy())
        preds = np.stack(preds, axis=0)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        ci_low = np.percentile(preds, 2.5, axis=0)
        ci_high = np.percentile(preds, 97.5, axis=0)
        return mean, std, ci_low, ci_high, preds


def ziln_loss(logits, labels):
    p_logit = logits[:, 0]
    mu = torch.clamp(logits[:, 1], min=-10, max=15)
    sigma = torch.clamp(F.softplus(logits[:, 2]) + 1e-4, min=0.01, max=3.0)
    labels_pos = (labels > 0).float()
    bce = F.binary_cross_entropy_with_logits(p_logit, labels_pos, reduction='none')
    safe_labels = torch.where(labels > 0, labels, torch.ones_like(labels))
    log_labels = torch.log(safe_labels)
    lognormal_nll = (
        log_labels + torch.log(sigma * np.sqrt(2 * np.pi))
        + (log_labels - mu) ** 2 / (2 * sigma ** 2)
    )
    return (bce + lognormal_nll * labels_pos).mean()


def train_mcd(X_train, y_train, X_val, y_val, input_dim, dropout=0.4,
              epochs=150, batch_size=64, lr=0.001, device='cpu'):
    torch.manual_seed(42)
    np.random.seed(42)

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)

    model = MCDZILNModel(input_dim, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = ziln_loss(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = ziln_loss(val_logits, y_val_t).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 25:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def evaluate_coverage(y_true, ci_low, ci_high):
    """Check how often actual values fall in predicted CI"""
    in_ci = (y_true >= ci_low) & (y_true <= ci_high)
    return in_ci.mean() * 100  # Should be ~95% if calibrated


def plot_uncertainty(y_test, mean_pred, ci_low, ci_high, save_path, n_show=200):
    """Plot predictions with uncertainty bands"""
    # Sort by true value
    sort_idx = np.argsort(y_test)[::-1][:n_show]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Sorted predictions with intervals
    ax = axes[0]
    x = np.arange(n_show)
    y_true_sorted = y_test[sort_idx]
    pred_sorted = mean_pred[sort_idx]
    ci_low_sorted = ci_low[sort_idx]
    ci_high_sorted = ci_high[sort_idx]

    ax.fill_between(x, ci_low_sorted, ci_high_sorted, alpha=0.3, color='#2E86AB',
                     label='95% CI')
    ax.plot(x, pred_sorted, color='#2E86AB', linewidth=1, label='MCD Mean')
    ax.scatter(x, y_true_sorted, s=10, color='#C73E1D', alpha=0.7, label='Actual', zorder=3)
    ax.set_xlabel('Customer Rank (top 200 by actual CLV)', fontsize=11)
    ax.set_ylabel('CLV ($)', fontsize=11)
    ax.set_yscale('symlog')
    ax.set_title(f'MCD Predictions with 95% CI (Top {n_show} Customers)',
                  fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Plot 2: CI width vs prediction
    ax = axes[1]
    ci_width = ci_high - ci_low
    ax.scatter(mean_pred, ci_width, alpha=0.4, s=12, color='#A23B72')
    ax.set_xlabel('Predicted CLV (Mean)', fontsize=11)
    ax.set_ylabel('95% CI Width', fontsize=11)
    ax.set_title('Uncertainty vs Predicted Value', fontsize=12, fontweight='bold')
    ax.set_xscale('symlog')
    ax.set_yscale('symlog')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_mcd():
    print("\n" + "="*70)
    print("[SOTA EXPERIMENT 2] Monte Carlo Dropout (MCD)")
    print("Bayesian uncertainty estimation for CLV predictions")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    windows = load_windows()
    all_results = []

    for window in windows:
        print(f"\n{'='*70}")
        print(f"Window {window['window_id']}: pred={window['pred_start']} to {window['pred_end']}")
        print(f"{'='*70}")

        features = window['features']
        X = get_features(features)
        y = features['ActualCLV'].values

        idx = np.arange(len(features))
        train_idx, test_idx = train_test_split(
            idx, test_size=0.2, random_state=42, stratify=features['IsVIP']
        )
        proper_idx, rest_idx = train_test_split(
            train_idx, test_size=0.30, random_state=42
        )
        cal_idx, val_idx = train_test_split(rest_idx, test_size=0.5, random_state=42)

        X_train, X_val, X_test = X[proper_idx], X[val_idx], X[test_idx]
        X_cal = X[cal_idx]
        y_train, y_val, y_test = y[proper_idx], y[val_idx], y[test_idx]
        y_cal = y[cal_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_cal_s = scaler.transform(X_cal)
        X_test_s = scaler.transform(X_test)

        print(f"  Training MCD-ZILN (dropout=0.4)...")
        model = train_mcd(X_train_s, y_train, X_val_s, y_val,
                          input_dim=X_train_s.shape[1], dropout=0.4,
                          epochs=200, batch_size=64, lr=0.001, device=device)

        # MCD inference with 100 samples
        print(f"  Running MCD inference (100 samples)...")
        X_cal_t = torch.FloatTensor(X_cal_s).to(device)
        X_test_t = torch.FloatTensor(X_test_s).to(device)
        _, _, ci_lo_cal, ci_hi_cal, _ = model.predict_with_uncertainty(X_cal_t, n_samples=100)
        mean_pred, std_pred, ci_low, ci_high, all_preds = model.predict_with_uncertainty(
            X_test_t, n_samples=100
        )

        # Split-conformal widening (alpha=0.05, same protocol as CQR)
        scores = np.maximum(ci_lo_cal - y_cal, y_cal - ci_hi_cal)
        q_hat = float(np.quantile(scores, 0.95, method='higher'))
        ci_lo_conf = np.maximum(0, ci_low - q_hat)
        ci_hi_conf = ci_high + q_hat

        # Evaluate point estimates
        metrics = comprehensive_metrics(y_test, mean_pred)
        print(f"  Point estimates:")
        print(f"    MAE: ${metrics['MAE']:,.2f} | R²: {metrics['R2']:.4f} | "
              f"Norm_Gini: {metrics['Norm_Gini']:.4f}")

        # Evaluate uncertainty quality
        coverage_95 = evaluate_coverage(y_test, ci_low, ci_high)
        coverage_conf = evaluate_coverage(y_test, ci_lo_conf, ci_hi_conf)
        mean_ci_width = (ci_high - ci_low).mean()
        mean_ci_width_conf = (ci_hi_conf - ci_lo_conf).mean()
        print(f"  Uncertainty:")
        print(f"    95% CI coverage (raw MC): {coverage_95:.1f}%")
        print(f"    95% CI coverage (conformal): {coverage_conf:.1f}% (q_hat={q_hat:.1f})")
        print(f"    Mean CI width: ${mean_ci_width:,.2f} -> ${mean_ci_width_conf:,.2f}")

        row = {'Window': window['window_id'], 'Model': 'MCD-ZILN'}
        row.update(metrics)
        row['Coverage_95'] = coverage_95
        row['Coverage_Conformal_95'] = coverage_conf
        row['Conformal_q_hat'] = q_hat
        row['Mean_CI_Width'] = mean_ci_width
        row['Mean_CI_Width_Conformal'] = mean_ci_width_conf
        all_results.append(row)

        # Plot uncertainty for Window 3
        if window['window_id'] == 3:
            plot_uncertainty(y_test, mean_pred, ci_low, ci_high,
                             FIGURES_DIR / 'mcd_uncertainty.png')
            print(f"  Saved: mcd_uncertainty.png")

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(RESULTS_DIR / 'mcd_walkforward.csv', index=False)

    print("\n" + "="*70)
    print("[SUMMARY] MCD-ZILN Results")
    print("="*70)
    print(f"\nMCD-ZILN")
    print(f"  MAE:                 ${df_results['MAE'].mean():.2f} ± {df_results['MAE'].std():.2f}")
    print(f"  R²:                  {df_results['R2'].mean():.4f} ± {df_results['R2'].std():.4f}")
    print(f"  Norm Gini:           {df_results['Norm_Gini'].mean():.4f} ± {df_results['Norm_Gini'].std():.4f}")
    print(f"  Revenue Capture@10%: {df_results['Revenue_Capture_10'].mean():.2f}% ± {df_results['Revenue_Capture_10'].std():.2f}%")
    print(f"  95% CI Coverage (raw):        {df_results['Coverage_95'].mean():.1f}% ± {df_results['Coverage_95'].std():.1f}%")
    print(f"  95% CI Coverage (conformal):  {df_results['Coverage_Conformal_95'].mean():.1f}% ± {df_results['Coverage_Conformal_95'].std():.1f}%")
    print(f"  Mean CI Width:       ${df_results['Mean_CI_Width'].mean():.2f}")

    print("\n" + "="*70)
    print("[DONE] MCD evaluation complete")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_mcd()
