"""
32_models_ziln.py
PHASE B1: Zero-Inflated Lognormal (ZILN) Neural Network
Based on Wang et al. (2019) - Google's deep probabilistic model for CLV.

ZILN handles:
1. Zero-inflation: 50% of customers have CLV=0
2. Heavy-tail: Skewness=23 for non-zero CLV

Loss = BCE(zero/non-zero) + Lognormal NLL(when y > 0)
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

# Try PyTorch first
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARN] PyTorch not installed, using numpy implementation")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

DATA_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_windows():
    with open(DATA_PROCESSED_DIR / 'walk_forward_windows.pkl', 'rb') as f:
        return pickle.load(f)


def get_features_for_dl(features_df):
    """Get feature matrix for neural network"""
    feature_cols = ['Recency', 'Frequency', 'Monetary', 'Tenure',
                    'ActiveMonths', 'ProductDiversity', 'AvgOrderValue',
                    'AvgDaysBetweenOrders', 'Regularity', 'IsUK']
    X = features_df[feature_cols].fillna(0).copy()

    # Log features (NN benefits from log-scale)
    X['log_Monetary'] = np.log1p(X['Monetary'])
    X['log_Frequency'] = np.log1p(X['Frequency'])
    X['log_AvgOrderValue'] = np.log1p(X['AvgOrderValue'])
    X['M_per_F'] = X['Monetary'] / np.maximum(X['Frequency'], 1)

    return X.values


# ============================================================
# ZILN Loss in PyTorch
# ============================================================
if HAS_TORCH:

    class ZILNLoss(nn.Module):
        """
        Zero-Inflated Lognormal Loss
        Output: logits[:, 0] = p (probability of positive)
                logits[:, 1] = mu (mean of lognormal)
                logits[:, 2] = sigma (raw, will be softplus)
        """
        def __init__(self):
            super().__init__()

        def forward(self, logits, labels):
            # Split logits
            p_logit = logits[:, 0]
            mu = logits[:, 1]
            sigma_raw = logits[:, 2]
            sigma = torch.nn.functional.softplus(sigma_raw) + 1e-4

            # Binary classification (zero vs non-zero)
            labels_pos = (labels > 0).float()
            bce_loss = nn.functional.binary_cross_entropy_with_logits(
                p_logit, labels_pos, reduction='none'
            )

            # Lognormal NLL (only for positive labels)
            safe_labels = torch.where(labels > 0, labels, torch.ones_like(labels))
            log_labels = torch.log(safe_labels)

            # Lognormal: log p(y) = -log(y) - log(sigma * sqrt(2pi)) - (log(y)-mu)^2 / (2 sigma^2)
            log_y = torch.log(safe_labels)
            lognormal_nll = (
                log_labels
                + torch.log(sigma * np.sqrt(2 * np.pi))
                + (log_y - mu) ** 2 / (2 * sigma ** 2)
            )

            regression_loss = lognormal_nll * labels_pos

            return (bce_loss + regression_loss).mean()


    class ZILNModel(nn.Module):
        """ZILN Neural Network"""
        def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.3):
            super().__init__()
            layers = []
            prev_dim = input_dim
            for h in hidden_dims:
                layers.append(nn.Linear(prev_dim, h))
                layers.append(nn.BatchNorm1d(h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                prev_dim = h

            self.backbone = nn.Sequential(*layers)
            # 3 outputs: p, mu, sigma
            self.head = nn.Linear(prev_dim, 3)

        def forward(self, x):
            features = self.backbone(x)
            return self.head(features)

        def predict_clv(self, x):
            """E[Y] for ZILN: p * exp(mu + sigma^2/2)"""
            logits = self.forward(x)
            p = torch.sigmoid(logits[:, 0])
            # Clip mu and sigma for numerical stability
            mu = torch.clamp(logits[:, 1], min=-10, max=15)
            sigma = torch.clamp(
                torch.nn.functional.softplus(logits[:, 2]) + 1e-4,
                min=0.01, max=3.0
            )
            expected = p * torch.exp(mu + sigma ** 2 / 2)
            # Clip to reasonable range (max 10x of typical CLV)
            expected = torch.clamp(expected, min=0, max=200000)
            return expected


def train_ziln(X_train, y_train, X_val, y_val, X_test, y_test,
               input_dim, epochs=100, batch_size=64, lr=0.001, device='cpu'):
    """Train ZILN model"""
    if not HAS_TORCH:
        return None

    torch.manual_seed(42)
    np.random.seed(42)

    # To tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)

    # Model + loss + optimizer
    model = ZILNModel(input_dim).to(device)
    criterion = ZILNLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # Train
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 20:
                break

    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)

    # Predict
    model.eval()
    with torch.no_grad():
        pred = model.predict_clv(X_test_t).cpu().numpy()

    return np.maximum(0, pred), model


def run_ziln():
    """Run ZILN model on walk-forward windows"""
    print("\n" + "="*70)
    print("[PHASE B1] ZILN NEURAL NETWORK")
    print("="*70)

    if not HAS_TORCH:
        print("[SKIP] PyTorch not available")
        return None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    windows = load_windows()
    all_results = []

    for window in windows:
        print(f"\n{'='*70}")
        print(f"Window {window['window_id']}: pred={window['pred_start']} to {window['pred_end']}")
        print(f"{'='*70}")

        features = window['features']
        X = get_features_for_dl(features)
        y = features['ActualCLV'].values

        # Train/val/test split
        np.random.seed(42)
        idx = np.arange(len(features))
        train_idx, test_idx = train_test_split(
            idx, test_size=0.2, random_state=42, stratify=features['IsVIP']
        )
        train_idx, val_idx = train_test_split(
            train_idx, test_size=0.15, random_state=42
        )

        X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
        y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
        print(f"  Input dim: {X_train_s.shape[1]}")
        print(f"  Training ZILN...")

        y_pred, model = train_ziln(
            X_train_s, y_train, X_val_s, y_val, X_test_s, y_test,
            input_dim=X_train_s.shape[1],
            epochs=150,
            batch_size=64,
            lr=0.001,
            device=device
        )

        metrics = comprehensive_metrics(y_test, y_pred)
        print(f"  MAE: ${metrics['MAE']:,.2f} | R²: {metrics['R2']:.4f} | "
              f"Norm_Gini: {metrics['Norm_Gini']:.4f} | "
              f"Revenue@10: {metrics['Revenue_Capture_10']:.2f}%")

        row = {'Window': window['window_id'], 'Model': 'ZILN (Deep Learning)'}
        row.update(metrics)
        all_results.append(row)

    # Save results
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(RESULTS_DIR / 'ziln_walkforward.csv', index=False)

    # Summary
    print("\n" + "="*70)
    print("[SUMMARY] ZILN Results")
    print("="*70)

    print(f"\nZILN (Deep Learning)")
    print(f"  MAE:                 ${df_results['MAE'].mean():.2f} ± {df_results['MAE'].std():.2f}")
    print(f"  R²:                  {df_results['R2'].mean():.4f} ± {df_results['R2'].std():.4f}")
    print(f"  Norm Gini:           {df_results['Norm_Gini'].mean():.4f} ± {df_results['Norm_Gini'].std():.4f}")
    print(f"  Revenue Capture@10%: {df_results['Revenue_Capture_10'].mean():.2f}% ± {df_results['Revenue_Capture_10'].std():.2f}%")
    print(f"  Top 5% MAPE:         {df_results['Top5_MAPE'].mean():.4f}")

    print("\n" + "="*70)
    print("[DONE] ZILN evaluation complete")
    print("="*70 + "\n")

    return df_results


if __name__ == "__main__":
    run_ziln()
