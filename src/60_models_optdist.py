"""
60_models_optdist.py
SOTA Implementation: OptDist (Tang et al., CIKM 2024)
- Multiple Sub-Distribution Networks (SDNs), each learning a ZILN
- Distribution Selection Module (DSM) with Gumbel-softmax
- Adapts distribution based on customer characteristics

Paper: "OptDist: Learning Optimal Distribution for CLV Prediction"
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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

DATA_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_windows():
    with open(DATA_PROCESSED_DIR / 'walk_forward_windows.pkl', 'rb') as f:
        return pickle.load(f)


def get_features_for_dl(features_df):
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
# OptDist Architecture
# ============================================================
class SubDistributionNetwork(nn.Module):
    """Single ZILN sub-distribution head"""
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
        )
        # Outputs: p, mu, sigma
        self.head = nn.Linear(hidden_dim // 2, 3)

    def forward(self, shared_features):
        h = self.layers(shared_features)
        return self.head(h)


class DistributionSelectionModule(nn.Module):
    """Selects which sub-distribution to use via Gumbel-softmax"""
    def __init__(self, input_dim, n_sub_distributions=3, hidden_dim=32):
        super().__init__()
        self.n_sub = n_sub_distributions
        self.selector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_sub_distributions),
        )

    def forward(self, x, tau=1.0, hard=False):
        logits = self.selector(x)
        # Gumbel-softmax for differentiable sampling
        return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)


class OptDistModel(nn.Module):
    """OptDist: Multi-distribution CLV prediction"""
    def __init__(self, input_dim, n_sub=3, shared_hidden=64):
        super().__init__()
        self.n_sub = n_sub

        # Shared representation
        self.shared = nn.Sequential(
            nn.Linear(input_dim, shared_hidden),
            nn.BatchNorm1d(shared_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Multiple sub-distribution networks (each is ZILN)
        self.sub_networks = nn.ModuleList([
            SubDistributionNetwork(shared_hidden) for _ in range(n_sub)
        ])

        # Distribution selection module
        self.selector = DistributionSelectionModule(shared_hidden, n_sub)

    def forward(self, x, tau=1.0, hard=False):
        # Shared representation
        shared = self.shared(x)

        # Get logits from each sub-distribution
        all_logits = torch.stack([sdn(shared) for sdn in self.sub_networks], dim=1)
        # shape: [batch, n_sub, 3]

        # Selection weights via Gumbel-softmax
        weights = self.selector(shared, tau=tau, hard=hard)
        # shape: [batch, n_sub]

        # Weighted combination of sub-distribution logits
        weighted_logits = (all_logits * weights.unsqueeze(-1)).sum(dim=1)
        # shape: [batch, 3]

        return weighted_logits, weights, all_logits

    def predict_clv(self, x):
        with torch.no_grad():
            logits, _, _ = self.forward(x, tau=0.5, hard=False)
            p = torch.sigmoid(logits[:, 0])
            mu = torch.clamp(logits[:, 1], min=-10, max=15)
            sigma = torch.clamp(F.softplus(logits[:, 2]) + 1e-4, min=0.01, max=3.0)
            expected = p * torch.exp(mu + sigma ** 2 / 2)
            return torch.clamp(expected, min=0, max=200000)


class OptDistLoss(nn.Module):
    """OptDist loss: ZILN + alignment loss"""
    def __init__(self, alignment_weight=0.1):
        super().__init__()
        self.alignment_weight = alignment_weight

    def ziln_loss(self, logits, labels):
        p_logit = logits[:, 0]
        mu = torch.clamp(logits[:, 1], min=-10, max=15)
        sigma = torch.clamp(F.softplus(logits[:, 2]) + 1e-4, min=0.01, max=3.0)

        labels_pos = (labels > 0).float()
        bce = F.binary_cross_entropy_with_logits(p_logit, labels_pos, reduction='none')

        safe_labels = torch.where(labels > 0, labels, torch.ones_like(labels))
        log_labels = torch.log(safe_labels)
        lognormal_nll = (
            log_labels
            + torch.log(sigma * np.sqrt(2 * np.pi))
            + (log_labels - mu) ** 2 / (2 * sigma ** 2)
        )
        return (bce + lognormal_nll * labels_pos).mean()

    def forward(self, weighted_logits, weights, all_logits, labels):
        # Main ZILN loss on combined distribution
        main_loss = self.ziln_loss(weighted_logits, labels)

        # Alignment loss: each sub-distribution should also be reasonable
        sub_losses = []
        for i in range(all_logits.shape[1]):
            sub_loss = self.ziln_loss(all_logits[:, i, :], labels)
            sub_losses.append(sub_loss)

        alignment_loss = torch.stack(sub_losses).mean()

        # Entropy regularization: prevent collapse to single distribution
        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()
        entropy_loss = -entropy * 0.01  # negative because we want to MAXIMIZE entropy

        return main_loss + self.alignment_weight * alignment_loss + entropy_loss


def train_optdist(X_train, y_train, X_val, y_val, X_test, y_test,
                   input_dim, n_sub=3, epochs=150, batch_size=64, lr=0.001, device='cpu'):
    """Train OptDist model"""
    torch.manual_seed(42)
    np.random.seed(42)

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)

    model = OptDistModel(input_dim, n_sub=n_sub).to(device)
    criterion = OptDistLoss(alignment_weight=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float('inf')
    best_state = None
    patience = 0

    for epoch in range(epochs):
        model.train()
        # Anneal temperature for Gumbel-softmax
        tau = max(0.5, 1.0 - epoch / epochs)

        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            weighted_logits, weights, all_logits = model(batch_x, tau=tau, hard=False)
            loss = criterion(weighted_logits, weights, all_logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            wl, w, al = model(X_val_t, tau=0.5, hard=False)
            val_loss = criterion(wl, w, al, y_val_t).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 25:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Predict
    model.eval()
    pred = model.predict_clv(X_test_t).cpu().numpy()

    # Get distribution usage stats
    with torch.no_grad():
        _, weights, _ = model(X_test_t, tau=0.5, hard=False)
        usage = weights.cpu().numpy().mean(axis=0)

    return np.maximum(0, pred), usage


def run_optdist():
    print("\n" + "="*70)
    print("[SOTA EXPERIMENT 1] OptDist (CIKM 2024)")
    print("Multi-distribution ZILN with Gumbel-softmax selection")
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
        X = get_features_for_dl(features)
        y = features['ActualCLV'].values

        idx = np.arange(len(features))
        train_idx, test_idx = train_test_split(
            idx, test_size=0.2, random_state=42, stratify=features['IsVIP']
        )
        train_idx, val_idx = train_test_split(train_idx, test_size=0.15, random_state=42)

        X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
        y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
        print(f"  Training OptDist (3 sub-distributions)...")

        y_pred, usage = train_optdist(
            X_train_s, y_train, X_val_s, y_val, X_test_s, y_test,
            input_dim=X_train_s.shape[1], n_sub=3,
            epochs=200, batch_size=64, lr=0.001, device=device
        )

        metrics = comprehensive_metrics(y_test, y_pred)
        print(f"  MAE: ${metrics['MAE']:,.2f} | R²: {metrics['R2']:.4f} | "
              f"Norm_Gini: {metrics['Norm_Gini']:.4f} | "
              f"Revenue@10: {metrics['Revenue_Capture_10']:.2f}%")
        print(f"  Sub-distribution usage: {usage}")

        row = {'Window': window['window_id'], 'Model': 'OptDist (Multi-ZILN)'}
        row.update(metrics)
        all_results.append(row)

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(RESULTS_DIR / 'optdist_walkforward.csv', index=False)

    print("\n" + "="*70)
    print("[SUMMARY] OptDist Results")
    print("="*70)
    print(f"\nOptDist (Multi-ZILN)")
    print(f"  MAE:                 ${df_results['MAE'].mean():.2f} ± {df_results['MAE'].std():.2f}")
    print(f"  R²:                  {df_results['R2'].mean():.4f} ± {df_results['R2'].std():.4f}")
    print(f"  Norm Gini:           {df_results['Norm_Gini'].mean():.4f} ± {df_results['Norm_Gini'].std():.4f}")
    print(f"  Revenue Capture@10%: {df_results['Revenue_Capture_10'].mean():.2f}% ± {df_results['Revenue_Capture_10'].std():.2f}%")
    print(f"  Top 5% MAPE:         {df_results['Top5_MAPE'].mean():.4f}")

    print("\n" + "="*70)
    print("[DONE] OptDist evaluation complete")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_optdist()
