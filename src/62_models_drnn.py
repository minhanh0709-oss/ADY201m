"""
62_models_drnn.py
SOTA Implementation: Dilated RNN (dRNN)
- Multi-layer RNN with dilated connections (delayed states)
- Custom cells with residual connections
- Reference: arXiv 2412.20295 (2024) - Meta/Uber CLV paper

Combines monthly sequence features with static RFM features.
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


# ============================================================
# Dilated LSTM Cell
# ============================================================
class DilatedLSTMCell(nn.Module):
    """LSTM cell with dilated connections"""
    def __init__(self, input_size, hidden_size, dilation=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.dilation = dilation
        self.lstm = nn.LSTMCell(input_size, hidden_size)

    def forward(self, x, prev_h_states, prev_c_states):
        """
        x: [batch, seq_len, input_size]
        prev_h_states: list of [batch, hidden_size] (length = dilation)
        prev_c_states: list of [batch, hidden_size]
        """
        seq_len = x.size(1)
        outputs = []
        h_states = list(prev_h_states)
        c_states = list(prev_c_states)

        for t in range(seq_len):
            # Use state from dilation steps back
            h_prev = h_states[-self.dilation]
            c_prev = c_states[-self.dilation]
            h_new, c_new = self.lstm(x[:, t, :], (h_prev, c_prev))
            h_states.append(h_new)
            c_states.append(c_new)
            outputs.append(h_new)

        return torch.stack(outputs, dim=1), h_states, c_states


# ============================================================
# dRNN Model (combines sequence + static features)
# ============================================================
class DRNNCLVModel(nn.Module):
    """Dilated RNN for CLV prediction with hybrid sequence+static features"""
    def __init__(self, seq_dim, static_dim, hidden_size=32, dilations=[1, 2, 4]):
        super().__init__()
        self.hidden_size = hidden_size
        self.dilations = dilations

        # Input projection for sequence
        self.input_proj = nn.Linear(seq_dim, hidden_size)

        # Stacked dilated LSTM layers
        self.dilated_cells = nn.ModuleList([
            DilatedLSTMCell(hidden_size, hidden_size, d) for d in dilations
        ])

        # Static features encoder
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Fusion + output head (ZILN: 3 outputs)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.head = nn.Linear(32, 3)  # p, mu, sigma for ZILN

    def forward(self, seq, static):
        """
        seq: [batch, seq_len, seq_dim]
        static: [batch, static_dim]
        """
        batch_size = seq.size(0)

        # Project sequence
        x = self.input_proj(seq)  # [batch, seq_len, hidden]

        # Pass through dilated LSTM layers
        for cell in self.dilated_cells:
            # Initialize h, c states (need dilation copies)
            h_init = [torch.zeros(batch_size, self.hidden_size, device=seq.device)
                      for _ in range(cell.dilation)]
            c_init = [torch.zeros(batch_size, self.hidden_size, device=seq.device)
                      for _ in range(cell.dilation)]
            x, _, _ = cell(x, h_init, c_init)

        # Take last hidden state
        seq_features = x[:, -1, :]  # [batch, hidden]

        # Encode static
        static_features = self.static_encoder(static)  # [batch, 32]

        # Fuse
        combined = torch.cat([seq_features, static_features], dim=1)
        h = self.fusion(combined)
        logits = self.head(h)
        return logits

    def predict_clv(self, seq, static):
        with torch.no_grad():
            logits = self.forward(seq, static)
            p = torch.sigmoid(logits[:, 0])
            mu = torch.clamp(logits[:, 1], min=-10, max=15)
            sigma = torch.clamp(F.softplus(logits[:, 2]) + 1e-4, min=0.01, max=3.0)
            expected = p * torch.exp(mu + sigma ** 2 / 2)
            return torch.clamp(expected, min=0, max=200000)


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


def train_drnn(seq_train, static_train, y_train,
                seq_val, static_val, y_val,
                seq_dim, static_dim,
                epochs=100, batch_size=64, lr=0.001, device='cpu'):
    torch.manual_seed(42)
    np.random.seed(42)

    seq_train_t = torch.FloatTensor(seq_train).to(device)
    static_train_t = torch.FloatTensor(static_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)

    seq_val_t = torch.FloatTensor(seq_val).to(device)
    static_val_t = torch.FloatTensor(static_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)

    model = DRNNCLVModel(seq_dim, static_dim, hidden_size=32, dilations=[1, 2, 4]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)

    dataset = TensorDataset(seq_train_t, static_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for batch_seq, batch_static, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_seq, batch_static)
            loss = ziln_loss(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(seq_val_t, static_val_t)
            val_loss = ziln_loss(val_logits, y_val_t).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 20:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def run_drnn():
    print("\n" + "="*70)
    print("[SOTA EXPERIMENT 3] Dilated RNN (dRNN)")
    print("Multi-scale temporal modeling with dilated LSTM cells")
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
        n_months = window['revenue_seq'].shape[1]

        # Sequence features: [n_customers, n_months, 2] (revenue + frequency)
        seq = np.stack([
            window['revenue_seq'],
            window['frequency_seq']
        ], axis=-1)  # [batch, time, 2]

        # Log transform for revenue
        seq[:, :, 0] = np.log1p(seq[:, :, 0])

        # Static features
        static_cols = ['Recency', 'Frequency', 'Monetary', 'Tenure',
                        'ActiveMonths', 'ProductDiversity', 'AvgOrderValue',
                        'AvgDaysBetweenOrders', 'Regularity', 'IsUK']
        static = features[static_cols].fillna(0).values

        y = features['ActualCLV'].values

        idx = np.arange(len(features))
        train_idx, test_idx = train_test_split(
            idx, test_size=0.2, random_state=42, stratify=features['IsVIP']
        )
        train_idx, val_idx = train_test_split(train_idx, test_size=0.15, random_state=42)

        seq_train, seq_val, seq_test = seq[train_idx], seq[val_idx], seq[test_idx]
        static_train, static_val, static_test = static[train_idx], static[val_idx], static[test_idx]
        y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

        # Scale static features
        static_scaler = StandardScaler()
        static_train_s = static_scaler.fit_transform(static_train)
        static_val_s = static_scaler.transform(static_val)
        static_test_s = static_scaler.transform(static_test)

        # Scale sequence features
        seq_mean = seq_train.mean(axis=(0, 1), keepdims=True)
        seq_std = seq_train.std(axis=(0, 1), keepdims=True) + 1e-6
        seq_train_s = (seq_train - seq_mean) / seq_std
        seq_val_s = (seq_val - seq_mean) / seq_std
        seq_test_s = (seq_test - seq_mean) / seq_std

        print(f"  Train: {len(seq_train):,} | Val: {len(seq_val):,} | Test: {len(seq_test):,}")
        print(f"  Seq shape: {seq_train_s.shape} | Static dim: {static_train_s.shape[1]}")
        print(f"  Training dRNN (dilations=[1,2,4])...")

        model = train_drnn(
            seq_train_s, static_train_s, y_train,
            seq_val_s, static_val_s, y_val,
            seq_dim=2, static_dim=static_train_s.shape[1],
            epochs=100, batch_size=64, lr=0.001, device=device
        )

        # Predict
        model.eval()
        seq_test_t = torch.FloatTensor(seq_test_s).to(device)
        static_test_t = torch.FloatTensor(static_test_s).to(device)
        y_pred = model.predict_clv(seq_test_t, static_test_t).cpu().numpy()

        metrics = comprehensive_metrics(y_test, y_pred)
        print(f"  MAE: ${metrics['MAE']:,.2f} | R²: {metrics['R2']:.4f} | "
              f"Norm_Gini: {metrics['Norm_Gini']:.4f} | "
              f"Revenue@10: {metrics['Revenue_Capture_10']:.2f}%")

        row = {'Window': window['window_id'], 'Model': 'dRNN (Dilated)'}
        row.update(metrics)
        all_results.append(row)

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(RESULTS_DIR / 'drnn_walkforward.csv', index=False)

    print("\n" + "="*70)
    print("[SUMMARY] dRNN Results")
    print("="*70)
    print(f"\ndRNN (Dilated)")
    print(f"  MAE:                 ${df_results['MAE'].mean():.2f} ± {df_results['MAE'].std():.2f}")
    print(f"  R²:                  {df_results['R2'].mean():.4f} ± {df_results['R2'].std():.4f}")
    print(f"  Norm Gini:           {df_results['Norm_Gini'].mean():.4f} ± {df_results['Norm_Gini'].std():.4f}")
    print(f"  Revenue Capture@10%: {df_results['Revenue_Capture_10'].mean():.2f}% ± {df_results['Revenue_Capture_10'].std():.2f}%")
    print(f"  Top 5% MAPE:         {df_results['Top5_MAPE'].mean():.4f}")

    print("\n" + "="*70)
    print("[DONE] dRNN evaluation complete")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_drnn()
