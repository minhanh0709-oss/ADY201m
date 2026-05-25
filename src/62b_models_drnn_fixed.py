"""
62b_models_drnn_fixed.py
Fixed dRNN with:
  1. Robust output clamping on raw log-space predictions
  2. Tighter gradient clipping (0.5 instead of 1.0)
  3. Learning-rate warmup via linear schedule
  4. Median-based outlier filter on per-sample losses
  5. Runs on the 5-fold pkl if available, else falls back to 3-fold

Root cause of Window-1 instability:
  - Sequences of length 12 with ~79% within-sequence zeros
  - ZILN loss: lognormal NLL on positive samples can explode when
    predicted sigma is small and log(y) - mu is large
  - Fix: clip per-sample loss before mean, add huber-like saturation
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
RESULTS_DIR        = Path(__file__).parent.parent / "results"


def load_windows():
    p5 = DATA_PROCESSED_DIR / 'walk_forward_windows_5fold.pkl'
    p3 = DATA_PROCESSED_DIR / 'walk_forward_windows.pkl'
    pkl = p5 if p5.exists() else p3
    with open(pkl, 'rb') as f:
        return pickle.load(f)


# ── Model (identical architecture to original) ────────────────────────────────

class DilatedLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, dilation=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.dilation    = dilation
        self.lstm        = nn.LSTMCell(input_size, hidden_size)

    def forward(self, x, prev_h, prev_c):
        seq_len = x.size(1)
        h_states, c_states = list(prev_h), list(prev_c)
        outputs = []
        for t in range(seq_len):
            h_prev = h_states[-self.dilation]
            c_prev = c_states[-self.dilation]
            h_new, c_new = self.lstm(x[:, t, :], (h_prev, c_prev))
            h_states.append(h_new)
            c_states.append(c_new)
            outputs.append(h_new)
        return torch.stack(outputs, dim=1), h_states, c_states


class DRNNCLVModel(nn.Module):
    def __init__(self, seq_dim, static_dim, hidden_size=32, dilations=(1, 2, 4)):
        super().__init__()
        self.hidden_size = hidden_size
        self.dilations   = dilations
        self.input_proj  = nn.Linear(seq_dim, hidden_size)
        self.dilated_cells = nn.ModuleList(
            [DilatedLSTMCell(hidden_size, hidden_size, d) for d in dilations]
        )
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, 32), nn.LayerNorm(32), nn.ReLU(), nn.Dropout(0.2),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size + 32, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.head = nn.Linear(32, 3)  # p, mu, sigma

    def forward(self, seq, static):
        B = seq.size(0)
        x = self.input_proj(seq)
        for cell in self.dilated_cells:
            h0 = [torch.zeros(B, self.hidden_size, device=seq.device)
                  for _ in range(cell.dilation)]
            c0 = [torch.zeros(B, self.hidden_size, device=seq.device)
                  for _ in range(cell.dilation)]
            x, _, _ = cell(x, h0, c0)
        seq_feat    = x[:, -1, :]
        static_feat = self.static_encoder(static)
        h = self.fusion(torch.cat([seq_feat, static_feat], dim=1))
        return self.head(h)

    @torch.no_grad()
    def predict_clv(self, seq, static):
        logits = self.forward(seq, static)
        p      = torch.sigmoid(logits[:, 0])
        mu     = torch.clamp(logits[:, 1], -8, 12)
        sigma  = torch.clamp(F.softplus(logits[:, 2]) + 1e-4, 0.05, 2.5)
        ev     = p * torch.exp(mu + sigma ** 2 / 2)
        return torch.clamp(ev, 0, 150_000)


def robust_ziln_loss(logits, labels, clip_percentile=95):
    """ZILN loss with per-sample loss clipping to prevent explosion."""
    p_logit    = logits[:, 0]
    mu         = torch.clamp(logits[:, 1], -8, 12)
    sigma      = torch.clamp(F.softplus(logits[:, 2]) + 1e-4, 0.05, 2.5)
    labels_pos = (labels > 0).float()

    bce = F.binary_cross_entropy_with_logits(p_logit, labels_pos, reduction='none')

    safe_labels = torch.where(labels > 0, labels, torch.ones_like(labels))
    log_y       = torch.log(safe_labels)
    lognorm_nll = (
        log_y
        + torch.log(sigma * (2 * np.pi) ** 0.5)
        + (log_y - mu) ** 2 / (2 * sigma ** 2)
    )
    per_sample  = bce + lognorm_nll * labels_pos

    # Clip at high percentile to discard blow-up samples
    if clip_percentile < 100:
        threshold   = torch.quantile(per_sample.detach(), clip_percentile / 100.0)
        per_sample  = torch.clamp(per_sample, max=threshold.item())

    return per_sample.mean()


def train_drnn(seq_tr, stat_tr, y_tr, seq_va, stat_va, y_va,
               seq_dim, static_dim, epochs=150, batch=64, lr=5e-4, device='cpu'):
    torch.manual_seed(42); np.random.seed(42)

    def to_t(x, y=False):
        t = torch.FloatTensor(x).to(device)
        return t

    S_tr, T_tr, Y_tr = to_t(seq_tr), to_t(stat_tr), to_t(y_tr)
    S_va, T_va, Y_va = to_t(seq_va), to_t(stat_va), to_t(y_va)

    model = DRNNCLVModel(seq_dim, static_dim, hidden_size=32, dilations=(1, 2, 4)).to(device)
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Linear LR warmup for 10 epochs, then cosine decay
    def lr_lambda(epoch):
        warmup = 10
        if epoch < warmup:
            return epoch / warmup
        progress = (epoch - warmup) / max(epochs - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    loader    = DataLoader(TensorDataset(S_tr, T_tr, Y_tr), batch_size=batch, shuffle=True)
    best_loss = float('inf')
    best_state = None
    patience = 0

    for epoch in range(epochs):
        model.train()
        for bs, bt, by in loader:
            opt.zero_grad()
            loss = robust_ziln_loss(model(bs, bt), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            opt.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            vl = robust_ziln_loss(model(S_va, T_va), Y_va).item()
        if vl < best_loss:
            best_loss  = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience   = 0
        else:
            patience += 1
            if patience >= 25:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def run_drnn_fixed():
    print("\n" + "="*70)
    print("[dRNN-FIXED] Dilated RNN with robust training")
    print("="*70)

    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    windows = load_windows()
    all_res = []

    for win in windows:
        wid = win['window_id']
        print(f"\nWindow {wid}: {win['pred_start']} -> {win['pred_end']}")

        feats    = win['features']
        seq      = np.stack([win['revenue_seq'], win['frequency_seq']], axis=-1)
        seq[:, :, 0] = np.log1p(seq[:, :, 0])

        static_cols = ['Recency', 'Frequency', 'Monetary', 'Tenure',
                        'ActiveMonths', 'ProductDiversity', 'AvgOrderValue',
                        'AvgDaysBetweenOrders', 'Regularity', 'IsUK']
        static  = feats[static_cols].fillna(0).values
        y       = feats['ActualCLV'].values

        idx = np.arange(len(feats))
        tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=42,
                                           stratify=feats['IsVIP'])
        tr_idx, va_idx = train_test_split(tr_idx, test_size=0.15, random_state=42)

        ss = StandardScaler()
        stat_tr = ss.fit_transform(static[tr_idx])
        stat_va = ss.transform(static[va_idx])
        stat_te = ss.transform(static[te_idx])

        seq_m = seq[tr_idx].mean((0, 1), keepdims=True)
        seq_s = seq[tr_idx].std((0, 1), keepdims=True) + 1e-6
        seq_tr = (seq[tr_idx] - seq_m) / seq_s
        seq_va = (seq[va_idx] - seq_m) / seq_s
        seq_te = (seq[te_idx] - seq_m) / seq_s

        print(f"  Train={len(tr_idx):,} Val={len(va_idx):,} Test={len(te_idx):,}")

        model = train_drnn(seq_tr, stat_tr, y[tr_idx],
                            seq_va, stat_va, y[va_idx],
                            seq_dim=2, static_dim=stat_tr.shape[1],
                            epochs=150, batch=64, lr=5e-4, device=device)

        model.eval()
        with torch.no_grad():
            y_pred = model.predict_clv(
                torch.FloatTensor(seq_te).to(device),
                torch.FloatTensor(stat_te).to(device)
            ).cpu().numpy()

        y_test  = y[te_idx]
        metrics = comprehensive_metrics(y_test, y_pred)
        print(f"  MAE={metrics['MAE']:,.0f}  NG={metrics['Norm_Gini']:.4f}  "
              f"RC@10={metrics['Revenue_Capture_10']:.2f}%")

        row = {'Window': wid, 'Model': 'dRNN-Fixed'}
        row.update(metrics)
        all_res.append(row)

    df = pd.DataFrame(all_res)
    out = RESULTS_DIR / 'drnn_fixed_walkforward.csv'
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}")

    print("\n[SUMMARY] dRNN-Fixed")
    print(f"  MAE: ${df['MAE'].mean():,.0f} ± {df['MAE'].std():,.0f}")
    print(f"  NG:  {df['Norm_Gini'].mean():.4f} ± {df['Norm_Gini'].std():.4f}")
    print(f"  RC@10: {df['Revenue_Capture_10'].mean():.2f}% ± {df['Revenue_Capture_10'].std():.2f}%")
    print("="*70)


if __name__ == "__main__":
    run_drnn_fixed()
