"""
Pilot Optuna tuning for ZILN on Window 3 only (10 trials).
Compares best-tuned NG vs fixed-hyperparameter baseline from 32_models_ziln.py.
"""
import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
exec(open(Path(__file__).parent / "21_utils_cv_metrics.py").read())

# Load ZILN module without triggering its __main__ block
_ziln_path = Path(__file__).parent / "32_models_ziln.py"
_spec = importlib.util.spec_from_file_location("ziln_mod", _ziln_path)
_ziln = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ziln)
ZILNModel = _ziln.ZILNModel
ZILNLoss = _ziln.ZILNLoss
train_ziln = _ziln.train_ziln
get_features_for_dl = _ziln.get_features_for_dl
normalized_gini = normalized_gini  # from utils exec

DATA_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"
N_TRIALS = 10
WINDOW = 3
SEED = 42


def load_window_data(window_id=3):
    with open(DATA_PROCESSED_DIR / "walk_forward_windows.pkl", "rb") as f:
        windows = pickle.load(f)
    w = windows[window_id - 1]
    features_df = w["features"]
    y = features_df["ActualCLV"].values
    X = get_features_for_dl(features_df)
    is_vip = features_df["IsVIP"].values
    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(
        idx, test_size=0.2, random_state=SEED, stratify=is_vip
    )
    tr_idx, va_idx = train_test_split(
        tr_idx, test_size=0.15, random_state=SEED, stratify=is_vip[tr_idx]
    )
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[tr_idx])
    X_va = scaler.transform(X[va_idx])
    X_te = scaler.transform(X[te_idx])
    return X_tr, y[tr_idx], X_va, y[va_idx], X_te, y[te_idx], X_tr.shape[1]


def norm_gini(y_true, y_pred):
    return normalized_gini(y_true, y_pred)


def objective(trial, X_tr, y_tr, X_va, y_va, input_dim, device="cpu"):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    h1 = trial.suggest_categorical("h1", [64, 128, 256])
    h2 = trial.suggest_categorical("h2", [32, 64, 128])

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = ZILNModel(input_dim, hidden_dims=[h1, h2, 32], dropout=dropout).to(device)
    criterion = ZILNLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    X_tr_t = torch.FloatTensor(X_tr).to(device)
    y_tr_t = torch.FloatTensor(y_tr).to(device)
    X_va_t = torch.FloatTensor(X_va).to(device)
    y_va_t = torch.FloatTensor(y_va).to(device)

    dataset = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_ng = -1.0
    patience = 0
    best_state = None
    for _ in range(80):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            pred = model.predict_clv(X_va_t).cpu().numpy()
        ng = norm_gini(y_va, pred)
        if ng > best_ng:
            best_ng = ng
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 12:
                break
    if best_state:
        model.load_state_dict(best_state)
    return best_ng


def eval_fixed(X_tr, y_tr, X_va, y_va, X_te, y_te, input_dim, device="cpu"):
    pred, _model = train_ziln(
        X_tr, y_tr, X_va, y_va, X_te, y_te,
        input_dim=input_dim, epochs=200, batch_size=64, lr=0.001, device=device,
    )
    if pred is None:
        return None, None
    return norm_gini(y_te, pred), pred


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_tr, y_tr, X_va, y_va, X_te, y_te, input_dim = load_window_data(WINDOW)

    fixed_ng, _ = eval_fixed(X_tr, y_tr, X_va, y_va, X_te, y_te, input_dim, device)
    print(f"Fixed ZILN W{WINDOW} test NG: {fixed_ng:.4f}")

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(
        lambda t: objective(t, X_tr, y_tr, X_va, y_va, input_dim, device),
        n_trials=N_TRIALS,
        show_progress_bar=False,
    )

    best = study.best_trial
    torch.manual_seed(SEED)
    model = ZILNModel(
        input_dim,
        hidden_dims=[best.params["h1"], best.params["h2"], 32],
        dropout=best.params["dropout"],
    ).to(device)
    criterion = ZILNLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best.params["lr"], weight_decay=1e-5)
    X_tr_t = torch.FloatTensor(X_tr).to(device)
    y_tr_t = torch.FloatTensor(y_tr).to(device)
    X_te_t = torch.FloatTensor(X_te).to(device)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr_t, y_tr_t),
        batch_size=best.params["batch_size"],
        shuffle=True,
    )
    for _ in range(120):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    with torch.no_grad():
        tuned_pred = model.predict_clv(X_te_t).cpu().numpy()
    tuned_ng = norm_gini(y_te, tuned_pred)

    row = {
        "window": WINDOW,
        "fixed_test_ng": fixed_ng,
        "optuna_val_ng": best.value,
        "optuna_test_ng": tuned_ng,
        "n_trials": N_TRIALS,
        **best.params,
    }
    out = RESULTS_DIR / "ziln_optuna_pilot_w3.csv"
    pd.DataFrame([row]).to_csv(out, index=False)
    print(f"Optuna best val NG: {best.value:.4f}, test NG: {tuned_ng:.4f}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
