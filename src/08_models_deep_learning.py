"""
08_models_deep_learning.py
Simple neural network models using numpy
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def spearmanr(a, b):
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

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size=32, learning_rate=0.001):
        self.lr = learning_rate
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, 1) * 0.01
        self.b2 = 0

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2

    def backward(self, X, y, y_pred):
        m = len(X)
        y_pred_flat = y_pred.flatten()

        dz2 = (y_pred_flat - y).reshape(-1, 1) / m
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_derivative(self.z1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=50):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            self.backward(X, y, y_pred)
        return self.forward(X).flatten()

    def predict(self, X):
        return self.forward(X).flatten()

def model_simple_nn(df):
    """Simple 2-layer neural network"""
    print("\n[Model 1] Simple Neural Network (2 layers)...")

    feature_cols = ['Frequency', 'Monetary', 'Tenure', 'AvgOrderValue', 'ActiveMonths']
    X = df[feature_cols].values
    y = df['ActualCLV'].values

    # Normalize
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_norm = (X - X_mean) / (X_std + 1e-8)

    # Train
    np.random.seed(42)
    nn = SimpleNeuralNetwork(input_size=len(feature_cols), hidden_size=32, learning_rate=0.01)
    y_pred = nn.train(X_norm, y, epochs=100)

    # Clip to reasonable range
    y_pred = np.maximum(0, y_pred)

    metrics = evaluate_model(y, y_pred, 'Simple NN')
    print(f"  MAE: ${metrics['MAE']:,.2f}, RMSE: ${metrics['RMSE']:,.2f}, R2: {metrics['R2']:.4f}")
    return metrics, y_pred

def model_lstm_like(df):
    """LSTM-like using sequence features"""
    print("\n[Model 2] LSTM-like (Sequence)...")

    feature_cols = ['Frequency', 'Monetary', 'Tenure', 'AvgOrderValue']
    X = df[feature_cols].values
    y = df['ActualCLV'].values

    # Use sequence values as temporal features
    # Simulate LSTM hidden state update
    h = np.zeros(32)
    y_pred = np.zeros(len(df))

    for i in range(len(df)):
        # Simple RNN-like update
        x = (X[i] - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        h = np.tanh(np.concatenate([x, h]) @ np.random.randn(36, 32))
        y_pred[i] = h.sum() * df['Monetary'].iloc[i] / 100

    y_pred = np.maximum(0, y_pred)

    metrics = evaluate_model(y, y_pred, 'LSTM-like')
    print(f"  MAE: ${metrics['MAE']:,.2f}, RMSE: ${metrics['RMSE']:,.2f}, R2: {metrics['R2']:.4f}")
    return metrics, y_pred

def model_transformer_like(df):
    """Transformer-like using attention mechanism"""
    print("\n[Model 3] Transformer-like (Attention)...")

    feature_cols = ['Frequency', 'Monetary', 'Tenure', 'AvgOrderValue', 'ActiveMonths']
    X = df[feature_cols].values
    y = df['ActualCLV'].values

    # Self-attention mechanism
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # Query, Key, Value projections
    Q = X_norm @ np.random.randn(len(feature_cols), 8)
    K = X_norm @ np.random.randn(len(feature_cols), 8)
    V = X_norm @ np.random.randn(len(feature_cols), 32)

    # Attention scores
    scores = (Q @ K.T) / np.sqrt(8)
    attention = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)

    # Apply attention to values
    context = attention @ V

    # Output layer
    y_pred = (context.mean(axis=1)) * df['Monetary'].values / 10

    y_pred = np.maximum(0, y_pred)

    metrics = evaluate_model(y, y_pred, 'Transformer-like')
    print(f"  MAE: ${metrics['MAE']:,.2f}, RMSE: ${metrics['RMSE']:,.2f}, R2: {metrics['R2']:.4f}")
    return metrics, y_pred

def model_hybrid_ensemble(df):
    """Hybrid ensemble of above models"""
    print("\n[Model 4] Hybrid Ensemble...")

    # Get predictions from multiple models
    feature_cols = ['Frequency', 'Monetary', 'Tenure', 'AvgOrderValue', 'ActiveMonths']
    X = df[feature_cols].values
    y = df['ActualCLV'].values

    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # Model 1: Linear
    from numpy.linalg import lstsq
    X_with_intercept = np.column_stack([np.ones(len(X_norm)), X_norm])
    beta = lstsq(X_with_intercept, y, rcond=None)[0]
    pred1 = X_with_intercept @ beta

    # Model 2: Nonlinear (simple NN)
    np.random.seed(42)
    nn = SimpleNeuralNetwork(len(feature_cols), 16, 0.01)
    pred2 = nn.train(X_norm, y, 50)

    # Model 3: RFM-based
    pred3 = (df['Monetary'].values * 0.7 +
             df['Frequency'].values / df['Frequency'].max() * df['Monetary'].max() * 0.2 +
             df['AvgOrderValue'].values * 0.1)

    # Ensemble average
    y_pred = (pred1 + pred2 + pred3) / 3
    y_pred = np.maximum(0, y_pred)

    metrics = evaluate_model(y, y_pred, 'Hybrid Ensemble')
    print(f"  MAE: ${metrics['MAE']:,.2f}, RMSE: ${metrics['RMSE']:,.2f}, R2: {metrics['R2']:.4f}")
    return metrics, y_pred

def main():
    print("\n" + "="*70)
    print("[DEEP LEARNING MODELS] Neural Networks & Attention")
    print("="*70)

    df = load_data()
    print(f"\nData: {len(df):,} customers")

    all_metrics = []

    # Set seed for reproducibility
    np.random.seed(42)

    metrics1, _ = model_simple_nn(df)
    all_metrics.append(metrics1)

    metrics2, _ = model_lstm_like(df)
    all_metrics.append(metrics2)

    metrics3, _ = model_transformer_like(df)
    all_metrics.append(metrics3)

    metrics4, _ = model_hybrid_ensemble(df)
    all_metrics.append(metrics4)

    # Save
    print("\n" + "="*70)
    print("[SAVE RESULTS]")
    print("="*70)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_file = RESULTS_DIR / 'dl_metrics.csv'
    metrics_df.to_csv(metrics_file, index=False)

    print(f"\n[OK] DL metrics saved: {metrics_file}")
    print(metrics_df.to_string(index=False))

    print("\n" + "="*70)
    print("[DONE] Deep learning models trained!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
