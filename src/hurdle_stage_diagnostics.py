"""
90_stage_wise_hurdle.py
Stage-wise evaluation of the two-stage Hurdle Model:
  Stage 1 (classifier): AUC-ROC, AUC-PR, F1, Brier score
  Stage 2 (regressor on positives): MAE, RMSE, R², MAPE
Uses window_N_features.csv + window_N_revenue_seq.npy
Saves results/hurdle_stage_diagnostics.csv
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, brier_score_loss,
                              mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split

RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_DIR    = Path(__file__).parent.parent / "data" / "processed"

STATIC_FEATS = ['Recency', 'Frequency', 'Monetary', 'Tenure', 'ActiveMonths',
                'ProductDiversity', 'AvgOrderValue', 'AvgDaysBetweenOrders',
                'Regularity', 'IsUK']
SEQ_FEATS    = ['seq_mean', 'seq_std', 'seq_max', 'seq_recent3_mean',
                'seq_recent3_max', 'seq_active_months', 'seq_trend']


def make_seq_features(seq_arr):
    n, m = seq_arr.shape
    mid = m // 2
    return pd.DataFrame({
        'seq_mean':          seq_arr.mean(axis=1),
        'seq_std':           seq_arr.std(axis=1),
        'seq_max':           seq_arr.max(axis=1),
        'seq_recent3_mean':  seq_arr[:, -3:].mean(axis=1),
        'seq_recent3_max':   seq_arr[:, -3:].max(axis=1),
        'seq_active_months': (seq_arr > 0).sum(axis=1).astype(float),
        'seq_trend':         seq_arr[:, mid:].mean(axis=1) - seq_arr[:, :mid].mean(axis=1),
    })


def mape_positive(y_true, y_pred):
    mask = y_true > 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_stages(X_tr, y_tr, X_te, y_te, window_id):
    buy_tr = (y_tr > 0).astype(int)
    buy_te = (y_te > 0).astype(int)
    vs = max(1, int(len(X_tr) * 0.15))

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    clf = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05,
                              num_leaves=31, subsample=0.8,
                              colsample_bytree=0.8, random_state=42, verbose=-1)
    clf.fit(X_tr[:-vs], buy_tr[:-vs],
            eval_set=[(X_tr[-vs:], buy_tr[-vs:])],
            callbacks=[lgb.early_stopping(30, verbose=False),
                       lgb.log_evaluation(-1)])

    prob = clf.predict_proba(X_te)[:, 1]
    pred_buy = (prob >= 0.5).astype(int)

    s1 = {
        'Window':             window_id,
        'Stage1_AUC_ROC':     round(roc_auc_score(buy_te, prob), 4),
        'Stage1_AUC_PR':      round(average_precision_score(buy_te, prob), 4),
        'Stage1_F1':          round(f1_score(buy_te, pred_buy), 4),
        'Stage1_Brier':       round(brier_score_loss(buy_te, prob), 4),
        'Zero_Rate_Test':     round(1 - buy_te.mean(), 4),
        'N_Positives_Test':   int(buy_te.sum()),
    }

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    pos_tr = y_tr > 0
    X_pos_tr, log_y_pos = X_tr[pos_tr], np.log1p(y_tr[pos_tr])
    vs2 = max(1, int(len(X_pos_tr) * 0.15))

    reg = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                             num_leaves=31, subsample=0.8,
                             colsample_bytree=0.8, random_state=42, verbose=-1)
    reg.fit(X_pos_tr[:-vs2], log_y_pos[:-vs2],
            eval_set=[(X_pos_tr[-vs2:], log_y_pos[-vs2:])],
            callbacks=[lgb.early_stopping(30, verbose=False),
                       lgb.log_evaluation(-1)])

    pos_te = y_te > 0
    X_pos_te, y_pos_te = X_te[pos_te], y_te[pos_te]
    log_pred  = reg.predict(X_pos_te)
    resid_var = np.var(log_y_pos - reg.predict(X_pos_tr))
    y_pred_pos = np.clip(np.expm1(log_pred) * np.exp(resid_var / 2), 0, None)

    s2 = {
        'Stage2_MAE':  round(mean_absolute_error(y_pos_te, y_pred_pos), 2),
        'Stage2_RMSE': round(np.sqrt(mean_squared_error(y_pos_te, y_pred_pos)), 2),
        'Stage2_R2':   round(r2_score(y_pos_te, y_pred_pos), 4),
        'Stage2_MAPE': round(mape_positive(y_pos_te, y_pred_pos), 2),
    }

    return {**s1, **s2}


def main():
    print("\n" + "="*65)
    print("[Stage-wise Hurdle Model Diagnostics]")
    print("="*65)

    records = []
    all_feats = STATIC_FEATS + SEQ_FEATS

    for w in [1, 2, 3]:
        feat_path = DATA_DIR / f"window_{w}_features.csv"
        seq_path  = DATA_DIR / f"window_{w}_revenue_seq.npy"

        if not feat_path.exists():
            print(f"  Window {w}: not found, skip.")
            continue

        df = pd.read_csv(feat_path).fillna(0)

        # Add interaction features
        df['LogMonetary']       = np.log1p(df['Monetary'])
        df['LogFrequency']      = np.log1p(df['Frequency'])
        df['LogAOV']            = np.log1p(df['AvgOrderValue'])
        df['MonetaryPerTenure'] = df['Monetary'] / (df['Tenure'] + 1)
        df['FreqPerTenure']     = df['Frequency'] / (df['Tenure'] + 1)
        df['ActiveRatio']       = df['ActiveMonths'] / (df['Tenure'] / 30 + 1)

        if seq_path.exists():
            seq_arr = np.load(seq_path)
            df_seq = make_seq_features(seq_arr)
            for col in df_seq.columns:
                df[col] = df_seq[col].values

        avail = [c for c in all_feats if c in df.columns]
        X = df[avail].values.astype(np.float32)
        y = df['ActualCLV'].values.astype(np.float32)
        strat = df['IsVIP'].values if 'IsVIP' in df.columns else None

        idx = np.arange(len(df))
        tr_idx, te_idx = train_test_split(
            idx, test_size=0.2, random_state=42,
            stratify=strat if strat is not None else None,
        )
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        print(f"\n  Window {w}  (n_test={len(y_te)}, "
              f"zero_rate={( y_te==0).mean():.1%})")

        try:
            res = evaluate_stages(X_tr, y_tr, X_te, y_te, w)
            records.append(res)
            print(f"  Stage1: AUC-ROC={res['Stage1_AUC_ROC']:.4f}  "
                  f"AUC-PR={res['Stage1_AUC_PR']:.4f}  "
                  f"F1={res['Stage1_F1']:.4f}  "
                  f"Brier={res['Stage1_Brier']:.4f}")
            print(f"  Stage2: MAE=${res['Stage2_MAE']:.0f}  "
                  f"RMSE=${res['Stage2_RMSE']:.0f}  "
                  f"R2={res['Stage2_R2']:.4f}  "
                  f"MAPE={res['Stage2_MAPE']:.1f}%")
        except Exception as e:
            print(f"  Window {w} ERROR: {e}")
            import traceback; traceback.print_exc()

    if records:
        df_out = pd.DataFrame(records)
        out_path = RESULTS_DIR / 'hurdle_stage_diagnostics.csv'
        df_out.to_csv(out_path, index=False)
        print(f"\n  Saved: {out_path}")

        print("\n[Mean +/- Std across windows]")
        num_cols = [c for c in df_out.columns if c != 'Window']
        for col in num_cols:
            print(f"  {col:30s}: {df_out[col].mean():.4f} +/- {df_out[col].std():.4f}")
    else:
        print("No results generated.")


if __name__ == "__main__":
    main()
