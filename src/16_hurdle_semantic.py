"""
16_hurdle_semantic.py
Hurdle Model with Semantic Product Graph Features

Compares 4 variants across walk-forward windows:
  (A) Hurdle-RFM        : 16 RFM + behavioral + interaction features
  (B) Hurdle-Seq        : A + 7 monthly sequence summary features  (existing best)
  (C) Hurdle-Sem        : A + 32 semantic product-graph features
  (D) Hurdle-All        : A + Seq + Sem  (full proposed model)

Hypothesis: Semantic features capture "what customers buy" (product taste),
which is orthogonal to "when customers buy" (captured by sequence features),
and complementary to "how much customers bought" (RFM).

SHAP analysis on Hurdle-All is also saved for paper Figure generation.
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

import lightgbm as lgb
import shap
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.model_selection import KFold, train_test_split
from sklearn.decomposition import PCA

DATA_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR        = Path(__file__).parent.parent / "results"

N_OPTUNA_TRIALS = 20   # per stage


# ── Feature builders ──────────────────────────────────────────────────────────

def rfm_features(feats_df):
    """Core 16 RFM + behavioral + interaction features."""
    cols = ['Recency', 'Frequency', 'Monetary', 'Tenure',
            'ActiveMonths', 'ProductDiversity', 'AvgOrderValue',
            'AvgDaysBetweenOrders', 'Regularity', 'IsUK']
    X = feats_df[cols].fillna(0).copy()
    X['log_Monetary']      = np.log1p(X['Monetary'])
    X['log_Frequency']     = np.log1p(X['Frequency'])
    X['log_AvgOrderValue'] = np.log1p(X['AvgOrderValue'])
    X['M_per_F']           = X['Monetary'] / np.maximum(X['Frequency'], 1)
    X['M_per_T']           = X['Monetary'] / np.maximum(X['Tenure'],    1)
    X['Active_ratio']      = X['ActiveMonths'] / np.maximum(X['Tenure'] / 30, 1)
    return X.values, list(X.columns)


def seq_features(win):
    """7 monthly sequence summary features."""
    seq = win['revenue_seq']
    feats = np.column_stack([
        seq.mean(axis=1),
        seq.max(axis=1),
        seq[:, -3:].mean(axis=1),
        seq[:, -3:].max(axis=1),
        (seq > 0).sum(axis=1).astype(float),
        seq.std(axis=1),
        seq[:, seq.shape[1]//2:].mean(axis=1) - seq[:, :seq.shape[1]//2].mean(axis=1),
    ])
    names = ['seq_mean','seq_max','seq_recent3_mean','seq_recent3_max',
             'seq_active_months','seq_std','seq_trend']
    return feats, names


def sem_features(win, n_pca=16):
    """Semantic product-graph features reduced to n_pca dims."""
    wid   = win['window_id']
    path  = DATA_PROCESSED_DIR / f'semantic_features_window_{wid}.npy'
    if not path.exists():
        raise FileNotFoundError(f"Run 14_customer_semantic_features.py first: {path}")
    sem   = np.load(path)                       # (n_customers, 32)
    if sem.shape[1] > n_pca:
        pca = PCA(n_components=n_pca, random_state=42)
        sem = pca.fit_transform(sem)
    names = [f'sem_{i}' for i in range(sem.shape[1])]
    return sem, names


def build_features(win, variant='all', n_pca=16):
    """Build feature matrix for a given variant."""
    X_rfm, n_rfm = rfm_features(win['features'])
    parts, names = [X_rfm], list(n_rfm)

    if variant in ('seq', 'all'):
        Xs, ns = seq_features(win)
        parts.append(Xs); names += ns

    if variant in ('sem', 'all'):
        Xsem, nsem = sem_features(win, n_pca)
        parts.append(Xsem); names += nsem

    return np.hstack(parts), names


# ── Hurdle training ───────────────────────────────────────────────────────────

def tune_lgbm(X, y, task='binary', n_trials=N_OPTUNA_TRIALS):
    metric = 'auc' if task == 'binary' else 'mae'
    obj    = 'binary' if task == 'binary' else 'regression'

    def objective(trial):
        params = {
            'objective': obj, 'metric': metric, 'verbosity': -1,
            'num_leaves':        trial.suggest_int('num_leaves', 15, 80),
            'learning_rate':     trial.suggest_float('lr', 0.01, 0.2, log=True),
            'feature_fraction':  trial.suggest_float('ff', 0.5, 1.0),
            'bagging_fraction':  trial.suggest_float('bf', 0.5, 1.0),
            'min_data_in_leaf':  trial.suggest_int('min_data', 5, 50),
            'random_state': 42,
        }
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for ti, vi in kf.split(X):
            Model = lgb.LGBMClassifier if task == 'binary' else lgb.LGBMRegressor
            m = Model(**params, n_estimators=300)
            m.fit(X[ti], y[ti], eval_set=[(X[vi], y[vi])],
                  callbacks=[lgb.early_stopping(20, verbose=False)])
            if task == 'binary':
                from sklearn.metrics import roc_auc_score
                scores.append(roc_auc_score(y[vi], m.predict_proba(X[vi])[:, 1]))
            else:
                scores.append(-np.mean(np.abs(y[vi] - m.predict(X[vi]))))
        return -np.mean(scores)

    study = optuna.create_study(direction='minimize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def train_hurdle(X_tr, y_tr, X_te, n_trials=N_OPTUNA_TRIALS):
    y_bin = (y_tr > 0).astype(int)

    # Stage 1: classifier
    clf_p = tune_lgbm(X_tr, y_bin, 'binary', n_trials)
    clf   = lgb.LGBMClassifier(**clf_p, n_estimators=500, verbosity=-1)
    Xtr2, Xva, ytr2, yva = train_test_split(X_tr, y_bin, test_size=0.15, random_state=42)
    clf.fit(Xtr2, ytr2, eval_set=[(Xva, yva)],
            callbacks=[lgb.early_stopping(30, verbose=False)])
    p_pos = clf.predict_proba(X_te)[:, 1]

    # Stage 2: regressor on positives only
    mask    = y_tr > 0
    X_pos   = X_tr[mask]
    y_log   = np.log1p(y_tr[mask])
    reg_p   = tune_lgbm(X_pos, y_log, 'regression', n_trials)
    reg     = lgb.LGBMRegressor(**reg_p, n_estimators=500, verbosity=-1)
    Xtr2, Xva, ytr2, yva = train_test_split(X_pos, y_log, test_size=0.15, random_state=42)
    reg.fit(Xtr2, ytr2, eval_set=[(Xva, yva)],
            callbacks=[lgb.early_stopping(30, verbose=False)])

    log_pred = reg.predict(X_te)
    res_var  = np.var(yva - reg.predict(Xva))
    pred     = p_pos * np.expm1(log_pred) * np.exp(res_var / 2)
    return np.maximum(pred, 0), clf, reg


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*70)
    print("[Phase B] Hurdle Model — Semantic Feature Variants")
    print("="*70)

    p5 = DATA_PROCESSED_DIR / 'walk_forward_windows_5fold.pkl'
    p3 = DATA_PROCESSED_DIR / 'walk_forward_windows.pkl'
    with open(p5 if p5.exists() else p3, 'rb') as f:
        windows = pickle.load(f)
    print(f"Loaded {len(windows)} windows")

    variants = {
        'Hurdle-RFM': 'rfm',
        'Hurdle-Seq': 'seq',
        'Hurdle-Sem': 'sem',
        'Hurdle-All': 'all',
    }

    all_rows   = []
    shap_saved = False   # save SHAP only for W3 (longest obs) of Hurdle-All

    for win in windows:
        wid = win['window_id']
        print(f"\n{'='*60}")
        print(f"Window {wid}: pred {win['pred_start']} -> {win['pred_end']}")
        print(f"{'='*60}")

        feats = win['features']
        y     = feats['ActualCLV'].values

        tr_idx, te_idx = train_test_split(
            range(len(feats)), test_size=0.2, random_state=42,
            stratify=feats['IsVIP']
        )
        tr_idx = np.array(tr_idx)
        te_idx = np.array(te_idx)

        for vname, vtype in variants.items():
            print(f"\n  [{vname}]  variant={vtype}")
            try:
                X, feat_names = build_features(win, vtype)
            except FileNotFoundError as e:
                print(f"    SKIP (missing semantic file): {e}")
                continue

            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            y_pred, clf, reg = train_hurdle(X_tr, y_tr, X_te)
            m = comprehensive_metrics(y_te, y_pred)
            print(f"  MAE={m['MAE']:,.0f}  NG={m['Norm_Gini']:.4f}  "
                  f"RC@10={m['Revenue_Capture_10']:.2f}%  "
                  f"Lift={m['Lift_10']:.2f}x")

            row = {'Window': wid, 'Model': vname, 'Variant': vtype}
            row.update(m)
            all_rows.append(row)

            # SHAP for Hurdle-All on Window 3 (largest obs window among original 3)
            if vname == 'Hurdle-All' and wid == 3 and not shap_saved:
                print(f"    Computing SHAP for {vname} Window {wid}...")
                try:
                    explainer = shap.TreeExplainer(reg)
                    pos_mask_te = y_te > 0
                    if pos_mask_te.sum() > 10:
                        X_pos_te  = X_te[pos_mask_te]
                        shap_vals = explainer.shap_values(X_pos_te)
                        shap_mean = np.abs(shap_vals).mean(axis=0)
                        shap_df   = pd.DataFrame({
                            'feature': feat_names,
                            'mean_abs_shap': shap_mean
                        }).sort_values('mean_abs_shap', ascending=False)
                        shap_df.to_csv(RESULTS_DIR / 'shap_hurdle_all_stage2.csv', index=False)
                        print(f"    SHAP saved. Top features: {shap_df['feature'].head(5).tolist()}")
                        shap_saved = True
                except Exception as e:
                    print(f"    SHAP failed: {e}")

    # ── Save results ──────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_DIR / 'semantic_walkforward.csv', index=False)
    print(f"\nSaved -> {RESULTS_DIR/'semantic_walkforward.csv'}")

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("[SUMMARY] Hurdle Variants (mean ± std across windows)")
    print("="*70)
    print(f"{'Variant':<18} {'NG':>8} {'RC@10%':>8} {'MAE':>8} {'Lift':>6}")
    print("-"*52)
    for vname in variants:
        sub = df[df['Model'] == vname]
        if len(sub) == 0:
            continue
        print(f"{vname:<18} "
              f"{sub['Norm_Gini'].mean():.4f}±{sub['Norm_Gini'].std():.3f}  "
              f"{sub['Revenue_Capture_10'].mean():.2f}±{sub['Revenue_Capture_10'].std():.2f}  "
              f"${sub['MAE'].mean():.0f}  "
              f"{sub['Lift_10'].mean():.2f}x")

    print("\n" + "="*70)
    print("[DONE] Hurdle semantic variants evaluation complete")
    print("="*70)


if __name__ == "__main__":
    main()
