"""
16b_hurdle_semantic_v2.py
Hurdle Model with IMPROVED Semantic Features (v2).

Key differences vs 16_hurdle_semantic.py:
  1. Supervised dim selection: select top-K semantic dimensions by
     |Spearman(dim, log(1+ActualCLV))| computed on the TRAIN fold only.
     Previously used unsupervised PCA which can ignore CLV-relevant dims.
  2. Recency-aware features: adds sem_recent (top-K dims), sem_drift scalar,
     and sem counts from semantic_v2_*.npz (built by 14b_*).
  3. Sem x Seq interaction: explicit product between top semantic dim and
     seq_recent3_mean (sem_seq_inter feature).

Variants:
  Hurdle-RFM       : same as v1
  Hurdle-Seq       : same as v1
  Hurdle-SemV2     : RFM + improved semantic (sem_full_topK + sem_recent_topK
                     + sem_drift + sem_counts)
  Hurdle-AllV2     : RFM + Seq + improved semantic + sem_seq interaction
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
from scipy.stats import spearmanr

DATA_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR        = Path(__file__).parent.parent / "results"

N_OPTUNA_TRIALS = 20
TOP_K_SEM       = 8     # top-K dims to select from each of full/recent (was 16 via PCA)


# ── Feature builders ──────────────────────────────────────────────────────────

def rfm_features(feats_df):
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


def load_semantic_v2(win):
    """Load recency-aware semantic data from semantic_v2_window_{wid}.npz."""
    wid  = win['window_id']
    path = DATA_PROCESSED_DIR / f'semantic_v2_window_{wid}.npz'
    if not path.exists():
        raise FileNotFoundError(f"Run 14b_customer_semantic_features_v2.py first")
    z = np.load(path, allow_pickle=True)
    return {
        'sem_full':     z['sem_full'],       # (n, 32)
        'sem_recent':   z['sem_recent'],     # (n, 32)
        'sem_drift':    z['sem_drift'],      # (n,)
        'full_count':   z['full_count'],     # (n,)
        'recent_count': z['recent_count'],   # (n,)
    }


def supervised_dim_select(sem_mat, y_log, tr_idx, k):
    """
    Select top-K dims by |Spearman(dim, y_log)| computed on TRAIN fold only.
    Returns indices of selected dims.
    """
    corrs = []
    for d in range(sem_mat.shape[1]):
        rho, _ = spearmanr(sem_mat[tr_idx, d], y_log[tr_idx])
        corrs.append(abs(rho) if not np.isnan(rho) else 0.0)
    corrs   = np.array(corrs)
    top_idx = np.argsort(corrs)[::-1][:k]
    return top_idx, corrs


def build_features_v2(win, variant, tr_idx, y_log):
    """Build feature matrix with supervised dim selection on train fold."""
    X_rfm, n_rfm = rfm_features(win['features'])
    parts, names = [X_rfm], list(n_rfm)

    if variant in ('seq', 'all_v2'):
        Xs, ns = seq_features(win)
        parts.append(Xs); names += ns

    if variant in ('sem_v2', 'all_v2'):
        sv2 = load_semantic_v2(win)

        # Supervised select top-K from full and recent
        idx_full, corr_full = supervised_dim_select(sv2['sem_full'],   y_log, tr_idx, TOP_K_SEM)
        idx_rec,  corr_rec  = supervised_dim_select(sv2['sem_recent'], y_log, tr_idx, TOP_K_SEM)

        Xfull = sv2['sem_full'][:, idx_full]
        Xrec  = sv2['sem_recent'][:, idx_rec]
        Xdrift = sv2['sem_drift'].reshape(-1, 1)
        Xfc    = sv2['full_count'].reshape(-1, 1)
        Xrc    = sv2['recent_count'].reshape(-1, 1)

        parts.extend([Xfull, Xrec, Xdrift, Xfc, Xrc])
        names += [f'sem_full_{i}' for i in idx_full]
        names += [f'sem_rec_{i}'  for i in idx_rec]
        names += ['sem_drift', 'sem_full_count', 'sem_recent_count']

    if variant == 'all_v2':
        # Sem x Seq interaction: top semantic dim * seq_recent3_mean
        Xs, _ = seq_features(win)
        sv2   = load_semantic_v2(win)
        idx_full, _ = supervised_dim_select(sv2['sem_full'], y_log, tr_idx, 1)
        # Use seq_recent3_mean (3rd column in seq matrix per seq_features)
        seq_recent3 = Xs[:, 2]
        top_sem = sv2['sem_full'][:, idx_full[0]]
        inter   = (top_sem * seq_recent3).reshape(-1, 1)
        parts.append(inter)
        names.append('sem_seq_inter')

    return np.hstack(parts), names


# ── Hurdle training (same as v1) ──────────────────────────────────────────────

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


def train_hurdle(X_tr, y_tr, X_te, y_te=None, n_trials=N_OPTUNA_TRIALS):
    y_bin = (y_tr > 0).astype(int)

    clf_p = tune_lgbm(X_tr, y_bin, 'binary', n_trials)
    clf   = lgb.LGBMClassifier(**clf_p, n_estimators=500, verbosity=-1)
    Xtr2, Xva, ytr2, yva = train_test_split(X_tr, y_bin, test_size=0.15, random_state=42)
    clf.fit(Xtr2, ytr2, eval_set=[(Xva, yva)],
            callbacks=[lgb.early_stopping(30, verbose=False)])
    p_pos = clf.predict_proba(X_te)[:, 1]

    from sklearn.metrics import roc_auc_score, average_precision_score
    if y_te is not None:
        y_te_bin = (y_te > 0).astype(int)
        stage1_auc  = roc_auc_score(y_te_bin, p_pos)
        stage1_aupr = average_precision_score(y_te_bin, p_pos)
    else:
        p_va = clf.predict_proba(Xva)[:, 1]
        stage1_auc  = roc_auc_score(yva, p_va)
        stage1_aupr = average_precision_score(yva, p_va)

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
    return np.maximum(pred, 0), clf, reg, stage1_auc, stage1_aupr


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*70)
    print("[Phase B-v2] Hurdle Model with IMPROVED Semantic Features")
    print(f"             Top-K supervised dim selection: K={TOP_K_SEM}")
    print("="*70)

    p5 = DATA_PROCESSED_DIR / 'walk_forward_windows_5fold.pkl'
    p3 = DATA_PROCESSED_DIR / 'walk_forward_windows.pkl'
    with open(p5 if p5.exists() else p3, 'rb') as f:
        windows = pickle.load(f)
    print(f"Loaded {len(windows)} windows")

    variants = {
        'Hurdle-RFM':    'rfm',
        'Hurdle-Seq':    'seq',
        'Hurdle-SemV2':  'sem_v2',
        'Hurdle-AllV2':  'all_v2',
    }

    all_rows   = []
    shap_saved = False

    for win in windows:
        wid = win['window_id']
        print(f"\n{'='*60}")
        print(f"Window {wid}: pred {win['pred_start']} -> {win['pred_end']}")
        print(f"{'='*60}")

        feats = win['features']
        y     = feats['ActualCLV'].values
        y_log = np.log1p(y)

        tr_idx, te_idx = train_test_split(
            range(len(feats)), test_size=0.2, random_state=42,
            stratify=feats['IsVIP']
        )
        tr_idx = np.array(tr_idx)
        te_idx = np.array(te_idx)

        for vname, vtype in variants.items():
            print(f"\n  [{vname}]  variant={vtype}")
            try:
                X, feat_names = build_features_v2(win, vtype, tr_idx, y_log)
            except FileNotFoundError as e:
                print(f"    SKIP: {e}")
                continue

            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            y_pred, clf, reg, auc, aupr = train_hurdle(X_tr, y_tr, X_te, y_te)
            m = comprehensive_metrics(y_te, y_pred)
            print(f"  N_feats={X.shape[1]}  S1_AUC={auc:.4f}  S1_AUPR={aupr:.4f}")
            print(f"  MAE={m['MAE']:,.0f}  NG={m['Norm_Gini']:.4f}  "
                  f"RC@10={m['Revenue_Capture_10']:.2f}%  "
                  f"Lift={m['Lift_10']:.2f}x")

            row = {'Window': wid, 'Model': vname, 'Variant': vtype,
                   'N_Features': X.shape[1],
                   'Stage1_AUC': auc, 'Stage1_AUPR': aupr}
            row.update(m)
            all_rows.append(row)

            # SHAP for AllV2 on W3
            if vname == 'Hurdle-AllV2' and wid == 3 and not shap_saved:
                print(f"    Computing SHAP for {vname} W{wid}...")
                try:
                    explainer = shap.TreeExplainer(reg)
                    pos_mask  = y_te > 0
                    if pos_mask.sum() > 10:
                        sv = explainer.shap_values(X_te[pos_mask])
                        shap_mean = np.abs(sv).mean(axis=0)
                        shap_df = pd.DataFrame({
                            'feature': feat_names,
                            'mean_abs_shap': shap_mean
                        }).sort_values('mean_abs_shap', ascending=False)
                        shap_df.to_csv(RESULTS_DIR / 'shap_hurdle_allv2_stage2.csv', index=False)
                        print(f"    SHAP saved. Top: {shap_df['feature'].head(5).tolist()}")
                        shap_saved = True
                except Exception as e:
                    print(f"    SHAP failed: {e}")

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_DIR / 'semantic_walkforward_v2.csv', index=False)
    print(f"\nSaved -> {RESULTS_DIR/'semantic_walkforward_v2.csv'}")

    print("\n" + "="*70)
    print("[SUMMARY] Hurdle V2 Variants (mean +/- std across windows)")
    print("="*70)
    print(f"{'Variant':<18} {'NFeat':>6} {'S1_AUC':>8} {'NG':>14} {'RC@10%':>14} {'MAE':>7}")
    print("-"*70)
    for vname in variants:
        sub = df[df['Model'] == vname]
        if len(sub) == 0:
            continue
        print(f"{vname:<18} {sub['N_Features'].mean():>5.0f}  "
              f"{sub['Stage1_AUC'].mean():>7.4f}  "
              f"{sub['Norm_Gini'].mean():.4f}+/-{sub['Norm_Gini'].std():.3f}  "
              f"{sub['Revenue_Capture_10'].mean():>5.2f}+/-{sub['Revenue_Capture_10'].std():.2f}  "
              f"${sub['MAE'].mean():>4.0f}")

    print("\n" + "="*70)
    print("[DONE] Hurdle semantic v2 evaluation complete")
    print("="*70)


if __name__ == "__main__":
    main()
