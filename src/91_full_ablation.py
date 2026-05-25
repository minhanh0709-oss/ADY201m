"""
91_full_ablation.py
Ablation study across ALL 3 walk-forward windows.
Feature groups: RFM → +Behavioral → +Interaction → +Sequence (Full)
Uses window_N_features.csv + window_N_revenue_seq.npy
Saves results/ablation_all_windows.csv
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb

RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_DIR    = Path(__file__).parent.parent / "data" / "processed"
FIGURES_DIR = Path(__file__).parent.parent / "figures"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STATIC_RFM   = ['Recency', 'Frequency', 'Monetary']
BEHAVIORAL   = ['Tenure', 'ActiveMonths', 'ProductDiversity',
                'AvgOrderValue', 'AvgDaysBetweenOrders', 'Regularity', 'IsUK']
INTERACTIONS = ['LogMonetary', 'LogFrequency', 'LogAOV',
                'MonetaryPerTenure', 'FreqPerTenure', 'ActiveRatio']
SEQ_SUMMARY  = ['seq_mean', 'seq_std', 'seq_max',
                'seq_recent3_mean', 'seq_recent3_max',
                'seq_active_months', 'seq_trend']

FEATURE_GROUPS = {
    'RFM only':              STATIC_RFM,
    'RFM + Behavioral':      STATIC_RFM + BEHAVIORAL,
    'RFM + Beh + Interact':  STATIC_RFM + BEHAVIORAL + INTERACTIONS,
    'Full (+ Sequence)':     STATIC_RFM + BEHAVIORAL + INTERACTIONS + SEQ_SUMMARY,
}

WINDOWS = [1, 2, 3]


def make_seq_features(seq_arr):
    """Summarise monthly revenue sequence → 7 scalar features."""
    n = seq_arr.shape[1]
    mid = n // 2
    df = pd.DataFrame({
        'seq_mean':          seq_arr.mean(axis=1),
        'seq_std':           seq_arr.std(axis=1),
        'seq_max':           seq_arr.max(axis=1),
        'seq_recent3_mean':  seq_arr[:, -3:].mean(axis=1),
        'seq_recent3_max':   seq_arr[:, -3:].max(axis=1),
        'seq_active_months': (seq_arr > 0).sum(axis=1).astype(float),
        'seq_trend':         seq_arr[:, mid:].mean(axis=1) - seq_arr[:, :mid].mean(axis=1),
    })
    return df


def make_interactions(df):
    out = df.copy()
    out['LogMonetary']      = np.log1p(df['Monetary'])
    out['LogFrequency']     = np.log1p(df['Frequency'])
    out['LogAOV']           = np.log1p(df['AvgOrderValue'])
    out['MonetaryPerTenure'] = df['Monetary'] / (df['Tenure'] + 1)
    out['FreqPerTenure']    = df['Frequency'] / (df['Tenure'] + 1)
    out['ActiveRatio']      = df['ActiveMonths'] / (df['Tenure'] / 30 + 1)
    return out


def normalized_gini(y_true, y_pred):
    def gini(a, p):
        n = len(a)
        idx = np.argsort(p)
        a_s = a[idx]
        cumsum = np.cumsum(a_s)
        total = cumsum[-1]
        if total == 0:
            return 0.0
        return (np.sum(cumsum[:-1]) / total - (n - 1) / 2) / n
    denom = gini(y_true, y_true)
    return gini(y_true, y_pred) / denom if denom != 0 else 0.0


def revenue_capture_at_10(y_true, y_pred):
    k = max(1, int(len(y_true) * 0.10))
    top_idx = np.argsort(y_pred)[-k:]
    total = y_true.sum()
    return y_true[top_idx].sum() / total * 100 if total > 0 else 0.0


def train_hurdle(X_tr, y_tr, X_te):
    buy_tr = (y_tr > 0).astype(int)
    vs = max(1, int(len(X_tr) * 0.15))

    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                              num_leaves=31, random_state=42, verbose=-1)
    clf.fit(X_tr[:-vs], buy_tr[:-vs],
            eval_set=[(X_tr[-vs:], buy_tr[-vs:])],
            callbacks=[lgb.early_stopping(20, verbose=False),
                       lgb.log_evaluation(-1)])
    prob = clf.predict_proba(X_te)[:, 1]

    pos = y_tr > 0
    X_pos, log_y_pos = X_tr[pos], np.log1p(y_tr[pos])
    vs2 = max(1, int(len(X_pos) * 0.15))

    reg = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                             num_leaves=31, random_state=42, verbose=-1)
    reg.fit(X_pos[:-vs2], log_y_pos[:-vs2],
            eval_set=[(X_pos[-vs2:], log_y_pos[-vs2:])],
            callbacks=[lgb.early_stopping(20, verbose=False),
                       lgb.log_evaluation(-1)])
    lp = reg.predict(X_te)
    rv = np.var(log_y_pos - reg.predict(X_pos))
    return prob * (np.expm1(lp) * np.exp(rv / 2))


def main():
    print("\n" + "="*65)
    print("[Full Ablation Across All 3 Windows]")
    print("="*65)

    records = []

    for w in WINDOWS:
        feat_path = DATA_DIR / f"window_{w}_features.csv"
        seq_path  = DATA_DIR / f"window_{w}_revenue_seq.npy"

        if not feat_path.exists():
            print(f"  Window {w}: features file not found, skip.")
            continue

        df_feat = pd.read_csv(feat_path).fillna(0)
        df_feat = make_interactions(df_feat)

        if seq_path.exists():
            seq_arr = np.load(seq_path)
            df_seq  = make_seq_features(seq_arr)
            for col in df_seq.columns:
                df_feat[col] = df_seq[col].values

        y = df_feat['ActualCLV'].values.astype(np.float32)
        n_tr = int(len(df_feat) * 0.8)

        print(f"\n  Window {w}  (n={len(df_feat)}, train={n_tr}, "
              f"test={len(df_feat)-n_tr}, zero_rate={( y==0).mean():.1%})")

        for gname, feat_list in FEATURE_GROUPS.items():
            avail = [c for c in feat_list if c in df_feat.columns]
            X = df_feat[avail].values.astype(np.float32)
            X_tr, X_te = X[:n_tr], X[n_tr:]
            y_tr, y_te = y[:n_tr], y[n_tr:]

            try:
                y_pred = train_hurdle(X_tr, y_tr, X_te)
                ng  = normalized_gini(y_te, y_pred)
                rc  = revenue_capture_at_10(y_te, y_pred)
                mae = float(np.mean(np.abs(y_te - y_pred)))
                records.append({'Window': w, 'Feature_Set': gname,
                                'N_Features': len(avail),
                                'Norm_Gini': round(ng, 4),
                                'Revenue_10': round(rc, 2),
                                'MAE': round(mae, 1)})
                print(f"    {gname:22s} ({len(avail):2d})  "
                      f"NG={ng:.4f}  RC@10={rc:.2f}%  MAE=${mae:.0f}")
            except Exception as e:
                print(f"    {gname}: ERROR — {e}")

    if not records:
        print("No results generated.")
        return

    df_out = pd.DataFrame(records)
    out_path = RESULTS_DIR / 'ablation_all_windows.csv'
    df_out.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    # ── Summary mean across windows ───────────────────────────────────────────
    print("\n[Mean +/- Std across 3 windows]")
    order = list(FEATURE_GROUPS.keys())
    summary = df_out.groupby('Feature_Set').agg(
        NG_mean=('Norm_Gini','mean'), NG_std=('Norm_Gini','std'),
        RC_mean=('Revenue_10','mean'), RC_std=('Revenue_10','std'),
    ).reindex(order)
    print(summary.round(4).to_string())

    base_ng = summary.loc['RFM only','NG_mean'] if 'RFM only' in summary.index else 0
    print("\n[Delta Norm Gini vs RFM-only]")
    for g in order:
        if g in summary.index:
            v = summary.loc[g,'NG_mean']
            print(f"  {g:25s}  {v:.4f}  ({v-base_ng:+.4f})")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ['#3498DB','#27AE60','#E67E22','#9B59B6']
    x = np.arange(len(WINDOWS))
    w_width = 0.18

    for ax, metric, ylabel, title in [
        (axes[0], 'Norm_Gini',   'Normalized Gini', '(a) Ablation: Normalized Gini'),
        (axes[1], 'Revenue_10',  'Revenue Capture @ 10% (%)', '(b) Ablation: Revenue Capture'),
    ]:
        for i, (gname, col) in enumerate(zip(order, colors)):
            vals = [df_out[(df_out['Window']==w) & (df_out['Feature_Set']==gname)][metric].values
                    for w in WINDOWS]
            vals = [v[0] if len(v)>0 else np.nan for v in vals]
            offset = (i - 1.5) * w_width
            ax.bar(x + offset, vals, width=w_width, color=col,
                   label=gname, alpha=0.88, edgecolor='white')

        ax.set_xticks(x)
        ax.set_xticklabels([f'Window {w}' for w in WINDOWS], fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)

    plt.suptitle('Feature Group Ablation Across All 3 Walk-Forward Windows\n'
                 '(Hurdle Model, LightGBM two-stage)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig_path = FIGURES_DIR / 'fig5_ablation.png'
    plt.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  Updated figure: {fig_path}")


if __name__ == "__main__":
    main()
