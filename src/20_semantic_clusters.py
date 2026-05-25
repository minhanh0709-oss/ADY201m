"""
20_semantic_clusters.py
Semantic Cluster Analysis for Customer Segmentation Interpretability.

Goal:
  Demonstrate that semantic product-graph profiles separate customers into
  meaningful taste-based clusters that are NOT predictable from RFM alone.
  This is the interpretability story that aligns with SIMC's theme of
  semantic knowledge representation.

Method (Window 3, longest observation):
  1. Load 32-dim sem_full semantic profiles
  2. Run K-Means (k=6) on profiles to get taste clusters
  3. For each cluster: compute mean RFM, ActualCLV, return rate, top products
  4. Statistical test: ANOVA / Kruskal-Wallis on ActualCLV across clusters
     (if p < 0.05, semantic clusters DO differentiate CLV beyond what RFM
     gives us)
  5. Compare with RFM-only clusters (K-Means on RFM): are they the same?
  6. Save:
     - results/semantic_clusters.csv  (per-cluster aggregates)
     - results/semantic_cluster_products.csv (top product per cluster)
     - figures/fig_semantic_clusters.png
"""

import pandas as pd
import numpy as np
import pickle
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway, kruskal
import warnings
warnings.filterwarnings('ignore')

DATA_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR        = Path(__file__).parent.parent / "results"
FIGURES_DIR        = Path(__file__).parent.parent / "figures"
CLEANED_FILE       = DATA_PROCESSED_DIR / "online_retail_cleaned.csv"

WID    = 3   # Window 3: longest observation
N_CLUS = 6


def main():
    print("\n" + "="*70)
    print(f"[20] Semantic Cluster Analysis (Window {WID}, k={N_CLUS})")
    print("="*70)

    # Load semantic profiles
    z = np.load(DATA_PROCESSED_DIR / f'semantic_v2_window_{WID}.npz', allow_pickle=True)
    sem   = z['sem_full']             # (n, 32)
    cids  = z['customer_ids']

    # Load features for Window 3
    feats = pd.read_csv(DATA_PROCESSED_DIR / f'window_{WID}_features.csv')
    # Re-order feats to match cids
    feats = feats.set_index('CustomerID').reindex(cids).reset_index()
    print(f"Loaded {len(feats):,} customers, sem shape={sem.shape}")

    # K-Means on semantic profiles
    print(f"\nClustering semantic profiles into {N_CLUS} taste groups...")
    valid = np.linalg.norm(sem, axis=1) > 0
    print(f"  Valid (non-zero) profiles: {valid.sum()}/{len(sem)}")
    km_sem = KMeans(n_clusters=N_CLUS, random_state=42, n_init=10)
    sem_labels = km_sem.fit_predict(sem)

    # K-Means on RFM (for comparison)
    rfm = feats[['Recency','Frequency','Monetary']].fillna(0).values
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    km_rfm = KMeans(n_clusters=N_CLUS, random_state=42, n_init=10)
    rfm_labels = km_rfm.fit_predict(rfm_scaled)

    feats['sem_cluster'] = sem_labels
    feats['rfm_cluster'] = rfm_labels

    # ── Per-semantic-cluster aggregates ──────────────────────────────────────
    agg = feats.groupby('sem_cluster').agg(
        n            = ('CustomerID',  'count'),
        mean_R       = ('Recency',     'mean'),
        mean_F       = ('Frequency',   'mean'),
        mean_M       = ('Monetary',    'mean'),
        mean_CLV     = ('ActualCLV',   'mean'),
        median_CLV   = ('ActualCLV',   'median'),
        zero_rate    = ('ActualCLV',   lambda s: (s == 0).mean() * 100),
        vip_rate     = ('IsVIP',       lambda s: s.mean() * 100),
        active_mo    = ('ActiveMonths','mean'),
        prod_div     = ('ProductDiversity','mean'),
    ).round(2)
    print("\nSemantic clusters (sorted by mean CLV desc):")
    agg = agg.sort_values('mean_CLV', ascending=False)
    print(agg.to_string())
    agg.to_csv(RESULTS_DIR / 'semantic_clusters.csv')

    # ── Statistical test: do clusters differ in CLV? ─────────────────────────
    clv_per_cluster = [feats[feats.sem_cluster == c]['ActualCLV'].values
                       for c in agg.index]
    f_stat, f_pval  = f_oneway(*clv_per_cluster)
    h_stat, h_pval  = kruskal(*clv_per_cluster)
    print(f"\nANOVA F-test  (CLV ~ sem_cluster): F={f_stat:.2f}, p={f_pval:.2e}")
    print(f"Kruskal-Wallis (CLV ~ sem_cluster): H={h_stat:.2f}, p={h_pval:.2e}")

    # Same test for RFM clusters
    clv_rfm = [feats[feats.rfm_cluster == c]['ActualCLV'].values
               for c in feats.rfm_cluster.unique()]
    f_rfm, p_rfm = f_oneway(*clv_rfm)
    print(f"\nANOVA F-test  (CLV ~ rfm_cluster): F={f_rfm:.2f}, p={p_rfm:.2e}")

    # ── How orthogonal are semantic and RFM clusters? ────────────────────────
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    ari = adjusted_rand_score(sem_labels, rfm_labels)
    nmi = normalized_mutual_info_score(sem_labels, rfm_labels)
    print(f"\nSemantic vs RFM clusters:")
    print(f"  ARI (chance=0, identical=1): {ari:.4f}")
    print(f"  NMI: {nmi:.4f}")
    print(f"  Interpretation: ARI < 0.1 means semantic clusters are orthogonal to RFM clusters")

    # ── Top products per semantic cluster ────────────────────────────────────
    print("\nLoading raw transactions to find characteristic products per cluster...")
    df = pd.read_csv(CLEANED_FILE)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    # Load window info
    with open(DATA_PROCESSED_DIR / 'walk_forward_windows_5fold.pkl', 'rb') as f:
        wins = pickle.load(f)
    win3 = [w for w in wins if w['window_id'] == WID][0]
    obs_start = pd.to_datetime(win3['obs_start'])
    obs_end   = pd.to_datetime(win3['obs_end'])
    df_obs = df[(df['InvoiceDate'] >= obs_start) & (df['InvoiceDate'] <= obs_end)]

    cust2cluster = dict(zip(feats['CustomerID'], feats['sem_cluster']))
    df_obs = df_obs.copy()
    df_obs['cluster'] = df_obs['CustomerID'].map(cust2cluster)
    df_obs = df_obs.dropna(subset=['cluster'])
    df_obs['cluster'] = df_obs['cluster'].astype(int)

    cluster_products = []
    for c in agg.index:
        sub = df_obs[df_obs.cluster == c]
        top_prods = sub.groupby('Description')['TotalPrice'].sum().nlargest(5)
        for prod, spend in top_prods.items():
            cluster_products.append({
                'cluster': c, 'product': prod, 'cluster_spend': spend,
                'cluster_n_customers': agg.loc[c, 'n'],
                'cluster_mean_CLV': agg.loc[c, 'mean_CLV'],
            })
    cp_df = pd.DataFrame(cluster_products)
    cp_df.to_csv(RESULTS_DIR / 'semantic_cluster_products.csv', index=False)

    print("\nTop products per semantic cluster:")
    for c in agg.index:
        sub = cp_df[cp_df.cluster == c]
        print(f"\n  Cluster {c} (n={agg.loc[c,'n']:.0f}, mean CLV=${agg.loc[c,'mean_CLV']:.0f}):")
        for _, row in sub.iterrows():
            print(f"    {row['product'][:50]:<50}  ${row['cluster_spend']:>10,.0f}")

    # ── Figure: cluster-wise mean CLV vs RFM (showing they differ) ──────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Sem clusters
    ax = axes[0]
    cluster_order = agg.index.tolist()
    means = agg['mean_CLV'].values
    stds  = [feats[feats.sem_cluster == c]['ActualCLV'].std() for c in cluster_order]
    colors = plt.cm.Set2(np.linspace(0, 1, N_CLUS))
    bars = ax.bar(range(N_CLUS), means, yerr=stds, capsize=4, color=colors, alpha=0.85)
    ax.set_xticks(range(N_CLUS))
    ax.set_xticklabels([f'C{c}\n(n={agg.loc[c,"n"]:.0f})' for c in cluster_order], fontsize=10)
    ax.set_ylabel('Mean Actual CLV ($)')
    ax.set_title(f'Semantic Clusters (ANOVA p={f_pval:.1e})')
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(stds)*0.05,
                f'${m:.0f}', ha='center', va='bottom', fontsize=9)

    # Right: RFM clusters
    ax = axes[1]
    rfm_agg = feats.groupby('rfm_cluster')['ActualCLV'].agg(['mean','std','count']).sort_values('mean', ascending=False)
    means = rfm_agg['mean'].values
    stds  = rfm_agg['std'].values
    bars = ax.bar(range(N_CLUS), means, yerr=stds, capsize=4,
                  color=plt.cm.Pastel1(np.linspace(0,1,N_CLUS)), alpha=0.85)
    ax.set_xticks(range(N_CLUS))
    ax.set_xticklabels([f'C{c}\n(n={rfm_agg.iloc[i]["count"]:.0f})'
                        for i, c in enumerate(rfm_agg.index)], fontsize=10)
    ax.set_ylabel('Mean Actual CLV ($)')
    ax.set_title(f'RFM Clusters (ANOVA p={p_rfm:.1e})')
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(stds)*0.05,
                f'${m:.0f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle(f'Customer Clusters: Semantic Taste vs RFM (Window 3, ARI={ari:.3f})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = FIGURES_DIR / 'fig_semantic_clusters.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {out}")

    # ── Save summary stats ──────────────────────────────────────────────────
    summary = {
        'window':                  WID,
        'n_clusters':              N_CLUS,
        'sem_anova_F':             f_stat,
        'sem_anova_p':             f_pval,
        'sem_kruskal_H':           h_stat,
        'sem_kruskal_p':           h_pval,
        'rfm_anova_F':             f_rfm,
        'rfm_anova_p':             p_rfm,
        'sem_vs_rfm_ARI':          ari,
        'sem_vs_rfm_NMI':          nmi,
    }
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / 'semantic_cluster_stats.csv', index=False)
    print(f"Summary stats saved: {RESULTS_DIR/'semantic_cluster_stats.csv'}")

    print("\n" + "="*70)
    print("[DONE] Semantic cluster analysis complete")
    print("="*70)


if __name__ == "__main__":
    main()
