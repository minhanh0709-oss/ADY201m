"""
19_semantic_figures.py
Generate figures for the semantic contribution sections:

  Fig A: t-SNE visualization of product embeddings (colored by category proxy)
  Fig B: Bar chart comparing 4 Hurdle variants across windows
  Fig C: CQR prediction interval coverage comparison (CQR vs MCD-ZILN)
  Fig D: SHAP importance plot showing semantic vs static features

All figures saved to figures/ directory at 300 dpi.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

DATA_DIR    = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

CLEANED_FILE = DATA_DIR / "online_retail_cleaned.csv"

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})


# ── Figure A: t-SNE of product embeddings ─────────────────────────────────────

def fig_tsne_product_embeddings():
    print("[Fig A] t-SNE product embeddings...")
    emb_path = DATA_DIR / 'product_embeddings.pkl'
    if not emb_path.exists():
        print("  SKIP: product_embeddings.pkl not found")
        return

    with open(emb_path, 'rb') as f:
        emb_data = pickle.load(f)

    embeddings  = emb_data['embeddings']
    product2idx = emb_data['product2idx']
    idx2product = emb_data['idx2product']

    # Load product descriptions for category proxy
    df = pd.read_csv(CLEANED_FILE)
    prod_desc = df.drop_duplicates('StockCode').set_index('StockCode')['Description'].to_dict()

    # Simple category keywords
    categories = {
        'Heart/Love':  ['HEART', 'LOVE', 'ROMANCE'],
        'Christmas':   ['CHRISTMAS', 'XMAS', 'SANTA', 'REINDEER', 'SNOWMAN'],
        'Kitchen':     ['CUP', 'MUG', 'TEACUP', 'KITCHEN', 'CAKE', 'BOWL'],
        'Bag/Storage': ['BAG', 'BOX', 'TIN', 'BASKET', 'STORAGE'],
        'Hot Water':   ['HOT WATER', 'HOTTIE'],
        'Bunting':     ['BUNTING', 'FLAG'],
    }

    # Assign categories
    cat_labels = ['Other'] * len(idx2product)
    for idx in range(len(idx2product)):
        prod = idx2product[idx]
        desc = str(prod_desc.get(prod, '')).upper()
        for cat, kws in categories.items():
            if any(kw in desc for kw in kws):
                cat_labels[idx] = cat
                break

    # t-SNE (subsample 500 for speed)
    from sklearn.manifold import TSNE
    np.random.seed(42)
    sample_idx = np.random.choice(len(embeddings), min(500, len(embeddings)), replace=False)
    emb_sample = embeddings[sample_idx]
    cat_sample = [cat_labels[i] for i in sample_idx]

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    coords = tsne.fit_transform(emb_sample)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    all_cats = ['Heart/Love', 'Christmas', 'Kitchen', 'Bag/Storage',
                'Hot Water', 'Bunting', 'Other']
    colors   = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12',
                 '#9b59b6', '#1abc9c', '#bdc3c7']
    cat2col  = dict(zip(all_cats, colors))

    for cat in all_cats:
        mask = np.array(cat_sample) == cat
        if mask.sum() == 0:
            continue
        alpha = 0.7 if cat != 'Other' else 0.3
        size  = 30  if cat != 'Other' else 8
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=cat2col[cat], label=cat, alpha=alpha, s=size, linewidths=0)

    ax.set_xlabel('t-SNE Dim 1')
    ax.set_ylabel('t-SNE Dim 2')
    ax.set_title('t-SNE Visualization of Product Semantic Embeddings\n'
                 '(Co-purchase PPMI + Truncated SVD, 32 dimensions)')
    ax.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()

    out = FIGURES_DIR / 'fig_product_tsne.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure B: Hurdle Variants Comparison ──────────────────────────────────────

def fig_hurdle_variants():
    print("[Fig B] Hurdle variants comparison...")
    path = RESULTS_DIR / 'semantic_walkforward.csv'
    if not path.exists():
        print("  SKIP: semantic_walkforward.csv not found")
        return

    df = pd.read_csv(path)

    variants = ['Hurdle-RFM', 'Hurdle-Seq', 'Hurdle-Sem', 'Hurdle-All']
    colors   = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    labels   = ['RFM Only', '+ Sequence', '+ Semantic', '+ Both (All)']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, ylabel, title in [
        (axes[0], 'Norm_Gini',          'Normalized Gini',        'Ranking Quality (Norm Gini)'),
        (axes[1], 'Revenue_Capture_10', 'Revenue Capture@10% (%)', 'VIP Targeting (Rev. Capture@10%)'),
    ]:
        means, stds = [], []
        for v in variants:
            sub = df[df['Model'] == v][metric]
            means.append(sub.mean())
            stds.append(sub.std())

        x = np.arange(len(variants))
        bars = ax.bar(x, means, color=colors, alpha=0.85,
                      yerr=stds, capsize=5, width=0.6, edgecolor='white')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        # Value labels
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(stds) * 0.3,
                    f'{m:.3f}', ha='center', va='bottom', fontsize=9)

        # Baseline line (Hurdle-RFM)
        ax.axhline(means[0], color='grey', linestyle='--', alpha=0.5, linewidth=1)

    plt.suptitle('Hurdle Model Variants: Impact of Feature Groups',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    out = FIGURES_DIR / 'fig_hurdle_variants.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure C: CQR Coverage Comparison ────────────────────────────────────────

def fig_cqr_coverage():
    print("[Fig C] CQR coverage comparison...")
    path = RESULTS_DIR / 'conformal_prediction.csv'
    if not path.exists():
        print("  SKIP: conformal_prediction.csv not found")
        return

    df = pd.read_csv(path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: Coverage per window
    ax = axes[0]
    windows = df['Window'].values
    coverage = df['CQR_Coverage'].values
    bars = ax.bar(windows, coverage, color='#2ecc71', alpha=0.8, edgecolor='white')
    ax.axhline(90, color='#e74c3c', linestyle='--', linewidth=2, label='Target 90%')
    ax.axhline(32, color='#3498db', linestyle=':', linewidth=2, label='MCD-ZILN 95% PI')
    ax.set_xlabel('Walk-forward Window')
    ax.set_ylabel('Empirical Coverage (%)')
    ax.set_title('Prediction Interval Coverage\n(CQR-Hurdle vs MCD-ZILN)')
    ax.set_ylim(0, 100)
    ax.legend()
    for bar, c in zip(bars, coverage):
        ax.text(bar.get_x() + bar.get_width()/2, c + 1,
                f'{c:.1f}%', ha='center', va='bottom', fontsize=10)

    # Right: Width comparison
    ax = axes[1]
    mean_w = df['Mean_Width'].values
    med_w  = df['Median_Width'].values
    x = np.arange(len(windows))
    ax.bar(x - 0.2, mean_w, width=0.35, label='Mean width', color='#9b59b6', alpha=0.8)
    ax.bar(x + 0.2, med_w,  width=0.35, label='Median width', color='#1abc9c', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'W{w}' for w in windows])
    ax.set_xlabel('Walk-forward Window')
    ax.set_ylabel('Prediction Interval Width ($)')
    ax.set_title('CQR Interval Width by Window\n(90% PI)')
    ax.legend()

    plt.suptitle('Conformalized Quantile Regression: Coverage & Width',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    out = FIGURES_DIR / 'fig_cqr_coverage.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure D: SHAP for Hurdle-All Stage 2 ─────────────────────────────────────

def fig_semantic_shap():
    print("[Fig D] SHAP importance for Hurdle-All Stage 2...")
    path = RESULTS_DIR / 'shap_hurdle_all_stage2.csv'
    if not path.exists():
        print("  SKIP: shap_hurdle_all_stage2.csv not found")
        return

    df = pd.read_csv(path).head(20)

    # Color by feature group
    def feat_group(name):
        if name.startswith('sem_'):   return 'Semantic'
        if name.startswith('seq_'):   return 'Sequence'
        if name.startswith('log_'):   return 'Interaction'
        if name in ['M_per_F','M_per_T','Active_ratio']: return 'Interaction'
        if name in ['Recency','Frequency','Monetary']:    return 'RFM'
        return 'Behavioral'

    grp_colors = {
        'Semantic':    '#e74c3c',
        'Sequence':    '#3498db',
        'Interaction': '#f39c12',
        'RFM':         '#2ecc71',
        'Behavioral':  '#9b59b6',
    }

    df['group'] = df['feature'].apply(feat_group)
    df['color'] = df['group'].map(grp_colors)

    fig, ax = plt.subplots(figsize=(9, 7))
    y = np.arange(len(df))
    ax.barh(y, df['mean_abs_shap'], color=df['color'], alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(df['feature'], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP value| (Stage 2: Revenue Regressor)')
    ax.set_title('Feature Importance — Hurdle-All Stage 2\n(SHAP TreeExplainer)')

    patches = [mpatches.Patch(color=c, label=g) for g, c in grp_colors.items()
               if g in df['group'].values]
    ax.legend(handles=patches, loc='lower right')

    plt.tight_layout()
    out = FIGURES_DIR / 'fig_semantic_shap.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    print("\n" + "="*70)
    print("[19] Generating semantic contribution figures")
    print("="*70)

    fig_tsne_product_embeddings()
    fig_hurdle_variants()
    fig_cqr_coverage()
    fig_semantic_shap()

    print("\n[DONE] Figures generated in", FIGURES_DIR)


if __name__ == "__main__":
    main()
