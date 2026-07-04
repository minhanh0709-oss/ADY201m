"""Phase 2 — cross-context CLV comparison (RQ1 scaffolding).

Summarises the two CLV contexts (Online Retail II vs Dunnhumby) on the dimensions that
determine whether the Hurdle-Semantic framework transfers: zero-inflation, tail heaviness,
frequency, value scale. Reads the feature tables built by p2_02_build_features.py.

Output: tables/context_comparison.csv, figures/context_comparison.png
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config as C  # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATASETS = {"Online Retail II": "online_retail", "Dunnhumby": "dunnhumby"}


def summarise(df):
    clv = df["actual_clv"]
    nz = clv[clv > 0]
    return {
        "n_customers": len(df),
        "zero_clv_rate": round(float((clv <= 0).mean()), 4),
        "mean_clv": round(float(clv.mean()), 2),
        "median_nonzero_clv": round(float(nz.median()), 2),
        "clv_skew_nonzero": round(float(sps.skew(nz)), 2),
        "mean_frequency": round(float(df["frequency"].mean()), 2),
        "median_frequency": float(df["frequency"].median()),
        "mean_recency_days": round(float(df["recency_days"].mean()), 1),
        "mean_product_diversity": round(float(df["product_diversity"].mean()), 1),
        "mean_taste_diversity": round(float(df["sem_taste_diversity"].mean()), 1),
    }


def main():
    rows = {}
    frames = {}
    for label, key in DATASETS.items():
        p = C.PROCESSED / f"{key}_features.parquet"
        if not p.exists():
            print(f"  [skip] {label}: {p.name} not built yet")
            continue
        df = pd.read_parquet(p)
        frames[label] = df
        rows[label] = summarise(df)

    comp = pd.DataFrame(rows)
    out = C.assert_phase2_path(C.TABLES / "context_comparison.csv")
    comp.to_csv(out)
    print(comp.to_string())
    print(f"\n  -> {out}")

    # figure: zero-rate + log-CLV distribution per context
    if frames:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        labels = list(frames)
        axes[0].bar(labels, [rows[l]["zero_clv_rate"] for l in labels],
                    color=["#2c7fb8", "#d95f0e"])
        axes[0].set_title("Zero-CLV rate by context (Hurdle Stage-1 relevance)")
        axes[0].set_ylabel("fraction with zero future spend")
        for l in labels:
            df = frames[l]
            nz = df["actual_clv"]; nz = nz[nz > 0]
            axes[1].hist(np.log1p(nz), bins=40, alpha=0.55, label=l, density=True)
        axes[1].set_title("log1p(non-zero CLV) density")
        axes[1].set_xlabel("log1p(future spend)")
        axes[1].legend()
        fig.tight_layout()
        figp = C.assert_phase2_path(C.FIGURES / "context_comparison.png")
        fig.savefig(figp, dpi=140)
        print(f"  -> {figp}")


if __name__ == "__main__":
    main()
