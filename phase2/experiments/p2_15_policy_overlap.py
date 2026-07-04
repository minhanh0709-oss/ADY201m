"""Phase 2 — REVISION P1.5 — Targeting policy overlap (RQ4 support).

Quantifies how differently the policies select customers by the overlap of their top-K% sets.
Low overlap between CLV-only and Uplift-only is direct evidence that value and persuadability
identify different customers -- the paper's thesis.

Output: results/x5_policy_overlap.csv, figures/x5_policy_overlap.png
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config as C  # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def topset(score, k):
    n_top = max(1, int(len(score) * k))
    return set(np.argsort(score)[-n_top:])


def overlap(a, b):
    return len(a & b) / len(a)   # fraction of A's top-set also in B's top-set (|A|=|B|)


def main():
    s = pd.read_parquet(C.PROCESSED / "x5_uplift_test_scores.parquet")
    feats = pd.read_parquet(C.PROCESSED / "x5_uplift_features.parquet")[
        ["client_id", "recency_days", "n_transactions", "monetary"]]
    s = s.merge(feats, on="client_id", how="left")
    v = s["monetary"].values.astype(float)
    mt = pd.read_csv(C.RESULTS / "x5_uplift_models.csv")
    best = mt.sort_values("Qini_AUC", ascending=False).iloc[0]["model"]
    up = s[f"uplift_{best}"].values

    def z(a):
        a = np.asarray(a, float); sd = a.std(); return (a - a.mean()) / sd if sd else a * 0
    rfm = -z(s["recency_days"]) + z(s["n_transactions"]) + z(s["monetary"])

    policies = {"RFM": rfm, "Value(CLV proxy)": v, f"Uplift({best})": up,
                "Value-adjusted": up * v}
    names = list(policies)
    rows = []
    for k in (0.10, 0.20):
        sets = {n: topset(policies[n], k) for n in names}
        for i in range(len(names)):
            for j in range(len(names)):
                if i < j:
                    ov = overlap(sets[names[i]], sets[names[j]])
                    rows.append({"K": f"{int(k*100)}%", "policy_A": names[i],
                                 "policy_B": names[j], "overlap": round(ov, 3)})
    res = pd.DataFrame(rows)
    res.to_csv(C.RESULTS / "x5_policy_overlap.csv", index=False)
    print("=== policy top-K overlap ===")
    print(res.to_string(index=False))

    # heatmap at K=10%
    k = 0.10
    sets = {n: topset(policies[n], k) for n in names}
    M = np.array([[overlap(sets[a], sets[b]) for b in names] for a in names])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Top-10% targeting overlap between policies")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(C.FIGURES / "x5_policy_overlap.png", dpi=140)
    cu = overlap(sets["Value(CLV proxy)"], sets[f"Uplift({best})"])
    print(f"\n  Value vs Uplift top-10% overlap = {cu:.3f} "
          f"(low overlap => value and persuadability pick different customers)")


if __name__ == "__main__":
    main()
