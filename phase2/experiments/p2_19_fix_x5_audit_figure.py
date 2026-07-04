# -*- coding: utf-8 -*-
"""Regenerate phase2/figures/audit_x5_uplift_signal.png as a CLEAN 2-panel figure.

The old version was drawn before purchases.csv.gz finished downloading, so its
right panel printed a "purchases.csv.gz not yet available" placeholder. This
rebuilds it entirely from the verified audit statistics (x5_feasibility.json),
so no 670 MB raw file is needed.

Left  : target rate Control vs Treated (with raw-uplift annotation).
Right : raw uplift point estimate with 95% CI (excludes 0 -> significant signal).
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
STATS = os.path.join(ROOT, "results", "x5_feasibility.json")
OUT = os.path.join(ROOT, "figures", "audit_x5_uplift_signal.png")

s = json.load(open(STATS))["stats"]
c, tr = s["target_rate_control"], s["target_rate_treated"]
up = s["raw_uplift"]
lo, hi = s["raw_uplift_ci95_lo"], s["raw_uplift_ci95_hi"]
n_t, n_c = s["train_n_treatment"], s["train_n_control"]

GREY, BLUE = "#9aa6b2", "#1f6fb0"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.4), gridspec_kw={"width_ratios": [1, 1]})

# --- Left: target rate by arm ---
bars = ax1.bar(["Control", "Treated"], [c, tr], color=[GREY, BLUE], width=0.6,
               edgecolor="white")
for b, v in zip(bars, [c, tr]):
    ax1.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
             ha="center", va="bottom", fontsize=11)
ax1.set_ylim(0, 0.74)
ax1.set_ylabel("P(target = 1)")
ax1.set_title(f"Target rate by arm  (raw uplift = +{up * 100:.2f} pp)", fontsize=11)
ax1.spines[["top", "right"]].set_visible(False)

# --- Right: raw uplift with 95% CI ---
ax2.errorbar([up], [0], xerr=[[up - lo], [hi - up]], fmt="o", color=BLUE,
             markersize=9, capsize=7, lw=2)
ax2.axvline(0, color="#c0392b", ls="--", lw=1.2)
ax2.text(0.0005, 0.32, "no effect", color="#c0392b", fontsize=9, rotation=90, va="bottom")
ax2.set_yticks([])
ax2.set_xlim(-0.006, 0.045)
ax2.set_xlabel("Raw uplift (treated − control)")
ax2.set_title("95% CI excludes 0  →  signal is significant", fontsize=11)
ax2.text(up, -0.45, f"{up:.4f}  [{lo:.4f}, {hi:.4f}]", ha="center", fontsize=10)
ax2.text(0.022, 0.62,
         f"RCT design: {n_t:,} treated\nvs {n_c:,} control\n(ratio {n_t/(n_t+n_c):.3f})",
         fontsize=9, color="#333",
         bbox=dict(boxstyle="round,pad=0.4", fc="#eef3f7", ec="#cdd7df"))
ax2.set_ylim(-0.7, 1.0)
ax2.spines[["top", "right", "left"]].set_visible(False)

fig.suptitle("X5 RetailHero — treatment/control uplift signal (Week-1 audit, n = 200,039)",
             fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("saved:", OUT)
