"""
create_fig1_pipeline.py
Generate Figure 1: Research Pipeline for CLV Prediction
Output: D:/SU26/ADY201m/paper/figures/fig.png  (4200x900 px, 300 DPI)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent.parent / "figures" / "fig.png"

# ── pipeline stages ──────────────────────────────────────────────────────────
STAGES = [
    {
        "line1": "Online Retail II",
        "line2": "Dataset",
        "color": "#C8C8C8",
        "edge":  "#909090",
    },
    {
        "line1": "Data Cleaning",
        "line2": "805K transactions",
        "color": "#F5C842",
        "edge":  "#C9A020",
    },
    {
        "line1": "Feature Engineering",
        "line2": "RFM + Behavioral\n+ Sequence",
        "color": "#7EC8A0",
        "edge":  "#4E9E70",
    },
    {
        "line1": "Walk-Forward CV",
        "line2": "3 temporal windows",
        "color": "#6BB8D4",
        "edge":  "#3A88A4",
    },
    {
        "line1": "Model Training",
        "line2": "17 models compared",
        "color": "#B39DDB",
        "edge":  "#7E57C2",
    },
    {
        "line1": "Evaluation",
        "line2": "Norm Gini\nTop-K MAPE",
        "color": "#F4A07A",
        "edge":  "#C46A3A",
    },
    {
        "line1": "VIP Targeting",
        "line2": "+ SHAP Analysis",
        "color": "#5FC9B5",
        "edge":  "#2A9980",
    },
]

N = len(STAGES)

# ── canvas ───────────────────────────────────────────────────────────────────
FIG_W_IN = 16.0       # inches  → 16 * 300 = 4800 px
FIG_H_IN = 3.2        # inches  → 3.2 * 300 = 960 px
DPI       = 300

fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI)
ax.set_xlim(-0.02, 1.02)   # extra margin so edge boxes are never clipped
ax.set_ylim(0, 1)
ax.axis('off')

# ── layout constants (in axes [0-1] coordinates) ─────────────────────────────
BOX_W   = 0.085          # box width  (7 boxes must fit in [0,1])
BOX_H   = 0.52           # box height
BOX_Y   = 0.19           # bottom y of boxes
GAP     = 0.010          # gap between box right edge and arrow start
ARROW_W = 0.018          # arrow horizontal span
STEP    = BOX_W + GAP + ARROW_W + GAP   # left-edge to left-edge distance
# Total occupied: 7*BOX_W + 6*(2*GAP + ARROW_W) = 0.595 + 0.228 = 0.823
TOTAL   = N * BOX_W + (N - 1) * (2 * GAP + ARROW_W)
X0      = (1.0 - TOTAL) / 2              # left edge of first box (≈ 0.089)

# ── draw boxes + labels ───────────────────────────────────────────────────────
box_centers = []
for i, stage in enumerate(STAGES):
    x_left = X0 + i * STEP
    xc     = x_left + BOX_W / 2
    box_centers.append(xc)

    # shadow (very subtle)
    shadow = mpatches.FancyBboxPatch(
        (x_left + 0.003, BOX_Y - 0.008),
        BOX_W, BOX_H,
        boxstyle="round,pad=0.015",
        linewidth=0,
        facecolor="#00000018",
        zorder=1,
    )
    ax.add_patch(shadow)

    # box
    box = mpatches.FancyBboxPatch(
        (x_left, BOX_Y),
        BOX_W, BOX_H,
        boxstyle="round,pad=0.015",
        linewidth=1.2,
        edgecolor=stage["edge"],
        facecolor=stage["color"],
        zorder=2,
    )
    ax.add_patch(box)

    yc = BOX_Y + BOX_H / 2

    # top divider line
    div_y = yc + 0.045
    ax.plot([x_left + 0.008, x_left + BOX_W - 0.008],
            [div_y, div_y],
            color=stage["edge"], linewidth=0.8, alpha=0.55, zorder=3)

    # line 1 — bold, above divider
    ax.text(xc, div_y + 0.07, stage["line1"],
            ha='center', va='center',
            fontsize=7.2, fontweight='bold', color='#1a1a1a',
            zorder=4, wrap=False)

    # line 2 — regular, below divider
    ax.text(xc, yc - 0.10, stage["line2"],
            ha='center', va='center',
            fontsize=6.2, color='#2e2e2e',
            zorder=4, multialignment='center',
            linespacing=1.35)

# ── draw arrows between boxes ─────────────────────────────────────────────────
arrow_y = BOX_Y + BOX_H / 2
for i in range(N - 1):
    x_start = X0 + i * STEP + BOX_W + GAP
    x_end   = x_start + ARROW_W

    ax.annotate(
        "", xy=(x_end, arrow_y), xytext=(x_start, arrow_y),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#444444",
            lw=1.4,
            mutation_scale=10,
        ),
        zorder=5,
    )

# ── step number badges — Ellipse corrected for axes aspect ratio ──────────────
# axes x-range = 1.04 (-0.02..1.02), y-range = 1.0
# fig  W = FIG_W_IN, H = FIG_H_IN
# For a visual circle: R_x/R_y = (y_range/x_range) * (FIG_H_IN/FIG_W_IN)
X_RANGE = 1.04
Y_RANGE = 1.0
R_Y = 0.052
R_X = R_Y * (Y_RANGE / X_RANGE) * (FIG_H_IN / FIG_W_IN)

for i, xc in enumerate(box_centers):
    cy = BOX_Y + BOX_H + 0.075
    badge = mpatches.Ellipse(
        (xc, cy),
        width=R_X * 2, height=R_Y * 2,
        facecolor=STAGES[i]["edge"],
        edgecolor="white", linewidth=1.0,
        zorder=6,
    )
    ax.add_patch(badge)
    ax.text(xc, cy, str(i + 1),
            ha='center', va='center',
            fontsize=6.8, fontweight='bold', color='white', zorder=7)

# ── title ─────────────────────────────────────────────────────────────────────
ax.text(0.5, 0.97,
        "Figure 1: Research Pipeline for CLV Prediction",
        ha='center', va='top', transform=ax.transAxes,
        fontsize=9.5, fontweight='bold', color='#111111',
        fontfamily='serif')

# ── save ──────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.3)
plt.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches='tight', pad_inches=0.15,
            facecolor='white', edgecolor='none')
plt.close()

w = OUTPUT_PATH.stat().st_size // 1024
print(f"Saved: {OUTPUT_PATH}")
print(f"Size:  {w} KB")
