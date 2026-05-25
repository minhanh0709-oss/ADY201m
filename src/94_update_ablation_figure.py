"""
94_update_ablation_figure.py
Redraw Fig 5 using the original Optuna-tuned ablation data across all 3 windows.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"

def main():
    df = pd.read_csv(RESULTS_DIR / "ablation_walkforward.csv")
    ORDER = ['RFM only', '+ Behavioral', '+ Interactions', '+ Sequence (Full)']
    WINDOWS = [1, 2, 3]
    COLORS = ['#3498DB', '#27AE60', '#E67E22', '#9B59B6']
    LABELS = ['RFM only (3)', 'RFM + Behavioral (10)',
              'RFM + Beh + Interact (16)', 'Full + Sequence (23)']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, (metric, ylabel, title) in zip(axes, [
        ('Norm_Gini',        'Normalized Gini',            '(a) Normalized Gini'),
        ('Revenue_Capture_10', 'Revenue Capture @ 10% (%)', '(b) Revenue Capture @ 10%'),
    ]):
        x = np.arange(len(WINDOWS))
        bar_w = 0.19

        for i, (grp, col, lab) in enumerate(zip(ORDER, COLORS, LABELS)):
            vals = []
            for w in WINDOWS:
                row = df[(df['Window'] == w) & (df['Feature Group'] == grp)]
                vals.append(row[metric].values[0] if len(row) > 0 else np.nan)

            offset = (i - 1.5) * bar_w
            bars = ax.bar(x + offset, vals, width=bar_w, color=col,
                          label=lab, alpha=0.88, edgecolor='white', linewidth=0.6)

            # Value labels on top of bars
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    fmt = f'{v:.3f}' if metric == 'Norm_Gini' else f'{v:.1f}%'
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                            fmt, ha='center', va='bottom', fontsize=6.5, color='#333')

        ax.set_xticks(x)
        ax.set_xticklabels([f'Window {w}' for w in WINDOWS], fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8.5, loc='lower right')
        ax.grid(True, axis='y', alpha=0.3)

        # y-axis: zoom in
        all_vals = [df[df['Feature Group']==g][metric].values for g in ORDER]
        all_vals = np.concatenate(all_vals)
        y_min = np.nanmin(all_vals) * 0.97
        y_max = np.nanmax(all_vals) * 1.025
        ax.set_ylim(y_min, y_max)

    # Summary table below
    summary = df.groupby('Feature Group').agg(
        NG_mean=('Norm_Gini', 'mean'), NG_std=('Norm_Gini', 'std')
    ).reindex(ORDER)
    base = summary.loc['RFM only', 'NG_mean']

    subtitle = '  |  '.join(
        f"{g}: NG={summary.loc[g,'NG_mean']:.4f} ({summary.loc[g,'NG_mean']-base:+.4f})"
        for g in ORDER
    )

    plt.suptitle(
        'Feature Group Ablation Study Across All 3 Walk-Forward Windows\n'
        f'Mean delta vs RFM-only:  {subtitle}',
        fontsize=9.5, fontweight='bold'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out = FIGURES_DIR / 'fig5_ablation.png'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out}  ({out.stat().st_size//1024} KB)")

    # Copy to both paper directories
    import shutil
    for dest in [
        Path(__file__).parent.parent / 'paper_latex' / 'figures' / 'fig5_ablation.png',
        Path(__file__).parent.parent / 'paper_sn'    / 'figures' / 'fig5_ablation.png',
    ]:
        shutil.copy(out, dest)
        print(f"  Copied to: {dest}")

    print("\n[Mean across 3 windows]")
    for g in ORDER:
        v = summary.loc[g,'NG_mean']
        print(f"  {g:25s}  NG={v:.4f}  dNG={v-base:+.4f}")

if __name__ == "__main__":
    main()
