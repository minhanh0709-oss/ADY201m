"""Compact cross-context SHAP figure: Stage-2 (spend) top features for Online Retail II vs
Dunnhumby, showing monetary drives e-commerce spend while recent-sequence spend drives grocery."""
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config as C  # noqa: E402

NICE = {  # readable feature labels
    "monetary": "monetary", "seq_spend_recent_half": "recent-seq spend",
    "spend_per_active_day": "spend/active day", "avg_order_value": "avg order value",
    "recency_days": "recency", "seq_active_periods": "active periods",
    "log_monetary": "log monetary", "frequency": "frequency", "n_baskets": "n baskets",
    "avg_basket_value": "avg basket value", "sem_taste_drift": "taste drift",
    "total_quantity": "total qty", "seq_spend_slope": "spend slope",
}


def panel(ax, key, title, color):
    d = pd.read_csv(C.RESULTS / f"{key}_shap_stage2.csv").head(6).iloc[::-1]
    labels = [NICE.get(f, f.replace("_", " ")) for f in d["feature"]]
    ax.barh(labels, d["mean_abs_shap_norm"], color=color)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("mean |SHAP| (normalised)", fontsize=9)
    ax.tick_params(labelsize=8)


fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.9))
panel(axes[0], "online_retail", "Online Retail II (e-commerce)", "#2c7fb8")
panel(axes[1], "dunnhumby", "Dunnhumby (grocery)", "#d95f0e")
fig.tight_layout()
out = C.FIGURES / "shap_crosscontext.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
