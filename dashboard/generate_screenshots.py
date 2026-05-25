"""Generate static dashboard screenshots for GitHub evidence."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dashboard" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=0.95)


def save(name: str):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close()
    print("saved", name)


# 01 Overview — monthly active customers
mac = pd.read_csv(ROOT / "notebooks" / "outputs" / "q6_mac.csv")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(mac["year_month"], mac["active_customers"], marker="o", linewidth=2)
ax.set_title("Dashboard — Overview: Monthly Active Customers")
ax.set_xlabel("Month")
ax.set_ylabel("Active customers")
save("01_overview.png")

# 02 RFM matrix — segment counts
rfm = pd.read_csv(ROOT / "notebooks" / "outputs" / "q7_rfm_segments.csv")
seg_col = "RFM_Segment" if "RFM_Segment" in rfm.columns else rfm.columns[-1]
counts = rfm[seg_col].value_counts().head(12)
fig, ax = plt.subplots(figsize=(8, 5))
counts.sort_values().plot(kind="barh", ax=ax, color="#2E86AB")
ax.set_title("Dashboard — RFM Matrix: Segment Distribution")
ax.set_xlabel("Customers")
save("02_rfm_matrix.png")

# 03 CLV forecast — predicted vs actual (top 50)
pred = pd.read_csv(ROOT / "dashboard" / "data" / "customer_predictions_w3.csv")
top = pred.nlargest(50, "PredictedCLV")
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(top["ActualCLV"], top["PredictedCLV"], alpha=0.7, s=40)
mx = max(top["ActualCLV"].max(), top["PredictedCLV"].max()) * 1.05
ax.plot([0, mx], [0, mx], "--", color="gray", lw=1)
ax.set_title("Dashboard — CLV Forecast (Window 3, top 50 VIP)")
ax.set_xlabel("Actual CLV ($)")
ax.set_ylabel("Predicted CLV ($)")
save("03_clv_forecast.png")

# 04 Campaign targeting — top VIP with intervals
camp = pred.nlargest(15, "PredictedCLV").sort_values("PredictedCLV")
y = range(len(camp))
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(list(y), camp["PredictedCLV"], color="#A23B72", alpha=0.85)
lower_err = (camp["PredictedCLV"] - camp["CQR_Lower"]).clip(lower=0)
upper_err = (camp["CQR_Upper"] - camp["PredictedCLV"]).clip(lower=0)
ax.errorbar(
    camp["PredictedCLV"],
    list(y),
    xerr=[lower_err, upper_err],
    fmt="none",
    ecolor="#333",
    capsize=3,
)
ax.set_yticks(list(y))
ax.set_yticklabels(camp["CustomerID"].astype(str))
ax.set_title("Dashboard — Campaign Targeting: Top VIP + CQR intervals")
ax.set_xlabel("Predicted CLV ($)")
save("04_campaign_targeting.png")

print("done:", OUT)
