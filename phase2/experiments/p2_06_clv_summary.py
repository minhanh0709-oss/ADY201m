"""Phase 2 — consolidate the CLV (RQ1/RQ2) results into one cross-context summary table.
Reads the per-dataset benchmark/ablation/CQR/SHAP CSVs and emits a paper-ready summary.

Output: tables/clv_crosscontext_summary.csv
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config as C  # noqa: E402

DS = {"online_retail": "Online Retail II (e-com)", "dunnhumby": "Dunnhumby (grocery)"}
rows = []
for key, label in DS.items():
    b = pd.read_csv(C.RESULTS / f"{key}_clv_benchmark.csv")
    a = pd.read_csv(C.RESULTS / f"{key}_clv_ablation.csv")
    cqr = pd.read_csv(C.RESULTS / f"{key}_cqr_coverage.csv")
    s1 = pd.read_csv(C.RESULTS / f"{key}_shap_stage1.csv")
    s2 = pd.read_csv(C.RESULTS / f"{key}_shap_stage2.csv")
    b = b.dropna(subset=["Norm_Gini"]).sort_values("Norm_Gini", ascending=False)
    hurdle = b[b["model"] == "Hurdle"].iloc[0]
    best = b.iloc[0]
    a_best = a.sort_values("Norm_Gini", ascending=False).iloc[0]
    cqr90 = cqr[cqr["nominal_coverage"] == 0.90].iloc[0]
    rows.append({
        "context": label,
        "best_model": best["model"],
        "best_NG": best["Norm_Gini"],
        "Hurdle_NG": hurdle["Norm_Gini"],
        "Hurdle_RC@10": hurdle["RevCapture_10"],
        "Hurdle_Precision@10": hurdle["Precision_10"],
        "best_ablation_variant": a_best["variant"],
        "best_ablation_NG": a_best["Norm_Gini"],
        "CQR90_empirical": cqr90["empirical_coverage"],
        "CQR90_width": cqr90["mean_interval_width"],
        "stage1_top_feature": s1.iloc[0]["feature"],
        "stage2_top_feature": s2.iloc[0]["feature"],
    })

summary = pd.DataFrame(rows).set_index("context").T
out = C.assert_phase2_path(C.TABLES / "clv_crosscontext_summary.csv")
summary.to_csv(out)
print(summary.to_string())
print(f"\n  -> {out}")
