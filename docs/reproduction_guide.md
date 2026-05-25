# Reproduction Guide

End-to-end pipeline for the CLV / VIP targeting project (Online Retail II).

## Prerequisites

- Python 3.10+
- Raw data: [Online Retail II (UCI)](https://archive.ics.uci.edu/dataset/502/online+retail+ii) → `data/raw/online_retail_II.xlsx`

```bash
pip install -r requirements.txt
```

## Phase 1 — Data pipeline

| Step | Script | Output |
|------|--------|--------|
| Download | `src/01_data_download.py` | `data/raw/online_retail_II.xlsx` |
| Clean | `src/02_data_cleaning.py` | `data/processed/online_retail_cleaned.csv` |
| EDA | `src/03_eda.py` | exploratory figures |
| Features | `src/04_feature_engineering.py` | customer-level features |
| CLV labels | `src/05_clv_labels.py` | 90-day forward spend labels |
| Windows | `src/22b_build_windows_5fold.py` | `window_*_features.csv`, walk-forward splits |

**Notebook:** `notebooks/01_data_cleaning.ipynb`

## Phase 2 — SQL analysis

Run all 7 queries in DuckDB:

**Notebook:** `notebooks/02_sql_analysis.ipynb`  
**SQL files:** `sql/01_rfm_metrics.sql` … `sql/07_rfm_segments.sql`

Core Project 8 queries: **01** (RFM), **07** (segments), **03** (top Monetary), **02** (cohort retention).

Outputs → `notebooks/outputs/*.csv` (consumed by dashboard).

## Phase 3 — Models (walk-forward)

Execute in order (or use `src/run_all_new_experiments.py` for batch runs):

| Script | Models |
|--------|--------|
| `src/30_models_baselines.py` | Mean, Monetary, RFM, BG/NBD+Gamma-Gamma |
| `src/31_models_gbm.py` | XGBoost, LightGBM (+ log / sequence variants) |
| `src/33_models_hurdle.py` | Two-stage Hurdle classifier + regressor |
| `src/32_models_ziln.py` | ZILN deep learning |
| `src/60_models_optdist.py` | OptDist multi-ZILN |
| `src/61_models_mcd.py` | MCD-ZILN |
| `src/62b_models_drnn_fixed.py` | Dilated RNN sequence model |
| `src/16b_hurdle_semantic_v2.py` | Hurdle + semantic product graph features |

Aggregate comparison:

```bash
python src/70_final_comparison_v2.py
```

→ `results/MASTER_TABLE.csv` (mirrored as `results/tables/main_model_comparison.csv`)

**Notebook:** `notebooks/03_model_benchmark.ipynb`

## Phase 4 — Uncertainty & VIP targeting

| Script | Purpose |
|--------|---------|
| `src/18b_conformal_hurdle_alpha05.py` | Conformalized quantile regression (CQR) |
| `src/93_profit_simulation.py` | VIP profit simulation |
| `src/52_revenue_capture_curve.py` | Revenue capture @ top-k% |
| `src/50_shap_hurdle.py` | SHAP feature importance |

## Phase 5 — Dashboard & report

```bash
python dashboard/_make_predictions.py   # optional: refresh predictions
python dashboard/app.py                 # http://127.0.0.1:8050
```

Report PDF: `report/CLV_Prediction_Framework.pdf` (compile from `report/sn-article.tex` if needed).

## Key result files

| File | Content |
|------|---------|
| `results/tables/main_model_comparison.csv` | 17-model master comparison |
| `results/tables/cqr_coverage.csv` | Conformal interval coverage |
| `results/tables/vip_profit_simulation.csv` | VIP campaign simulation |
| `results/tables/semantic_variants.csv` | Semantic graph ablation |
| `results/figures/fig3_master_comparison.png` | Main comparison chart |

## src/ script index (core only)

Scripts prefixed with `_` or `99_` are diagnostics — not required for reproduction.

```
01_data_download.py      → 02_data_cleaning.py → 03_eda.py
→ 04_feature_engineering.py → 05_clv_labels.py → 22b_build_windows_5fold.py
→ 30_models_baselines.py → 31_models_gbm.py → 33_models_hurdle.py
→ 32_models_ziln.py → 60_models_optdist.py → 61_models_mcd.py → 62b_models_drnn_fixed.py
→ 70_final_comparison_v2.py → 18b_conformal_hurdle_alpha05.py → 93_profit_simulation.py
```

Semantic extension (optional): `13_product_graph.py` → `14b_customer_semantic_features_v2.py` → `16b_hurdle_semantic_v2.py`
