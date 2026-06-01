# Reproduction Guide

End-to-end pipeline for CLV / VIP targeting (Online Retail II).

## Prerequisites

- Python 3.10+
- Raw data: [Online Retail II (UCI)](https://archive.ics.uci.edu/dataset/502/online+retail+ii) → `data/raw/online_retail_II.xlsx`

```bash
pip install -r requirements.txt
```

## Phase 1 — Data pipeline

| Step | Script | Output |
|------|--------|--------|
| Download | `src/data_download.py` | `data/raw/online_retail_II.xlsx` |
| Clean | `src/data_cleaning.py` | `data/processed/online_retail_cleaned.csv` |
| EDA | `src/eda.py` | exploratory figures |
| Features | `src/feature_engineering.py` | customer-level features |
| CLV labels | `src/clv_labels.py` | 90-day forward spend labels |
| Windows | `src/build_walkforward_windows.py` | `window_*_features.csv` |

**Notebook:** `notebooks/01_data_cleaning.ipynb`

## Phase 2 — SQL analysis

**Notebook:** `notebooks/02_sql_analysis.ipynb`  
**SQL files:** `sql/01_rfm_metrics.sql` … `sql/07_rfm_segments.sql`

Core Project 8: **01** (RFM), **07** (segments), **03** (top Monetary), **02** (cohort).

Outputs → `notebooks/outputs/*.csv`

## Phase 3 — Models (walk-forward)

| Script | Models |
|--------|--------|
| `src/models_baselines.py` | Mean, Monetary, RFM, BG/NBD+Gamma-Gamma |
| `src/models_gbm.py` | XGBoost, LightGBM |
| `src/models_hurdle.py` | Two-stage Hurdle |
| `src/models_ziln.py` | ZILN |
| `src/models_optdist.py` | OptDist |
| `src/models_mcd.py` | MCD-ZILN |
| `src/models_drnn.py` | Dilated RNN |

Aggregate:

```bash
python src/final_model_comparison.py
```

→ `results/tables/main_model_comparison.csv`

**Notebook:** `notebooks/03_model_benchmark.ipynb`

## Phase 4 — Uncertainty & VIP targeting

| Script | Purpose |
|--------|---------|
| `src/conformal_cqr.py` | Conformalized quantile regression |
| `src/profit_simulation.py` | VIP profit simulation |
| `src/revenue_capture.py` | Revenue capture @ top-k% |
| `src/shap_hurdle.py` | SHAP feature importance |

Semantic extension: `product_graph.py` → `customer_semantic_features.py` → `hurdle_semantic.py`

## Phase 5 — Dashboard & report

```bash
python dashboard/app.py
```

Report PDF: `report/CLV_Prediction_Framework.pdf`  
Figures (single copy): `results/figures/`

## Key result files

| File | Content |
|------|---------|
| `results/tables/main_model_comparison.csv` | 17-model comparison |
| `results/tables/cqr_coverage.csv` | Conformal coverage |
| `results/tables/vip_profit_simulation.csv` | VIP simulation |
