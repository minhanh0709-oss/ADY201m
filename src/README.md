# Python pipeline scripts

Scripts use **descriptive names** (no numeric prefixes). Run in this order:

## Core pipeline

| Step | Script | Output |
|------|--------|--------|
| Download | `data_download.py` | `data/raw/online_retail_II.xlsx` |
| Clean | `data_cleaning.py` | `data/processed/online_retail_cleaned.csv` |
| EDA | `eda.py` | exploratory figures |
| Features | `feature_engineering.py` | customer-level features |
| Labels | `clv_labels.py` | 90-day forward CLV labels |
| Windows | `build_walkforward_windows.py` | `window_*_features.csv` |

## Models (walk-forward)

| Script | Models |
|--------|--------|
| `models_baselines.py` | Mean, Monetary, RFM, BG/NBD+GG |
| `models_gbm.py` | XGBoost, LightGBM variants |
| `models_hurdle.py` | Two-stage Hurdle (proposed) |
| `models_ziln.py` | ZILN deep learning |
| `models_optdist.py` | OptDist |
| `models_mcd.py` | MCD-ZILN |
| `models_drnn.py` | Dilated RNN |
| `final_model_comparison.py` | Aggregate → `results/MASTER_TABLE.csv` |

## Extensions

| Script | Purpose |
|--------|---------|
| `product_graph.py` | Product co-purchase embeddings |
| `customer_semantic_features.py` | Semantic features per window |
| `hurdle_semantic.py` | Hurdle + semantic variants |
| `conformal_cqr.py` | CQR prediction intervals |
| `profit_simulation.py` | VIP campaign simulation |
| `shap_hurdle.py` | SHAP interpretability |
| `run_experiments.py` | Batch runner for semantic + CQR + dRNN |

See [docs/reproduction_guide.md](../docs/reproduction_guide.md) for full details.
