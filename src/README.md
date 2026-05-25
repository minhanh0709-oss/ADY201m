# Python pipeline scripts

Numbered scripts run in pipeline order. See [docs/reproduction_guide.md](../docs/reproduction_guide.md) for the full execution guide.

## Core path (minimum reproduction)

```
02_data_cleaning.py → 04_feature_engineering.py → 22b_build_windows_5fold.py
→ 30_models_baselines.py → 31_models_gbm.py → 33_models_hurdle.py
→ 70_final_comparison_v2.py
```

## Key modules

| Script | Role |
|--------|------|
| `02_data_cleaning.py` | Transaction cleaning |
| `04_feature_engineering.py` | RFM + behavioral features |
| `22b_build_windows_5fold.py` | Walk-forward window splits |
| `30–33, 32, 60–62` | Model training (baselines → DL) |
| `70_final_comparison_v2.py` | Aggregate `MASTER_TABLE.csv` |
| `18b_conformal_hurdle_alpha05.py` | CQR prediction intervals |
| `93_profit_simulation.py` | VIP profit simulation |

Diagnostic scripts (`99_*`, `_*.py`) are for local debugging only.
