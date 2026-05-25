# Processed data

Full processed files are generated locally and are **not committed** to Git (size).

After running the cleaning pipeline, this folder contains:

- `online_retail_cleaned.csv` — transaction-level cleaned data
- `window_*_features.csv` — walk-forward customer features per window
- `walk_forward_windows*.pkl` — window definitions

## Sample

A small sample for inspection is committed as `sample_window_3_features.csv` (first 100 customers, Window 3).

## Regenerate

```bash
python src/02_data_cleaning.py
python src/04_feature_engineering.py
python src/22b_build_windows_5fold.py
```

See [docs/reproduction_guide.md](../docs/reproduction_guide.md).
