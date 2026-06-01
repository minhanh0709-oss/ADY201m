# CLV Prediction & VIP Targeting — Online Retail II

**Course:** ADY201m — Applied Data Science  
**Student:** Vu Minh Anh Nguyen (SE203412)  
**Topic:** Project 8 — Customer Lifetime Value (CLV) prediction and VIP targeting on the UCI Online Retail II dataset.

This repository contains the full analytics pipeline: data cleaning, SQL analysis, feature engineering, walk-forward model benchmarking (17+ models), conformal prediction intervals, and a Plotly Dash dashboard.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 1. Download raw data → data/raw/ (see data/raw/README.md)
python src/data_cleaning.py
python src/feature_engineering.py
python src/build_walkforward_windows.py

# 2. SQL analysis (DuckDB)
jupyter notebook notebooks/02_sql_analysis.ipynb

# 3. Model benchmark (pre-computed results in results/tables/)
jupyter notebook notebooks/03_model_benchmark.ipynb

# 4. Dashboard
python dashboard/app.py
# → http://127.0.0.1:8050
```

See [docs/reproduction_guide.md](docs/reproduction_guide.md) for the full script execution order.

## Repository structure

```text
├── README.md
├── requirements.txt
├── data/raw/              # Raw UCI download (not committed)
├── data/processed/        # Cleaned features + sample CSV
├── sql/                   # 7 DuckDB queries (Project 8 core: 01, 02, 03, 07)
├── notebooks/             # 01 cleaning, 02 SQL, 03 model benchmark
├── src/                   # Python pipeline scripts
├── results/tables/        # Model comparison & experiment tables
├── results/figures/       # Key figures for report
├── dashboard/             # Plotly Dash app + predictions export
├── report/                # Final Springer-format PDF + LaTeX source
└── docs/                  # Reproduction guide + project planning PDF
```

## SQL — Project 8 core queries

| File | ADY201m requirement |
|------|---------------------|
| `sql/01_rfm_metrics.sql` | RFM metrics per customer |
| `sql/07_rfm_segments.sql` | RFM segment groups |
| `sql/03_top_customers_monetary.sql` | Top customers by Monetary |
| `sql/02_cohort_retention.sql` | Cohort retention rate |

See [sql/README.md](sql/README.md) for the full query list.

## Models

Walk-forward evaluation across 5 temporal windows. Primary comparison table:

**`results/tables/main_model_comparison.csv`** — 17 models including baselines (Mean, Monetary, RFM, BG/NBD), linear/GBM models, two-stage Hurdle, ZILN-family deep models, and sequence extensions.

## Dashboard

Plotly Dash application with four views:

1. **Overview** — dataset summary & KPIs  
2. **RFM Matrix** — RFM segmentation heatmap  
3. **CLV Forecast** — predicted customer value  
4. **Campaign Targeting** — VIP list with prediction intervals  

Run: `python dashboard/app.py`

## Final report

- PDF: [report/CLV_Prediction_Framework.pdf](report/CLV_Prediction_Framework.pdf)  
- LaTeX source: `report/sn-article.tex`

## Course Requirement Mapping

| ADY201m Requirement | Repository Evidence |
|---|---|
| Data ingestion, cleaning, EDA, feature engineering, model | `notebooks/01_data_cleaning.ipynb`, `src/data_cleaning.py`, `src/feature_engineering.py` |
| SQL analysis | `sql/`, `notebooks/02_sql_analysis.ipynb` |
| At least 5 ML models | `results/tables/main_model_comparison.csv` (17 models) |
| Regression analysis | `notebooks/03_model_benchmark.ipynb`, `src/models_*.py` |
| Visualization | `results/figures/` (single copy — no duplicate in `report/`) |
| Dashboard / tool | `dashboard/` (RFM matrix, CLV forecast, VIP list) |
| Final report (Springer 10–12 pp.) | `report/CLV_Prediction_Framework.pdf` |
| AI Audit Log | **Submitted separately via LMS** (not in this repo) |

## AI Audit Log

The AI Audit Log is submitted separately via LMS (ADY201m requirement).  
It is intentionally excluded from this repository to avoid exposing private reflection, prompt history, and evidence materials.

## Data license

Online Retail II is provided by UCI ML Repository. Download instructions: [data/raw/README.md](data/raw/README.md).

## License

MIT — see [LICENSE](LICENSE).
