# GitHub submission checklist (ADY201m)

Use this when pushing to a public repo (e.g. `clv-online-retail`).

## Before push

1. Close Excel if `AI_AuditLog*.xlsx` is open locally.
2. Confirm `.gitignore` excludes audit log, raw data, large binaries, zip archives.
3. Do **not** commit: `data/raw/*.xlsx`, full `data/processed/*`, audit scripts, course PDFs.

## Reset messy staging (if needed)

```powershell
cd d:\SU26\ADY201m\paper
git reset
```

## Add only submission files

```powershell
git add .gitignore LICENSE README.md requirements.txt
git add data/raw/README.md data/processed/README.md data/processed/sample_window_3_features.csv
git add sql/
git add notebooks/01_data_cleaning.ipynb notebooks/02_sql_analysis.ipynb notebooks/03_model_benchmark.ipynb
git add notebooks/outputs/
git add src/*.py src/README.md
git add results/tables/ results/figures/
git add dashboard/app.py dashboard/data_loader.py dashboard/data/ dashboard/pages/
git add dashboard/README.md dashboard/generate_screenshots.py dashboard/screenshots/
git add report/
git add docs/
```

## Verify nothing sensitive staged

```powershell
git status
git diff --cached --stat
```

Check that these do **not** appear:

- `AI_AuditLog*`, `audit_evidence/`, `fill_audit_log.py`
- `data/raw/online_retail_II.xlsx`
- `data/processed/window_*.csv`, `*.npy`, `*.pkl`
- `*.zip`, `OVERLEAF*`, `paper_sn/`, `paper_latex/`

## Commit & push (when ready)

```powershell
git commit -m "Add ADY201m CLV project: pipeline, SQL, models, dashboard, report"
git remote add origin https://github.com/YOUR_USERNAME/clv-online-retail.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub account.

## LMS (separate from GitHub)

Submit on LMS:

- Project Planning PDF → `docs/project_plan.pdf`
- Final Report PDF → `report/CLV_Prediction_Framework.pdf`
- AI Audit Log → `dashboard/AI_AuditLog_VuMinhAnhNguyen_ADY201m_Group5.xlsx` (local only)

## Quick rubric self-check

| Requirement | Evidence |
|-------------|----------|
| SQL (RFM, segments, top Monetary, cohort) | `sql/01`, `07`, `03`, `02` |
| ≥ 5 ML models | `results/tables/main_model_comparison.csv` (17 rows) |
| 3+ notebooks | `01`, `02`, `03` |
| Dashboard | `dashboard/` + `screenshots/` |
| Final report | `report/CLV_Prediction_Framework.pdf` |
| AI Audit Log | LMS only (note in README) |
