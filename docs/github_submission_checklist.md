# GitHub submission checklist (ADY201m)

## What is on GitHub (clean layout)

| Folder | Contents |
|--------|----------|
| `src/` | **Descriptive script names** (no numeric prefixes) — see `src/README.md` |
| `results/figures/` | **Single copy** of all paper figures |
| `results/tables/` | Summary CSVs only (`main_model_comparison.csv`, etc.) |
| `report/` | PDF + LaTeX (figures referenced from `../results/figures/`) |
| `sql/` | 7 queries with clear names `01_rfm_metrics.sql` … |
| `notebooks/` | 01, 02, 03 only |
| `dashboard/` | App + screenshots (no audit log) |

**Not on GitHub:** raw xlsx, full processed data, audit log, duplicate figures, legacy numbered scripts, local `presentation/`.

## Push updates

```powershell
cd d:\SU26\ADY201m\paper
git add -u
git add README.md .gitignore docs/ data/ notebooks/ src/README.md
git status
git commit -m "Clean repo: remove duplicate figures, rename src scripts for clarity"
git push origin main
```

## Self-check

- [ ] No `report/figures/` duplicate of `results/figures/`
- [ ] No `AI_AuditLog*` on remote
- [ ] `src/` uses names like `data_cleaning.py`, `models_hurdle.py`
- [ ] `main_model_comparison.csv` present (17 models)
