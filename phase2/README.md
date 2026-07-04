# Phase 2 — Explainable CLV Ranking & Uplift-Aware Targeting Across Retail Contexts

> **Isolation guarantee.** Phase 2 lives entirely in `phase2/`. It **never writes** to
> Phase-1 paths (`../data`, `../src`, `../results`, `../figures`, `../dashboard`, `../paper_final`).
> Phase-1 artifacts are only ever *read* (e.g. Online Retail II cleaned data for the
> cross-context CLV comparison), never modified.

## Research framing

CLV tells us *who is valuable*. Uplift tells us *who is responsive*.
**Value-adjusted uplift** tells us *who is worth targeting*.

### Research questions
- **RQ1 — Cross-context CLV robustness.** How robust is the explainable Hurdle–Semantic
  CLV framework when re-estimated across UK online retail and US grocery retail?
- **RQ2 — Semantic & behavioural value.** Do semantic product profiles + behavioural
  sequence features add incremental value beyond RFM for CLV ranking / calibration /
  interpretation?
- **RQ3 — Uplift-aware targeting.** On campaign data with treatment/control labels, does
  uplift-aware targeting beat CLV-only, RFM-only, and response-model targeting?
- **RQ4 — Value-adjusted uplift.** Does combining predicted value with uplift improve
  simulated campaign efficiency, and what transferable principles follow for
  budget-constrained emerging retail markets?

## Dataset roles (causal-clean)

| Dataset            | Role                       | Used for                                          | Do **not** claim |
|--------------------|----------------------------|---------------------------------------------------|------------------|
| Online Retail II   | Baseline CLV (Phase 1)     | CLV ranking, Hurdle, SHAP, CQR (read-only reuse)  | No treatment     |
| Dunnhumby          | External CLV validation    | Household CLV, grocery context, semantic profiling| Coupon redemption is **not** a clean causal treatment |
| X5 RetailHero      | Main uplift dataset        | treatment/control, uplift, value-adjusted policy  | Not the primary monetary-CLV dataset |

## Folder layout

```
phase2/
  data/raw/x5/            X5 RetailHero gz files (uplift_train, clients, purchases)
  data/raw/dunnhumby/     Dunnhumby Complete Journey CSVs (user-supplied zip)
  data/processed/         engineered features (parquet/csv)
  data/windows/           walk-forward CLV windows
  src/                    feature / split / model modules
  experiments/            runnable p2_*.py scripts
  results/                metric CSVs
  figures/                figures
  tables/                 audit / statistics tables
  audit/                  AI audit log (Phase 2)
```

## Data acquisition

- **X5 RetailHero** — downloaded programmatically from the public scikit-uplift S3 bucket
  (`uplift_train.csv.gz`, `clients.csv.gz`, `purchases.csv.gz`). Reproduced by
  `experiments/p2_00_download_x5.sh` / the curl commands logged in the audit log.
- **Dunnhumby Complete Journey** — gated (Kaggle login). Download the Kaggle dataset
  `frtgnn/dunnhumby-the-complete-journey` and drop the `archive.zip` (or the extracted
  CSVs) into `data/raw/dunnhumby/`. Then run `experiments/p2_01_dunnhumby_audit.py`,
  which auto-detects and extracts the zip.

## Status

Week 1 — Dataset audit (feasibility gating). See `audit/AI_AUDIT_LOG_PHASE2.md`.
