#!/usr/bin/env bash
# Phase 2 — reproduce the full pipeline end to end.
# Prereqs:
#   - pip install -r requirements.txt
#   - X5: bash experiments/_robust_dl_x5.sh   (downloads uplift_train/clients/purchases.csv.gz)
#   - Dunnhumby: drop Kaggle zip 'frtgnn/dunnhumby-the-complete-journey' into data/raw/dunnhumby/
#   - Online Retail II: reuses ../data/processed/online_retail_cleaned.csv (Phase 1, read-only)
set -e
PY="${PYTHON:-python}"
cd "$(dirname "$0")"

echo "== Week 1: dataset audits =="
$PY experiments/p2_01_x5_audit.py
$PY experiments/p2_01_dunnhumby_audit.py

echo "== Week 2: feature engineering (both CLV contexts) =="
$PY experiments/p2_02_build_features.py --dataset online_retail
$PY experiments/p2_02_build_features.py --dataset dunnhumby
$PY experiments/p2_03_context_comparison.py

echo "== Week 4-5: CLV benchmark + ablation + SHAP + CQR =="
for ds in online_retail dunnhumby; do
  $PY experiments/p2_04_clv_benchmark.py --dataset $ds
  $PY experiments/p2_05_shap_cqr.py --dataset $ds
done
$PY experiments/p2_06_clv_summary.py

echo "== Week 6-8: X5 uplift features, models, policy comparison =="
$PY experiments/p2_07_x5_uplift_features.py
$PY experiments/p2_08_x5_uplift_models.py
$PY experiments/p2_09_policy_comparison.py

echo "== Week 9-10: robustness + paper tables =="
$PY experiments/p2_10_robustness.py
$PY experiments/p2_11_make_paper_tables.py

echo "== DONE. Results in results/, figures/, tables/, paper2/ =="
