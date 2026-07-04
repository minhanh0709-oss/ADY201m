# Phase 2 raw data

The small data files (< 100 MB) are committed to the repository. **Two large files exceed
GitHub's 100 MB limit and must be downloaded separately** (see below). Intermediate/derived
artifacts (`data/processed/`, `*.parquet`, `*.npz`) are gitignored and reproduced from raw + code.

> **Data licensing.** These datasets are provided by their owners for research/education.
> They are included/linked here only for coursework reproducibility (ADY201m). Respect the
> original terms; do not redistribute beyond this academic use.

## x5/ — X5 RetailHero (uplift)
Source: public scikit-uplift S3 bucket (URLs verified from `sklift/datasets/datasets.py`).
Reproduce with `phase2/experiments/p2_00_download_x5.sh` or `_robust_dl_x5.sh` (resumable).

| file                 | rows        | in repo? | columns |
|----------------------|-------------|----------|---------|
| uplift_train.csv.gz  | 200,039     | yes      | client_id, treatment_flg, target |
| clients.csv.gz       | 400,162     | yes      | client_id, first_issue_date, first_redeem_date, age, gender |
| uplift_test.csv      | —           | yes      | client_id |
| products.csv         | —           | yes      | product metadata |
| **purchases.csv.gz** | 45,786,568  | **NO — ~639 MB, download** | client_id, transaction_id, transaction_datetime, purchase_sum, product_id, product_quantity, points… |

> `purchases.csv.gz` (~639 MB) is only needed to rebuild the X5 uplift **features**; the
> Week-1 feasibility (treatment balance + raw uplift) and the uplift **labels** need only
> `uplift_train.csv.gz`, which is in the repo.

## dunnhumby/ — Dunnhumby Complete Journey (external CLV validation)
Source: Kaggle `frtgnn/dunnhumby-the-complete-journey` (login required).

| file                 | rows      | in repo? |
|----------------------|-----------|----------|
| product.csv          | 92,353    | yes      |
| hh_demographic.csv   | 801       | yes      |
| campaign_table.csv, campaign_desc.csv, coupon.csv, coupon_redempt.csv | — | yes (metadata; **coupon/redemption NOT used as treatment**) |
| **transaction_data.csv** | 2,595,732 | **NO — ~136 MB, download** |

Columns of `transaction_data.csv`: household_key, BASKET_ID, DAY, PRODUCT_ID, QUANTITY,
SALES_VALUE, WEEK_NO, STORE_ID, TRANS_TIME, RETAIL_DISC, COUPON_DISC, COUPON_MATCH_DISC.
Download the Kaggle zip and drop `transaction_data.csv` (or the whole `archive.zip`) here;
`p2_01_dunnhumby_audit.py` auto-extracts any `*.zip` found.

## Online Retail II (baseline CLV, e-commerce)
Source: **UCI Machine Learning Repository** (open) — reused read-only from Phase 1
(`data/processed/online_retail_cleaned.csv`, 805,549 transactions / 5,038 customers).
