#!/usr/bin/env bash
# Phase 2 — reproducible X5 RetailHero download (public scikit-uplift S3 bucket).
# URLs verified from sklift/datasets/datasets.py (fetch_x5). ~647MB compressed total.
set -euo pipefail
DEST="$(cd "$(dirname "$0")/.." && pwd)/data/raw/x5"
mkdir -p "$DEST"
BASE="https://sklift.s3.eu-west-2.amazonaws.com"
for f in uplift_train.csv.gz clients.csv.gz purchases.csv.gz; do
  echo "downloading $f ..."
  curl -fL --retry 3 -m 3600 "$BASE/$f" -o "$DEST/$f"
done
echo "done -> $DEST"
ls -la "$DEST"
