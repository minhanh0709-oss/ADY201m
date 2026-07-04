#!/usr/bin/env bash
# Robust resumable X5 download: tolerant of mid-transfer connection drops.
# Loops curl -C - (resume) until local size == server Content-Length, then verifies gzip.
DEST="/d/SU26/ADY201m/paper/phase2/data/raw/x5"
BASE="https://sklift.s3.eu-west-2.amazonaws.com"
mkdir -p "$DEST"; cd "$DEST"

dl() {
  local f="$1"
  local url="$BASE/$f"
  local expected
  expected=$(curl -sIL -m 60 "$url" | tr -d '\r' | awk -F': ' 'tolower($1)=="content-length"{print $2; exit}')
  echo "[$f] expected bytes: ${expected:-unknown}"
  local attempt=0
  while :; do
    attempt=$((attempt+1))
    local have=0
    [ -f "$f" ] && have=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if [ -n "$expected" ] && [ "$have" -ge "$expected" ] 2>/dev/null; then
      echo "[$f] size OK ($have) — verifying gzip..."
      if gzip -t "$f" 2>/dev/null; then echo "[$f] gzip OK"; return 0; fi
      echo "[$f] gzip FAILED — restarting from scratch"; rm -f "$f"
    fi
    echo "[$f] attempt $attempt (have ${have}/${expected:-?}) ..."
    # IMPORTANT: NO curl internal --retry here. curl's --retry latches the -C - resume
    # offset at invocation start and restarts from it on retry, TRUNCATING progress.
    # Single attempt per curl; the outer while-loop does the retrying with a fresh
    # -C - offset read from the on-disk size each pass. --speed-limit aborts stalls.
    curl -fL -C - --connect-timeout 30 --speed-limit 1500 --speed-time 20 \
         -m 0 -s -S "$url" -o "$f" || true
    sleep 2
    # very high cap: this link stalls every ~20s (each stall = 1 attempt), so 670MB needs
    # thousands of resume cycles. Effectively "never give up" while still bounding runaway.
    [ "$attempt" -ge 100000 ] && { echo "[$f] giving up after $attempt"; return 1; }
  done
}

for f in uplift_train.csv.gz clients.csv.gz purchases.csv.gz; do
  dl "$f" || { echo "FAILED: $f"; exit 1; }
done
echo "ALL DONE"; ls -la
