"""Phase 2 path configuration. All paths are anchored to phase2/ — never to Phase-1 dirs.

Isolation rule: Phase 2 code WRITES only under PHASE2_ROOT. Phase-1 directories
(../data, ../results, ...) may be READ for cross-context reuse, never written.
"""
from pathlib import Path

PHASE2_ROOT = Path(__file__).resolve().parents[1]      # .../paper/phase2
PROJECT_ROOT = PHASE2_ROOT.parent                      # .../paper  (Phase 1, read-only)

# --- Phase 2 (writable) ---
RAW = PHASE2_ROOT / "data" / "raw"
X5_RAW = RAW / "x5"
DUNN_RAW = RAW / "dunnhumby"
PROCESSED = PHASE2_ROOT / "data" / "processed"
WINDOWS = PHASE2_ROOT / "data" / "windows"
RESULTS = PHASE2_ROOT / "results"
FIGURES = PHASE2_ROOT / "figures"
TABLES = PHASE2_ROOT / "tables"
AUDIT = PHASE2_ROOT / "audit"

# --- Phase 1 (READ-ONLY reuse for cross-context) ---
P1_PROCESSED = PROJECT_ROOT / "data" / "processed"
P1_ONLINE_RETAIL_CLEANED = P1_PROCESSED / "online_retail_cleaned.csv"

for _d in (PROCESSED, WINDOWS, RESULTS, FIGURES, TABLES, AUDIT):
    _d.mkdir(parents=True, exist_ok=True)


def assert_phase2_path(p) -> Path:
    """Guard: refuse to write outside phase2/."""
    p = Path(p).resolve()
    if PHASE2_ROOT not in p.parents and p != PHASE2_ROOT:
        raise RuntimeError(f"Refusing to write outside phase2/: {p}")
    return p
