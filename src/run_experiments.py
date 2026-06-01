"""
run_experiments.py — batch runner for semantic graph, CQR, and dRNN experiments.

Usage:
    python src/run_experiments.py
    python src/run_experiments.py --skip-drnn
    python src/run_experiments.py --only 1,2,3
"""

import sys
import time
import subprocess
from pathlib import Path

SRC = Path(__file__).parent

STEPS = [
    (1, "build_walkforward_windows.py", "Extend to 5 walk-forward windows"),
    (2, "product_graph.py", "Product co-purchase PPMI embeddings"),
    (3, "customer_semantic_features.py", "Customer semantic profiles"),
    (4, "hurdle_semantic.py", "Hurdle variants with semantic features"),
    (5, "conformal_cqr.py", "CQR calibrated prediction intervals"),
    (6, "models_drnn.py", "Dilated RNN (slow)"),
]


def run_step(step_id, script_name, description):
    script = SRC / script_name
    print(f"\n{'='*70}")
    print(f"[STEP {step_id}/6]  {description}")
    print(f"  Script: {script_name}")
    print(f"{'='*70}")

    if not script.exists():
        print(f"  ERROR: script not found at {script}")
        return False

    t0 = time.time()
    res = subprocess.run([sys.executable, str(script)], capture_output=False, text=True)
    elapsed = time.time() - t0

    if res.returncode == 0:
        print(f"\n  [OK] Completed in {elapsed:.1f}s")
        return True
    print(f"\n  [FAILED] Exit code {res.returncode} after {elapsed:.1f}s")
    return False


def main():
    skip_drnn = "--skip-drnn" in sys.argv
    only_ids = None
    for arg in sys.argv[1:]:
        if arg.startswith("--only="):
            only_ids = {int(x) for x in arg.split("=")[1].split(",")}
        elif arg.startswith("--only"):
            idx = sys.argv.index(arg)
            if idx + 1 < len(sys.argv):
                only_ids = {int(x) for x in sys.argv[idx + 1].split(",")}

    print("\n" + "=" * 70)
    print("BATCH RUNNER: semantic + CQR + dRNN")
    print("=" * 70)

    total_t0 = time.time()
    results = {}

    for sid, script, desc in STEPS:
        if only_ids and sid not in only_ids:
            continue
        if skip_drnn and sid == 6:
            continue
        ok = run_step(sid, script, desc)
        results[sid] = ok
        if not ok and sid in (1, 2, 3):
            print(f"\nCritical step {sid} failed. Aborting.")
            break

    print(f"\nTotal: {(time.time() - total_t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
