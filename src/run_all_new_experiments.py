"""
run_all_new_experiments.py
Master runner for all new Phase A–D experiments.

Run order:
  1. 22b_build_windows_5fold.py   — extend to 5 walk-forward windows
  2. 13_product_graph.py          — product co-purchase PPMI embeddings
  3. 14_customer_semantic_features.py — per-window customer semantic profiles
  4. 16_hurdle_semantic.py        — Hurdle variants (RFM / Seq / Sem / All)
  5. 18_conformal_hurdle.py       — CQR calibrated prediction intervals
  6. 62b_models_drnn_fixed.py     — dRNN with robust training

Usage:
    python src/run_all_new_experiments.py
    python src/run_all_new_experiments.py --skip-drnn   (skip slow dRNN)
    python src/run_all_new_experiments.py --only 1,2,3  (run specific steps)
"""

import sys
import time
import subprocess
from pathlib import Path

SRC = Path(__file__).parent

STEPS = [
    (1, '22b_build_windows_5fold.py',          'Extend to 5 walk-forward windows'),
    (2, '13_product_graph.py',                  'Build product co-purchase graph embeddings'),
    (3, '14_customer_semantic_features.py',     'Compute customer semantic profiles'),
    (4, '16_hurdle_semantic.py',                'Hurdle variants with semantic features'),
    (5, '18_conformal_hurdle.py',               'CQR calibrated prediction intervals'),
    (6, '62b_models_drnn_fixed.py',             'dRNN with robust training (slow)'),
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

    t0  = time.time()
    res = subprocess.run(
        [sys.executable, str(script)],
        capture_output=False,
        text=True,
    )
    elapsed = time.time() - t0

    if res.returncode == 0:
        print(f"\n  [OK] Completed in {elapsed:.1f}s")
        return True
    else:
        print(f"\n  [FAILED] Exit code {res.returncode} after {elapsed:.1f}s")
        return False


def main():
    # Parse --skip-drnn and --only flags
    skip_drnn = '--skip-drnn' in sys.argv
    only_ids  = None
    for arg in sys.argv[1:]:
        if arg.startswith('--only='):
            only_ids = {int(x) for x in arg.split('=')[1].split(',')}
        elif arg.startswith('--only'):
            idx = sys.argv.index(arg)
            if idx + 1 < len(sys.argv):
                only_ids = {int(x) for x in sys.argv[idx+1].split(',')}

    print("\n" + "="*70)
    print("MASTER RUNNER: New Experiments (Phase A-D)")
    print("="*70)

    total_t0 = time.time()
    results  = {}

    for (sid, script, desc) in STEPS:
        if only_ids and sid not in only_ids:
            print(f"\n[STEP {sid}/6] SKIP (not in --only list): {desc}")
            continue
        if skip_drnn and sid == 6:
            print(f"\n[STEP {sid}/6] SKIP (--skip-drnn): {desc}")
            continue

        ok = run_step(sid, script, desc)
        results[sid] = ok
        if not ok and sid in (1, 2, 3):   # critical steps — abort pipeline
            print(f"\nCritical step {sid} failed. Aborting.")
            break

    total_elapsed = time.time() - total_t0
    print("\n" + "="*70)
    print(f"ALL STEPS COMPLETE  ({total_elapsed/60:.1f} min total)")
    print("="*70)
    print(f"{'Step':<6} {'Status':<10} {'Description'}")
    print("-"*60)
    for sid, script, desc in STEPS:
        if sid in results:
            status = 'OK' if results[sid] else 'FAILED'
        elif skip_drnn and sid == 6:
            status = 'SKIPPED'
        elif only_ids and sid not in only_ids:
            status = 'SKIPPED'
        else:
            status = 'NOT RUN'
        print(f"  {sid:<4} {status:<10} {desc}")

    print("\nOutput files to check:")
    DATA = SRC.parent / 'data' / 'processed'
    RES  = SRC.parent / 'results'
    checks = [
        DATA / 'walk_forward_windows_5fold.pkl',
        DATA / 'product_embeddings.pkl',
        DATA / 'semantic_features_window_1.npy',
        DATA / 'semantic_features_window_3.npy',
        RES  / 'semantic_walkforward.csv',
        RES  / 'conformal_prediction.csv',
        RES  / 'drnn_fixed_walkforward.csv',
    ]
    for p in checks:
        status = 'OK' if p.exists() else 'MISSING'
        print(f"  [{status}]  {p.name}")


if __name__ == "__main__":
    main()
