"""Wall-time benchmark for the GSM golden scenarios.

Usage (from the repository root, single-threaded)::

    OMP_NUM_THREADS=1 python -m tests.bench --record    # store baseline timings
    OMP_NUM_THREADS=1 python -m tests.bench --compare   # fail if >5 % slower

Each scenario is run ``--repeat`` times and the minimum is used, which is the
most stable estimate on a shared machine.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .golden.make_golden import run_scenario

BENCH_FILE = Path(__file__).resolve().parent / 'golden' / 'bench_baseline.json'
SCENARIOS = ['degsm_ef_climb_qc_launcher', 'degsm_lbfgs_climb', 'segsm_ef']
TOLERANCE = 0.05


def measure(name: str, repeat: int) -> float:
    best = float('inf')
    for _ in range(repeat):
        start = time.perf_counter()
        run_scenario(name)
        best = min(best, time.perf_counter() - start)
    return best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--record', action='store_true')
    mode.add_argument('--compare', action='store_true')
    parser.add_argument('--repeat', type=int, default=3)
    args = parser.parse_args(argv)

    timings = {name: measure(name, args.repeat) for name in SCENARIOS}
    if args.record:
        BENCH_FILE.write_text(json.dumps(timings, indent=2) + '\n')
        for name, t in timings.items():
            print(f'{name:32s} {t:7.2f} s')
        return 0

    baseline = json.loads(BENCH_FILE.read_text())
    failed = False
    for name, t in timings.items():
        ref = baseline[name]
        change = t / ref - 1.0
        flag = 'SLOWER' if change > TOLERANCE else 'ok'
        failed |= change > TOLERANCE
        print(f'{name:32s} {ref:7.2f} s -> {t:7.2f} s ({change:+6.1%}) {flag}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
