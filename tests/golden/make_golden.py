"""Record numerical baselines for the golden regression tests.

Usage (from the repository root)::

    OMP_NUM_THREADS=1 python -m tests.golden.make_golden            # all scenarios
    OMP_NUM_THREADS=1 python -m tests.golden.make_golden segsm_ef   # selected ones

Only re-record a baseline in a commit that intentionally changes numerics.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from .scenarios import SCENARIOS

BASELINE_DIR = Path(__file__).resolve().parent / 'baselines'


def run_scenario(name: str) -> dict[str, np.ndarray]:
    """Run one scenario inside a fresh temporary working directory."""
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix=f'pygsm_{name}_') as tmp:
        os.chdir(tmp)
        try:
            return SCENARIOS[name]()
        finally:
            os.chdir(cwd)


def main(names: list[str]) -> None:
    BASELINE_DIR.mkdir(exist_ok=True)
    for name in names or list(SCENARIOS):
        start = time.perf_counter()
        result = run_scenario(name)
        np.savez_compressed(BASELINE_DIR / f'{name}.npz', **result)
        print(f'{name:32s} {time.perf_counter() - start:7.2f} s  keys={sorted(result)}')


if __name__ == '__main__':
    main(sys.argv[1:])
