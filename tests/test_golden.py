"""Numerical regression tests: fresh runs must reproduce the stored baselines.

Baselines live in ``tests/golden/baselines`` and are produced by
``python -m tests.golden.make_golden``.  Comparisons are exact for integers
and strings and use a near-bitwise tolerance for floats.
"""

import os

import numpy as np
import pytest

pytest.importorskip('ase')
pytest.importorskip('tblite')

from .golden.make_golden import BASELINE_DIR, run_scenario
from .golden.scenarios import SCENARIOS

RTOL = 1e-10
ATOL = 1e-12


@pytest.fixture(scope='module', autouse=True)
def _single_thread():
    if os.environ.get('OMP_NUM_THREADS') != '1':
        pytest.skip('golden tests require OMP_NUM_THREADS=1 for deterministic tblite results')


@pytest.mark.golden
@pytest.mark.parametrize('name', list(SCENARIOS))
def test_golden(name):
    path = BASELINE_DIR / f'{name}.npz'
    if not path.exists():
        pytest.skip(f'no baseline recorded for {name}')
    expected = dict(np.load(path))
    actual = run_scenario(name)
    assert sorted(actual) == sorted(expected)
    for key, ref in expected.items():
        got = np.asarray(actual[key])
        assert got.shape == ref.shape, f'{name}.{key}: shape {got.shape} != {ref.shape}'
        if np.issubdtype(ref.dtype, np.floating):
            np.testing.assert_allclose(got, ref, rtol=RTOL, atol=ATOL, err_msg=f'{name}.{key}')
        else:
            np.testing.assert_array_equal(got, ref, err_msg=f'{name}.{key}')
