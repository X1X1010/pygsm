import numpy as np

from pygsm.coordinate_systems.slots import Rotator


def test_reset_behaves_like_a_fresh_rotator():
    rng = np.random.default_rng(1)
    x0 = rng.standard_normal((4, 3))
    x1 = x0 + 0.2 * rng.standard_normal((4, 3))
    x2 = x0 + 0.2 * rng.standard_normal((4, 3))
    rotator = Rotator([0, 1, 2, 3], x0)
    rotator.calcDiff(x1, x2)  # caches values relative to x0
    rotator.derivative(x1)

    new_reference = x0 + 0.3 * rng.standard_normal((4, 3))
    rotator.reset(new_reference)
    fresh = Rotator([0, 1, 2, 3], new_reference)

    np.testing.assert_allclose(rotator.value(x1), fresh.value(x1))
    np.testing.assert_allclose(rotator.calcDiff(x1, x2), fresh.calcDiff(x1, x2))
    np.testing.assert_allclose(rotator.derivative(x1), fresh.derivative(x1))
