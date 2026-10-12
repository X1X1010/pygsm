import numpy as np
import pytest

from pygsm.lot.base import Lot
from pygsm.utils import manage_xyz


class RecordingLot(Lot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []
        self.fail_state = None

    def run(self, geom, mult, ad_idx, runtype='gradient'):
        state = (mult, ad_idx)
        self.calls.append((state, runtype))
        if state == self.fail_state:
            raise RuntimeError('calculation failed')
        xyz = manage_xyz.xyz_to_np(geom)
        self.Energies[state] = self.Energy(float(xyz.sum()) + ad_idx, 'kcal/mol')
        if runtype == 'gradient':
            self.Gradients[state] = self.Gradient(np.ones_like(xyz), 'kcal/mol/Angstrom')
        # Some external backends set this per state, before runall has finished.
        self.hasRanForCurrentCoords = True


GEOM = [('H', 0.0, 0.0, 0.0)]


@pytest.mark.parametrize('settings', [{'calc_grad': False}, {'gradient_states': []}])
def test_energy_only_calculation(settings):
    lot = RecordingLot(geom=GEOM, **settings)
    xyz = np.zeros((1, 3))
    assert lot.get_energy(xyz, 1, 0) == 0.0
    assert lot.calls == [((1, 0), 'energy')]
    assert lot.Gradients == {}


def test_only_requested_states_get_gradients():
    lot = RecordingLot(geom=GEOM, states=[(1, 0), (3, 0)], gradient_states=[(3, 0)])
    xyz = np.zeros((1, 3))
    lot.get_energy(xyz, 1, 0)
    lot.get_gradient(xyz, 3, 0)
    assert lot.calls == [((1, 0), 'energy'), ((3, 0), 'gradient')]


def test_failed_calculation_is_not_cached():
    lot = RecordingLot(geom=GEOM, states=[(1, 0), (1, 1)])
    xyz = np.zeros((1, 3))
    assert lot.get_energy(xyz, 1, 0) == 0.0
    displaced = xyz + 1.0
    lot.fail_state = (1, 1)
    for _ in range(2):
        with pytest.raises(RuntimeError, match='calculation failed'):
            lot.get_energy(displaced, 1, 0)
        assert not lot.hasRanForCurrentCoords

    lot.fail_state = None
    assert lot.get_energy(displaced, 1, 1) == 4.0
    calls = len(lot.calls)
    lot.get_gradient(displaced, 1, 1)
    assert len(lot.calls) == calls
    np.testing.assert_array_equal(lot.currentCoords, displaced)
    displaced[:] = 2.0
    assert lot.get_energy(displaced, 1, 0) == 6.0


def test_state_normalization_keeps_all_spin_multiplicities():
    lot = RecordingLot(geom=GEOM, states=[(1, 2), (2, 0), (3, 1), (6, 0)])
    lot.get_energy(np.zeros((1, 3)), 6, 0)
    assert set(lot.Energies) == {(1, 0), (1, 1), (1, 2), (2, 0), (3, 0), (3, 1), (6, 0)}
    assert set(lot.Gradients) == {(1, 2), (2, 0), (3, 1), (6, 0)}
