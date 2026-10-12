import numpy as np
import pytest

pytest.importorskip('ase')

from ase.calculators.morse import MorsePotential

from pygsm import PES, ASELoT
from pygsm.utils import manage_xyz, units

from .golden.scenarios import DATA_DIR


def _pes(**kwargs):
    geom = manage_xyz.read_xyzs(str(DATA_DIR / 'ethylene.xyz'))[0]
    return PES(ASELoT(MorsePotential(), geom=geom), **kwargs), manage_xyz.xyz_to_np(geom)


def test_restraints_energy_and_gradient_are_consistent():
    pes, xyz0 = _pes(RESTRAINTS=[(0, 50.0), (3, 20.0)])
    plain = PES(pes.lot.clone())
    x = xyz0 + 0.05 * np.random.default_rng(0).standard_normal(xyz0.shape)

    energy = pes.get_energy(x)
    assert np.ndim(energy) == 0
    d0, d3 = x[0] - xyz0[0], x[3] - xyz0[3]
    assert energy - plain.get_energy(x) == pytest.approx(0.5 * 50.0 * d0 @ d0 + 0.5 * 20.0 * d3 @ d3)

    gradient = pes.get_gradient(x).reshape(-1, 3)  # Ha/Angstrom
    h = 1e-5
    finite_difference = np.zeros_like(x)
    for atom in range(x.shape[0]):
        for axis in range(3):
            xp, xm = x.copy(), x.copy()
            xp[atom, axis] += h
            xm[atom, axis] -= h
            finite_difference[atom, axis] = (pes.get_energy(xp) - pes.get_energy(xm)) / (2 * h)
    np.testing.assert_allclose(gradient, finite_difference * units.KCAL_MOL_TO_AU, atol=1e-8)


def test_restraint_reference_is_the_input_geometry_for_every_copy():
    pes, xyz0 = _pes(RESTRAINTS=[(0, 50.0)])
    copy = PES.create_pes_from(pes, node_id=4)
    np.testing.assert_array_equal(pes.reference_xyz, xyz0)
    np.testing.assert_array_equal(copy.reference_xyz, xyz0)
