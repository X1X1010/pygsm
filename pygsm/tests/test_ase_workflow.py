from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ase")

from ase.calculators.morse import MorsePotential

from pygsm.coordinate_systems import DelocalizedInternalCoordinates, Topology
from pygsm.growing_string_methods import GSM
from pygsm.level_of_theories.ase import ASELoT
from pygsm.molecule import Molecule
from pygsm.optimizers import eigenvector_follow
from pygsm.potential_energy_surfaces import PES
from pygsm.utilities import elements, manage_xyz


DATA_DIR = Path(__file__).resolve().parents[1] / 'data'


def _load_ethene_geom():
    return manage_xyz.read_xyzs(str(DATA_DIR / 'ethylene.xyz'))[0]


def _build_atom_list(geom):
    element_table = elements.ElementData()
    return [element_table.from_symbol(atom) for atom in manage_xyz.get_atoms(geom)]


def _build_ase_molecule():
    geom = _load_ethene_geom()
    xyz = manage_xyz.xyz_to_np(geom)
    atoms = _build_atom_list(geom)
    coord_obj = DelocalizedInternalCoordinates.from_options(
        xyz=xyz,
        atoms=atoms,
        addtr=False,
    )
    lot = ASELoT.from_options(
        MorsePotential(),
        geom=geom,
        ID=0,
        node_id=0,
        states=[(1, 0)],
        gradient_states=[(1, 0)],
    )
    pes = PES.from_options(lot=lot, multiplicity=1, ad_idx=0)
    molecule = Molecule.from_options(
        geom=geom,
        PES=pes,
        coord_obj=coord_obj,
        Form_Hessian=False,
    )
    return geom, xyz, atoms, molecule


def test_topology_builds_expected_bonds_for_ethene():
    geom = _load_ethene_geom()
    xyz = manage_xyz.xyz_to_np(geom)
    atoms = _build_atom_list(geom)
    topology = Topology.build_topology(xyz, atoms)

    assert topology.number_of_nodes() == 6
    assert topology.number_of_edges() == 5


def test_topology_handles_fragmented_and_hybrid_indices():
    element_table = elements.ElementData()
    geom = [
        ('H', 0.0, 0.0, 0.0),
        ('H', 0.0, 0.0, 0.74),
        ('H', 0.0, 0.0, 4.0),
        ('H', 0.0, 0.0, 4.74),
    ]
    atoms = [element_table.from_symbol(atom[0]) for atom in geom]
    xyz = manage_xyz.xyz_to_np(geom)

    fragmented = Topology.build_topology(xyz, atoms)
    assert fragmented.number_of_edges() == 2

    hybrid = Topology.build_topology(xyz, atoms, hybrid_indices=[1, 2])
    assert set(hybrid.nodes()) == {0, 3}


def test_ase_lot_energy_and_gradient_are_finite():
    _, xyz, _, molecule = _build_ase_molecule()

    assert np.isfinite(molecule.energy)
    assert molecule.gradx.shape == xyz.shape
    assert molecule.gradient.shape[0] == molecule.num_coordinates


def test_gsm_energy_cache_invalidates_when_node_geometry_changes():
    _, xyz, _, reactant = _build_ase_molecule()
    displaced = xyz.copy()
    displaced[0, 1] += 0.05
    midpoint = 0.5 * (xyz + displaced)

    middle = Molecule.copy_from_options(reactant, xyz=midpoint, new_node_id=1, copy_wavefunction=False)
    product = Molecule.copy_from_options(reactant, xyz=displaced, new_node_id=2, copy_wavefunction=False)
    optimizer = eigenvector_follow.from_options(Linesearch='backtrack', OPTTHRESH=0.0005, DMAX=0.1)
    gsm = GSM.from_options(reactant=reactant, product=product, nnodes=3, optimizer=optimizer)
    gsm.nodes[1] = middle

    energies_before = list(gsm.energies)
    signature_before = gsm._cached_energy_signature
    energies_repeat = list(gsm.energies)

    assert energies_before == pytest.approx(energies_repeat)

    middle.xyz = middle.xyz + 0.01
    energies_after = list(gsm.energies)

    assert gsm._cached_energy_signature != signature_before
    assert len(energies_after) == 3
