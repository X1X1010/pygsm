from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('ase')

from ase.calculators.morse import MorsePotential

from pygsm.coords import DelocalizedInternalCoordinates, build_topology
from pygsm.gsm import GSM
from pygsm.lot.ase import ASELoT
from pygsm.molecule import Molecule
from pygsm.optimizers import EigenvectorFollow
from pygsm.pes import PES
from pygsm.utils import elements, manage_xyz

DATA_DIR = Path(__file__).resolve().parent / 'data'


def _load_ethene_geom():
    return manage_xyz.read_xyzs(str(DATA_DIR / 'ethylene.xyz'))[0]


def _build_atom_list(geom):
    element_table = elements.ElementData()
    return [element_table.from_symbol(atom) for atom in manage_xyz.get_atoms(geom)]


def _build_ase_molecule():
    geom = _load_ethene_geom()
    xyz = manage_xyz.xyz_to_np(geom)
    atoms = _build_atom_list(geom)
    coord_obj = DelocalizedInternalCoordinates(
        xyz=xyz,
        atoms=atoms,
        addtr=False,
    )
    lot = ASELoT(
        MorsePotential(),
        geom=geom,
        string_id=0,
        node_id=0,
        states=[(1, 0)],
        gradient_states=[(1, 0)],
    )
    pes = PES(lot=lot, multiplicity=1, ad_idx=0)
    molecule = Molecule(
        geom=geom,
        pes=pes,
        coord_obj=coord_obj,
        form_hessian=False,
    )
    return geom, xyz, atoms, molecule


def test_topology_builds_expected_bonds_for_ethene():
    geom = _load_ethene_geom()
    xyz = manage_xyz.xyz_to_np(geom)
    atoms = _build_atom_list(geom)
    topology = build_topology(xyz, atoms)

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

    fragmented = build_topology(xyz, atoms)
    assert fragmented.number_of_edges() == 2

    hybrid = build_topology(xyz, atoms, hybrid_indices=[1, 2])
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

    middle = Molecule.copy_from(reactant, xyz=midpoint, new_node_id=1)
    product = Molecule.copy_from(reactant, xyz=displaced, new_node_id=2)
    optimizer = EigenvectorFollow(linesearch='backtrack', conv_grms=0.0005, dmax=0.1)
    gsm = GSM(reactant=reactant, product=product, nnodes=3, optimizer=optimizer)
    gsm.nodes[1] = middle

    energies_before = list(gsm.energies)
    signature_before = gsm._cached_energy_signature
    energies_repeat = list(gsm.energies)

    assert energies_before == pytest.approx(energies_repeat)

    middle.xyz = middle.xyz + 0.01
    energies_after = list(gsm.energies)

    assert gsm._cached_energy_signature != signature_before
    assert len(energies_after) == 3


@pytest.mark.parametrize('mode', ['DE_GSM', 'SE_GSM', 'SE_Cross'])
@pytest.mark.parametrize('use_defaults', [False, True])
def test_cli_convergence_settings_reach_gsm_and_node_optimizers(mode, use_defaults):
    from pygsm._cli.factory import choose_optimizer
    from pygsm._cli.runtime import _build_gsm
    from pygsm.cli import parse_arguments

    convergence_options = [] if use_defaults else [
        '--conv-tol', '0.002', '--conv-gmax', '0.03', '--conv-ediff', '0.04', '--conv-de', '0.05',
    ]  # fmt: skip
    config = parse_arguments(
        verbose=False,
        argv=[
            '--xyzfile', str(DATA_DIR / 'ethylene.xyz'), '--mode', mode, '--num-nodes', '5',
            '--pes-type', 'Penalty_PES', '--multiplicity', '1', '3', '--adiabatic-index', '0', '0',
            *convergence_options,
        ],
    )  # fmt: skip
    _, xyz, _, reactant = _build_ase_molecule()
    product = Molecule.copy_from(reactant, xyz=xyz + 0.01) if mode == 'DE_GSM' else None
    optimizer = choose_optimizer(config)
    gsm = _build_gsm(config, reactant, product, optimizer, [('BREAK', 1, 2)])

    tolerance = 0.0005 if use_defaults else 0.002
    node_thresholds = (100.0, 100.0, 1.0) if use_defaults else (0.03, 0.04, 0.05)
    string_thresholds = (0.001, 0.1, 0.5) if use_defaults and mode != 'DE_GSM' else node_thresholds
    assert (gsm.conv_tol, optimizer.conv_grms) == (tolerance, tolerance)
    assert (gsm.conv_gmax, gsm.conv_ediff, gsm.conv_de) == string_thresholds
    for node_optimizer in gsm.optimizer:
        assert (node_optimizer.conv_gmax, node_optimizer.conv_ediff, node_optimizer.conv_de) == node_thresholds
        assert node_optimizer.opt_cross is False  # crossing criteria are for the end points only
