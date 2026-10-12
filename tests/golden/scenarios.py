"""Deterministic end-to-end scenarios used as numerical regression baselines.

Every scenario returns a flat ``dict[str, np.ndarray]``.  ``make_golden.py``
stores them as ``.npz`` files and ``test_golden.py`` compares fresh runs
against them.  When the public API changes, only this module should need to
be ported -- the stored baselines are the contract and must not move unless a
commit intentionally changes numerics.

All scenarios use tblite GFN2-xTB through ASE (deterministic with
``OMP_NUM_THREADS=1``). They are run from an empty temporary directory
because the CLI scenarios write their result files to the working directory.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _tblite(**kwargs):
    from tblite.ase import TBLite

    return TBLite(method='GFN2-xTB', verbosity=0, **kwargs)


class CallCounter:
    """Wraps an ASE calculator class so the number of SCF calls is recorded."""

    def __init__(self):
        self.count = 0

    def wrap(self, calc):
        original = calc.calculate
        counter = self

        def calculate(*args, **kwargs):
            counter.count += 1
            return original(*args, **kwargs)

        calc.calculate = calculate
        return calc


def _da_geoms():
    from pygsm.utils import manage_xyz

    return manage_xyz.read_xyzs(str(DATA_DIR / 'diels_alder.xyz'))


def _atoms_of(geom):
    from pygsm.utils import elements, manage_xyz

    table = elements.ElementData()
    return [table.from_symbol(sym) for sym in manage_xyz.get_atoms(geom)]


def _xyz(geom):
    from pygsm.utils import manage_xyz

    return manage_xyz.xyz_to_np(geom)


def _coord_flags(coordinate_type):
    return dict(
        connect=coordinate_type == 'DLC',
        addtr=coordinate_type == 'TRIC',
        addcart=coordinate_type == 'HDLC',
    )


def _union_topology(xyz1, xyz2, atoms):
    from pygsm.coords import build_topology

    top1 = build_topology(xyz=xyz1, atoms=atoms)
    top2 = build_topology(xyz=xyz2, atoms=atoms)
    for bond in top2.edges():
        if bond in top1.edges() or (bond[1], bond[0]) in top1.edges():
            continue
        if bond[0] > bond[1]:
            top1.add_edge(bond[0], bond[1])
        else:
            top1.add_edge(bond[1], bond[0])
    return top1


def _union_dlc(xyz1, xyz2, atoms, coordinate_type):
    from pygsm.coords import DelocalizedInternalCoordinates, PrimitiveInternalCoordinates

    flags = _coord_flags(coordinate_type)
    top = _union_topology(xyz1, xyz2, atoms)
    p1 = PrimitiveInternalCoordinates(xyz=xyz1, atoms=atoms, topology=top, **flags)
    p2 = PrimitiveInternalCoordinates(xyz=xyz2, atoms=atoms, topology=top, **flags)
    p1.add_union_primitives(p2)
    dlc = DelocalizedInternalCoordinates(xyz=xyz1, atoms=atoms, primitives=p1, **flags)
    return top, p1, dlc


def _dense(m):
    from pygsm.utils import block_matrix

    return np.asarray(block_matrix.full_matrix(m)) if isinstance(m, block_matrix) else np.asarray(m)


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ---------------------------------------------------------------------------
# coordinate scenarios
# ---------------------------------------------------------------------------


def _coords(coordinate_type, perm=None):
    geoms = _da_geoms()
    xyz1, xyz2 = _xyz(geoms[0]), _xyz(geoms[1])
    atoms = _atoms_of(geoms[0])
    if perm is not None:
        xyz1, xyz2 = xyz1[perm], xyz2[perm]
        atoms = [atoms[i] for i in perm]
    with _quiet():
        top, prims, dlc = _union_dlc(xyz1, xyz2, atoms, coordinate_type)
        dq = 0.05 * dlc.calcDiff(xyz2, xyz1)
        new_xyz = dlc.newCartesian(xyz1, dq, verbose=False)
        out = {
            'edges': np.array(sorted(tuple(sorted(e)) for e in top.edges())),
            'prim_repr': np.array([repr(p) for p in prims.Internals]),
            'prim_values': np.asarray(prims.calculate(xyz1)),
            'prim_diff': np.asarray(prims.calcDiff(xyz2, xyz1)),
            'prim_B': _dense(prims.wilsonB(xyz1)),
            'prim_hess_guess': _dense(prims.guess_hessian(xyz1)),
            'dlc_vecs': _dense(dlc.Vecs),
            'dlc_values': np.asarray(dlc.calculate(xyz2)),
            'dlc_B': _dense(dlc.wilsonB(xyz1)),
            'dlc_dq': np.asarray(dq),
            'dlc_new_xyz': np.asarray(new_xyz),
        }
    return out


def coords_tric():
    return _coords('TRIC')


def coords_dlc():
    return _coords('DLC')


def coords_hdlc():
    return _coords('HDLC')


def coords_tric_interleaved():
    # interleave the ethylene atoms (10-15) into the butadiene block
    perm = [0, 10, 1, 11, 2, 12, 3, 13, 4, 14, 5, 15, 6, 7, 8, 9]
    return _coords('TRIC', perm=perm)


# ---------------------------------------------------------------------------
# molecule / optimizer scenarios
# ---------------------------------------------------------------------------


def _molecule(geom, xyz2=None, coordinate_type='TRIC', form_hessian=True, counter=None, calc=None):
    from pygsm.lot.ase import ASELoT
    from pygsm.molecule import Molecule
    from pygsm.pes import PES

    atoms = _atoms_of(geom)
    xyz1 = _xyz(geom)
    _, _, dlc = _union_dlc(xyz1, xyz1 if xyz2 is None else xyz2, atoms, coordinate_type)
    calc = calc if calc is not None else _tblite()
    if counter is not None:
        counter.wrap(calc)
    lot = ASELoT(calculator=calc, geom=geom, ID=0)
    pes = PES(lot=lot, ad_idx=0)
    return Molecule(geom=geom, PES=pes, coord_obj=dlc, Form_Hessian=form_hessian)


def _optimizer(method, **opts):
    from pygsm.optimizers import LBFGS, EigenvectorFollow

    cls = {'ef': EigenvectorFollow, 'lbfgs': LBFGS}[method]
    return cls(**opts)


def _opt(method, linesearch):
    counter = CallCounter()
    geom = _da_geoms()[0]
    with _quiet():
        mol = _molecule(geom, form_hessian=method == 'ef', counter=counter)
        opt = _optimizer(method, Linesearch=linesearch, DMAX=0.1, print_level=1)
        _geoms, energies = opt.optimize(molecule=mol, refE=mol.energy, opt_steps=15)
    return {
        'energies': np.asarray(energies, dtype=float),
        'final_xyz': np.asarray(mol.xyz),
        'final_gradient': np.asarray(mol.gradient),
        'final_coords': np.asarray(mol.coordinates),
        'ncalls': np.array(counter.count),
    }


def opt_ef_nols():
    return _opt('ef', 'NoLineSearch')


def opt_ef_backtrack():
    return _opt('ef', 'backtrack')


def opt_lbfgs_nols():
    return _opt('lbfgs', 'NoLineSearch')


def opt_lbfgs_backtrack():
    return _opt('lbfgs', 'backtrack')


# ---------------------------------------------------------------------------
# DE-GSM: mirrors qc_launcher/tasks/gsm.py::run_gsm
# ---------------------------------------------------------------------------


def _gsm_result(gsm, counter):
    return {
        'energies': np.asarray(gsm.energies, dtype=float),
        'geometries': np.asarray([_xyz(g) for g in gsm.geometries]),
        'TSnode': np.array(gsm.TSnode),
        'nnodes': np.array(len([n for n in gsm.nodes if n is not None])),
        'ncalls': np.array(counter.count),
    }


def _de_gsm(
    method='ef', climb=True, optimize_endpoints=False, num_nodes=9, max_gsm_steps=20, coordinate_type='TRIC', mp_cores=1
):
    from pygsm.gsm import DEGSM
    from pygsm.molecule import Molecule

    counter = CallCounter()
    geoms = _da_geoms()
    with _quiet():
        reactant = _molecule(
            geoms[0],
            xyz2=_xyz(geoms[1]),
            coordinate_type=coordinate_type,
            form_hessian=method == 'ef',
            counter=counter,
        )
        product = Molecule.copy_from(reactant, xyz=_xyz(geoms[1]), new_node_id=num_nodes - 1)
        opt = _optimizer(
            method,
            print_level=1,
            Linesearch='NoLineSearch',
            update_hess_in_bg=not (climb or method == 'lbfgs'),
            conv_Ediff=100.0,
            conv_gmax=100.0,
            DMAX=0.1,
        )
        gsm = DEGSM(
            reactant=reactant,
            product=product,
            nnodes=num_nodes,
            CONV_TOL=5e-4,
            CONV_gmax=100.0,
            CONV_Ediff=100.0,
            ADD_NODE_TOL=0.1,
            growth_direction=0,
            optimizer=opt,
            ID=0,
            print_level=1,
            mp_cores=mp_cores,
            interp_method='DLC',
        )
        if optimize_endpoints:
            opt.optimize(molecule=reactant, refE=reactant.energy, opt_steps=100)
            opt.optimize(molecule=product, refE=product.energy, opt_steps=100)
        gsm.go_gsm(max_iters=max_gsm_steps, opt_steps=3, rtype=1 if climb else 2)
    return _gsm_result(gsm, counter)


def degsm_ef_climb_qc_launcher():
    """Exact qc_launcher recipe (endpoints optimised, climbing on)."""
    return _de_gsm('ef', climb=True, optimize_endpoints=True)


def degsm_ef_noclimb():
    return _de_gsm('ef', climb=False)


def degsm_lbfgs_climb():
    return _de_gsm('lbfgs', climb=True)


def degsm_ef_mp2():
    """Parallel node updates (mp_cores=2); must reproduce the serial run exactly."""
    return _de_gsm('ef', climb=True, mp_cores=2, max_gsm_steps=10)


def degsm_ef_dlc():
    return _de_gsm('ef', climb=True, coordinate_type='DLC', max_gsm_steps=8)


# ---------------------------------------------------------------------------
# SE-GSM: mirrors pygsm/_cli/runtime.py
# ---------------------------------------------------------------------------


def segsm_ef():
    from pygsm.coords import (
        DelocalizedInternalCoordinates,
        Distance,
        PrimitiveInternalCoordinates,
        build_topology,
    )
    from pygsm.gsm import SEGSM
    from pygsm.lot.ase import ASELoT
    from pygsm.molecule import Molecule
    from pygsm.pes import PES

    counter = CallCounter()
    geom = _da_geoms()[0]
    atoms = _atoms_of(geom)
    xyz1 = _xyz(geom)
    driving = [['ADD', 4, 12], ['ADD', 1, 11]]
    with _quiet():
        top = build_topology(xyz1, atoms)
        for dc in driving:
            a, b = sorted((dc[1] - 1, dc[2] - 1))
            prim = Distance(a, b)
            bond = tuple(prim.atoms)
            if bond in top.edges or (bond[1], bond[0]) in top.edges():
                continue
            top.add_edge(bond[0], bond[1])
        flags = _coord_flags('TRIC')
        p1 = PrimitiveInternalCoordinates(xyz=xyz1, atoms=atoms, topology=top, **flags)
        p1.newMakePrimitives(xyz1)
        dlc = DelocalizedInternalCoordinates(xyz=xyz1, atoms=atoms, primitives=p1, **flags)
        lot = ASELoT(calculator=counter.wrap(_tblite()), geom=geom, ID=0)
        pes = PES(lot=lot, ad_idx=0)
        reactant = Molecule(geom=geom, PES=pes, coord_obj=dlc, Form_Hessian=True)
        opt = _optimizer('ef', Linesearch='NoLineSearch', DMAX=0.1, conv_Ediff=100.0, conv_gmax=100.0, print_level=1)
        gsm = SEGSM(
            reactant=reactant,
            nnodes=30,
            DQMAG_MAX=0.8,
            BDIST_RATIO=0.5,
            CONV_TOL=5e-4,
            ADD_NODE_TOL=0.01,
            optimizer=opt,
            print_level=1,
            driving_coords=driving,
            ID=0,
            mp_cores=1,
            interp_method='DLC',
        )
        opt.optimize(molecule=reactant, refE=reactant.energy, opt_steps=100)
        gsm.go_gsm(max_iters=20, opt_steps=10, rtype=2)
    return _gsm_result(gsm, counter)


# ---------------------------------------------------------------------------
# Penalty PES (MECP) with a two-state tblite level of theory
# ---------------------------------------------------------------------------


def _two_state_lot_class():
    from ase import units

    from pygsm.lot.ase import ASELoT, xyz_to_ase

    class TwoStateTBLite(ASELoT):
        """Test-only LoT: one tblite calculator per multiplicity."""

        def run(self, geom, mult, ad_idx, runtype='gradient'):
            for m, _ in self.states:
                atoms = xyz_to_ase(geom)
                atoms.calc = _tblite(multiplicity=m)
                self._Gradients[(m, 0)] = self.Gradient(-atoms.get_forces() / units.Ha * units.Bohr, 'Hartree/Bohr')
                self._Energies[(m, 0)] = self.Energy(atoms.get_potential_energy() / units.Ha, 'Hartree')
            self.hasRanForCurrentCoords = True

    return TwoStateTBLite


def mecp_penalty_ethylene():
    from pygsm.coords import DelocalizedInternalCoordinates, PrimitiveInternalCoordinates, build_topology
    from pygsm.molecule import Molecule
    from pygsm.pes import PES, PenaltyPES
    from pygsm.utils import manage_xyz

    geom = manage_xyz.read_xyzs(str(DATA_DIR / 'ethylene.xyz'))[0]
    xyz = _xyz(geom)
    # twist one CH2 by ~50 degrees about the C=C axis (y axis)
    theta = np.deg2rad(50.0)
    rot = np.array([[np.cos(theta), 0.0, np.sin(theta)], [0.0, 1.0, 0.0], [-np.sin(theta), 0.0, np.cos(theta)]])
    xyz[[1, 2]] = xyz[[1, 2]] @ rot.T
    geom = manage_xyz.np_to_xyz(geom, xyz)
    atoms = _atoms_of(geom)

    with _quiet():
        TwoStateTBLite = _two_state_lot_class()
        lot = TwoStateTBLite(
            calculator=_tblite(), geom=geom, ID=0, states=[(1, 0), (3, 0)], gradient_states=[(1, 0), (3, 0)]
        )
        pes1 = PES(lot=lot, multiplicity=1, ad_idx=0)
        pes2 = PES(lot=lot, multiplicity=3, ad_idx=0)
        pes = PenaltyPES(PES1=pes1, PES2=pes2, lot=lot, sigma=3.5)
        top = build_topology(xyz, atoms)
        flags = _coord_flags('TRIC')
        p1 = PrimitiveInternalCoordinates(xyz=xyz, atoms=atoms, topology=top, **flags)
        dlc = DelocalizedInternalCoordinates(xyz=xyz, atoms=atoms, primitives=p1, **flags)
        mol = Molecule(geom=geom, PES=pes, coord_obj=dlc, Form_Hessian=True)
        opt = _optimizer(
            'ef', Linesearch='backtrack', DMAX=0.05, conv_Ediff=0.1, conv_dE=0.1, conv_gmax=0.005, opt_cross=True
        )
        _geoms, energies = opt.optimize(molecule=mol, refE=mol.energy, opt_steps=30, opt_type='UNCONSTRAINED')
    return {
        'energies': np.asarray(energies, dtype=float),
        'final_xyz': np.asarray(mol.xyz),
        'difference_energy': np.array(mol.difference_energy),
    }


# ---------------------------------------------------------------------------
# CLI end-to-end (string-level interface is stable across API changes)
# ---------------------------------------------------------------------------


def _cli(argv, result_file='TSnode_000.xyz'):
    from pygsm.cli import main
    from pygsm.utils import manage_xyz

    with _quiet():
        main(argv)
    ts = manage_xyz.read_xyzs(result_file)[0]
    return {'ts_xyz': _xyz(ts)}


def cli_de_gsm():
    xyz = str(DATA_DIR / 'diels_alder.xyz')
    return _cli([
        '-xyzfile', xyz, '-mode', 'DE_GSM', '-package', 'ase',
        '--ase-class', 'tblite.ase.TBLite', '--ase-kwargs', '{"method": "GFN2-xTB", "verbosity": 0}',
        '-num_nodes', '9', '-max_gsm_iters', '15', '-reactant_geom_fixed', '-product_geom_fixed',
    ])  # fmt: skip


def cli_de_gsm_xtb():
    """DE-GSM through the native tblite-backed xTB level of theory."""
    xyz = str(DATA_DIR / 'diels_alder.xyz')
    return _cli([
        '-xyzfile', xyz, '-mode', 'DE_GSM', '-package', 'xTB',
        '-num_nodes', '9', '-max_gsm_iters', '15', '-reactant_geom_fixed', '-product_geom_fixed',
    ])  # fmt: skip


def cli_se_gsm():
    xyz = str(DATA_DIR / 'diels_alder.xyz')
    Path('isomers.txt').write_text('ADD 4 12\nADD 1 11\n')
    return _cli([
        '-xyzfile', xyz, '-mode', 'SE_GSM', '-isomers', 'isomers.txt', '-package', 'ase',
        '--ase-class', 'tblite.ase.TBLite', '--ase-kwargs', '{"method": "GFN2-xTB", "verbosity": 0}',
        '-num_nodes', '30', '-max_gsm_iters', '15',
    ])  # fmt: skip


def cli_se_cross():
    """Singlet/triplet crossing search on twisted ethylene (SE-Cross, penalty PES, native xTB)."""
    xyz = str(DATA_DIR / 'ethylene.xyz')
    Path('isomers.txt').write_text('TORSION 2 1 4 5 90.0\n')
    return _cli(
        [
            '-xyzfile', xyz, '-mode', 'SE_Cross', '-isomers', 'isomers.txt', '-package', 'xTB',
            '-multiplicity', '1', '3', '-adiabatic_index', '0', '0', '-num_nodes', '20', '-max_gsm_iters', '15',
        ],
        result_file='meci_000.xyz',
    )  # fmt: skip


SCENARIOS = {
    fn.__name__: fn
    for fn in (
        coords_tric,
        coords_dlc,
        coords_hdlc,
        coords_tric_interleaved,
        opt_ef_nols,
        opt_ef_backtrack,
        opt_lbfgs_nols,
        opt_lbfgs_backtrack,
        degsm_ef_climb_qc_launcher,
        degsm_ef_noclimb,
        degsm_lbfgs_climb,
        degsm_ef_dlc,
        degsm_ef_mp2,
        segsm_ef,
        mecp_penalty_ethylene,
        cli_de_gsm,
        cli_de_gsm_xtb,
        cli_se_gsm,
        cli_se_cross,
    )
}
