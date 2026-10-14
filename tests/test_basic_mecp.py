from pathlib import Path

import pytest

pytest.importorskip('tblite')

from pygsm.coords.dlc import (
    DelocalizedInternalCoordinates,
)
from pygsm.coords.primitives import PrimitiveInternalCoordinates
from pygsm.coords.topology import build_topology
from pygsm.lot.xtb import XTBLoT
from pygsm.molecule.molecule import Molecule
from pygsm.optimizers import EigenvectorFollow
from pygsm.pes.penalty import PenaltyPES
from pygsm.pes.pes import PES
from pygsm.utils import elements, manage_xyz, output


def test_basic_penalty_opt():
    geom = manage_xyz.read_xyzs(str(Path(__file__).resolve().parent / 'data' / 'diels_alder.xyz'))[0]

    coupling_states = []

    lot1 = XTBLoT(
        string_id=0,
        states=[(1, 0)],
        gradient_states=[(1, 0)],
        coupling_states=coupling_states,
        geom=geom,
    )

    lot2 = XTBLoT(
        string_id=1,
        states=[(1, 0)],
        gradient_states=[(1, 0)],
        coupling_states=coupling_states,
        geom=geom,
    )

    pes1 = PES(
        lot=lot1,
        multiplicity=1,
        ad_idx=0,
    )

    pes2 = PES(
        lot=lot2,
        multiplicity=1,
        ad_idx=0,
    )

    pes = PenaltyPES(pes1=pes1, pes2=pes2, lot=lot1)

    output.printcool('Building the topology')
    atom_symbols = manage_xyz.get_atoms(geom)
    ELEMENT_TABLE = elements.ElementData()
    atoms = [ELEMENT_TABLE.from_symbol(atom) for atom in atom_symbols]
    xyz1 = manage_xyz.xyz_to_np(geom)
    top1 = build_topology(
        xyz1,
        atoms,
        bondlistfile=None,
    )

    output.printcool('Building Primitive Internal Coordinates')
    connect = False
    addtr = True
    addcart = False
    p1 = PrimitiveInternalCoordinates(
        xyz=xyz1,
        atoms=atoms,
        connect=connect,
        addtr=addtr,
        addcart=addcart,
        topology=top1,
    )

    output.printcool('Building Delocalized Internal Coordinates')
    coord_obj1 = DelocalizedInternalCoordinates(
        xyz=xyz1,
        atoms=atoms,
        addtr=addtr,
        addcart=addcart,
        connect=connect,
        primitives=p1,
    )

    output.printcool('Building the molecule')

    Form_Hessian = True
    initial = Molecule(
        geom=geom,
        pes=pes,
        coord_obj=coord_obj1,
        form_hessian=Form_Hessian,
    )

    optimizer = EigenvectorFollow(
        linesearch='backtrack',  # a step size algorithm
        conv_grms=0.0005,  # The gradrms threshold, this is generally easy to reach for large systems
        dmax=0.01,  # The initial max step size, will be adjusted if optimizer is doing well. Max is 0.5
        conv_ediff=0.1,  # convergence of difference energy
        conv_de=0.1,  # convergence of energy difference between optimization steps
        conv_gmax=0.005,  # convergence of max gradient
        opt_cross=True,  # use difference energy criteria to determine if you are at crossing
    )

    print(' MECP optimization')
    _geoms, energies = optimizer.optimize(
        molecule=initial,
        ref_energy=initial.energy,
        opt_steps=150,  # The max number of optimization steps, use a small number until you have your final sigma
        verbose=True,
        opt_type='UNCONSTRAINED',
        xyzframerate=1,
    )

    print(f'{energies = }')
    # tblite GFN2-xTB reference (xtb-python gave -1.1495296961093118)
    assert energies[-1] == pytest.approx(-1.1495308976482193)
    print('Finished!')
