"""Double-ended GSM through the pygsm API with an ASE calculator.

Run from the repository root (needs the ``ase`` and ``xtb`` extras)::

    python examples/ase_api_de_gsm.py

It writes ``<name>_gsm.xyz`` (the converged string, energies in kcal/mol in the
comment lines) and ``<name>_ts.xyz`` (the highest-energy node).
"""

from pathlib import Path

import ase.io
from ase import Atoms

from pygsm.coordinate_systems import DelocalizedInternalCoordinates, PrimitiveInternalCoordinates, build_topology
from pygsm.growing_string_methods import DEGSM
from pygsm.level_of_theories.ase import ASELoT
from pygsm.molecule import Molecule
from pygsm.optimizers import LBFGS, EigenvectorFollow
from pygsm.potential_energy_surfaces import PES
from pygsm.utilities import output
from pygsm.utilities.elements import ElementData

# go_gsm(rtype=...): 0 no climbing image, 1 climbing image, 2 climbing image + exact TS search
CLIMB_ONLY = 1


def run_de_gsm(
    reactant: Atoms,
    product: Atoms,
    calculator,
    *,
    coordinate_type: str = 'TRIC',  # TRIC, DLC or HDLC
    optimizer_method: str = 'eigenvector_follow',  # or 'lbfgs'
    num_nodes: int = 11,
    max_gsm_iterations: int = 100,
    max_opt_steps: int = 3,
    optimize_endpoints: bool = True,
    rtype: int = CLIMB_ONLY,
) -> DEGSM:
    element_table = ElementData()
    elements = [element_table.from_symbol(symbol) for symbol in reactant.get_chemical_symbols()]
    geom = [[atom.symbol, *atom.position] for atom in reactant]
    flags = dict(connect=coordinate_type == 'DLC', addtr=coordinate_type == 'TRIC', addcart=coordinate_type == 'HDLC')

    # Level of theory and potential energy surface (ground state)
    lot = ASELoT(calculator=calculator, geom=geom, ID=0)
    pes = PES(lot=lot, ad_idx=0, multiplicity=1)

    # Topology: reactant bonds plus any bond formed in the product
    topology = build_topology(xyz=reactant.get_positions(), atoms=elements)
    for i, j in build_topology(xyz=product.get_positions(), atoms=elements).edges():
        if not topology.has_edge(i, j):
            topology.add_edge(max(i, j), min(i, j))

    # Primitive internals from both endpoints, then the delocalised coordinates
    prims = PrimitiveInternalCoordinates(xyz=reactant.get_positions(), atoms=elements, topology=topology, **flags)
    prims.add_union_primitives(
        PrimitiveInternalCoordinates(xyz=product.get_positions(), atoms=elements, topology=topology, **flags)
    )
    coords = DelocalizedInternalCoordinates(xyz=reactant.get_positions(), atoms=elements, primitives=prims, **flags)

    # Endpoints
    reactant_mol = Molecule(geom=geom, PES=pes, coord_obj=coords, Form_Hessian=optimizer_method == 'eigenvector_follow')
    product_mol = Molecule.copy_from(
        reactant_mol, xyz=product.get_positions(), new_node_id=num_nodes - 1, copy_wavefunction=False
    )

    optimizer_class = {'eigenvector_follow': EigenvectorFollow, 'lbfgs': LBFGS}[optimizer_method]
    optimizer = optimizer_class(
        Linesearch='NoLineSearch',
        DMAX=0.1,
        conv_Ediff=100.0,
        conv_gmax=100.0,
        update_hess_in_bg=not (rtype == CLIMB_ONLY or optimizer_method == 'lbfgs'),
    )

    gsm = DEGSM(
        reactant=reactant_mol,
        product=product_mol,
        nnodes=num_nodes,
        CONV_TOL=0.0005,
        CONV_gmax=100.0,
        CONV_Ediff=100.0,
        ADD_NODE_TOL=0.1,
        optimizer=optimizer,
        interp_method='DLC',
    )

    if optimize_endpoints:
        output.printcool('Optimizing the endpoints')
        optimizer.optimize(molecule=reactant_mol, refE=reactant_mol.energy, opt_steps=100)
        optimizer.optimize(molecule=product_mol, refE=product_mol.energy, opt_steps=100)

    gsm.go_gsm(max_gsm_iterations, max_opt_steps, rtype=rtype)
    return gsm


def write_results(gsm: DEGSM, name: str) -> None:
    frames = []
    for energy, geom in zip(gsm.energies, gsm.geometries, strict=True):
        frame = Atoms(symbols=[x[0] for x in geom], positions=[x[1:4] for x in geom])
        frame.info['energy'] = energy
        frames.append(frame)
    ase.io.write(f'{name}_gsm.xyz', frames)
    ase.io.write(f'{name}_ts.xyz', frames[gsm.TSnode])


if __name__ == '__main__':
    from tblite.ase import TBLite

    output.setup_logging()  # pygsm only logs; show INFO messages on stdout
    data_file = Path(__file__).resolve().parents[1] / 'tests' / 'data' / 'diels_alder.xyz'
    reactant, product = ase.io.read(data_file, index=':')
    gsm = run_de_gsm(reactant, product, TBLite(method='GFN2-xTB', verbosity=0), num_nodes=9, max_gsm_iterations=20)
    write_results(gsm, 'diels_alder')
