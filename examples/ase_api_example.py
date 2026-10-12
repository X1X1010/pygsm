"""Geometry optimisation through the pygsm API with an ASE calculator.

The Morse potential keeps the example dependency-free (only ``ase``); the
resulting structure is chemically meaningless. Swap in any ASE calculator,
e.g. ``tblite.ase.TBLite(method='GFN2-xTB')``.
"""

from pathlib import Path

import ase.io
from ase.calculators.morse import MorsePotential

from pygsm.coords import DelocalizedInternalCoordinates
from pygsm.lot.ase import ASELoT
from pygsm.molecule import Molecule
from pygsm.optimizers import EigenvectorFollow
from pygsm.pes import PES
from pygsm.utils import elements, manage_xyz, output


def optimize(atoms, calculator, opt_steps=500):
    """Minimise ``atoms`` in delocalised internal coordinates; return the final geometry and energy."""
    geom = [[atom.symbol, *atom.position] for atom in atoms]
    element_table = elements.ElementData()

    lot = ASELoT(calculator=calculator, geom=geom)
    pes = PES(lot=lot, ad_idx=0, multiplicity=1)
    coords = DelocalizedInternalCoordinates(
        xyz=atoms.get_positions(),
        atoms=[element_table.from_symbol(symbol) for symbol in atoms.get_chemical_symbols()],
        addtr=False,
    )
    molecule = Molecule(geom=geom, PES=pes, coord_obj=coords, Form_Hessian=True)

    optimizer = EigenvectorFollow(Linesearch='backtrack', OPTTHRESH=0.0005, DMAX=0.5, abs_max_step=0.5, conv_Ediff=0.5)
    output.printcool(f'Initial energy is {molecule.energy:5.4f} kcal/mol')
    geoms, energies = optimizer.optimize(molecule=molecule, refE=molecule.energy, opt_steps=opt_steps)
    output.printcool(f'Final energy is {molecule.energy:5.4f} kcal/mol')
    return geoms[-1], energies[-1]


if __name__ == '__main__':
    output.setup_logging()
    data_file = Path(__file__).resolve().parents[1] / 'tests' / 'data' / 'diels_alder.xyz'
    reactant = ase.io.read(data_file, index=0)
    geom, energy = optimize(reactant, MorsePotential())
    manage_xyz.write_xyz('minimized.xyz', geom, energy, scale=1.0)
