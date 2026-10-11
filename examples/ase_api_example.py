"""
API example modified for an ASE calculator.

The result is meaningless, apart from showing that the code runs, because of using a Morse potential.

"""

from pathlib import Path

import ase.io
import numpy as np
from ase.calculators.morse import MorsePotential

from pygsm.coordinate_systems import DelocalizedInternalCoordinates
from pygsm.level_of_theories.ase import ASELoT
from pygsm.molecule import Molecule
from pygsm.optimizers import EigenvectorFollow
from pygsm.potential_energy_surfaces import PES
from pygsm.utilities import elements, manage_xyz, output


def main(geom):
    output.printcool(' Building the LOT')
    lot = ASELoT(MorsePotential(), geom=geom)

    output.printcool(' Building the PES')
    pes = PES(
        lot=lot,
        ad_idx=0,
        multiplicity=1,
    )

    output.printcool('Building the topology')
    atom_symbols = manage_xyz.get_atoms(geom)
    ELEMENT_TABLE = elements.ElementData()
    atoms = [ELEMENT_TABLE.from_symbol(atom) for atom in atom_symbols]
    xyz = manage_xyz.xyz_to_np(geom)
    #     xyz,
    #     atoms,

    output.printcool('Building Delocalized Internal Coordinates')
    coord_obj1 = DelocalizedInternalCoordinates(
        xyz=xyz,
        atoms=atoms,
        addtr=False,  # Add TRIC
    )

    output.printcool('Building Molecule')
    reactant = Molecule(
        geom=geom,
        PES=pes,
        coord_obj=coord_obj1,
        Form_Hessian=True,
    )

    output.printcool('Creating optimizer')
    optimizer = EigenvectorFollow(Linesearch='backtrack', OPTTHRESH=0.0005, DMAX=0.5, abs_max_step=0.5, conv_Ediff=0.5)

    output.printcool(f'initial energy is {reactant.energy:5.4f} kcal/mol')
    geoms, energies = optimizer.optimize(
        molecule=reactant,
        refE=reactant.energy,
        opt_steps=500,
        verbose=True,
    )

    output.printcool(f'Final energy is {reactant.energy:5.4f}')
    manage_xyz.write_xyz('minimized.xyz', geoms[-1], energies[-1], scale=1.0)


if __name__ == '__main__':
    data_file = Path(__file__).resolve().parents[1] / 'tests' / 'data' / 'diels_alder.xyz'
    diels_adler = ase.io.read(data_file, ':')
    xyz = diels_adler[0].positions

    # this is a hack
    geom = np.column_stack([diels_adler[0].symbols, xyz]).tolist()
    for i in range(len(geom)):
        for j in [1, 2, 3]:
            geom[i][j] = float(geom[i][j])
    # --------------------------

    main(geom)
