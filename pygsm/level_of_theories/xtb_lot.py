"""GFN-xTB level of theory through the tblite Python API.

https://github.com/tblite/tblite
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utilities import elements, manage_xyz, units
from ..utilities.config import option
from .base_lot import Lot, LotConfig, LoTError

# Boltzmann constant in Hartree/K, used to convert the electronic temperature.
KB_HARTREE_PER_KELVIN = 3.166811563455546e-06


def _require_tblite():
    try:
        from tblite.interface import Calculator
    except ModuleNotFoundError as error:
        raise LoTError(
            'xTB backend requires the optional dependency `tblite`. Install with `pip install pygsm[xtb]`.'
        ) from error
    return Calculator


@dataclass(kw_only=True)
class XTBConfig(LotConfig):
    xTB_Hamiltonian: str = option('GFN2-xTB', doc='xTB Hamiltonian', choices=('GFN2-xTB', 'GFN1-xTB', 'IPEA1-xTB'))
    xTB_accuracy: float = option(1.0, doc='xTB numerical accuracy')
    xTB_electronic_temperature: float = option(300.0, doc='electronic temperature in K')
    solvent: str | None = option(None, doc='ALPB implicit solvent name')


class XTBLoT(Lot):
    """GFN1/GFN2-xTB ground states, one calculation per requested spin multiplicity.

    Only adiabatic index 0 is available for each multiplicity, which is enough
    for ground-state strings and for minimum-energy crossing points between
    spin states (``Penalty_PES``).
    """

    config_class = XTBConfig

    def __init__(self, options=None, **kwargs):
        super().__init__(options, **kwargs)
        self.xTB_Hamiltonian = self.options.xTB_Hamiltonian
        self.xTB_accuracy = self.options.xTB_accuracy
        self.xTB_electronic_temperature = self.options.xTB_electronic_temperature
        self.solvent = self.options.solvent
        table = elements.ElementData()
        self.numbers = np.asarray([table.from_symbol(a).atomic_num for a in manage_xyz.get_atoms(self.geom)])

    def run(self, geom, multiplicity, state, runtype='gradient'):
        if state != 0:
            raise LoTError(
                f'xTB only provides the ground state of each multiplicity, got state {(multiplicity, state)}'
            )
        Calculator = _require_tblite()

        positions = manage_xyz.xyz_to_np(geom) * units.ANGSTROM_TO_AU
        calc = Calculator(self.xTB_Hamiltonian, self.numbers, positions, charge=self.charge, uhf=multiplicity - 1)
        calc.set('verbosity', 0)
        calc.set('accuracy', self.xTB_accuracy)
        calc.set('temperature', self.xTB_electronic_temperature * KB_HARTREE_PER_KELVIN)
        if self.solvent is not None:
            calc.add('alpb-solvation', self.solvent)
        res = calc.singlepoint()

        self._Energies[(multiplicity, state)] = self.Energy(res.get('energy'), 'Hartree')
        self._Gradients[(multiplicity, state)] = self.Gradient(res.get('gradient'), 'Hartree/Bohr')
        return res
