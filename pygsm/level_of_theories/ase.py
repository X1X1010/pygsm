"""
Level Of Theory for ASE calculators
https://gitlab.com/ase/ase

Written by Tamas K. Stenczel in 2021
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .base_lot import Lot, LoTError

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator


def _require_ase():
    try:
        from ase import Atoms, units
        from ase.calculators.calculator import Calculator
        from ase.data import atomic_numbers
    except ModuleNotFoundError as error:
        raise LoTError(
            'ASE backend requires the optional dependency `ase`. Install with `pip install pygsm[ase]`.'
        ) from error
    return Atoms, Calculator, atomic_numbers, units


class ASELoT(Lot):
    """Level of theory backed by any ASE calculator.

    ``calculator`` is the ASE calculator; the other parameters are those of :class:`Lot`.
    The calculator ignores the multiplicity.
    """

    def __init__(self, calculator: Calculator, **kwargs):
        super().__init__(**kwargs)
        self.calculator = calculator

    @classmethod
    def from_calculator_string(cls, calculator_import: str, calculator_kwargs=None, **kwargs):
        _, Calculator, _, _ = _require_ase()
        if calculator_kwargs is None:
            calculator_kwargs = {}
        # this imports the calculator
        module_name = '.'.join(calculator_import.split('.')[:-1])
        class_name = calculator_import.split('.')[-1]

        # import the module of the calculator
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            raise LoTError(f"ASE-calculator's module is not found: {module_name}") from error

        # class of the calculator
        if hasattr(module, class_name):
            calc_class = getattr(module, class_name)
            assert issubclass(calc_class, Calculator)
        else:
            raise LoTError(f"ASE-calculator's class ({class_name}) not found in module {module_name}")

        # the calculator is built here, so drop one passed by mistake
        _ = kwargs.pop('calculator', None)

        # construct from the constructor
        return cls(calc_class(**calculator_kwargs), **kwargs)

    def run(self, geom, mult, ad_idx, runtype='gradient'):
        # run ASE
        self.run_ase_atoms(xyz_to_ase(geom), mult, ad_idx, runtype)

    def run_ase_atoms(self, atoms: Atoms, mult, ad_idx, runtype='gradient'):
        _, _, _, units = _require_ase()
        # set the calculator
        atoms.calc = self.calculator

        # perform gradient calculation if needed
        if runtype == 'gradient':
            self._Gradients[(mult, ad_idx)] = self.Gradient(-atoms.get_forces() / units.Ha * units.Bohr, 'Hartree/Bohr')
        elif runtype == 'energy':
            pass
        else:
            raise NotImplementedError(f'Run type {runtype} is not implemented in the ASE calculator interface')

        # energy is always calculated -> cached if force calculation was done
        self._Energies[(mult, ad_idx)] = self.Energy(atoms.get_potential_energy() / units.Ha, 'Hartree')

        self.hasRanForCurrentCoords = True


def xyz_to_ase(xyz):
    """

    Parameters
    ----------
    xyz : np.ndarray, shape=(N, 4)


    Returns
    -------
    atoms : ase.Atoms
        ASE's Atoms object

    """

    # compatible with list-of-list as well
    _, _, atomic_numbers, _ = _require_ase()
    numbers = [atomic_numbers[x[0]] for x in xyz]
    pos = [x[1:4] for x in xyz]
    return geom_to_ase(numbers, pos)


def geom_to_ase(numbers, positions, **kwargs):
    """Geometry to ASE atoms object

    Parameters
    ----------
    numbers : array_like, shape=(N_atoms,)
        atomic numbers
    positions : array_like, shape=(N_atoms,3)
        positions of atoms in Angstroms
    kwargs

    Returns
    -------
    atoms : ase.Atoms
        ASE's Atoms object
    """

    Atoms, _, _, _ = _require_ase()
    return Atoms(numbers=numbers, positions=positions, **kwargs)
