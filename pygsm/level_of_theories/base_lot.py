# standard library imports
import logging
import os
from collections import namedtuple
from dataclasses import dataclass
from typing import Any

# third party
import numpy as np

# local application imports
from ..utilities import manage_xyz, units
from ..utilities.config import Config, option
from ._lot_helpers import infer_gradient_states, normalize_states

logger = logging.getLogger(__name__)


class LoTError(Exception):
    pass


Energy = namedtuple('Energy', 'value unit')
Gradient = namedtuple('Gradient', 'value unit')
Coupling = namedtuple('Coupling', 'value unit')


@dataclass(kw_only=True)
class LotConfig(Config):
    fnm: str | None = option(None, doc='File name to create the LOT object from. Only used if geom is none.')
    geom: Any = option(None, doc='geometry object required to get the atom names and initial coords')
    states: list = option(
        default_factory=lambda: [(1, 0)], doc='list of (multiplicity, adiabatic index) states, 0-indexed'
    )
    gradient_states: list | None = option(None, doc='states to calculate gradients for; defaults to states')
    coupling_states: Any = option(
        None, doc='states to calculate the derivative coupling between (one pair per level of theory)'
    )
    charge: int = option(0, doc='charge of molecule')
    nproc: int = option(1, doc='number of processors')
    do_coupling: bool = option(False, doc='calculate the derivative coupling')
    node_id: int = option(0, doc='node id used for storing orbs, etc.')
    ID: int = option(0, doc='string id used for storing orbs, etc.')
    calc_grad: bool = option(True, doc='calculate gradient or not')


class Lot:
    """Level-of-theory base class: caches energies/gradients/couplings per (multiplicity, state)."""

    config_class = LotConfig

    def __init__(self, options=None, **kwargs):
        self.options = self.config_class.build(options, **kwargs)
        # properties
        self.Energy = Energy
        self.Gradient = Gradient
        self.Coupling = Coupling
        self._Energies = {}
        self._Gradients = {}
        self._Couplings = {}

        # count number of states
        self.options.gradient_states = infer_gradient_states(
            self.states,
            self.options.gradient_states,
            self.calc_grad,
        )
        normalized_states, did_normalize = normalize_states(self.states)
        if did_normalize:
            logger.debug('fixing states to be proper length')
            self.states = normalized_states
            logger.debug(f' New states  {self.states}')

        self.geom = self.options.geom
        if self.geom is not None:
            logger.debug(' initializing LOT from geom')
        elif self.options.fnm is not None:
            logger.debug(' initializing LOT from file')
            if not os.path.exists(self.options.fnm):
                raise OSError
            self.geom = manage_xyz.read_xyz(self.options.fnm, scale=1.0)
        else:
            raise RuntimeError('Need to initialize LOT object')

        # Cache some useful atributes - other useful attributes are properties
        self.currentCoords = manage_xyz.xyz_to_np(self.geom)
        self.atoms = manage_xyz.get_atoms(self.geom)
        self.ID = self.options.ID
        self.nproc = self.options.nproc
        self.charge = self.options.charge
        self.node_id = self.options.node_id

        # Bools for running
        self.hasRanForCurrentCoords = False

    @property
    def Energies(self):
        """
        A list of tuples with multiplicity, state and energy
        """
        return self._Energies

    @Energies.setter
    def Energies(self, value):
        self._Energies = value

    @property
    def Gradients(self):
        """
        A list of tuples with multiplicity, state and energy
        """

        return self._Gradients

    @Gradients.setter
    def Gradients(self, value):
        assert type(value) is dict, 'grada must be dictionary'
        self._Gradients = value

    @property
    def Couplings(self):
        """ """
        return self._Couplings

    @Couplings.setter
    def Couplings(self, value):
        self._Couplings = value

    @property
    def do_coupling(self):
        return self.options.do_coupling

    @do_coupling.setter
    def do_coupling(self, value):
        assert isinstance(value, bool), 'incorrect type for do_coupling'
        self.options.do_coupling = value

    @property
    def coupling_states(self):
        return self.options.coupling_states

    @coupling_states.setter
    def coupling_states(self, value):
        assert isinstance(value, tuple), 'incorrect type for coupling,currently only support a tuple'
        self.options.coupling_states = value

    @property
    def gradient_states(self):
        return self.options.gradient_states

    @gradient_states.setter
    def gradient_states(self, value):
        assert isinstance(value, list), 'incorrect type for gradient'
        self.options.gradient_states = value

    @property
    def states(self):
        return self.options.states

    @states.setter
    def states(self, value):
        assert isinstance(value, list), 'incorrect type for gradient'
        self.options.states = value

    @property
    def calc_grad(self):
        return self.options.calc_grad

    @calc_grad.setter
    def calc_grad(self, value):
        assert isinstance(value, bool), 'incorrect type for calc_grad'
        self.options.calc_grad = value

    @classmethod
    def copy(cls, lot, options=None, copy_wavefunction=True):
        return cls(lot.options.replace(**(options or {})))

    def get_energy(self, coords, multiplicity, state, runtype=None):
        if self._coords_changed(coords):
            self.currentCoords = coords.copy()
            geom = manage_xyz.np_to_xyz(self.geom, self.currentCoords)
            self.runall(geom, runtype)
            self.hasRanForCurrentCoords = True

        Energy = self.Energies[(multiplicity, state)]
        if Energy.unit == 'Hartree':
            return Energy.value * units.KCAL_MOL_PER_AU
        elif Energy.unit == 'kcal/mol' or Energy.unit is None:
            return Energy.value

    def get_gradient(self, coords, multiplicity, state, frozen_atoms=None):
        if self._coords_changed(coords):
            self.currentCoords = coords.copy()
            geom = manage_xyz.np_to_xyz(self.geom, self.currentCoords)
            self.runall(geom)
            self.hasRanForCurrentCoords = True
        Gradient = self.Gradients[(multiplicity, state)]
        if Gradient.value is not None:
            gradient = Gradient.value.copy()
            if frozen_atoms is not None:
                for a in frozen_atoms:
                    gradient[a, :] = 0.0
            if Gradient.unit == 'Hartree/Bohr':
                return gradient * units.ANGSTROM_TO_AU  # Ha/bohr*bohr/ang=Ha/ang
            elif Gradient.unit == 'kcal/mol/Angstrom':
                return gradient * units.KCAL_MOL_TO_AU  # kcalmol/A*Ha/kcalmol=Ha/ang
            else:
                raise NotImplementedError
        else:
            return None

    def get_coupling(self, coords, multiplicity, state1, state2, frozen_atoms=None):
        if self._coords_changed(coords):
            self.currentCoords = coords.copy()
            geom = manage_xyz.np_to_xyz(self.geom, self.currentCoords)
            self.runall(geom)
            self.hasRanForCurrentCoords = True
        Coupling = self.Couplings[(state1, state2)]

        if Coupling.value is not None:
            coupling = Coupling.value.copy()
            if frozen_atoms is not None:
                for a in [3 * i for i in frozen_atoms]:
                    coupling[a : a + 3, 0] = 0.0
            if Coupling.unit == 'Hartree/Bohr':
                return coupling * units.ANGSTROM_TO_AU  # Ha/bohr*bohr/ang=Ha/ang
            else:
                raise NotImplementedError
        else:
            return None

    def run(self, geom, mult, ad_idx, runtype='gradient'):
        raise NotImplementedError

    def runall(self, geom, runtype=None):
        self.Gradients = {}
        self.Energies = {}
        self.Couplings = {}
        for state in self.states:
            mult, ad_idx = state
            if state in self.gradient_states or runtype == 'gradient':
                self.run(geom, mult, ad_idx)
            elif state in self.coupling_states:
                self.run(geom, mult, ad_idx, 'coupling')
            else:
                self.run(geom, mult, ad_idx, 'energy')

    def _coords_changed(self, coords):
        return self.hasRanForCurrentCoords is False or not np.array_equal(coords, self.currentCoords)
