# standard library imports
import logging
import os
from collections import namedtuple

# third party
import numpy as np

# local application imports
from ..utilities import manage_xyz, units
from ..utilities.config import Configurable
from ._lot_helpers import infer_gradient_states, normalize_states

logger = logging.getLogger(__name__)


class LoTError(Exception):
    pass


Energy = namedtuple('Energy', 'value unit')
Gradient = namedtuple('Gradient', 'value unit')
Coupling = namedtuple('Coupling', 'value unit')


class Lot(Configurable):
    """Level-of-theory base class: caches energies/gradients/couplings per (multiplicity, state).

    Parameters
    ----------
    geom : geometry as (symbol, x, y, z) rows; gives the atoms and initial coordinates.
    fnm : xyz file to read the geometry from when ``geom`` is not given.
    states : (multiplicity, adiabatic index) states to compute, 0-indexed; default ``[(1, 0)]``.
    gradient_states : states to compute gradients for; defaults to ``states``.
    coupling_states : pair of states to compute the derivative coupling between.
    charge : molecular charge.
    nproc : number of processors.
    do_coupling : compute the derivative coupling.
    node_id : index of the string node this level of theory belongs to.
    ID : string id.
    calc_grad : compute gradients.
    """

    def __init__(
        self,
        *,
        geom: list | None = None,
        fnm: str | None = None,
        states: list | None = None,
        gradient_states: list | None = None,
        coupling_states: tuple | None = None,
        charge: int = 0,
        nproc: int = 1,
        do_coupling: bool = False,
        node_id: int = 0,
        ID: int = 0,
        calc_grad: bool = True,
    ):
        self.geom = geom
        self.fnm = fnm
        self.states = [(1, 0)] if states is None else states
        self.coupling_states = coupling_states
        self.charge = charge
        self.nproc = nproc
        self.do_coupling = do_coupling
        self.node_id = node_id
        self.ID = ID
        self.calc_grad = calc_grad

        # properties
        self.Energy = Energy
        self.Gradient = Gradient
        self.Coupling = Coupling
        self._Energies = {}
        self._Gradients = {}
        self._Couplings = {}

        # count number of states
        self.gradient_states = infer_gradient_states(self.states, gradient_states, self.calc_grad)
        normalized_states, did_normalize = normalize_states(self.states)
        if did_normalize:
            logger.debug('fixing states to be proper length')
            self.states = normalized_states
            logger.debug(f' New states  {self.states}')

        if self.geom is not None:
            logger.debug(' initializing LOT from geom')
        elif self.fnm is not None:
            logger.debug(' initializing LOT from file')
            if not os.path.exists(self.fnm):
                raise OSError
            self.geom = manage_xyz.read_xyz(self.fnm, scale=1.0)
        else:
            raise RuntimeError('Need to initialize LOT object')

        # Cache some useful atributes - other useful attributes are properties
        self.currentCoords = manage_xyz.xyz_to_np(self.geom)
        self.atoms = manage_xyz.get_atoms(self.geom)

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
