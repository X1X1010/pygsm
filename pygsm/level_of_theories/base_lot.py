# standard library imports
import os
from collections import namedtuple

# third party
import numpy as np

# local application imports
from ..utilities import manage_xyz, options, units
from ._lot_helpers import (
    energy_file_path,
    ensure_node_scratch_dir,
    infer_gradient_states,
    node_scratch_dir,
    normalize_states,
)


class LoTError(Exception):
    pass

Energy = namedtuple('Energy','value unit')
Gradient = namedtuple('Gradient','value unit')
Coupling = namedtuple('Coupling','value unit')

class Lot:
    """ Lot object for level of theory calculators """

    @staticmethod
    def default_options():
        """ Lot default options. """

        if hasattr(Lot, '_default_options'):
            return Lot._default_options.copy()
        opt = options.Options()

        opt.add_option(
            key='fnm',
            value=None,
            required=False,
            allowed_types=[str],
            doc='File name to create the LOT object from. Only used if geom is none.'
        )

        opt.add_option(
            key='geom',
            value=None,
            required=False,
            doc='geometry object required to get the atom names and initial coords'
        )

        opt.add_option(
            key='states',
            value=[(1, 0)],
            required=False,
            doc='list of states 0-indexed')

        opt.add_option(
            key='gradient_states',
            value=None,
            required=False,
            doc='list of states to calculate gradients for, will assume same as states if not given'
        )

        opt.add_option(
            key='coupling_states',
            value=None,
            required=False,
            doc='states to calculate derivative coupling. Currently only one coupling can be calculated per level of theory object.'
        )

        opt.add_option(
            key='charge',
            value=0,
            required=False,
            allowed_types=[int],
            doc='charge of molecule')

        opt.add_option(
            key='nproc',
            required=False,
            value=1,
            allowed_types=[int],
            doc="number of processors",
        )

        opt.add_option(
            key='do_coupling',
            required=False,
            value=False,
            doc='derivative coupling'
        )

        opt.add_option(
            key="node_id",
            required=False,
            value=0,
            allowed_types=[int],
            doc='node id used for storing orbs,etc'
        )

        opt.add_option(
            key="ID",
            required=False,
            value=0,
            allowed_types=[int],
            doc=' id used for storing orbs,etc for string'
        )

        opt.add_option(
            key="calc_grad",
            required=False,
            value=True,
            allowed_types=[bool],
            doc=' calculate gradient or not'
        )

        opt.add_option(
                key='xTB_Hamiltonian',
                value='GFN2-xTB',
                required=False,
                allowed_types=[str],
                doc='xTB hamiltonian'
                )

        opt.add_option(
                key='xTB_accuracy',
                value=1.0,
                required=False,
                allowed_types=[float],
                doc='xTB accuracy'
                )

        opt.add_option(
                key='xTB_electronic_temperature',
                value=300,
                required=False,
                allowed_types=[float],
                doc='xTB electronic_temperature'
                )

        opt.add_option(
            key='solvent',
            value=None,
            required=False,
            allowed_types=[str],
            doc='xTB solvent'
        )

        opt.add_option(
            key='scratch_root',
            value='scratch',
            required=False,
            allowed_types=[str],
            doc='root directory for scratch outputs',
        )

        opt.add_option(
            key='write_energy_file',
            value=True,
            required=False,
            allowed_types=[bool],
            doc='write cached state energies to scratch output',
        )

        Lot._default_options = opt
        return Lot._default_options.copy()

    def __init__(self,
                 options,
                 ):
        """ Constructor """
        self.options = options
        # properties
        self.Energy = Energy
        self.Gradient = Gradient
        self.Coupling = Coupling
        self._Energies={}
        self._Gradients={}
        self._Couplings={}

        # count number of states
        self.options['gradient_states'] = infer_gradient_states(
            self.states,
            self.options['gradient_states'],
            self.calc_grad,
        )
        normalized_states, did_normalize = normalize_states(self.states)
        if did_normalize:
            print('fixing states to be proper length')
            self.states = normalized_states
            print(' New states ', self.states)

        self.geom = self.options['geom']
        if self.geom is not None:
            print(" initializing LOT from geom")
        elif self.options['fnm'] is not None:
            print(" initializing LOT from file")
            if not os.path.exists(self.options['fnm']):
                raise OSError
            self.geom = manage_xyz.read_xyz(self.options['fnm'], scale=1.)
        else:
            raise RuntimeError("Need to initialize LOT object")

        # Cache some useful atributes - other useful attributes are properties
        self.currentCoords = manage_xyz.xyz_to_np(self.geom)
        self.atoms = manage_xyz.get_atoms(self.geom)
        self.ID = self.options['ID']
        self.nproc = self.options['nproc']
        self.charge = self.options['charge']
        self.node_id = self.options['node_id']
        self.xTB_Hamiltonian = self.options['xTB_Hamiltonian']
        self.xTB_accuracy = self.options['xTB_accuracy']
        self.xTB_electronic_temperature = self.options['xTB_electronic_temperature']
        self.solvent = self.options['solvent']
        self.scratch_root = self.options['scratch_root']

        # Bools for running
        self.hasRanForCurrentCoords = False

        ensure_node_scratch_dir(self.scratch_root, self.ID, self.node_id)

    @classmethod
    def from_options(cls, **kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return cls(cls.default_options().set_values(kwargs))

    @property
    def Energies(self):
        '''
        A list of tuples with multiplicity, state and energy
        '''
        return self._Energies

    @Energies.setter
    def Energies(self, value):
        self._Energies = value

    @property
    def Gradients(self):
        '''
        A list of tuples with multiplicity, state and energy
        '''

        return self._Gradients

    @Gradients.setter
    def Gradients(self, value):
        assert type(value) is dict, "grada must be dictionary"
        self._Gradients = value

    @property
    def Couplings(self):
        '''
        '''
        return self._Couplings

    @Couplings.setter
    def Couplings(self, value):
        self._Couplings = value

    @property
    def do_coupling(self):
        return self.options['do_coupling']

    @do_coupling.setter
    def do_coupling(self, value):
        assert isinstance(value, bool), "incorrect type for do_coupling"
        self.options['do_coupling'] = value

    @property
    def coupling_states(self):
        return self.options['coupling_states']

    @coupling_states.setter
    def coupling_states(self, value):
        assert isinstance(value, tuple), "incorrect type for coupling,currently only support a tuple"
        self.options['coupling_states'] = value

    @property
    def gradient_states(self):
        return self.options['gradient_states']

    @gradient_states.setter
    def gradient_states(self, value):
        assert isinstance(value, list), "incorrect type for gradient"
        self.options['gradient_states'] = value

    @property
    def states(self):
        return self.options['states']

    @states.setter
    def states(self, value):
        assert isinstance(value, list), "incorrect type for gradient"
        self.options['states'] = value

    @property
    def calc_grad(self):
        return self.options['calc_grad']

    @calc_grad.setter
    def calc_grad(self, value):
        assert isinstance(value, bool), "incorrect type for calc_grad"
        self.options['calc_grad'] = value

    @classmethod
    def copy(cls, lot, options=None, copy_wavefunction=True):
        return cls(lot.options.copy().set_values(options or {}))

    def get_energy(self, coords, multiplicity, state, runtype=None):
        if self._coords_changed(coords):
            self.currentCoords = coords.copy()
            geom = manage_xyz.np_to_xyz(self.geom, self.currentCoords)
            self.runall(geom, runtype)
            self.hasRanForCurrentCoords = True

        Energy = self.Energies[(multiplicity,state)]
        if Energy.unit=="Hartree":
            return Energy.value*units.KCAL_MOL_PER_AU
        elif Energy.unit == 'kcal/mol' or Energy.unit is None:
            return Energy.value

    def get_gradient(self, coords, multiplicity, state, frozen_atoms=None):
        if self._coords_changed(coords):
            self.currentCoords = coords.copy()
            geom = manage_xyz.np_to_xyz(self.geom, self.currentCoords)
            self.runall(geom)
            self.hasRanForCurrentCoords = True
        Gradient = self.Gradients[(multiplicity,state)]
        if Gradient.value is not None:
            gradient = Gradient.value.copy()
            if frozen_atoms is not None:
                for a in frozen_atoms:
                    gradient[a, :] = 0.
            if Gradient.unit == "Hartree/Bohr":
                return gradient * units.ANGSTROM_TO_AU  # Ha/bohr*bohr/ang=Ha/ang
            elif Gradient.unit == "kcal/mol/Angstrom":
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
        Coupling = self.Couplings[(state1,state2)]

        if Coupling.value is not None:
            coupling = Coupling.value.copy()
            if frozen_atoms is not None:
                for a in [3*i for i in frozen_atoms]:
                    coupling[a:a+3, 0] = 0.
            if Coupling.unit == "Hartree/Bohr":
                return coupling * units.ANGSTROM_TO_AU  # Ha/bohr*bohr/ang=Ha/ang
            else:
                raise NotImplementedError
        else:
            return None

    def write_E_to_file(self):
        if not self.options['write_energy_file']:
            return
        with open(self.energy_file_path, 'w') as f:
            for key, Energy in self.Energies.items():
                f.write(f'{key[0]} {key[1]} {Energy.value:9.7f} Hartree\n')

    def run(self, geom, mult, ad_idx, runtype='gradient'):
        raise NotImplementedError

    def runall(self, geom, runtype=None):
        self.Gradients = {}
        self.Energies = {}
        self.Couplings = {}
        for state in self.states:
            mult, ad_idx = state
            if state in self.gradient_states or runtype == "gradient":
                self.run(geom, mult, ad_idx)
            elif state in self.coupling_states:
                self.run(geom, mult, ad_idx, 'coupling')
            else:
                self.run(geom, mult, ad_idx, 'energy')

    @property
    def node_scratch_dir(self):
        return node_scratch_dir(self.scratch_root, self.ID, self.node_id)

    @property
    def energy_file_path(self):
        return energy_file_path(self.scratch_root, self.ID, self.node_id)

    def _coords_changed(self, coords):
        return self.hasRanForCurrentCoords is False or not np.array_equal(coords, self.currentCoords)
