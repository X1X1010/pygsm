"""Class structures of important chemical concepts
This class is the combination of Martinez group and Lee Ping's molecule class.
"""

from dataclasses import dataclass
from time import time
from typing import Any

import numpy as np

from pygsm.coordinate_systems import CartesianCoordinates
from pygsm.utilities import block_matrix, elements, manage_xyz, units
from pygsm.utilities.config import Config, option

from ._helpers import copy_geometry_inputs, copy_pes, load_geometry_inputs, validate_geometry_inputs

ELEMENT_TABLE = elements.ElementData()

# TOC:
# constructors
# methods
# properties

"""Specify a molecule by its atom composition, coordinates, charge
and spin multiplicities.
"""


@dataclass(kw_only=True)
class MoleculeConfig(Config):
    PES: Any = option(doc='potential energy surface used to evaluate energies, gradients, etc.')
    geom: list | None = option(None, doc='geometry as (symbol, x, y, z) tuples in Angstrom')
    fnm: str | None = option(None, doc='xyz file to read the geometry from when geom is not given')
    ftype: str | None = option(None, doc='file type (inferred from the extension if not given)')
    xyz: np.ndarray | None = option(None, doc='Cartesian coordinates in Angstrom')
    coord_obj: Any = option(None, doc='coordinate system object (required)')
    coordinate_type: str = option(
        'Cartesian', doc='type of coordinate system', choices=('Cartesian', 'DLC', 'HDLC', 'TRIC')
    )
    copy_wavefunction: bool = option(
        True, doc='copy the level-of-theory data when copying the PES (not always helpful, e.g. at the start of DE-GSM)'
    )
    Primitive_Hessian: Any = option(None, doc='primitive Hessian to start from')
    Hessian: Any = option(None, doc='Hessian in the current coordinate basis')
    Form_Hessian: bool = option(True, doc='form the Hessian in the current basis (slow for large molecules)')
    top_settings: dict = option(default_factory=dict, doc='extra settings for forming the coordinate object')
    comment: str = option('', doc='descriptive comment saved on the molecule')
    node_id: int = option(0, doc='node id passed on to the level of theory')
    frozen_atoms: list | None = option(None, doc='indices of frozen atoms')


class Molecule:
    """Geometry + PES + coordinate system of one string node."""

    config_class = MoleculeConfig

    @staticmethod
    def _copy_coord_object(coord_obj, xyz=None):
        coord_options = coord_obj.options.replace() if xyz is None else coord_obj.options.replace(xyz=xyz)
        return type(coord_obj)(coord_options)

    @classmethod
    def _copy_fast(cls, source, xyz=None, fnm=None, new_node_id=1, copy_wavefunction=True):
        new_geom, xyz_array = copy_geometry_inputs(source.geometry, source.xyz, xyz=xyz, fnm=fnm)
        pes = copy_pes(source.PES, new_node_id, copy_wavefunction)
        coord_obj = cls._copy_coord_object(source.coord_obj, xyz_array if xyz is not None or fnm is not None else None)

        new = cls.__new__(cls)
        new.options = source.options.replace(
            PES=pes,
            coord_obj=coord_obj,
            geom=new_geom,
            node_id=new_node_id,
            copy_wavefunction=copy_wavefunction,
        )
        new.PES = pes
        new.coord_obj = coord_obj
        new.options.xyz = xyz_array.copy()
        new.options.coord_obj = coord_obj
        new.atoms = list(source.atoms)
        new._initialize_runtime_state()

        if new.options.Hessian is None and new.options.Form_Hessian:
            if new.options.Primitive_Hessian is None and type(new.coord_obj) is not CartesianCoordinates:
                new.form_Primitive_Hessian()
            if new.options.Primitive_Hessian is not None:
                new.form_Hessian_in_basis()
            else:
                new.form_Hessian()

        return new

    def __repr__(self):
        lines = [' molecule object']
        lines.append(self.options.__str__())
        return '\n'.join(lines)

    @classmethod
    def copy_from(cls, MoleculeA, xyz=None, fnm=None, new_node_id=1, copy_wavefunction=True):
        """Copy ``MoleculeA``, optionally with new coordinates, node id and a fresh wavefunction."""
        print(f' Copying from MoleculA {MoleculeA.node_id}')
        return cls._copy_fast(
            MoleculeA,
            xyz=xyz,
            fnm=fnm,
            new_node_id=new_node_id,
            copy_wavefunction=copy_wavefunction,
        )

    def __init__(self, options=None, **kwargs):
        self.options = self.config_class.build(options, **kwargs)

        # => Read in the coordinates <= #
        # important first try to read in geom

        t0 = time()
        _geom, atoms, xyz = load_geometry_inputs(self.options)

        t1 = time()
        print(f' Time to get coords= {t1 - t0:.3f}')

        # Perform all the sanity checks and cache some useful attributes

        # TODO make PES property
        self.PES = copy_pes(self.options.PES, self.options.node_id, self.options.copy_wavefunction)
        validate_geometry_inputs(atoms, xyz)
        self.options.xyz = xyz.copy()

        # create a dictionary from atoms
        # atoms contain info you need to know about the atoms
        self.atoms = [ELEMENT_TABLE.from_symbol(atom) for atom in atoms]

        tx = time()
        print(f' Time to create PES,elements {tx - t1:.3f}')
        t1 = tx

        if not isinstance(self.options.comment, str):
            raise TypeError('comment for a Molecule must be a string')

        # Create the coordinate system
        if self.options.coord_obj is not None:
            print(' getting coord_object from options')
            self.coord_obj = self.options.coord_obj
        else:
            print(' Disabling this feature, Molecule cannot create coordinate system')
            raise NotImplementedError
        self.options.coord_obj = self.coord_obj

        t2 = time()
        print(f' Time  to build coordinate system= {t2 - t1:.3f}')
        self._initialize_runtime_state()

        if self.options.Hessian is None and self.options.Form_Hessian:
            if self.options.Primitive_Hessian is None and type(self.coord_obj) is not CartesianCoordinates:
                self.form_Primitive_Hessian()
            t3 = time()
            print(f' Time to build Prim Hessian {t3 - t2:.3f}')
            if self.options.Primitive_Hessian is not None:
                print(' forming Hessian in basis')
                self.form_Hessian_in_basis()
            else:
                self.form_Hessian()

        print(' molecule constructed')

    def _initialize_runtime_state(self):
        self.gradrms = 0.0
        self.isTSnode = False
        self.bdist = 0.0
        self.newHess = 5
        self._state_token = 0
        self._cached_energy_token = -1
        self._cached_energy = None

    def _invalidate_runtime_caches(self):
        if not hasattr(self, '_state_token'):
            return
        self._state_token += 1
        self._cached_energy_token = -1
        self._cached_energy = None

    def __add__(self, other):
        """add method for molecule objects. Concatenates"""
        raise NotImplementedError

    def center(self, center_mass=False):
        """Move geometric center to the origin."""
        if center_mass:
            com = self.center_of_mass
            self.xyz = self.xyz - com
        else:
            self.xyz = self.xyz - self.xyz.mean(0)

    @property
    def atomic_mass(self):
        return np.array([units.AMU_TO_AU * ele.mass_amu for ele in self.atoms])

    @property
    def mass_amu(self):
        return np.array([ele.mass_amu for ele in self.atoms])

    @property
    def atomic_num(self):
        return [ele.atomic_num for ele in self.atoms]

    @property
    def total_mass_au(self):
        """Returns the total mass of the molecule"""
        return np.sum(self.atomic_mass)

    @property
    def natoms(self):
        """The number of atoms in the molecule"""
        return len(self.atoms)

    @property
    def center_of_mass(self):
        M = self.total_mass_au
        atomic_masses = self.atomic_mass
        return np.sum([self.xyz[i, :] * atomic_masses[i] / M for i in range(self.natoms)], axis=0)

    @property
    def geometry(self):
        return manage_xyz.combine_atom_xyz(self.atom_symbols, self.xyz)

    @property
    def atom_symbols(self):
        return [a.symbol for a in self.atoms]

    @property
    def energy(self):
        if self._cached_energy_token != self._state_token:
            self._cached_energy = self.PES.get_energy(self.xyz)
            self._cached_energy_token = self._state_token
        return self._cached_energy

    @property
    def gradx(self):
        return np.reshape(self.PES.get_gradient(self.xyz, frozen_atoms=self.frozen_atoms), (-1, 3))

    @property
    def gradient(self):
        gradx = self.PES.get_gradient(self.xyz, frozen_atoms=self.frozen_atoms)
        return self.coord_obj.calcGrad(self.xyz, gradx)  # CartesianCoordinate just returns gradx

    # for PES seams
    @property
    def derivative_coupling(self):
        dvecx = self.PES.get_coupling(self.xyz, frozen_atoms=self.frozen_atoms)
        return self.coord_obj.calcGrad(self.xyz, dvecx)

    @property
    def difference_gradient(self):
        dgradx = self.PES.get_dgrad(self.xyz, frozen_atoms=self.frozen_atoms)
        return self.coord_obj.calcGrad(self.xyz, dgradx)

    @property
    def difference_energy(self):
        _ = self.energy  # evaluating the energy refreshes PES.dE
        return self.PES.dE

    @property
    def Primitive_Hessian(self):
        return self.options.Primitive_Hessian

    @Primitive_Hessian.setter
    def Primitive_Hessian(self, value):
        self.options.Primitive_Hessian = value

    def form_Primitive_Hessian(self):
        print(' making primitive Hessian')
        self.options.Primitive_Hessian = self.coord_obj.Prims.guess_hessian(self.xyz)
        self.newHess = 10

    def update_Primitive_Hessian(self, change=None):
        print(' updating prim hess')
        if change is not None:
            self.Primitive_Hessian += change
        return self.Primitive_Hessian

    @property
    def Hessian(self):
        return self.options.Hessian

    @Hessian.setter
    def Hessian(self, value):
        self.options.Hessian = value

    def form_Hessian(self):
        self.options.Hessian = self.coord_obj.guess_hessian(self.xyz)
        self.newHess = 5

    def update_Hessian(self, change=None):
        if change is not None:
            self.Hessian += change
        return self.Hessian

    def form_Hessian_in_basis(self):
        self.Hessian = block_matrix.dot(
            block_matrix.dot(block_matrix.transpose(self.coord_basis), self.Primitive_Hessian), self.coord_basis
        )
        return self.Hessian

    @property
    def xyz(self):
        return self.options.xyz

    @xyz.setter
    def xyz(self, newxyz=None):
        if newxyz is not None:
            self.options.xyz = np.asarray(newxyz)
            self._invalidate_runtime_caches()

    @property
    def frozen_atoms(self):
        return self.options.frozen_atoms

    @property
    def num_frozen_atoms(self):
        if self.frozen_atoms is not None:
            return len(self.frozen_atoms)
        else:
            return 0

    def update_xyz(self, dq=None, verbose=True):
        if dq is not None:
            self.xyz = self.coord_obj.newCartesian(self.xyz, dq, frozen_atoms=self.frozen_atoms, verbose=verbose)
        return self.xyz

    @property
    def primitive_internal_coordinates(self):
        return self.coord_obj.Prims.Internals

    @property
    def num_primitives(self):
        return len(self.primitive_internal_coordinates)

    @property
    def primitive_internal_values(self):
        ans = self.coord_obj.Prims.calculate(self.xyz)
        return np.asarray(ans)

    @property
    def coord_basis(self):
        return self.coord_obj.Vecs

    @coord_basis.setter
    def coord_basis(self, value):
        self.coord_obj.Vecs = value

    def update_coordinate_basis(self, constraints=None):
        if isinstance(self.coord_obj, CartesianCoordinates):
            return

        print(' updating coord basis')
        self.coord_obj.clearCache()
        self.coord_obj.build_dlc(self.xyz, constraints)
        return self.coord_basis

    @property
    def constraints(self):
        return self.coord_obj.Vecs.cnorms

    @property
    def coordinates(self):
        return np.reshape(self.coord_obj.calculate(self.xyz), (-1, 1))

    @property
    def BMatrix(self):
        return self.coord_obj.Prims.wilsonB(self.xyz)

    @property
    def WilsonB(self):
        return self.coord_obj.wilsonB(self.xyz)

    @property
    def num_coordinates(self):
        return len(self.coordinates)

    @property
    def node_id(self):
        return self.options.node_id

    @node_id.setter
    def node_id(self, value):
        self.options.node_id = value
        self.PES.lot.node_id = value

    @property
    def state_token(self):
        return self._state_token
