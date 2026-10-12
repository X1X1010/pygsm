"""Class structures of important chemical concepts
This class is the combination of Martinez group and Lee Ping's molecule class.
"""

from __future__ import annotations

import logging
from time import time

import numpy as np

from pygsm.coordinate_systems import CartesianCoordinates
from pygsm.utilities import block_matrix, elements, manage_xyz, units

from ._helpers import copy_geometry_inputs, copy_pes, load_geometry_inputs, validate_geometry_inputs

logger = logging.getLogger(__name__)

ELEMENT_TABLE = elements.ElementData()

# TOC:
# constructors
# methods
# properties

"""Specify a molecule by its atom composition, coordinates, charge
and spin multiplicities.
"""


class Molecule:
    """Geometry + PES + coordinate system of one string node.

    Parameters
    ----------
    PES : potential energy surface; the molecule gets its own copy for ``node_id``.
    coord_obj : coordinate system (e.g. ``DelocalizedInternalCoordinates``).
    geom : geometry as (symbol, x, y, z) rows in Angstrom.
    fnm : xyz file to read the geometry from when ``geom`` is not given.
    Form_Hessian : form the Hessian in the current basis (slow for large molecules).
    Primitive_Hessian : primitive Hessian to start from.
    Hessian : Hessian in the current coordinate basis to start from.
    node_id : index of the string node, passed on to the level of theory.
    frozen_atoms : indices of frozen atoms.
    comment : descriptive comment.
    """

    @staticmethod
    def _copy_coord_object(coord_obj, xyz=None):
        return coord_obj.clone() if xyz is None else coord_obj.clone(xyz=xyz)

    @classmethod
    def _copy_fast(cls, source, xyz=None, fnm=None, new_node_id=1):
        _geom, xyz_array = copy_geometry_inputs(source.geometry, source.xyz, xyz=xyz, fnm=fnm)
        pes = copy_pes(source.PES, new_node_id)
        coord_obj = cls._copy_coord_object(source.coord_obj, xyz_array if xyz is not None or fnm is not None else None)

        new = cls.__new__(cls)
        new.PES = pes
        new.coord_obj = coord_obj
        new.Form_Hessian = source.Form_Hessian
        new.Primitive_Hessian = source.Primitive_Hessian
        new.Hessian = source.Hessian
        new._node_id = new_node_id
        new.frozen_atoms = source.frozen_atoms
        new.comment = source.comment
        new._xyz = xyz_array.copy()
        new.atoms = list(source.atoms)
        new._initialize_runtime_state()

        if new.Hessian is None and new.Form_Hessian:
            if new.Primitive_Hessian is None and type(new.coord_obj) is not CartesianCoordinates:
                new.form_Primitive_Hessian()
            if new.Primitive_Hessian is not None:
                new.form_Hessian_in_basis()
            else:
                new.form_Hessian()

        return new

    def __repr__(self):
        return f'{type(self).__name__}(node_id={self.node_id}, natoms={self.natoms}, coord_obj={type(self.coord_obj).__name__})'

    @classmethod
    def copy_from(
        cls,
        MoleculeA: Molecule,
        xyz: np.ndarray | None = None,
        fnm: str | None = None,
        new_node_id: int = 1,
    ) -> Molecule:
        """Copy ``MoleculeA`` for string node ``new_node_id``, optionally with new coordinates."""
        logger.debug(f' Copying from MoleculA {MoleculeA.node_id}')
        return cls._copy_fast(
            MoleculeA,
            xyz=xyz,
            fnm=fnm,
            new_node_id=new_node_id,
        )

    def __init__(
        self,
        PES,
        coord_obj,
        *,
        geom: list | None = None,
        fnm: str | None = None,
        Form_Hessian: bool = True,
        Primitive_Hessian=None,
        Hessian=None,
        node_id: int = 0,
        frozen_atoms: list | None = None,
        comment: str = '',
    ):
        if coord_obj is None:
            raise ValueError('Molecule needs a coordinate system (coord_obj)')
        if not isinstance(comment, str):
            raise TypeError('comment for a Molecule must be a string')
        self.Form_Hessian = Form_Hessian
        self.Primitive_Hessian = Primitive_Hessian
        self.Hessian = Hessian
        self._node_id = node_id
        self.frozen_atoms = frozen_atoms
        self.comment = comment

        t0 = time()
        _geom, atoms, xyz = load_geometry_inputs(geom, fnm)
        t1 = time()
        logger.debug(f' Time to get coords= {t1 - t0:.3f}')

        self.PES = copy_pes(PES, node_id)
        validate_geometry_inputs(atoms, xyz)
        self._xyz = xyz.copy()

        # atoms contain info you need to know about the atoms
        self.atoms = [ELEMENT_TABLE.from_symbol(atom) for atom in atoms]
        logger.debug(f' Time to create PES,elements {time() - t1:.3f}')

        self.coord_obj = coord_obj
        t2 = time()
        self._initialize_runtime_state()

        if self.Hessian is None and self.Form_Hessian:
            if self.Primitive_Hessian is None and type(self.coord_obj) is not CartesianCoordinates:
                self.form_Primitive_Hessian()
            t3 = time()
            logger.debug(f' Time to build Prim Hessian {t3 - t2:.3f}')
            if self.Primitive_Hessian is not None:
                logger.debug(' forming Hessian in basis')
                self.form_Hessian_in_basis()
            else:
                self.form_Hessian()

        logger.debug(' molecule constructed')

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

    def form_Primitive_Hessian(self):
        logger.debug(' making primitive Hessian')
        self.Primitive_Hessian = self.coord_obj.Prims.guess_hessian(self.xyz)
        self.newHess = 10

    def update_Primitive_Hessian(self, change=None):
        logger.debug(' updating prim hess')
        if change is not None:
            self.Primitive_Hessian += change
        return self.Primitive_Hessian

    def form_Hessian(self):
        self.Hessian = self.coord_obj.guess_hessian(self.xyz)
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
        return self._xyz

    @xyz.setter
    def xyz(self, newxyz=None):
        if newxyz is not None:
            self._xyz = np.asarray(newxyz)
            self._invalidate_runtime_caches()

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

        logger.debug(' updating coord basis')
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
        return self._node_id

    @node_id.setter
    def node_id(self, value):
        self._node_id = value
        self.PES.lot.node_id = value

    @property
    def state_token(self):
        return self._state_token
