"""Internal helpers for molecule construction and copy paths."""

from __future__ import annotations

import os

import numpy as np

from pygsm.utilities import manage_xyz


def load_geometry_inputs(data):
    if data['geom'] is not None:
        print(' getting cartesian coordinates from geom')
        geom = data['geom']
        atoms = manage_xyz.get_atoms(geom)
        xyz = manage_xyz.xyz_to_np(geom)
        return geom, atoms, xyz

    if data['fnm'] is not None:
        print(' reading cartesian coordinates from file')
        if data['ftype'] is None:
            data['ftype'] = os.path.splitext(data['fnm'])[1][1:]
        if not os.path.exists(data['fnm']):
            raise IOError
        geom = manage_xyz.read_xyz(data['fnm'], scale=1.0)
        atoms = manage_xyz.get_atoms(geom)
        xyz = manage_xyz.xyz_to_np(geom)
        return geom, atoms, xyz

    raise RuntimeError


def copy_geometry_inputs(source_geometry, source_xyz, xyz=None, fnm=None):
    if xyz is not None and fnm is not None:
        raise ValueError('Specify at most one of xyz or fnm when copying a Molecule.')

    if fnm is not None:
        new_geom = manage_xyz.read_xyz(fnm, scale=1.0)
        return new_geom, manage_xyz.xyz_to_np(new_geom)

    if xyz is not None:
        xyz_array = np.asarray(xyz)
        return manage_xyz.np_to_xyz(source_geometry, xyz_array), xyz_array

    return source_geometry, source_xyz.copy()


def validate_geometry_inputs(atoms, xyz):
    if not hasattr(atoms, '__getitem__'):
        raise TypeError('atoms must be a sequence of atomic symbols')
    for atom in atoms:
        if not isinstance(atom, str):
            raise TypeError('atom symbols must be strings')
    if type(xyz) is not np.ndarray:
        raise TypeError('xyz must be a numpy ndarray')
    if xyz.shape != (len(atoms), 3):
        raise ValueError('xyz must have shape natoms x 3')


def copy_pes(pes, node_id, copy_wavefunction):
    return type(pes).create_pes_from(
        PES=pes,
        options={'node_id': node_id},
        copy_wavefunction=copy_wavefunction,
    )
