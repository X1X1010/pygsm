import numpy as np

try:
    pass
except:
    pass

import re

# => XYZ File Utility <= #


def read_xyz(
        filename,
        scale=1.):
    """ Read xyz file

    Params:
        filename (str) - name of xyz file to read

    Returns:
        geom ((natoms,4) np.ndarray) - system geometry (atom symbol, x,y,z)

    """

    lines = open(filename).readlines()
    lines = lines[2:]
    geom = []
    for line in lines:
        mobj = re.match(r'^\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$', line)
        geom.append((
            mobj.group(1),
            scale*float(mobj.group(2)),
            scale*float(mobj.group(3)),
            scale*float(mobj.group(4)),
        ))
    return geom


def read_xyzs(
    filename,
    scale=1.
):
    """ Read xyz file

    Params:
        filename (str) - name of xyz file to read

    Returns:
        geom ((natoms,4) np.ndarray) - system geometry (atom symbol, x,y,z)

    """

    lines = open(filename).readlines()
    natoms = int(lines[0])
    total_lines = len(lines)
    num_geoms = total_lines/(natoms+2)

    geoms = []
    sa = 2
    for i in range(int(num_geoms)):
        ea = sa+natoms
        geom = []
        for line in lines[sa:ea]:
            mobj = re.match(r'^\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$', line)
            geom.append((
                mobj.group(1),
                scale*float(mobj.group(2)),
                scale*float(mobj.group(3)),
                scale*float(mobj.group(4)),
            ))
        sa = ea+2
        geoms.append(geom)
    return geoms


def read_molden_geoms(
    filename,
    scale=1.
):

    lines = open(filename).readlines()
    natoms = int(lines[2])
    nlines = len(lines)

    # this is for three blocks after GEOCON
    num_geoms = (nlines-6) / (natoms+5)
    num_geoms = int(num_geoms)
    print(num_geoms)
    geoms = []

    sa = 4
    for i in range(int(num_geoms)):
        ea = sa+natoms
        geom = []
        for line in lines[sa:ea]:
            mobj = re.match(r'^\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$', line)
            geom.append((
                mobj.group(1),
                scale*float(mobj.group(2)),
                scale*float(mobj.group(3)),
                scale*float(mobj.group(4)),
            ))
        sa = ea+2
        geoms.append(geom)
    return geoms


def write_molden_geoms(
        filename,
        geoms,
        energies,
        gradrms,
        dEs,
):
    with open(filename, 'w') as f:
        f.write("[Molden Format]\n[Geometries] (XYZ)\n")
        for geom in geoms:
            f.write(f'{len(geom)}\n\n')
            for atom in geom:
                f.write(f'{atom[0]:<2} {atom[1]:14.6f} {atom[2]:14.6f} {atom[3]:14.6f}\n')
        f.write("[GEOCONV]\n")
        f.write('energy\n')
        V0 = energies[0]
        for energy in energies:
            f.write(f'{energy-V0}\n')
        f.write("max-force\n")
        for grad in gradrms:
            f.write(f"{grad:.6f}\n")
        f.write("max-step\n")
        for dE in dEs:
            f.write(f"{dE:.6f}\n")

def get_atoms(
        geom,
):

    atoms = []
    for atom in geom:
        atoms.append(atom[0])
    return atoms


def write_xyz(
    filename,
    geom,
    comment=0,
    scale=1.0  # (1.0/units.ANGSTROM_TO_AU),
):
    """ Writes xyz file with single frame

    Params:
        filename (str) - name of xyz file to write
        geom ((natoms,4) np.ndarray) - system geometry (atom symbol, x,y,z)

    """
    fh = open(filename, 'w')
    fh.write(f'{len(geom)}\n')
    fh.write(f'{comment}\n')
    for atom in geom:
        fh.write(f'{atom[0]:<2} {scale*atom[1]:14.6f} {scale*atom[2]:14.6f} {scale*atom[3]:14.6f}\n')


def write_xyzs(
    filename,
    geoms,
    scale=1.,
):
    """ Writes xyz trajectory file with multiple frames

    Params:
        filename (str) - name of xyz file to write
        geom ((natoms,4) np.ndarray) - system geometry (atom symbol, x,y,z)

    Returns:

    """

    fh = open(filename, 'w')
    for geom in geoms:
        fh.write(f'{len(geom)}\n\n')
        for atom in geom:
            fh.write(f'{atom[0]:<2} {scale*atom[1]:14.6f} {scale*atom[2]:14.6f} {scale*atom[3]:14.6f}\n')


def write_std_multixyz(
        filename,
        geoms,
        energies,
        gradrms,
        dEs,
):
    with open(filename, 'w') as f:
        for E, geom in zip(energies, geoms):
            f.write(f'{len(geom)}\n')
            # Energies are already in kcal/mol (see level_of_theories/base_lot.py
            # get_energy, which returns Hartree * KCAL_MOL_PER_AU). Write them in
            # that native unit with an explicit label so the output is a clean,
            # documented unit and downstream readers can detect this format.
            # Previously this multiplied E by KJ_MOL_TO_AU, which treated a
            # kcal/mol value as if it were kJ/mol and produced a garbled
            # AU-scaled number (see ZimmermanGroup/pyGSM#65).
            f.write(f'{E:.6f} kcal/mol\n')
            for atom in geom:
                f.write(f'{atom[0]:<2} {atom[1]:14.6f} {atom[2]:14.6f} {atom[3]:14.6f}\n')


def write_xyzs_w_comments(
    filename,
    geoms,
    comments,
    scale=1.0  # (1.0/units.ANGSTROM_TO_AU),
):
    """ Writes xyz trajectory file with multiple frames

    Params:
        filename (str) - name of xyz file to write
        geom ((natoms,4) np.ndarray) - system geometry (atom symbol, x,y,z)

    Returns:

    """

    fh = open(filename, 'w')
    for geom, comment in zip(geoms, comments):
        fh.write(f'{len(geom)}\n')
        fh.write(f'{comment}\n')
        for atom in geom:
            fh.write(f'{atom[0]:<2} {scale*atom[1]:14.6f} {scale*atom[2]:14.6f} {scale*atom[3]:14.6f}\n')


def xyz_to_np(
    geom,
):
    """ Convert from xyz file format xyz array for editing geometry

    Params:
        geom ((natoms,4) np.ndarray) - system geometry (atom symbol, x,y,z)

    Returns:
        xyz ((natoms,3) np.ndarray) - system geometry (x,y,z)

    """

    xyz2 = np.zeros((len(geom), 3))
    for A, atom in enumerate(geom):
        xyz2[A, 0] = atom[1]
        xyz2[A, 1] = atom[2]
        xyz2[A, 2] = atom[3]
    return xyz2


def np_to_xyz(
    geom,
    xyz2,
):
    """ Convert from xyz array to xyz file format in order to write xyz

    Params:
        geom ((natoms,4) np.ndarray) - system reference geometry
            (atom symbol, x,y,z) from xyz file
        xyz2 ((natoms,3) np.ndarray) - system geometry (x,y,z)

    Returns:
        geom2 ((natoms,4) np.ndarray) - new system geometry
            (atom symbol, x,y,z)

    """

    geom2 = []
    for A, atom in enumerate(geom):
        geom2.append((
            atom[0],
            xyz2[A, 0],
            xyz2[A, 1],
            xyz2[A, 2],
        ))
    return geom2


def combine_atom_xyz(
    atoms,
    xyz,
):
    """ Combines atom list with xyz array
     Params:
        atom list
        geom ((natoms,3) np.ndarray) - system geometry (atom symbol, x,y,z)

    Returns:
        geom2 ((natoms,4) np.ndarray) - new system geometry
            (atom symbol, x,y,z)

    """
    geom2 = []
    for A, atom in enumerate(atoms):
        geom2.append((
            atom,
            xyz[A, 0],
            xyz[A, 1],
            xyz[A, 2],
        ))
    return geom2


XYZ_WRITERS = {
    'molden': write_molden_geoms,
    'multixyz': write_std_multixyz,
}
