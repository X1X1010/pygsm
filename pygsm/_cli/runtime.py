"""Workflow orchestration for the ``gsm`` command."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from pygsm.coords import (
    DelocalizedInternalCoordinates,
    Distance,
    PrimitiveInternalCoordinates,
    build_topology,
)
from pygsm.coords.primitives import get_driving_coord_prim
from pygsm.gsm import DEGSM, SEGSM, GrowthError, SECross
from pygsm.molecule import Molecule
from pygsm.utils import elements, manage_xyz, output
from pygsm.utils.cli_utils import plot
from pygsm.utils.manage_xyz import XYZ_WRITERS

from .arguments import parse_arguments
from .factory import choose_optimizer, choose_pes, create_lot

logger = logging.getLogger(__name__)


def _read_nonempty_lines(path: str) -> list[str]:
    with open(path) as handle:
        return [line.rstrip() for line in handle if line.rstrip()]


def _read_force_like_file(path: str, integer_prefix: int) -> list[tuple]:
    records = []
    for line in _read_nonempty_lines(path):
        record = []
        for index, entry in enumerate(line.split()):
            if index < integer_prefix:
                record.append(int(entry))
            else:
                record.append(float(entry))
        records.append(tuple(record))
    return records


def _read_index_list(path: str | None) -> list[int] | None:
    if path is None:
        return None
    with open(path) as handle:
        return [int(value) for value in handle.read().splitlines() if value]


def _read_primitive_regions(path: str | None):
    if path is None:
        return None
    prim_indices = np.loadtxt(path)
    if prim_indices.ndim == 2:
        return [(int(prim_indices[i, 0]), int(prim_indices[i, 1]) - 1) for i in range(len(prim_indices))]
    if prim_indices.ndim == 1:
        return [(int(prim_indices[0]), int(prim_indices[1]) - 1)]
    raise ValueError(f'Unsupported primitive region format in {path}')


def _coordinate_flags(coordinate_type: str) -> tuple[bool, bool, bool]:
    return (
        coordinate_type == 'DLC',
        coordinate_type == 'TRIC',
        coordinate_type == 'HDLC',
    )


def _build_topology_data(inpfileq: dict, geoms):
    atom_symbols = manage_xyz.get_atoms(geoms[0])
    element_table = elements.ElementData()
    atoms = [element_table.from_symbol(atom) for atom in atom_symbols]
    xyz1 = manage_xyz.xyz_to_np(geoms[0])

    hybrid_indices = _read_index_list(inpfileq['hybrid_coord_idx_file'])
    if hybrid_indices is not None:
        output.printcool(' Using Hybrid COORDINATES :)')
        assert inpfileq['coordinate_type'] == 'TRIC', (
            "hybrid indices won't work (currently) with other coordinate systems"
        )

    prim_indices = _read_primitive_regions(inpfileq['prim_idx_file'])
    if prim_indices is not None:
        output.printcool(' Defining primitive internal region :)')
        assert inpfileq['coordinate_type'] == 'TRIC', "won't work (currently) with other coordinate systems"
        logger.debug(prim_indices)

    top1 = build_topology(
        xyz1,
        atoms,
        hybrid_indices=hybrid_indices,
        prim_idx_start_stop=prim_indices,
        bondlistfile=inpfileq['bonds_file'],
    )

    xyz2 = None
    driving_coordinates = None
    driving_coord_prims = []
    if inpfileq['gsm_type'] == 'DE_GSM':
        xyz2 = manage_xyz.xyz_to_np(geoms[-1])
        top2 = build_topology(
            xyz2,
            atoms,
            hybrid_indices=hybrid_indices,
            prim_idx_start_stop=prim_indices,
        )
        for bond in top2.edges():
            if bond in top1.edges or (bond[1], bond[0]) in top1.edges():
                continue
            logger.debug(f' Adding bond {bond} to top1')
            if bond[0] > bond[1]:
                top1.add_edge(bond[0], bond[1])
            else:
                top1.add_edge(bond[1], bond[0])
    else:
        driving_coordinates = read_isomers_file(inpfileq['isomers_file'])
        for driving_coordinate in driving_coordinates:
            primitive = get_driving_coord_prim(driving_coordinate)
            if primitive is not None:
                driving_coord_prims.append(primitive)
        for primitive in driving_coord_prims:
            if not isinstance(primitive, Distance):
                continue
            bond = (primitive.atoms[0], primitive.atoms[1])
            if bond in top1.edges or (bond[1], bond[0]) in top1.edges():
                continue
            logger.debug(f' Adding bond {bond} to top1')
            top1.add_edge(bond[0], bond[1])

    return atoms, xyz1, xyz2, top1, hybrid_indices, driving_coordinates, driving_coord_prims


def _build_coordinate_object(inpfileq: dict, atoms, xyz1, xyz2, top1, hybrid_indices, driving_coord_prims):
    connect, addtr, addcart = _coordinate_flags(inpfileq['coordinate_type'])

    output.printcool('Building Primitive Internal Coordinates')
    p1 = PrimitiveInternalCoordinates(
        xyz=xyz1,
        atoms=atoms,
        connect=connect,
        addtr=addtr,
        addcart=addcart,
        topology=top1,
    )
    if hybrid_indices:
        p1.get_hybrid_indices(xyz1)
    p1.newMakePrimitives(xyz1)
    logger.debug(' done making primitives p1')

    if inpfileq['gsm_type'] == 'DE_GSM':
        output.printcool('Building Primitive Internal Coordinates 2')
        p2 = PrimitiveInternalCoordinates(
            xyz=xyz2,
            atoms=atoms,
            addtr=addtr,
            addcart=addcart,
            connect=connect,
            topology=top1,
        )
        if hybrid_indices:
            p2.get_hybrid_indices(xyz2)
        p2.newMakePrimitives(xyz2)
        logger.debug(' done making primitives p2')
        output.printcool('Forming Union of Primitives')
        p1.add_union_primitives(p2)
        logger.debug(f'check {len(p1.Internals)}')
    else:
        for primitive in driving_coord_prims:
            if isinstance(primitive, Distance):
                continue
            if primitive not in p1.Internals:
                logger.info(f'Adding driving coord prim {primitive} to Internals')
                p1.append_prim_to_block(primitive)

    output.printcool('Building Delocalized Internal Coordinates')
    return DelocalizedInternalCoordinates(
        xyz=xyz1,
        atoms=atoms,
        addtr=addtr,
        addcart=addcart,
        connect=connect,
        primitives=p1,
    )


def _build_gsm(inpfileq: dict, reactant: Molecule, product: Molecule | None, optimizer, driving_coordinates):
    options = dict(
        reactant=reactant,
        nnodes=inpfileq['num_nodes'],
        conv_tol=inpfileq['CONV_TOL'],
        conv_gmax=inpfileq['CONV_gmax'],
        conv_ediff=inpfileq['CONV_Ediff'],
        conv_de=inpfileq['CONV_dE'],
        add_node_tol=inpfileq['ADD_NODE_TOL'],
        optimizer=optimizer,
        print_level=inpfileq['gsm_print_level'],
        string_id=inpfileq['ID'],
        xyz_writer=XYZ_WRITERS[inpfileq['xyz_output_format']],
        output_dir='.',
        mp_cores=inpfileq['mp_cores'],
        interp_method=inpfileq['interp_method'],
    )
    if inpfileq['gsm_type'] == 'DE_GSM':
        return DEGSM(product=product, growth_direction=inpfileq['growth_direction'], **options)
    gsm_class = SEGSM if inpfileq['gsm_type'] == 'SE_GSM' else SECross
    return gsm_class(
        driving_coords=driving_coordinates,
        dqmag_max=inpfileq['DQMAG_MAX'],
        bdist_ratio=inpfileq['BDIST_RATIO'],
        **options,
    )


def _run_only_drive_mode(gsm) -> None:
    for _ in range(gsm.nnodes - 1):
        try:
            gsm.add_GSM_nodeR()
        except (IndexError, RuntimeError, ValueError):
            break
    geometries = [node.geometry for node in gsm.nodes if node is not None]
    manage_xyz.write_xyzs('interpolated.xyz', geometries)


def _resolve_rtype(inpfileq: dict) -> int:
    if inpfileq['only_climb']:
        return 1
    if inpfileq['no_climb'] or inpfileq['optimize_meci']:
        return 0
    if inpfileq['optimize_mesx'] or inpfileq['gsm_type'] == 'SE_Cross':
        return 1
    return 2


def _load_pes_side_inputs(inpfileq: dict) -> None:
    if inpfileq['FORCE_FILE']:
        inpfileq['FORCE'] = _read_force_like_file(inpfileq['FORCE_FILE'], integer_prefix=2)
        logger.info(inpfileq['FORCE'])
    if inpfileq['RESTRAINT_FILE']:
        inpfileq['RESTRAINTS'] = _read_force_like_file(inpfileq['RESTRAINT_FILE'], integer_prefix=1)
        logger.info(inpfileq['RESTRAINTS'])


def main(argv: Sequence[str] | None = None):
    inpfileq = parse_arguments(verbose=True, argv=argv)

    if inpfileq['restart_file']:
        geoms = manage_xyz.read_molden_geoms(inpfileq['restart_file'])
    else:
        geoms = manage_xyz.read_xyzs(inpfileq['xyzfile'])

    output.printcool(f'Build the {inpfileq["EST_Package"]} level of theory (LOT) object')
    lot = create_lot(inpfileq, geoms[0])

    _load_pes_side_inputs(inpfileq)
    output.printcool(f'Building the {inpfileq["PES_type"]} objects')
    pes = choose_pes(lot, inpfileq)

    form_hessian = inpfileq['optimizer'] == 'eigenvector_follow'
    frozen_indices = _read_index_list(inpfileq['frozen_coord_idx_file'])

    output.printcool('Building the topology')
    atoms, xyz1, xyz2, top1, hybrid_indices, driving_coordinates, driving_coord_prims = _build_topology_data(
        inpfileq, geoms
    )
    coord_obj1 = _build_coordinate_object(inpfileq, atoms, xyz1, xyz2, top1, hybrid_indices, driving_coord_prims)

    output.printcool(f'Building the reactant object with {inpfileq["coordinate_type"]}')
    reactant = Molecule(
        geom=geoms[0],
        pes=pes,
        coord_obj=coord_obj1,
        form_hessian=form_hessian,
        frozen_atoms=frozen_indices,
    )

    product = None
    if inpfileq['gsm_type'] == 'DE_GSM':
        output.printcool('Building the product object')
        product = Molecule.copy_from(
            reactant,
            xyz=xyz2,
            new_node_id=inpfileq['num_nodes'] - 1,
        )

    output.printcool('Building the Optimizer object')
    optimizer = choose_optimizer(inpfileq)

    output.printcool('Building the GSM object')
    gsm = _build_gsm(inpfileq, reactant, product, optimizer, driving_coordinates)

    if inpfileq['only_drive']:
        _run_only_drive_mode(gsm)
        return

    # Crossing convergence (energy gap) for the end-point optimisations only: the string's node
    # optimizers were copied from this one above and keep the ordinary criteria.
    if inpfileq['gsm_type'] != 'SE_Cross' and inpfileq['PES_type'] in ('Avg_PES', 'Penalty_PES'):
        optimizer.opt_cross = True

    if not inpfileq['reactant_geom_fixed'] and inpfileq['gsm_type'] != 'SE_Cross':
        output.printcool('REACTANT GEOMETRY NOT FIXED!!! OPTIMIZING')
        optimizer.optimize(molecule=reactant, ref_energy=reactant.energy, opt_steps=100)

    if product is not None and not inpfileq['product_geom_fixed']:
        output.printcool('PRODUCT GEOMETRY NOT FIXED!!! OPTIMIZING')
        optimizer.optimize(molecule=product, ref_energy=reactant.energy, opt_steps=100)

    if inpfileq['max_opt_steps'] is None:
        inpfileq['max_opt_steps'] = 3 if inpfileq['gsm_type'] == 'DE_GSM' else 20

    if inpfileq['restart_file'] is not None:
        gsm.setup_from_geometries(
            geoms,
            reparametrize=inpfileq['reparametrize'],
            start_climb_immediately=inpfileq['start_climb_immediately'],
        )

    try:
        gsm.go_gsm(inpfileq['max_gsm_iters'], inpfileq['max_opt_steps'], _resolve_rtype(inpfileq))
    except GrowthError as error:
        logger.error(f' GSM stopped: {error} (--max-gsm-iters); the last string is in string_{gsm.string_id:03d}.xyz')
        raise SystemExit(1) from error
    if inpfileq['gsm_type'] == 'SE_Cross':
        post_processing(gsm, analyze_ICs=inpfileq['analyze_ICs'], have_TS=False)
        manage_xyz.write_xyz(f'meci_{gsm.string_id:03d}.xyz', gsm.nodes[gsm.nR].geometry)
        if not gsm.end_early:
            manage_xyz.write_xyz(f'TSnode_{gsm.string_id:03d}.xyz', gsm.nodes[gsm.TSnode].geometry)
    else:
        post_processing(gsm, analyze_ICs=inpfileq['analyze_ICs'], have_TS=True)
        manage_xyz.write_xyz(f'TSnode_{gsm.string_id:03d}.xyz', gsm.nodes[gsm.TSnode].geometry)


def read_isomers_file(isomers_file):
    lines = _read_nonempty_lines(isomers_file)
    driving_coordinates = []
    start = 1 if lines and lines[0] == 'NEW' else 0

    for line in lines[start:]:
        driving_coordinate = []
        two_ints = False
        three_ints = False
        four_ints = False
        for index, entry in enumerate(line.split()):
            if index == 0:
                driving_coordinate.append(entry)
                if entry in {'ADD', 'BREAK'}:
                    two_ints = True
                elif entry in {'ANGLE', 'ROTATE'}:
                    three_ints = True
                elif entry in {'TORSION', 'OOP'}:
                    four_ints = True
            else:
                if (
                    (two_ints and index > 2)
                    or (two_ints and index > 3)
                    or (three_ints and index > 3)
                    or (four_ints and index > 4)
                ):
                    driving_coordinate.append(float(entry))
                else:
                    driving_coordinate.append(int(entry))
        driving_coordinates.append(driving_coordinate)

    output.printcool(f'driving coordinates {driving_coordinates}')
    return driving_coordinates


def _write_ic_data(gsm, header: str, node_indices) -> None:
    """Write the primitive internal coordinates of the given nodes to IC_data_<ID>.txt, one per line."""
    columns = [gsm.nodes[0].primitive_internal_coordinates]
    columns += [gsm.nodes[n].primitive_internal_values for n in node_indices]
    with open(f'IC_data_{gsm.string_id:03d}.txt', 'w') as handle:
        handle.write(header + '\n')
        for record in zip(*columns):
            handle.write('\t'.join(str(v) for v in record) + '\n')


def post_processing(gsm, analyze_ICs=True, have_TS=True):
    plot(fx=gsm.energies, x=range(len(gsm.energies)), title=gsm.string_id)

    if have_TS:
        minnodeR = np.argmin(gsm.energies[: gsm.TSnode])
        ts_energy = gsm.energies[gsm.TSnode] - gsm.energies[minnodeR]
        logger.info(f' TS energy: {ts_energy:5.4f}')
        logger.info(f' absolute energy TS node {gsm.nodes[gsm.TSnode].energy:5.4f}')
        minnodeP = gsm.TSnode + np.argmin(gsm.energies[gsm.TSnode :])
        logger.info(f' min reactant node: {minnodeR} min product node {minnodeP} TS node is {gsm.TSnode}')
        if analyze_ICs:
            header = f'Internals \t minnodeR: {minnodeR} \t TSnode: {gsm.TSnode} \t minnodeP: {minnodeP}'
            _write_ic_data(gsm, header, (minnodeR, gsm.TSnode, minnodeP))
    else:
        minnodeR = 0
        minnodeP = gsm.nR
        logger.info(f' absolute energy end node {gsm.nodes[gsm.nR].energy:5.4f}')
        logger.info(f' difference energy end node {gsm.nodes[gsm.nR].difference_energy:5.4f}')
        if analyze_ICs:
            _write_ic_data(gsm, f'Internals \t Beginning: {minnodeR} \t End: {minnodeP}', (minnodeR, minnodeP))

    delta_e = gsm.energies[minnodeP] - gsm.energies[minnodeR]
    logger.info(f' Delta E is {delta_e:5.4f}')
