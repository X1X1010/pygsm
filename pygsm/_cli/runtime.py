"""Workflow orchestration for the ``gsm`` command."""

from __future__ import annotations

import os
from pathlib import Path
from collections.abc import Sequence

import numpy as np

from pygsm.coordinate_systems import DelocalizedInternalCoordinates, Distance, PrimitiveInternalCoordinates, Topology
from pygsm.growing_string_methods import DE_GSM, SE_Cross, SE_GSM
from pygsm.molecule import Molecule
from pygsm.potential_energy_surfaces import Penalty_PES
from pygsm.utilities import elements, manage_xyz, nifty
from pygsm.utilities.cli_utils import get_driving_coord_prim, plot
from pygsm.utilities.manage_xyz import XYZ_WRITERS

from .arguments import parse_arguments
from .factory import choose_optimizer, choose_pes, create_lot


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
        nifty.printcool(' Using Hybrid COORDINATES :)')
        assert inpfileq['coordinate_type'] == 'TRIC', "hybrid indices won't work (currently) with other coordinate systems"

    prim_indices = _read_primitive_regions(inpfileq['prim_idx_file'])
    if prim_indices is not None:
        nifty.printcool(' Defining primitive internal region :)')
        assert inpfileq['coordinate_type'] == 'TRIC', "won't work (currently) with other coordinate systems"
        print(prim_indices)

    top1 = Topology.build_topology(
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
        top2 = Topology.build_topology(
            xyz2,
            atoms,
            hybrid_indices=hybrid_indices,
            prim_idx_start_stop=prim_indices,
        )
        for bond in top2.edges():
            if bond in top1.edges or (bond[1], bond[0]) in top1.edges():
                continue
            print(f' Adding bond {bond} to top1')
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
            print(f' Adding bond {bond} to top1')
            top1.add_edge(bond[0], bond[1])

    return atoms, xyz1, xyz2, top1, hybrid_indices, driving_coordinates, driving_coord_prims


def _build_coordinate_object(inpfileq: dict, atoms, xyz1, xyz2, top1, hybrid_indices, driving_coord_prims):
    connect, addtr, addcart = _coordinate_flags(inpfileq['coordinate_type'])

    nifty.printcool('Building Primitive Internal Coordinates')
    p1 = PrimitiveInternalCoordinates.from_options(
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
    print(' done making primitives p1')

    if inpfileq['gsm_type'] == 'DE_GSM':
        nifty.printcool('Building Primitive Internal Coordinates 2')
        p2 = PrimitiveInternalCoordinates.from_options(
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
        print(' done making primitives p2')
        nifty.printcool('Forming Union of Primitives')
        p1.add_union_primitives(p2)
        print(f'check {len(p1.Internals)}')
    else:
        for primitive in driving_coord_prims:
            if isinstance(primitive, Distance):
                continue
            if primitive not in p1.Internals:
                print(f'Adding driving coord prim {primitive} to Internals')
                p1.append_prim_to_block(primitive)

    nifty.printcool('Building Delocalized Internal Coordinates')
    return DelocalizedInternalCoordinates.from_options(
        xyz=xyz1,
        atoms=atoms,
        addtr=addtr,
        addcart=addcart,
        connect=connect,
        primitives=p1,
    )


def _build_gsm(inpfileq: dict, reactant: Molecule, product: Molecule | None, optimizer, driving_coordinates):
    xyz_writer = XYZ_WRITERS[inpfileq['xyz_output_format']]
    if inpfileq['gsm_type'] == 'DE_GSM':
        return DE_GSM.from_options(
            reactant=reactant,
            product=product,
            nnodes=inpfileq['num_nodes'],
            CONV_TOL=inpfileq['CONV_TOL'],
            CONV_gmax=inpfileq['conv_gmax'],
            CONV_Ediff=inpfileq['conv_Ediff'],
            CONV_dE=inpfileq['conv_dE'],
            ADD_NODE_TOL=inpfileq['ADD_NODE_TOL'],
            growth_direction=inpfileq['growth_direction'],
            optimizer=optimizer,
            ID=inpfileq['ID'],
            print_level=inpfileq['gsm_print_level'],
            xyz_writer=xyz_writer,
            mp_cores=inpfileq['mp_cores'],
            interp_method=inpfileq['interp_method'],
        )

    gsm_class = SE_GSM if inpfileq['gsm_type'] == 'SE_GSM' else SE_Cross
    return gsm_class.from_options(
        reactant=reactant,
        nnodes=inpfileq['num_nodes'],
        DQMAG_MAX=inpfileq['DQMAG_MAX'],
        BDIST_RATIO=inpfileq['BDIST_RATIO'],
        CONV_TOL=inpfileq['CONV_TOL'],
        ADD_NODE_TOL=inpfileq['ADD_NODE_TOL'],
        optimizer=optimizer,
        print_level=inpfileq['gsm_print_level'],
        driving_coords=driving_coordinates,
        ID=inpfileq['ID'],
        xyz_writer=xyz_writer,
        mp_cores=inpfileq['mp_cores'],
        interp_method=inpfileq['interp_method'],
    )


def _run_only_drive_mode(gsm) -> None:
    for _ in range(gsm.nnodes - 1):
        try:
            gsm.add_GSM_nodeR()
        except (IndexError, RuntimeError, ValueError):
            break
    geometries = [node.geometry for node in gsm.nodes if node is not None]
    manage_xyz.write_xyzs('interpolated.xyz', geometries)


def _optimize_endpoint_if_needed(optimizer, molecule: Molecule, ref_energy: float, opt_steps: int, path: str) -> None:
    optimizer.optimize(
        molecule=molecule,
        refE=ref_energy,
        opt_steps=opt_steps,
        path=path,
    )


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
        print(inpfileq['FORCE'])
    if inpfileq['RESTRAINT_FILE']:
        inpfileq['RESTRAINTS'] = _read_force_like_file(inpfileq['RESTRAINT_FILE'], integer_prefix=1)
        print(inpfileq['RESTRAINTS'])


def main(argv: Sequence[str] | None = None):
    inpfileq = parse_arguments(verbose=True, argv=argv)

    if inpfileq['restart_file']:
        geoms = manage_xyz.read_molden_geoms(inpfileq['restart_file'])
    else:
        geoms = manage_xyz.read_xyzs(inpfileq['xyzfile'])

    nifty.printcool(f"Build the {inpfileq['EST_Package']} level of theory (LOT) object")
    lot = create_lot(inpfileq, geoms[0])

    if inpfileq['gsm_type'] == 'SE_Cross' and inpfileq['PES_type'] != 'Penalty_PES':
        print(' setting PES type to Penalty')
        inpfileq['PES_type'] = 'Penalty_PES'
    if inpfileq['optimize_mesx'] or inpfileq['optimize_meci'] or inpfileq['gsm_type'] == 'SE_Cross':
        assert inpfileq['PES_type'] == 'Penalty_PES', 'Need penalty pes for optimizing MESX/MECI'

    _load_pes_side_inputs(inpfileq)
    nifty.printcool(f"Building the {inpfileq['PES_type']} objects")
    pes = choose_pes(lot, inpfileq)

    form_hessian = inpfileq['optimizer'] == 'eigenvector_follow'
    frozen_indices = _read_index_list(inpfileq['frozen_coord_idx_file'])

    nifty.printcool('Building the topology')
    atoms, xyz1, xyz2, top1, hybrid_indices, driving_coordinates, driving_coord_prims = _build_topology_data(inpfileq, geoms)
    coord_obj1 = _build_coordinate_object(inpfileq, atoms, xyz1, xyz2, top1, hybrid_indices, driving_coord_prims)

    nifty.printcool(f"Building the reactant object with {inpfileq['coordinate_type']}")
    reactant = Molecule.from_options(
        geom=geoms[0],
        PES=pes,
        coord_obj=coord_obj1,
        Form_Hessian=form_hessian,
        frozen_atoms=frozen_indices,
    )

    product = None
    if inpfileq['gsm_type'] == 'DE_GSM':
        nifty.printcool('Building the product object')
        product = Molecule.copy_from_options(
            reactant,
            xyz=xyz2,
            new_node_id=inpfileq['num_nodes'] - 1,
            copy_wavefunction=False,
        )

    nifty.printcool('Building the Optimizer object')
    optimizer = choose_optimizer(inpfileq)

    nifty.printcool('Building the GSM object')
    gsm = _build_gsm(inpfileq, reactant, product, optimizer, driving_coordinates)

    if inpfileq['only_drive']:
        _run_only_drive_mode(gsm)
        return

    if inpfileq['gsm_type'] != 'SE_Cross' and isinstance(pes, (Penalty_PES,)):
        optimizer.opt_cross = True
    elif inpfileq['gsm_type'] != 'SE_Cross' and inpfileq['PES_type'] == 'Avg_PES':
        optimizer.opt_cross = True

    if not inpfileq['reactant_geom_fixed'] and inpfileq['gsm_type'] != 'SE_Cross':
        path = os.path.join(os.getcwd(), f"scratch/{inpfileq['ID']:03}/0")
        nifty.printcool('REACTANT GEOMETRY NOT FIXED!!! OPTIMIZING')
        _optimize_endpoint_if_needed(optimizer, reactant, reactant.energy, 100, path)

    if product is not None and not inpfileq['product_geom_fixed']:
        path = os.path.join(os.getcwd(), f"scratch/{inpfileq['ID']:03}/{inpfileq['num_nodes'] - 1}")
        nifty.printcool('PRODUCT GEOMETRY NOT FIXED!!! OPTIMIZING')
        _optimize_endpoint_if_needed(optimizer, product, reactant.energy, 100, path)

    if inpfileq['max_opt_steps'] is None:
        inpfileq['max_opt_steps'] = 3 if inpfileq['gsm_type'] == 'DE_GSM' else 20

    if inpfileq['restart_file'] is not None:
        gsm.setup_from_geometries(
            geoms,
            reparametrize=inpfileq['reparametrize'],
            start_climb_immediately=inpfileq['start_climb_immediately'],
        )

    gsm.go_gsm(inpfileq['max_gsm_iters'], inpfileq['max_opt_steps'], _resolve_rtype(inpfileq))
    if inpfileq['gsm_type'] == 'SE_Cross':
        post_processing(gsm, analyze_ICs=inpfileq['dont_analyze_ICs'], have_TS=False)
        manage_xyz.write_xyz(f'meci_{gsm.ID}.xyz', gsm.nodes[gsm.nR].geometry)
        if not gsm.end_early:
            manage_xyz.write_xyz(f'TSnode_{gsm.ID}.xyz', gsm.nodes[gsm.TSnode].geometry)
    else:
        post_processing(gsm, analyze_ICs=inpfileq['dont_analyze_ICs'], have_TS=True)
        manage_xyz.write_xyz(f'TSnode_{gsm.ID}.xyz', gsm.nodes[gsm.TSnode].geometry)

    cleanup_scratch(gsm.ID)
    return


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
                if two_ints and index > 2:
                    driving_coordinate.append(float(entry))
                elif two_ints and index > 3:
                    driving_coordinate.append(float(entry))
                elif three_ints and index > 3:
                    driving_coordinate.append(float(entry))
                elif four_ints and index > 4:
                    driving_coordinate.append(float(entry))
                else:
                    driving_coordinate.append(int(entry))
        driving_coordinates.append(driving_coordinate)

    nifty.printcool(f'driving coordinates {driving_coordinates}')
    return driving_coordinates


def cleanup_scratch(ID):
    scratch_root = Path('scratch')
    for pattern in (f'growth_iters_{ID:03d}_*.xyz', f'opt_iters_{ID:03d}_*.xyz'):
        for path in scratch_root.glob(pattern):
            path.unlink(missing_ok=True)


def post_processing(gsm, analyze_ICs=False, have_TS=True):
    plot(fx=gsm.energies, x=range(len(gsm.energies)), title=gsm.ID)

    internals = [gsm.nodes[0].primitive_internal_coordinates]
    if have_TS:
        minnodeR = np.argmin(gsm.energies[:gsm.TSnode])
        ts_energy = gsm.energies[gsm.TSnode] - gsm.energies[minnodeR]
        print(f' TS energy: {ts_energy:5.4f}')
        print(f' absolute energy TS node {gsm.nodes[gsm.TSnode].energy:5.4f}')
        minnodeP = gsm.TSnode + np.argmin(gsm.energies[gsm.TSnode:])
        print(f' min reactant node: {minnodeR} min product node {minnodeP} TS node is {gsm.TSnode}')

        internals.append(gsm.nodes[minnodeR].primitive_internal_values)
        internals.append(gsm.nodes[gsm.TSnode].primitive_internal_values)
        internals.append(gsm.nodes[minnodeP].primitive_internal_values)
        with open(f'IC_data_{gsm.ID:04d}.txt', 'w') as handle:
            handle.write(f'Internals \t minnodeR: {minnodeR} \t TSnode: {gsm.TSnode} \t minnodeP: {minnodeP}\n')
            for record in zip(*internals):
                handle.write('{0}\t{1}\t{2}\t{3}\n'.format(*record))
    else:
        minnodeR = 0
        minnodeP = gsm.nR
        print(f' absolute energy end node {gsm.nodes[gsm.nR].energy:5.4f}')
        print(f' difference energy end node {gsm.nodes[gsm.nR].difference_energy:5.4f}')
        internals.append(gsm.nodes[minnodeR].primitive_internal_values)
        internals.append(gsm.nodes[minnodeP].primitive_internal_values)
        with open(f'IC_data_{gsm.ID}.txt', 'w') as handle:
            handle.write(f'Internals \t Beginning: {minnodeR} \t End: {minnodeP}')
            for record in zip(*internals):
                handle.write('{0}\t{1}\t{2}\n'.format(*record))

    delta_e = gsm.energies[minnodeP] - gsm.energies[minnodeR]
    print(f' Delta E is {delta_e:5.4f}')
