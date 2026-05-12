"""Factory helpers for CLI-selected workflow components."""

from __future__ import annotations

import importlib
import json

from pygsm.level_of_theories.ase import ASELoT
from pygsm.level_of_theories.xtb_lot import xTB_lot
from pygsm.optimizers import beales_cg, conjugate_gradient, eigenvector_follow, lbfgs
from pygsm.potential_energy_surfaces import Avg_PES, PES, Penalty_PES


def create_lot(inpfileq: dict, geom):
    inpfileq['states'] = [
        (int(multiplicity), int(state))
        for multiplicity, state in zip(inpfileq['multiplicity'], inpfileq['adiabatic_index'])
    ]
    do_coupling = inpfileq['PES_type'] == 'Avg_PES'
    coupling_states = inpfileq['states'] if do_coupling else []

    lot_options = dict(
        ID=inpfileq['ID'],
        lot_inp_file=inpfileq['lot_inp_file'],
        states=inpfileq['states'],
        gradient_states=inpfileq['states'],
        coupling_states=coupling_states,
        geom=geom,
        nproc=inpfileq['nproc'],
        charge=inpfileq['charge'],
        do_coupling=do_coupling,
    )

    lot_name = inpfileq['EST_Package']
    if lot_name.lower() == 'ase':
        ase_kwargs = json.loads(inpfileq['ase_kwargs'] or '{}')
        return ASELoT.from_calculator_string(
            calculator_import=inpfileq['ase_class'],
            calculator_kwargs=dict(ase_kwargs),
            **lot_options,
        )

    if lot_name == 'xTB_lot':
        return xTB_lot.from_options(
            xTB_Hamiltonian=inpfileq['xTB_Hamiltonian'],
            xTB_accuracy=inpfileq['xTB_accuracy'],
            xTB_electronic_temperature=inpfileq['xTB_electronic_temperature'],
            solvent=inpfileq['solvent'],
            **lot_options,
        )

    est_package = importlib.import_module(f"pygsm.level_of_theories.{lot_name.lower()}")
    lot_class = getattr(est_package, lot_name)
    return lot_class.from_options(**lot_options)


def choose_pes(lot, inpfileq: dict):
    if inpfileq['PES_type'] == 'PES':
        return PES.from_options(
            lot=lot,
            ad_idx=inpfileq['adiabatic_index'][0],
            multiplicity=inpfileq['multiplicity'][0],
            FORCE=inpfileq['FORCE'],
            RESTRAINTS=inpfileq['RESTRAINTS'],
        )

    pes1 = PES.from_options(
        lot=lot,
        multiplicity=inpfileq['states'][0][0],
        ad_idx=inpfileq['states'][0][1],
        FORCE=inpfileq['FORCE'],
        RESTRAINTS=inpfileq['RESTRAINTS'],
    )
    pes2 = PES.from_options(
        lot=lot,
        multiplicity=inpfileq['states'][1][0],
        ad_idx=inpfileq['states'][1][1],
        FORCE=inpfileq['FORCE'],
        RESTRAINTS=inpfileq['RESTRAINTS'],
    )

    if inpfileq['PES_type'] == 'Avg_PES':
        return Avg_PES(PES1=pes1, PES2=pes2, lot=lot)
    if inpfileq['PES_type'] == 'Penalty_PES':
        return Penalty_PES(PES1=pes1, PES2=pes2, lot=lot, sigma=inpfileq['sigma'])
    raise NotImplementedError(f"Unsupported PES type `{inpfileq['PES_type']}`")


def choose_optimizer(inpfileq: dict):
    update_hess_in_bg = not (inpfileq['only_climb'] or inpfileq['optimizer'] == 'lbfgs')
    optimizer_map = {
        'conjugate_gradient': conjugate_gradient,
        'eigenvector_follow': eigenvector_follow,
        'lbfgs': lbfgs,
        'beales_cg': beales_cg,
    }
    try:
        opt_class = optimizer_map[inpfileq['optimizer']]
    except KeyError as exc:
        raise NotImplementedError(f"Optimizer `{inpfileq['optimizer']}` not implemented") from exc

    return opt_class.from_options(
        print_level=inpfileq['opt_print_level'],
        Linesearch=inpfileq['linesearch'],
        update_hess_in_bg=update_hess_in_bg,
        conv_Ediff=inpfileq['conv_Ediff'],
        conv_dE=inpfileq['conv_dE'],
        conv_gmax=inpfileq['conv_gmax'],
        DMAX=inpfileq['DMAX'],
    )
