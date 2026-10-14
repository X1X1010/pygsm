"""Factory helpers for CLI-selected workflow components."""

from __future__ import annotations

import json

from pygsm.lot.ase import ASELoT
from pygsm.lot.xtb import XTBLoT
from pygsm.optimizers import LBFGS, BealesCG, ConjugateGradient, EigenvectorFollow
from pygsm.pes import PES, AvgPES, PenaltyPES


def create_lot(inpfileq: dict, geom):
    inpfileq['states'] = [
        (int(multiplicity), int(state))
        for multiplicity, state in zip(inpfileq['multiplicity'], inpfileq['adiabatic_index'])
    ]
    do_coupling = inpfileq['PES_type'] == 'Avg_PES'
    lot_options = dict(
        string_id=inpfileq['ID'],
        states=inpfileq['states'],
        gradient_states=inpfileq['states'],
        coupling_states=inpfileq['states'] if do_coupling else [],
        geom=geom,
        nproc=inpfileq['nproc'],
        charge=inpfileq['charge'],
        do_coupling=do_coupling,
    )

    if inpfileq['EST_Package'] == 'ase':
        ase_kwargs = json.loads(inpfileq['ase_kwargs'] or '{}')
        return ASELoT.from_calculator_string(
            calculator_import=inpfileq['ase_class'],
            calculator_kwargs=ase_kwargs,
            **lot_options,
        )
    if inpfileq['EST_Package'] == 'xTB':
        return XTBLoT(
            hamiltonian=inpfileq['xTB_Hamiltonian'],
            accuracy=inpfileq['xTB_accuracy'],
            electronic_temperature=inpfileq['xTB_electronic_temperature'],
            solvent=inpfileq['solvent'],
            **lot_options,
        )
    raise NotImplementedError(f'Unsupported level of theory package `{inpfileq["EST_Package"]}`')


def choose_pes(lot, inpfileq: dict):
    if inpfileq['PES_type'] == 'PES':
        return PES(
            lot=lot,
            ad_idx=inpfileq['adiabatic_index'][0],
            multiplicity=inpfileq['multiplicity'][0],
            forces=inpfileq['FORCE'],
            restraints=inpfileq['RESTRAINTS'],
        )

    pes1 = PES(
        lot=lot,
        multiplicity=inpfileq['states'][0][0],
        ad_idx=inpfileq['states'][0][1],
        forces=inpfileq['FORCE'],
        restraints=inpfileq['RESTRAINTS'],
    )
    pes2 = PES(
        lot=lot,
        multiplicity=inpfileq['states'][1][0],
        ad_idx=inpfileq['states'][1][1],
        forces=inpfileq['FORCE'],
        restraints=inpfileq['RESTRAINTS'],
    )

    if inpfileq['PES_type'] == 'Avg_PES':
        return AvgPES(pes1=pes1, pes2=pes2, lot=lot)
    if inpfileq['PES_type'] == 'Penalty_PES':
        return PenaltyPES(pes1=pes1, pes2=pes2, lot=lot, sigma=inpfileq['sigma'])
    raise NotImplementedError(f'Unsupported PES type `{inpfileq["PES_type"]}`')


def choose_optimizer(inpfileq: dict):
    update_hess_in_bg = not (inpfileq['only_climb'] or inpfileq['optimizer'] == 'lbfgs')
    optimizer_map = {
        'conjugate_gradient': ConjugateGradient,
        'eigenvector_follow': EigenvectorFollow,
        'lbfgs': LBFGS,
        'beales_cg': BealesCG,
    }
    try:
        opt_class = optimizer_map[inpfileq['optimizer']]
    except KeyError as exc:
        raise NotImplementedError(f'Optimizer `{inpfileq["optimizer"]}` not implemented') from exc

    return opt_class(
        conv_grms=inpfileq['CONV_TOL'],
        print_level=inpfileq['opt_print_level'],
        linesearch=inpfileq['linesearch'],
        update_hess_in_bg=update_hess_in_bg,
        conv_ediff=inpfileq['conv_Ediff'],
        conv_de=inpfileq['conv_dE'],
        conv_gmax=inpfileq['conv_gmax'],
        dmax=inpfileq['DMAX'],
    )
