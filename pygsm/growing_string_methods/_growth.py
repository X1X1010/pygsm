"""Growth-phase helpers for :class:`pygsm.growing_string_methods.MainGSM`."""

from __future__ import annotations

import os

from pygsm.molecule import Molecule
from pygsm.utilities.nifty import printcool


def grow_string(gsm, max_iters=30, max_opt_steps=3, nconstraints=1):
    """Run the GSM growth phase."""
    printcool('In growth_iters')

    gsm.make_difference_node_list()
    gsm.ictan, gsm.dqmaga = gsm.get_tangents_growing()
    gsm.refresh_coordinates()
    gsm.set_active(gsm.nR - 1, gsm.nnodes - gsm.nP)

    is_grown = False
    iteration = 0
    while not is_grown:
        if iteration > max_iters:
            print(' Ran out of iterations')
            return
        printcool('Starting growth iteration %i' % iteration)
        gsm.optimize_iteration(max_opt_steps)
        totalgrad, gradrms, _ = gsm.calc_optimization_metrics(gsm.nodes)
        gsm.xyz_writer(
            'scratch/growth_iters_{:03}_{:03}.xyz'.format(gsm.ID, iteration),
            gsm.geometries,
            gsm.energies,
            gsm.gradrmss,
            gsm.dEs,
        )
        print(
            ' gopt_iter: {:2} totalgrad: {:4.3} gradrms: {:5.4} max E: {:5.4}\n'.format(
                iteration,
                float(totalgrad),
                float(gradrms),
                float(gsm.emax),
            )
        )

        try:
            gsm.grow_nodes()
        except (IndexError, RuntimeError, ValueError) as error:
            print(f"can't add anymore nodes, bdist too small: {error}")

            if gsm.__class__.__name__ == 'SE_GSM':
                opt_type = 'MECI' if gsm.nodes[gsm.nR - 1].PES.lot.do_coupling else 'UNCONSTRAINED'
                print(' optimizing last node')
                gsm.optimizer[gsm.nR - 1].conv_grms = gsm.CONV_TOL
                print(gsm.optimizer[gsm.nR - 1].conv_grms)
                path = os.path.join(os.getcwd(), 'scratch/{:03d}/{}'.format(gsm.ID, gsm.nR - 1))
                gsm.optimizer[gsm.nR - 1].optimize(
                    molecule=gsm.nodes[gsm.nR - 1],
                    refE=gsm.nodes[0].V0,
                    opt_steps=50,
                    opt_type=opt_type,
                    path=path,
                )
            elif gsm.__class__.__name__ == 'SE_Cross':
                print(' Will do extra optimization of this node in SE-Cross')
            else:
                raise RuntimeError
            break

        gsm.set_active(gsm.nR - 1, gsm.nnodes - gsm.nP)
        gsm.ic_reparam_g()
        gsm.ictan, gsm.dqmaga = gsm.get_tangents_growing()
        gsm.refresh_coordinates()

        iteration += 1
        is_grown = gsm.check_if_grown()

    print(' creating newic molecule--used for ic_reparam')
    gsm.newic = Molecule.copy_from_options(gsm.nodes[0])

    if gsm.growth_direction == 1:
        print('Setting LOT of last node')
        gsm.nodes[-1] = Molecule.copy_from_options(
            MoleculeA=gsm.nodes[-2],
            xyz=gsm.nodes[-1].xyz,
            new_node_id=gsm.nnodes - 1,
        )
