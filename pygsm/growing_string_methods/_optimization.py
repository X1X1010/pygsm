"""Optimization-phase helpers for :class:`pygsm.growing_string_methods.MainGSM`."""

from __future__ import annotations

import numpy as np

from pygsm.utilities.nifty import printcool


def optimize_string(gsm, max_iter=30, nconstraints=1, opt_steps=1, rtype=2):
    """Optimize a grown string until convergence."""
    printcool('In opt_iters Cody')

    gsm.nclimb = 0
    gsm.nhessreset = 10
    gsm.hessrcount = 0
    gsm.newclimbscale = 2.0
    gsm.set_finder(rtype)

    gsm.isConverged = False
    opt_iteration = 0

    while opt_iteration < max_iter:
        printcool('Starting opt iter %i' % opt_iteration)
        if gsm.climb and not gsm.find:
            print(' CLIMBING')
        elif gsm.find:
            print(' TS SEARCHING')

        gsm.pTSnode = gsm.TSnode
        gsm.emaxp = gsm.emax
        ts_node_changed = False

        print(' V_profile (beginning of iteration): ', end=' ')
        gsm.print_energies()

        gsm.get_tangents_opting()
        gsm.refresh_coordinates()

        gsm.set_node_convergence()
        gsm.optimize_iteration(opt_steps)

        print(' V_profile: ', end=' ')
        gsm.print_energies()

        if gsm.TSnode == gsm.nnodes - 2 and (gsm.climb or gsm.find):
            printcool("WARNING\n: TS node shouldn't be second to last node for tangent reasons")
            gsm.add_node_after_TS()
            added = True
        elif gsm.TSnode == 1 and (gsm.climb or gsm.find):
            printcool("WARNING\n: TS node shouldn't be first  node for tangent reasons")
            gsm.add_node_before_TS()
            added = True
        else:
            added = False

        fp = gsm.find_peaks('opting')

        ts_cgradq = 0.0
        if not gsm.find:
            ts_cgradq = np.linalg.norm(
                np.dot(gsm.nodes[gsm.TSnode].gradient.T, gsm.nodes[gsm.TSnode].constraints[:, 0])
                * gsm.nodes[gsm.TSnode].constraints[:, 0]
            )
            print(' ts_cgradq %5.4f' % ts_cgradq)

        ts_gradrms = gsm.nodes[gsm.TSnode].gradrms
        gsm.dE_iter = abs(gsm.emax - gsm.emaxp)
        print(' dE_iter ={:2.2f}'.format(gsm.dE_iter))

        totalgrad, gradrms, sum_gradrms = gsm.calc_optimization_metrics(gsm.nodes)

        energies = np.array(gsm.energies)
        if (np.all(energies[1:] + 0.5 >= energies[:-1]) or np.all(energies[1:] - 0.5 <= energies[:-1])) and (
            gsm.climber or gsm.finder
        ):
            printcool(' There is no TS, turning off TS search')
            rtype = 0
            gsm.climber = gsm.finder = gsm.find = gsm.climb = False
            gsm.CONV_TOL = gsm.options['CONV_TOL'] * 5

        gsm.isConverged = gsm.is_converged(totalgrad, fp, rtype, ts_cgradq)
        stage_changed = gsm.set_stage(totalgrad, sum_gradrms, ts_cgradq, ts_gradrms, fp)

        if not stage_changed:
            if gsm.climb:
                gsm.nclimb -= 1
            gsm.nhessreset -= 1
            if gsm.nopt_intermediate > 0:
                gsm.nopt_intermediate -= 1

            if gsm.pTSnode != gsm.TSnode and gsm.climb:
                print('TS node changed after opting')
                gsm.climb = False
                ts_node_changed = True
                gsm.pTSnode = gsm.TSnode

            if gsm.find and (not gsm.optimizer[gsm.TSnode].maxol_good or added):
                gsm.ictan, gsm.dqmaga = gsm.get_three_way_tangents(gsm.nodes, gsm.energies)
                gsm.modify_TS_Hess()
            elif gsm.find and (
                gsm.optimizer[gsm.TSnode].nneg > 3
                or gsm.optimizer[gsm.TSnode].nneg == 0
                or gsm.hess_counter > 10
                or np.abs(gsm.TS_E_0 - gsm.emax) > 10.0
            ) and not gsm.optimizer[gsm.TSnode].converged:
                gsm.nodes[gsm.TSnode].form_Primitive_Hessian()
                if gsm.hessrcount < 1 and gsm.pTSnode == gsm.TSnode:
                    print(' resetting TS node coords Ut (and Hessian)')
                    gsm.ictan, gsm.dqmaga = gsm.get_three_way_tangents(gsm.nodes, gsm.energies)
                    gsm.modify_TS_Hess()
                    gsm.nhessreset = 10
                    gsm.hessrcount = 1
                else:
                    print(' Hessian consistently bad, going back to climb (for 3 iterations)')
                    gsm.find = False
                    gsm.nclimb = 2
            elif gsm.find and gsm.optimizer[gsm.TSnode].nneg <= 3:
                gsm.hessrcount -= 1
                gsm.hess_counter += 1
        print(f'{stage_changed=}: {gsm.climb=} {gsm.find=}')

        filename = 'scratch/opt_iters_{:03}_{:03}.xyz'.format(gsm.ID, opt_iteration)
        gsm.xyz_writer(filename, gsm.geometries, gsm.energies, gsm.gradrmss, gsm.dEs)

        print(' End early counter {}'.format(gsm.endearly_counter))
        print(
            'opt_iter: {:2} totalgrad: {:4.3} gradrms: {:5.4} max E({}) {:5.4}\n'.format(
                opt_iteration,
                float(totalgrad),
                float(gradrms),
                gsm.TSnode,
                float(gsm.emax),
            )
        )
        opt_iteration += 1

        if gsm.isConverged and not added and not ts_node_changed and not stage_changed:
            print('Converged')
            return

        if opt_iteration < max_iter and not gsm.isConverged and not stage_changed:
            gsm.reparameterize(nconstraints=nconstraints)
            gsm.get_tangents_opting()
            gsm.refresh_coordinates()
            if gsm.pTSnode != gsm.TSnode and gsm.climb:
                print('TS node changed after reparameterizing')
                gsm.slow_down_climb()
        elif opt_iteration == max_iter and not gsm.isConverged:
            gsm.ran_out = True
            print(' Ran out of iterations')
            return
