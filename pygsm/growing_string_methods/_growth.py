"""Growth-phase helpers for :class:`pygsm.growing_string_methods.MainGSM`."""

from __future__ import annotations

from pygsm.molecule import Molecule
from pygsm.utilities.output import printcool


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
        printcool(f'Starting growth iteration {iteration}')
        gsm.optimize_iteration(max_opt_steps)
        totalgrad, gradrms, _ = gsm.calc_optimization_metrics(gsm.nodes)
        gsm.xyz_writer(
            f'scratch/growth_iters_{gsm.ID:03}_{iteration:03}.xyz',
            gsm.geometries,
            gsm.energies,
            gsm.gradrmss,
            gsm.dEs,
        )
        print(
            f' gopt_iter: {iteration:2} totalgrad: {float(totalgrad):4.3} gradrms: {float(gradrms):5.4} max E: {float(gsm.emax):5.4}\n'
        )

        try:
            gsm.grow_nodes()
        except (IndexError, RuntimeError, ValueError) as error:
            print(f"can't add anymore nodes, bdist too small: {error}")

            gsm.on_growth_exhausted(error)
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
