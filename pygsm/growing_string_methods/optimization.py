"""Optimization phase of the growing string method (climbing image / exact TS search)."""

import numpy as np

from pygsm.molecule.molecule import Molecule
from pygsm.optimizers import EigenvectorFollow
from pygsm.potential_energy_surfaces import AvgPES
from pygsm.utilities import block_matrix, units
from pygsm.utilities.output import printcool


class OptimizationMixin:
    """Optimizes a grown string, optionally climbing to and refining the TS (mixed into :class:`MainGSM`)."""

    def optimize_string(self, max_iter=30, nconstraints=1, opt_steps=1, rtype=2):
        """Optimize a grown string until convergence."""
        printcool('In opt_iters Cody')

        self.nclimb = 0
        self.nhessreset = 10
        self.hessrcount = 0
        self.newclimbscale = 2.0
        self.set_finder(rtype)

        self.isConverged = False
        opt_iteration = 0

        while opt_iteration < max_iter:
            printcool(f'Starting opt iter {opt_iteration}')
            if self.climb and not self.find:
                print(' CLIMBING')
            elif self.find:
                print(' TS SEARCHING')

            self.pTSnode = self.TSnode
            self.emaxp = self.emax
            ts_node_changed = False

            print(' V_profile (beginning of iteration): ', end=' ')
            self.print_energies()

            self.get_tangents_opting()
            self.refresh_coordinates()

            self.set_node_convergence()
            self.optimize_iteration(opt_steps)

            print(' V_profile: ', end=' ')
            self.print_energies()

            if self.TSnode == self.nnodes - 2 and (self.climb or self.find):
                printcool("WARNING\n: TS node shouldn't be second to last node for tangent reasons")
                self.add_node_after_TS()
                added = True
            elif self.TSnode == 1 and (self.climb or self.find):
                printcool("WARNING\n: TS node shouldn't be first  node for tangent reasons")
                self.add_node_before_TS()
                added = True
            else:
                added = False

            fp = self.find_peaks('opting')

            ts_cgradq = 0.0
            if not self.find:
                ts_cgradq = np.linalg.norm(
                    np.dot(self.nodes[self.TSnode].gradient.T, self.nodes[self.TSnode].constraints[:, 0])
                    * self.nodes[self.TSnode].constraints[:, 0]
                )
                print(f' ts_cgradq {ts_cgradq:5.4f}')

            ts_gradrms = self.nodes[self.TSnode].gradrms
            self.dE_iter = abs(self.emax - self.emaxp)
            print(f' dE_iter ={self.dE_iter:2.2f}')

            totalgrad, gradrms, sum_gradrms = self.calc_optimization_metrics(self.nodes)

            energies = np.array(self.energies)
            if (np.all(energies[1:] + 0.5 >= energies[:-1]) or np.all(energies[1:] - 0.5 <= energies[:-1])) and (
                self.climber or self.finder
            ):
                printcool(' There is no TS, turning off TS search')
                rtype = 0
                self.climber = self.finder = self.find = self.climb = False
                self.CONV_TOL = self.options.CONV_TOL * 5

            self.isConverged = self.is_converged(totalgrad, fp, rtype, ts_cgradq)
            stage_changed = self.set_stage(totalgrad, sum_gradrms, ts_cgradq, ts_gradrms, fp)

            if not stage_changed:
                if self.climb:
                    self.nclimb -= 1
                self.nhessreset -= 1
                if self.nopt_intermediate > 0:
                    self.nopt_intermediate -= 1

                if self.pTSnode != self.TSnode and self.climb:
                    print('TS node changed after opting')
                    self.climb = False
                    ts_node_changed = True
                    self.pTSnode = self.TSnode

                if self.find and (not self.optimizer[self.TSnode].maxol_good or added):
                    self.ictan, self.dqmaga = self.get_three_way_tangents(self.nodes, self.energies)
                    self.modify_TS_Hess()
                elif (
                    self.find
                    and (
                        self.optimizer[self.TSnode].nneg > 3
                        or self.optimizer[self.TSnode].nneg == 0
                        or self.hess_counter > 10
                        or np.abs(self.TS_E_0 - self.emax) > 10.0
                    )
                    and not self.optimizer[self.TSnode].converged
                ):
                    self.nodes[self.TSnode].form_Primitive_Hessian()
                    if self.hessrcount < 1 and self.pTSnode == self.TSnode:
                        print(' resetting TS node coords Ut (and Hessian)')
                        self.ictan, self.dqmaga = self.get_three_way_tangents(self.nodes, self.energies)
                        self.modify_TS_Hess()
                        self.nhessreset = 10
                        self.hessrcount = 1
                    else:
                        print(' Hessian consistently bad, going back to climb (for 3 iterations)')
                        self.find = False
                        self.nclimb = 2
                elif self.find and self.optimizer[self.TSnode].nneg <= 3:
                    self.hessrcount -= 1
                    self.hess_counter += 1
            print(f'{stage_changed=}: {self.climb=} {self.find=}')

            filename = f'scratch/opt_iters_{self.ID:03}_{opt_iteration:03}.xyz'
            self.xyz_writer(filename, self.geometries, self.energies, self.gradrmss, self.dEs)

            print(f' End early counter {self.endearly_counter}')
            print(
                f'opt_iter: {opt_iteration:2} totalgrad: {float(totalgrad):4.3} gradrms: {float(gradrms):5.4} max E({self.TSnode}) {float(self.emax):5.4}\n'
            )
            opt_iteration += 1

            if self.isConverged and not added and not ts_node_changed and not stage_changed:
                print('Converged')
                return

            if opt_iteration < max_iter and not self.isConverged and not stage_changed:
                self.reparameterize(nconstraints=nconstraints)
                self.get_tangents_opting()
                self.refresh_coordinates()
                if self.pTSnode != self.TSnode and self.climb:
                    print('TS node changed after reparameterizing')
                    self.slow_down_climb()
            elif opt_iteration == max_iter and not self.isConverged:
                self.ran_out = True
                print(' Ran out of iterations')
                return

    def get_tangents_opting(self, print_level=1):
        if self.climb or self.find:
            self.ictan, self.dqmaga = self.get_three_way_tangents(self.nodes, self.energies)
        else:
            self.ictan, self.dqmaga = self.get_tangents(self.nodes)

    def set_stage(self, totalgrad, sumgradrms, ts_cgradq, ts_gradrms, fp):
        totalgrad = totalgrad.item() if isinstance(totalgrad, np.ndarray) else totalgrad
        ts_cgradq = ts_cgradq.item() if isinstance(ts_cgradq, np.ndarray) else ts_cgradq
        # checking sum gradrms is not good because if one node is converged a lot while others a re not this is bad
        print('In set stage')
        all_converged = all(
            [self.nodes[n].gradrms < self.optimizer[n].conv_grms * 1.1 for n in range(1, self.nnodes - 1)]
        )
        all([self.nodes[n].gradrms < self.optimizer[n].conv_grms * 2.5 for n in range(1, self.nnodes - 1)])
        stage_changed = False
        print('set stage ts_cgradq')
        print(ts_cgradq)

        if fp > 0:  # and all_converged_climb and self.dE_iter < 2.:  # and self.nopt_intermediate<1:
            if not self.climb and self.climber:
                print(' ** starting climb **')
                self.climb = True
                print(f' totalgrad {totalgrad:.4f} gradrms: {ts_gradrms:.4f} gts: {ts_cgradq:.4f}')
                # overwrite this here just in case TSnode changed wont cause slow down climb
                self.pTSnode = self.TSnode
                stage_changed = True

            # TODO deserves to be rethought 3/2021
            # (totalgrad < 0.2 and ts_gradrms < self.CONV_TOL*20. and ts_cgradq < 0.02) or  #
            elif (
                self.climb
                and not self.find
                and self.finder
                and self.nclimb < 1
                and (
                    (totalgrad < 0.3 and ts_gradrms < self.CONV_TOL * 20.0)  # I hate totalgrad  and ts_cgradq < 0.02
                    or (all_converged)
                    or (ts_gradrms < self.CONV_TOL * 2.5 and ts_cgradq < 0.01)  # used to be 5
                )
            ) and self.dE_iter < 5.0:
                print(' ** starting exact climb **')
                print(f' totalgrad {totalgrad:.4f} gradrms: {ts_gradrms:.4f} gts: {ts_cgradq:.4f}')
                self.find = True

                # Modify TS Hessian
                self.ictan, self.dqmaga = self.get_three_way_tangents(self.nodes, self.energies)
                self.modify_TS_Hess()

                if self.optimizer[self.TSnode].options.DMAX > 0.1:
                    self.optimizer[self.TSnode].options.DMAX = 0.1
                self.optimizer[self.TSnode] = EigenvectorFollow(self.optimizer[self.TSnode].options.replace())
                self.optimizer[self.TSnode].options.SCALEQN = 1.0
                self.nhessreset = 10  # are these used??? TODO
                self.hessrcount = 0  # are these used?!  TODO
                stage_changed = True

        return stage_changed

    def modify_TS_Hess(self):
        """Modifies Hessian using RP direction"""
        print(f'modifying {self.TSnode} Hessian with RP')

        TSnode = self.TSnode
        # a variable to determine how many time since last modify
        self.hess_counter = 0
        self.TS_E_0 = self.energies[TSnode]
        print(f'{TSnode=}')

        E0 = self.energies[TSnode] / units.KCAL_MOL_PER_AU
        Em1 = self.energies[TSnode - 1] / units.KCAL_MOL_PER_AU
        if self.TSnode + 1 < self.nnodes:
            Ep1 = self.energies[TSnode + 1] / units.KCAL_MOL_PER_AU
        else:
            Ep1 = Em1

        # Update TS node coord basis
        Vecs = self.nodes[TSnode].update_coordinate_basis(constraints=None)

        # get constrained coord basis
        self.newic.xyz = self.nodes[TSnode].xyz.copy()
        const_vec = self.newic.update_coordinate_basis(constraints=self.ictan[TSnode])
        q0 = self.newic.coordinates[0]
        constraint = self.newic.constraints[:, 0]

        # this should just give back ictan[TSnode]?
        tan0 = block_matrix.dot(const_vec, constraint)

        # get qm1 (don't update basis)
        self.newic.xyz = self.nodes[TSnode - 1].xyz.copy()
        qm1 = self.newic.coordinates[0]

        if TSnode + 1 < self.nnodes:
            # get qp1 (don't update basis)
            self.newic.xyz = self.nodes[TSnode + 1].xyz.copy()
            qp1 = self.newic.coordinates[0]
        else:
            qp1 = qm1

        print(" TS Hess init'd w/ existing Hintp")

        # Go to non-constrained basis
        self.newic.xyz = self.nodes[TSnode].xyz.copy()
        self.newic.coord_basis = Vecs
        self.newic.Primitive_Hessian = self.nodes[TSnode].Primitive_Hessian.copy()
        self.newic.form_Hessian_in_basis()

        tan = block_matrix.dot(block_matrix.transpose(Vecs), tan0)  # (nicd,1
        Ht = np.dot(self.newic.Hessian, tan)  # (nicd,nicd)(nicd,1) = nicd,1
        tHt = np.dot(tan.T, Ht)

        a = abs(q0 - qm1)
        b = abs(qp1 - q0)
        c = 2 * (Em1 / a / (a + b) - E0 / a / b + Ep1 / b / (a + b))
        print(
            ' tHt {:1.3f} a: {:1.1f} b: {:1.1f} c: {:1.3f}'.format(
                *tuple(np.asarray(v).item() for v in (tHt, a[0], b[0], c[0]))
            )
        )

        ttt = np.outer(tan, tan)

        # Hint before
        # print eig

        # Finalize Hessian
        self.newic.Hessian += (c - tHt) * ttt
        self.nodes[TSnode].Hessian = self.newic.Hessian.copy()

        # Hint after

        self.nodes[TSnode].newHess = 5

        if False:
            print(f'newHess of node {TSnode} {self.nodes[TSnode].newHess}')
            eigen, _tmph = np.linalg.eigh(self.nodes[TSnode].Hessian)  # nicd,nicd
            print('eigenvalues of new Hess')
            print(eigen)

    def mult_steps(self, n, opt_steps):
        exsteps = 1
        tsnode = int(self.TSnode)

        if ((self.find or self.climb) and self.energies[n] > self.energies[self.TSnode] * 0.9 and n != tsnode) or (
            self.find
            and n == tsnode
            and self.energies[tsnode] > self.energies[tsnode - 1] * 1.1
            and self.energies[tsnode] > self.energies[tsnode + 1] * 1.1
        ):  #
            exsteps = 2
            print(f' multiplying steps for node {n} by {exsteps}')

        return exsteps * opt_steps

    def set_opt_type(self, n, quiet=False):
        # TODO error for seam climb
        opt_type = 'ICTAN'
        if self.climb and n == self.TSnode and not self.find and not isinstance(self.nodes[n].PES, AvgPES):
            opt_type = 'CLIMB'
        elif self.find and n == self.TSnode:
            opt_type = 'TS'
        elif isinstance(self.nodes[n].PES, AvgPES):
            opt_type = 'SEAM'
            if self.climb and n == self.TSnode:
                opt_type = 'TS-SEAM'
        if not quiet:
            print(f' setting node {n} opt_type to {opt_type}')

        return opt_type

    def set_finder(self, rtype):
        assert rtype in [0, 1, 2], 'rtype not defined'
        print('')
        print('*********************************************************************')
        if rtype == 2:
            print('****************** set climber and finder to True *******************')
            self.climber = True
            self.finder = True
        elif rtype == 1:
            print('***************** setting climber to True*************************')
            self.climber = True
        else:
            print('******** Turning off climbing image and exact TS search **********')
        print('*********************************************************************')

    def is_converged(self, totalgrad, fp, rtype, ts_cgradq):
        """
        Check if optimization is converged
        """

        # Important the factor 5 here corresponds to the same convergence criteria in the TS optimizer
        TS_conv = self.CONV_TOL * 5
        # => Check if intermediate exists
        # ALEX REMOVED CLIMB REQUIREMENT
        if self.has_intermediate(self.noise):
            print(f'New pot min: {self.get_intermediate(self.noise)}')
            print(f'Old pot min: {self.pot_min}')
            if self.get_intermediate(self.noise) == self.pot_min:
                self.endearly_counter += 1
            else:
                self.pot_min = self.get_intermediate(self.noise)
                self.endearly_counter = 1
            if self.endearly_counter >= 3:
                self.end_early = True
                self.tscontinue = False
                printcool(' THERE IS AN INTERMEDIATE, OPTIMIZE THE INTERMEDIATE AND TRY AGAIN')
                return True

        elif not self.has_intermediate(self.noise):
            self.endearly_counter = 0
            self.pot_min = self.get_intermediate(self.noise)

        # or (totalgrad<0.1 and self.nodes[self.TSnode].gradrms<2.5*TS_conv and self.dE_iter<0.02 and self.optimizer[self.TSnode].nneg <2)  #TODO extra crit here

        # 4/8/2022 Bug with this because it isn't recalculated and abs(ts_cgradq) < TS_conv
        # 5/10/2022 self.optimizer[self.TSnode].nneg < 3
        if self.finder and self.find:
            return (
                self.nodes[self.TSnode].gradrms < self.CONV_TOL
                and self.dE_iter < self.optimizer[self.TSnode].conv_Ediff * 3
            )
        elif self.climber and self.climb:
            return (
                self.nodes[self.TSnode].gradrms < self.CONV_TOL
                and abs(ts_cgradq) < TS_conv
                and self.dE_iter < self.optimizer[self.TSnode].conv_Ediff * 3
            )
        elif not self.climber and not self.finder:
            print(f' CONV_TOL={self.CONV_TOL:.4f}')
            return all([self.optimizer[n].converged for n in range(1, self.nnodes - 1)])

        return False

    def add_node_before_TS(self):
        """ """

        # New node is TS node
        new_node = self.add_node(
            self.nodes[self.TSnode - 1],
            self.nodes[self.TSnode],
            stepsize=0.5,
            node_id=self.TSnode,
        )
        new_node_list = [None] * (self.nnodes + 1)
        new_optimizers = [None] * (self.nnodes + 1)

        for n in range(0, self.TSnode):
            new_node_list[n] = self.nodes[n]
            new_optimizers[n] = self.optimizer[n]

        new_node_list[self.TSnode] = new_node
        new_optimizers[self.TSnode] = self.optimizer[0].__class__(self.optimizer[0].options.replace())

        for n in range(self.TSnode + 1, self.nnodes + 1):
            new_node_list[n] = Molecule.copy_from(MoleculeA=self.nodes[n - 1], new_node_id=n)
            new_optimizers[n] = self.optimizer[n - 1]

        self.nodes = new_node_list
        self.optimizer = new_optimizers
        self.nnodes = len(self.nodes)
        print(f' New number of nodes {self.nnodes}')
        self.active = [True] * self.nnodes
        self.active[0] = False
        self.active[self.nnodes - 1] = False
        print('0')
        print(self.nodes[0].xyz)
        print('1')
        print(self.nodes[1].xyz)
        print('-1')
        print(self.nodes[-1].xyz)

    def add_node_after_TS(self):
        """ """
        new_node = self.add_node(
            self.nodes[self.TSnode],
            self.nodes[self.TSnode + 1],
            stepsize=0.5,
            node_id=self.TSnode + 1,
        )
        new_node_list = [None] * (self.nnodes + 1)
        new_optimizers = [None] * (self.nnodes + 1)
        for n in range(0, self.TSnode + 1):
            new_node_list[n] = self.nodes[n]
            new_optimizers[n] = self.optimizer[n]
        new_node_list[self.TSnode + 1] = new_node
        new_optimizers[self.TSnode + 1] = self.optimizer[0].__class__(self.optimizer[0].options.replace())

        for n in range(self.TSnode + 2, self.nnodes + 1):
            new_node_list[n] = Molecule.copy_from(MoleculeA=self.nodes[n - 1], new_node_id=n)
            new_optimizers[n] = self.optimizer[n - 1]
        self.nodes = new_node_list
        self.optimizer = new_optimizers
        self.nnodes = len(self.nodes)
        print(f' New number of nodes {self.nnodes}')
        self.active = [True] * self.nnodes
        self.active[0] = False
        self.active[self.nnodes - 1] = False

    def set_node_convergence(self):
        """set convergence for nodes"""

        factor = 5.0 if (self.climber or self.finder) else 1.0
        TSnode = self.TSnode
        for n in range(1, self.nnodes - 1):
            if self.nodes[n] is not None:
                self.optimizer[n].conv_grms = self.CONV_TOL * factor
                self.optimizer[n].conv_gmax = self.options.CONV_gmax * factor
                self.optimizer[n].conv_Ediff = self.options.CONV_Ediff * factor
                if self.optimizer[n].converged and n != TSnode:
                    self.optimizer[n].check_only_grad_converged = True
                if (self.climb or self.find) and self.energies[n] > self.energies[TSnode] * 0.75 and n != TSnode:
                    self.optimizer[n].conv_grms = self.CONV_TOL
                    self.optimizer[n].conv_gmax = self.options.CONV_gmax
                    self.optimizer[n].conv_Ediff = self.options.CONV_Ediff
                    self.optimizer[n].check_only_grad_converged = False
                if n == self.TSnode and (self.climb or self.find):
                    self.optimizer[n].conv_grms = self.CONV_TOL
                    self.optimizer[n].conv_gmax = self.options.CONV_gmax
                    self.optimizer[n].conv_Ediff = self.options.CONV_Ediff
                    self.optimizer[n].check_only_grad_converged = False

    def slow_down_climb(self):
        if self.climb and not self.find:
            print(' slowing down climb optimization')
            self.optimizer[self.TSnode].options.DMAX /= self.newclimbscale
            self.optimizer[self.TSnode].options.SCALEQN = 2.0
            if self.optimizer[self.TSnode].SCALE_CLIMB < 5.0:
                self.optimizer[self.TSnode].SCALE_CLIMB += 1.0
            self.optimizer[self.pTSnode].options.SCALEQN = 1.0
            self.ts_exsteps = 1
            if self.newclimbscale < 5.0:
                self.newclimbscale += 1.0
        elif self.find:
            self.find = False
            self.climb = True
            self.nclimb = 1
            print(' Find bad, going back to climb')
