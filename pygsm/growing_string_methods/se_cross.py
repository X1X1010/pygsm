import os

from ..molecule import Molecule

# local application imports
from ..potential_energy_surfaces import PES, Avg_PES
from ..utilities import output
from .se_gsm import SE_GSM

# standard library imports


class SE_Cross(SE_GSM):
    def go_gsm(self, max_iters=50, opt_steps=3, rtype=0):
        """rtype=0 MECI search
        rtype=1 MESX search
        """
        assert rtype in [0, 1], 'rtype not defined'
        if rtype == 0:
            output.printcool('Doing SE-MECI search')
        else:
            output.printcool('Doing SE-MESX search')

        self.nodes[0].gradrms = 0.0
        self.nodes[0].V0 = self.nodes[0].energy
        print(f' Initial energy is {self.nodes[0].energy:1.4f}')

        # stash bdist for node 0
        _, self.nodes[0].bdist = self.get_tangent(self.nodes[0], None, driving_coords=self.driving_coords)
        print(f' Initial bdist is {self.nodes[0].bdist:1.3f}')

        # interpolate first node
        self.add_GSM_nodeR()

        # grow string
        self.grow_string(max_iters=max_iters, max_opt_steps=opt_steps)
        print(' SE_Cross growth phase over')
        print(' Warning last node still not fully optimized')

        if True:
            path = os.path.join(os.getcwd(), f'scratch/{self.ID:03d}/{self.nR - 1}')
            # doing extra constrained penalty optimization for MECI
            print(f' extra constrained optimization for the nnR-1 = {self.nR - 1}')
            self.optimizer[self.nR - 1].conv_grms = self.options['CONV_TOL'] * 5
            ictan = self.get_tangent_xyz(
                self.nodes[self.nR - 1].xyz, self.nodes[self.nR - 2].xyz, self.newic.primitive_internal_coordinates
            )
            self.nodes[self.nR - 1].PES.sigma = 1.5
            self.optimizer[self.nR - 1].optimize(
                molecule=self.nodes[self.nR - 1],
                refE=self.nodes[0].V0,
                opt_type='ICTAN',
                opt_steps=5,
                ictan=ictan,
                path=path,
            )
            ictan = self.get_tangent_xyz(
                self.nodes[self.nR - 1].xyz, self.nodes[self.nR - 2].xyz, self.newic.primitive_internal_coordinates
            )
            self.nodes[self.nR - 1].PES.sigma = 2.5
            self.optimizer[self.nR - 1].optimize(
                molecule=self.nodes[self.nR - 1],
                refE=self.nodes[0].V0,
                opt_type='ICTAN',
                opt_steps=5,
                ictan=ictan,
                path=path,
            )
            ictan = self.get_tangent_xyz(
                self.nodes[self.nR - 1].xyz, self.nodes[self.nR - 2].xyz, self.newic.primitive_internal_coordinates
            )
            self.nodes[self.nR - 1].PES.sigma = 3.5
            self.optimizer[self.nR - 1].optimize(
                molecule=self.nodes[self.nR - 1],
                refE=self.nodes[0].V0,
                opt_type='ICTAN',
                opt_steps=5,
                ictan=ictan,
                path=path,
            )

        self.xyz_writer(f'after_penalty_{self.ID:03}.xyz', self.geometries, self.energies, self.gradrmss, self.dEs)
        self.optimizer[self.nR].opt_cross = True
        self.nodes[0].V0 = self.nodes[0].PES.PES2.energy
        if rtype == 0:
            # MECI optimization
            self.nodes[self.nR] = Molecule.copy_from_options(self.nodes[self.nR - 1], new_node_id=self.nR)
            avg_pes = Avg_PES.create_pes_from(self.nodes[self.nR].PES)
            self.nodes[self.nR].PES = avg_pes
            path = os.path.join(os.getcwd(), f'scratch/{self.ID:03d}/{self.nR}')
            self.optimizer[self.nR].conv_grms = self.options['CONV_TOL']
            self.optimizer[self.nR].conv_gmax = 0.1  # self.options['CONV_gmax']
            self.optimizer[self.nR].conv_Ediff = self.options['CONV_Ediff']
            self.optimizer[self.nR].conv_dE = self.options['CONV_dE']
            self.optimizer[self.nR].optimize(
                molecule=self.nodes[self.nR],
                refE=self.nodes[0].V0,
                opt_type='MECI',
                opt_steps=100,
                verbose=True,
                path=path,
            )
            if not self.optimizer[self.nR].converged:
                print('doing extra optimization in hopes that the MECI will converge.')
                if self.nodes[self.nR].PES.PES2.energy - self.nodes[0].V0 < 20:
                    self.optimizer[self.nR].optimize(
                        molecule=self.nodes[self.nR],
                        refE=self.nodes[0].V0,
                        opt_type='MECI',
                        opt_steps=100,
                        verbose=True,
                        path=path,
                    )
        else:
            # unconstrained penalty optimization
            # TODO make unctonstrained "CROSSING" which checks for dE convergence
            self.nodes[self.nR] = Molecule.copy_from_options(self.nodes[self.nR - 1], new_node_id=self.nR)
            self.nodes[self.nR].PES.sigma = 10.0
            print(f' sigma for node {self.nR} is {self.nodes[self.nR].PES.sigma:.3f}')
            path = os.path.join(os.getcwd(), f'scratch/{self.ID:03d}/{self.nR}')
            self.optimizer[self.nR].opt_cross = True
            self.optimizer[self.nR].conv_grms = self.options['CONV_TOL']
            self.optimizer[self.nR].conv_Ediff = self.options['CONV_Ediff']
            self.optimizer[self.nR].conv_dE = self.options['CONV_dE']
            self.optimizer[self.nR].optimize(
                molecule=self.nodes[self.nR],
                refE=self.nodes[0].V0,
                opt_type='UNCONSTRAINED',
                opt_steps=200,
                verbose=True,
                path=path,
            )
        self.xyz_writer(f'grown_string_{self.ID:03}.xyz', self.geometries, self.energies, self.gradrmss, self.dEs)

        if self.optimizer[self.nR].converged:
            self.nnodes = self.nR + 1
            self.nodes = self.nodes[: self.nnodes]
            print('Setting all interior nodes to active')
            for n in range(1, self.nnodes - 1):
                self.active[n] = True
            self.active[self.nnodes - 1] = False
            self.active[0] = False

            # Convert all the PES to excited-states
            for n in range(self.nnodes):
                self.nodes[n].PES = PES.create_pes_from(self.nodes[n].PES.PES2, options={'gradient_states': [(1, 1)]})

            print(' initial ic_reparam')
            self.reparameterize(ic_reparam_steps=25)
            print(' V_profile (after reparam): ', end=' ')
            energies = self.energies
            for n in range(self.nnodes):
                print(f' {float(energies[n]):7.3f}', end=' ')
            print()
            self.xyz_writer(f'grown_string1_{self.ID:03}.xyz', self.geometries, self.energies, self.gradrmss, self.dEs)

            deltaE = energies[-1] - energies[0]
            if deltaE > 20:
                print(f" MECI energy is too high {deltaE:5.4f}. Don't try to optimize pathway")
                print('Exiting early')
                self.end_early = True
            else:
                print(f' deltaE s1-minimum and MECI {deltaE:5.4f}')
                try:
                    self.optimize_string(max_iter=max_iters, opt_steps=3, rtype=1)
                except Exception as error:
                    if str(error) == 'Ran out of iterations':
                        print(error)
                        self.end_early = True
                    else:
                        print(error)
                        self.end_early = True
        else:
            print('Exiting early')
            self.end_early = True

    def check_if_grown(self):
        isDone = False
        pes1dE = self.nodes[self.nR - 1].PES.dE
        pes2dE = self.nodes[self.nR - 2].PES.dE
        condition1 = abs(self.nodes[self.nR - 1].bdist) <= (1 - self.BDIST_RATIO) * abs(self.nodes[0].bdist) and (
            abs(pes1dE) > abs(pes2dE)
        )
        if condition1:
            print(' Condition 1 satisfied')
            print(f' bdist current {abs(self.nodes[self.nR - 1].bdist):1.3f}')
            print(f' bdist target {abs(self.nodes[0].bdist) * (1 - self.BDIST_RATIO):1.3f}')
            print(' Growth-phase over')
            isDone = True
        return isDone

    def set_frontier_convergence(self, nR):
        self.optimizer[nR].conv_grms = self.options['ADD_NODE_TOL']
        self.optimizer[
            nR
        ].conv_gmax = 100.0  # self.options['ADD_NODE_TOL'] # could use some multiplier times CONV_GMAX...
        self.optimizer[nR].conv_Ediff = 1000.0  # 2.5
        print(f' conv_tol of node {nR} is {self.optimizer[nR].conv_grms:.4f}')
