import logging

from ..molecule import Molecule

# local application imports
from ..potential_energy_surfaces import PES, AvgPES
from ..utilities import output
from .se_gsm import SEGSM

logger = logging.getLogger(__name__)

# standard library imports


class SECross(SEGSM):
    """Single-ended search for a minimum-energy crossing point (MESX/MECI) on a penalty PES."""

    is_crossing_search = True

    def on_growth_exhausted(self, error):
        # SE-Cross optimises the last node itself after growth (see go_gsm).
        logger.debug(' Will do extra optimization of this node in SE-Cross')

    def go_gsm(self, max_iters: int = 50, opt_steps: int = 3, rtype: int = 0) -> None:
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
        logger.info(f' Initial energy is {self.nodes[0].energy:1.4f}')

        # stash bdist for node 0
        _, self.nodes[0].bdist = self.get_tangent(self.nodes[0], None, driving_coords=self.driving_coords)
        logger.debug(f' Initial bdist is {self.nodes[0].bdist:1.3f}')

        # interpolate first node
        self.add_GSM_nodeR()

        # grow string
        self.grow_string(max_iters=max_iters, max_opt_steps=opt_steps)
        logger.info(' SE_Cross growth phase over')
        logger.warning(' Warning last node still not fully optimized')

        # doing extra constrained penalty optimization for MECI, tightening the penalty step by step
        logger.info(f' extra constrained optimization for the nnR-1 = {self.nR - 1}')
        self.optimizer[self.nR - 1].conv_grms = self.CONV_TOL * 5
        for sigma in (1.5, 2.5, 3.5):
            ictan = self.get_tangent_xyz(
                self.nodes[self.nR - 1].xyz, self.nodes[self.nR - 2].xyz, self.newic.primitive_internal_coordinates
            )
            self.nodes[self.nR - 1].PES.sigma = sigma
            self.optimizer[self.nR - 1].optimize(
                molecule=self.nodes[self.nR - 1],
                refE=self.nodes[0].V0,
                opt_type='ICTAN',
                opt_steps=5,
                ictan=ictan,
            )

        self.optimizer[self.nR].opt_cross = True
        self.nodes[0].V0 = self.nodes[0].PES.PES2.energy
        if rtype == 0:
            # MECI optimization
            self.nodes[self.nR] = Molecule.copy_from(self.nodes[self.nR - 1], new_node_id=self.nR)
            avg_pes = AvgPES.create_pes_from(self.nodes[self.nR].PES)
            self.nodes[self.nR].PES = avg_pes
            self.optimizer[self.nR].conv_grms = self.CONV_TOL
            self.optimizer[self.nR].conv_gmax = 0.1  # self.CONV_gmax
            self.optimizer[self.nR].conv_Ediff = self.CONV_Ediff
            self.optimizer[self.nR].conv_dE = self.CONV_dE
            self.optimizer[self.nR].optimize(
                molecule=self.nodes[self.nR],
                refE=self.nodes[0].V0,
                opt_type='MECI',
                opt_steps=100,
                verbose=True,
            )
            if not self.optimizer[self.nR].converged:
                logger.info('doing extra optimization in hopes that the MECI will converge.')
                if self.nodes[self.nR].PES.PES2.energy - self.nodes[0].V0 < 20:
                    self.optimizer[self.nR].optimize(
                        molecule=self.nodes[self.nR],
                        refE=self.nodes[0].V0,
                        opt_type='MECI',
                        opt_steps=100,
                        verbose=True,
                    )
        else:
            # unconstrained penalty optimization
            # TODO make unctonstrained "CROSSING" which checks for dE convergence
            self.nodes[self.nR] = Molecule.copy_from(self.nodes[self.nR - 1], new_node_id=self.nR)
            self.nodes[self.nR].PES.sigma = 10.0
            logger.debug(f' sigma for node {self.nR} is {self.nodes[self.nR].PES.sigma:.3f}')
            self.optimizer[self.nR].opt_cross = True
            self.optimizer[self.nR].conv_grms = self.CONV_TOL
            self.optimizer[self.nR].conv_Ediff = self.CONV_Ediff
            self.optimizer[self.nR].conv_dE = self.CONV_dE
            self.optimizer[self.nR].optimize(
                molecule=self.nodes[self.nR],
                refE=self.nodes[0].V0,
                opt_type='UNCONSTRAINED',
                opt_steps=200,
                verbose=True,
            )
        self.write_string('grown_string')

        if self.optimizer[self.nR].converged:
            self.nnodes = self.nR + 1
            self.nodes = self.nodes[: self.nnodes]
            logger.debug('Setting all interior nodes to active')
            for n in range(1, self.nnodes - 1):
                self.active[n] = True
            self.active[self.nnodes - 1] = False
            self.active[0] = False

            # Convert all the PES to excited-states
            for n in range(self.nnodes):
                self.nodes[n].PES = PES.create_pes_from(self.nodes[n].PES.PES2, gradient_states=[(1, 1)])

            logger.debug(' initial ic_reparam')
            self.reparameterize(ic_reparam_steps=25)
            self.print_energies(' V_profile (after reparam):')
            energies = self.energies

            deltaE = energies[-1] - energies[0]
            if deltaE > 20:
                logger.info(f" MECI energy is too high {deltaE:5.4f}. Don't try to optimize pathway")
                logger.info('Exiting early')
                self.end_early = True
            else:
                logger.info(f' deltaE s1-minimum and MECI {deltaE:5.4f}')
                try:
                    self.optimize_string(max_iter=max_iters, opt_steps=3, rtype=1)
                except Exception as error:
                    if str(error) == 'Ran out of iterations':
                        logger.warning(error)
                        self.end_early = True
                    else:
                        logger.warning(error)
                        self.end_early = True
        else:
            logger.info('Exiting early')
            self.end_early = True
        self.write_string('string')

    def check_if_grown(self):
        isDone = False
        pes1dE = self.nodes[self.nR - 1].PES.dE
        pes2dE = self.nodes[self.nR - 2].PES.dE
        condition1 = abs(self.nodes[self.nR - 1].bdist) <= (1 - self.BDIST_RATIO) * abs(self.nodes[0].bdist) and (
            abs(pes1dE) > abs(pes2dE)
        )
        if condition1:
            logger.debug(' Condition 1 satisfied')
            logger.debug(f' bdist current {abs(self.nodes[self.nR - 1].bdist):1.3f}')
            logger.debug(f' bdist target {abs(self.nodes[0].bdist) * (1 - self.BDIST_RATIO):1.3f}')
            logger.info(' Growth-phase over')
            isDone = True
        return isDone

    def set_frontier_convergence(self, nR):
        self.optimizer[nR].conv_grms = self.ADD_NODE_TOL
        self.optimizer[nR].conv_gmax = 100.0  # self.ADD_NODE_TOL # could use some multiplier times CONV_GMAX...
        self.optimizer[nR].conv_Ediff = 1000.0  # 2.5
        logger.debug(f' conv_tol of node {nR} is {self.optimizer[nR].conv_grms:.4f}')
