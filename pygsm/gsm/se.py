# local application imports
import logging

# standard library imports
# third party
from collections import Counter

import numpy as np

from ..coords import Distance
from ..molecule import Molecule
from ..utils import output
from .main import MainGSM

logger = logging.getLogger(__name__)


class SEGSM(MainGSM):
    """Single-ended GSM: grows from the reactant along driving coordinates, then optimises towards the TS.

    Parameters: see :class:`GSM` (no ``product``; ``driving_coords`` is required).
    """

    def __init__(self, reactant, **kwargs):
        super().__init__(reactant, **kwargs)
        self.current_nnodes = 1

        logger.debug(' Assuming the isomers are initialized!')
        logger.debug(' Primitive Internal Coordinates')
        logger.debug(self.nodes[0].primitive_internal_coordinates[0:50])
        logger.debug(f' number of primitives is {self.nodes[0].num_primitives}')

        logger.info('Driving Coordinates: ')
        logger.debug(self.driving_coords)

        # stash bdist for node 0
        ictan, self.nodes[0].bdist = self.get_tangent(
            self.nodes[0],
            None,
            driving_coords=self.driving_coords,
        )
        self.nodes[0].update_coordinate_basis(constraints=ictan)

    def set_V0(self):
        self.nodes[0].V0 = self.nodes[0].energy
        # TODO should be actual gradient
        self.nodes[0].gradrms = 0.0

    default_growth_opt_steps = 10

    def _grow_phase(self, max_iters, opt_steps):
        self.nodes[0].gradrms = 0.0
        self.nodes[0].V0 = self.nodes[0].energy
        logger.info(f' Initial energy is {self.nodes[0].energy:1.4f}')
        self.add_GSM_nodeR()
        self.grow_string(max_iters=max_iters, max_opt_steps=opt_steps)
        if self.tscontinue:
            self.pastts = self.past_ts()
            logger.debug(f'pastts {self.pastts}')
            try:
                if self.pastts == 1:  # normal over the hill
                    self.add_GSM_nodeR(1)
                    self.add_last_node(2)
                elif self.pastts in (2, 3):  # 2: constraint gradient positive, 3: product bonds formed
                    self.add_last_node(1)
                    if self.nodes[self.nR - 1].gradrms > 5.0 * self.CONV_TOL:
                        self.add_last_node(1)
            except (IndexError, RuntimeError, ValueError) as error:
                logger.warning(f'Failed to add last node, continuing: {error}')
                # probably need to make sure last node is optimized

        self.nnodes = self.nR
        self.nodes = self.nodes[: self.nR]

        if self.TSnode == self.nR - 1:
            logger.info(' The highest energy node is the last')
            logger.info(' not continuing with TS optimization.')
            self.tscontinue = False

        logger.info(f' Number of nodes is  {self.nnodes}')
        logger.warning(' Warning last node still not optimized fully')
        self.write_string('grown_string')
        logger.info(' SSM growth phase over')
        self.done_growing = True

        logger.info(' beginning opt phase')
        logger.debug('Setting all interior nodes to active')
        for n in range(1, self.nnodes - 1):
            self.active[n] = True
        self.active[self.nnodes - 1] = False
        self.active[0] = False

        logger.debug(' initial ic_reparam')
        self.reparameterize(ic_reparam_steps=25)
        self.print_energies(' V_profile (after reparam):')

    def _string_opt_steps(self, opt_steps):
        # opt steps are fixed at 3 in the optimisation phase
        return 3

    def add_last_node(self, rtype):
        assert rtype == 1 or rtype == 2, 'rtype must be 1 or 2'
        noptsteps = 100
        if self.nodes[self.nR - 1].PES.lot.do_coupling:
            opt_type = 'MECI'
        else:
            opt_type = 'UNCONSTRAINED'

        if rtype == 1:
            logger.debug(' copying last node, opting')
            self.nodes[self.nR] = Molecule.copy_from(self.nodes[self.nR - 1], new_node_id=self.nR)
            logger.debug(f' Optimizing node {self.nR}')
            self.optimizer[self.nR].conv_grms = self.CONV_TOL
            self.optimizer[self.nR].conv_gmax = self.CONV_gmax
            self.optimizer[self.nR].conv_Ediff = self.CONV_Ediff
            self.optimizer[self.nR].conv_dE = self.CONV_dE
            self.optimizer[self.nR].optimize(
                molecule=self.nodes[self.nR],
                refE=self.nodes[0].V0,
                opt_steps=noptsteps,
                opt_type=opt_type,
            )
            self.active[self.nR] = True
            if (self.nodes[self.nR].xyz == self.nodes[self.nR - 1].xyz).all():
                logger.info(' Opt did not produce new geometry')
            else:
                self.nR += 1
        elif rtype == 2:
            logger.debug(' already created node, opting')
            self.optimizer[self.nR - 1].conv_grms = self.CONV_TOL
            self.optimizer[self.nR - 1].conv_gmax = self.CONV_gmax
            self.optimizer[self.nR - 1].conv_Ediff = self.CONV_Ediff
            self.optimizer[self.nR - 1].conv_dE = self.CONV_dE
            self.optimizer[self.nR - 1].optimize(
                molecule=self.nodes[self.nR - 1],
                refE=self.nodes[0].V0,
                opt_steps=noptsteps,
                opt_type=opt_type,
            )
        return

    def on_growth_exhausted(self, error):
        """Optimise the last grown node once the driving coordinates are exhausted."""
        opt_type = 'MECI' if self.nodes[self.nR - 1].PES.lot.do_coupling else 'UNCONSTRAINED'
        logger.info(' optimizing last node')
        self.optimizer[self.nR - 1].conv_grms = self.CONV_TOL
        logger.debug(self.optimizer[self.nR - 1].conv_grms)
        self.optimizer[self.nR - 1].optimize(
            molecule=self.nodes[self.nR - 1],
            refE=self.nodes[0].V0,
            opt_steps=50,
            opt_type=opt_type,
        )

    def grow_nodes(self):
        if self.nodes[self.nR - 1].gradrms < self.ADD_NODE_TOL:
            if self.nR == self.nnodes:
                logger.warning(' Ran out of nodes, exiting GSM')
                raise ValueError
            if self.nodes[self.nR] is None:
                self.add_GSM_nodeR()
                logger.debug(
                    f' getting energy for node {self.nR - 1}: {self.nodes[self.nR - 1].energy - self.nodes[0].V0:5.4f}'
                )
        return

    def add_GSM_nodes(self, newnodes=1):
        if self.current_nnodes + newnodes > self.nnodes:
            logger.debug('Adding too many nodes, cannot interpolate')
        for _i in range(newnodes):
            self.add_GSM_nodeR()

    def ic_reparam_g(self, ic_reparam_steps=4, n0=0, nconstraints=1):  # see line 3863 of gstring.cpp
        """
        Dont do ic_reparam_g for SE-GSM
        """
        return

    def set_frontier_convergence(self, nR):
        # set
        self.optimizer[nR].conv_grms = self.ADD_NODE_TOL
        self.optimizer[nR].conv_gmax = 100.0  # self.ADD_NODE_TOL # could use some multiplier times CONV_GMAX...
        self.optimizer[nR].conv_Ediff = 1000.0  # 2.5
        logger.debug(f' conv_tol of node {nR} is {self.optimizer[nR].conv_grms:.4f}')

    def set_active(self, nR, nP=None):
        logger.debug(f' setting active node to {nR} ')

        for i in range(self.nnodes):
            if self.nodes[i] is not None:
                self.active[i] = False

        self.set_frontier_convergence(nR)
        self.active[nR] = True

    def make_tan_list(self):
        ncurrent, nlist = self.make_difference_node_list()
        param_list = []
        for n in range(ncurrent - 1):
            if nlist[2 * n] not in param_list:
                param_list.append(nlist[2 * n])
        return param_list

    def make_move_list(self):
        ncurrent, nlist = self.make_difference_node_list()
        param_list = []
        for n in range(ncurrent):
            if nlist[2 * n + 1] not in param_list:
                param_list.append(nlist[2 * n + 1])
        return param_list

    def make_difference_node_list(self):
        ncurrent = 0
        nlist = [0] * (2 * self.nnodes)
        for n in range(self.nR - 1):
            nlist[2 * ncurrent] = n
            nlist[2 * ncurrent + 1] = n + 1
            ncurrent += 1
        nlist[2 * ncurrent + 1] = self.nR - 1
        nlist[2 * ncurrent] = self.nR - 1
        ncurrent += 1
        return ncurrent, nlist

    def past_ts(self):
        """ """
        ispast = ispast1 = ispast2 = ispast3 = 0
        THRESH1 = 5.0
        THRESH2 = 3.0
        THRESH3 = -1.0
        CTHRESH = 0.005
        OTHRESH = -0.015
        emax = -100.0
        nodemax = 1
        # n0 is zero until after finished growing
        ns = self.n0 - 1
        if ns < nodemax:
            ns = nodemax

        energies = self.energies
        logger.debug(' Energies ' + ' '.join(f'{energies[n]:4.3f}' for n in range(ns, self.nR)))
        for n in range(ns, self.nR):
            if energies[n] > emax:
                nodemax = n
                emax = energies[n]
        logger.debug(f' nodemax  {nodemax}')

        for n in range(nodemax, self.nR):
            if energies[n] < emax - THRESH1:
                ispast1 += 1
            if energies[n] < emax - THRESH2:
                ispast2 += 1
            if energies[n] < emax - THRESH3:
                ispast3 += 1
            if ispast1 > 1:
                break
        logger.debug(f' ispast1 {ispast1}')
        logger.debug(f' ispast2 {ispast2}')
        logger.debug(f' ispast3 {ispast3}')

        # TODO 5/9/2019 what about multiple constraints
        # Done 6/23/2019
        constraints = self.nodes[self.nR - 1].constraints[:, 0]
        gradient = self.nodes[self.nR - 1].gradient

        overlap = np.dot(gradient.T, constraints)
        cgrad = overlap * constraints

        cgrad = np.linalg.norm(cgrad) * np.sign(overlap)

        logger.debug(f' cgrad: {np.asarray(cgrad).item():4.3f} nodemax: {nodemax} nR: {self.nR}')

        # 6/17 THIS should check if the last node is high in energy
        if cgrad > CTHRESH and not self.nodes[self.nR - 1].PES.lot.do_coupling and nodemax != self.TSnode:
            logger.debug(' constraint gradient positive')
            ispast = 2
        elif ispast1 > 0 and cgrad > OTHRESH:
            logger.info(' over the hill(1)')
            ispast = 1
        elif ispast2 > 1:
            logger.info(' over the hill(2)')
            ispast = 1
        else:
            ispast = 0

        if ispast == 0:
            bch = self.check_for_reaction_g(1, self.driving_coords)
            if ispast3 > 1 and bch:
                logger.info(f'over the hill(3) connection changed {bch!r} ')
                ispast = 3
        logger.debug(f' ispast= {ispast}')
        return ispast

    def check_if_grown(self):
        """
        Check if the string is grown
        Returns True if grown
        """

        self.pastts = self.past_ts()
        isDone = False
        # TODO break planes
        condition1 = abs(self.nodes[self.nR - 1].bdist) <= (1 - self.BDIST_RATIO) * abs(self.nodes[0].bdist)
        logger.debug(f' bdist {self.nodes[self.nR - 1].bdist:.3f}')

        fp = self.find_peaks('growing')
        if self.pastts and self.current_nnodes > 3 and condition1:  # TODO extra criterion here
            logger.debug(f' pastts is  {self.pastts}')
            if self.TSnode == self.nR - 1:
                logger.info(' The highest energy node is the last')
                logger.info(' not continuing with TS optimization.')
                self.tscontinue = False
            output.printcool('Over the hill')
            isDone = True
        elif fp == -1 and self.energies[self.nR - 1] > 200.0 and self.nodes[self.nR - 1].gradrms > self.CONV_TOL * 5:
            logger.info('growth_iters over: all uphill and high energy')
            self.end_early = 2
            self.tscontinue = False
            self.nnodes = self.nR
            isDone = True
        elif fp == -2:
            logger.info('growth_iters over: all uphill and flattening out')
            self.end_early = 2
            self.tscontinue = False
            self.nnodes = self.nR
            isDone = True

        # ADD extra criteria here to check if TS is higher energy than product
        return isDone

    def is_converged(self, totalgrad, fp, rtype, ts_cgradq):
        if self.TSnode == self.nnodes - 2 and (self.find or totalgrad < 0.2) and fp == 1:
            if self.nodes[self.nR - 1].gradrms > self.CONV_TOL:
                logger.debug('TS node is second to last node, adding one more node')
                self.add_last_node(1)
                self.nnodes = self.nR
                self.active[self.nnodes - 1] = False  # GSM makes self.active[self.nnodes-1]=True as well
                self.active[self.nnodes - 2] = True  # GSM makes self.active[self.nnodes-1]=True as well
                logger.debug('done adding node')
                logger.debug(f'nnodes =  {self.nnodes}')
                self.ictan, self.dqmaga = self.get_tangents(self.nodes)
                self.refresh_coordinates()
            return False

        # => check string profile <= #
        if fp == -1:  # total string is uphill
            logger.debug('fp == -1, check V_profile')
            logger.info('total dissociation')
            self.endearly = True  # bools
            self.tscontinue = False
            return True
        elif fp == -2:
            logger.info('termination due to dissociation')
            self.tscontinue = False
            self.endearly = True  # bools
            return True
        elif fp == 0:
            self.tscontinue = False
            self.endearly = True  # bools
            return True
        elif self.climb and fp > 0 and self.finder:
            fp = self.find_peaks('opting')
            if fp > 1:
                rxnocc, wint = self.check_for_reaction()
            if fp > 1 and rxnocc and wint < self.nnodes - 1:
                logger.info('Need to trim string')
                self.tscontinue = False
                self.endearly = True  # bools
                return True
            else:
                return False
        else:
            return super().is_converged(totalgrad, fp, rtype, ts_cgradq)

    def check_for_reaction(self):
        """ """
        isrxn = self.check_for_reaction_g(1, self.driving_coords)
        minnodes = []
        maxnodes = []
        wint = 0
        energies = self.energies
        if energies[1] > energies[0]:
            minnodes.append(0)
        if energies[self.nnodes - 1] < energies[self.nnodes - 2]:
            minnodes.append(self.nnodes - 1)
        for n in range(self.n0, self.nnodes - 1):
            if energies[n + 1] > energies[n] and energies[n] < energies[n - 1]:
                minnodes.append(n)
            if energies[n + 1] < energies[n] and energies[n] > energies[n - 1]:
                maxnodes.append(n)
        if len(minnodes) > 2 and len(maxnodes) > 1:
            wint = minnodes[1]  # the real reaction ends at first minimum
            logger.debug(f' wint  {wint}')

        return isrxn, wint

    def check_for_reaction_g(self, rtype, driving_coords):
        """ """

        c = Counter(elem[0] for elem in driving_coords)
        nadds = c['ADD']
        nbreaks = c['BREAK']
        isrxn = False

        if (nadds + nbreaks) < 1:
            return False
        nadded = 0
        nbroken = 0
        nnR = self.nR - 1
        xyz = self.nodes[nnR].xyz
        atoms = self.nodes[nnR].atoms

        for i in driving_coords:
            if 'ADD' in i:
                index = [i[1] - 1, i[2] - 1]
                bond = Distance(index[0], index[1])
                d = bond.value(xyz)
                d0 = (atoms[index[0]].vdw_radius + atoms[index[1]].vdw_radius) / 2
                if d < d0:
                    nadded += 1
            if 'BREAK' in i:
                index = [i[1] - 1, i[2] - 1]
                bond = Distance(index[0], index[1])
                d = bond.value(xyz)
                d0 = (atoms[index[0]].vdw_radius + atoms[index[1]].vdw_radius) / 2
                if d > d0:
                    nbroken += 1
        if rtype == 1:
            if (nadded + nbroken) >= (nadds + nbreaks):
                isrxn = True
        else:
            isrxn = True
        logger.debug(f' check_for_reaction_g isrxn: {isrxn!r} nadd+nbrk: {nadds + nbreaks}')
        return isrxn

        # # => Convergence Criteria
