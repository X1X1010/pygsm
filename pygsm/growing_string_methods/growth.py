"""Growth phase of the growing string method."""

import logging

import numpy as np

from pygsm.molecule.molecule import Molecule
from pygsm.utilities.output import printcool

from ._shared import map_method

logger = logging.getLogger(__name__)


class GrowthMixin:
    """Adds nodes from the end points until the string is grown (mixed into :class:`MainGSM`)."""

    def grow_string(self, max_iters=30, max_opt_steps=3, nconstraints=1):
        """Run the GSM growth phase."""

        self.make_difference_node_list()
        self.ictan, self.dqmaga = self.get_tangents_growing()
        self.refresh_coordinates()
        self.set_active(self.nR - 1, self.nnodes - self.nP)

        is_grown = False
        iteration = 0
        while not is_grown:
            if iteration > max_iters:
                logger.warning(' Ran out of iterations')
                return
            printcool(f'Starting growth iteration {iteration}')
            self.optimize_iteration(max_opt_steps)
            totalgrad, gradrms, _ = self.calc_optimization_metrics(self.nodes)
            self.write_string('string')
            logger.info(
                f' gopt_iter: {iteration:2} totalgrad: {float(totalgrad):4.3} gradrms: {float(gradrms):5.4} max E: {float(self.emax):5.4}'
            )

            try:
                self.grow_nodes()
            except (IndexError, RuntimeError, ValueError) as error:
                logger.warning(f"can't add anymore nodes, bdist too small: {error}")

                self.on_growth_exhausted(error)
                break

            self.set_active(self.nR - 1, self.nnodes - self.nP)
            self.ic_reparam_g()
            self.ictan, self.dqmaga = self.get_tangents_growing()
            self.refresh_coordinates()

            iteration += 1
            is_grown = self.check_if_grown()

        logger.debug(' creating newic molecule--used for ic_reparam')
        self.newic = Molecule.copy_from(self.nodes[0])

        if self.growth_direction == 1:
            logger.debug('Setting LOT of last node')
            self.nodes[-1] = Molecule.copy_from(
                MoleculeA=self.nodes[-2],
                xyz=self.nodes[-1].xyz,
                new_node_id=self.nnodes - 1,
            )

    def on_growth_exhausted(self, error):
        """Called when growth stops because no further node can be added."""
        raise RuntimeError(f'cannot grow the string any further: {error}') from error

    def get_tangents_growing(self, print_level=1):
        """
        Finds the tangents during the growth phase.
        Tangents referenced to left or right during growing phase.
        Also updates coordinates
        Not a static method beause no one should ever call this outside of GSM
        """

        ncurrent, nlist = self.make_difference_node_list()
        dqmaga = [0.0] * self.nnodes
        ictan = [[]] * self.nnodes

        if self.print_level > 1:
            logger.debug('ncurrent, nlist')
            logger.debug(ncurrent)
            logger.debug(nlist)

        for n in range(ncurrent):
            if self.is_double_ended:
                logger.debug(
                    f' getting tangent [{nlist[2 * n]} ]from between {nlist[2 * n]} {nlist[2 * n + 1]} pointing towards {nlist[2 * n]}'
                )
                ictan0 = self.get_tangent_xyz(
                    self.nodes[nlist[2 * n]].xyz,
                    self.nodes[nlist[2 * n + 1]].xyz,
                    self.nodes[0].primitive_internal_coordinates,
                )
            else:
                ictan0, _ = self.get_tangent(
                    node1=self.nodes[nlist[2 * n]],
                    node2=self.nodes[nlist[2 * n + 1]],
                    driving_coords=self.driving_coords,
                )

            if self.print_level > 1:
                logger.debug(f'forming space for {nlist[2 * n + 1]}')
            if self.print_level > 1:
                logger.debug(f'forming tangent for  {nlist[2 * n]}')

            if (ictan0[:] == 0.0).all():
                logger.debug(' ICTAN IS ZERO!')
                logger.debug(nlist[2 * n])
                logger.debug(nlist[2 * n + 1])
                raise RuntimeError

            # normalize ictan
            norm = np.linalg.norm(ictan0)
            ictan[nlist[2 * n]] = ictan0 / norm

            # NOTE regular GSM does something weird here
            # but this is not followed here anymore 7/1/2020

            dqmaga[nlist[2 * n]] = norm

        if print_level > 0:
            logger.debug(' dqmaga: ' + ' '.join(f'{dqmaga[n]:5.3}' for n in range(self.nnodes)))

        if print_level > 1:
            for n in range(ncurrent):
                logger.debug(f'dqmag[{nlist[2 * n]}] ={self.dqmaga[nlist[2 * n]]:1.2f}')
                logger.debug(f'printing ictan[{nlist[2 * n]}]')
                logger.debug(self.ictan[nlist[2 * n]].T)
        for i, tan in enumerate(ictan):
            if np.all(tan == 0.0):
                logger.debug(f'tan {i} of the tangents is 0')
                raise RuntimeError

        return ictan, dqmaga

    def add_GSM_nodeR(self, newnodes=1):
        """
        Add a node between endpoints on the reactant side, should only be called inside GSM
        """
        printcool('Adding reactant node', level=logging.DEBUG)

        if self.current_nnodes + newnodes > self.nnodes:
            raise ValueError('Adding too many nodes, cannot interpolate')
        for _i in range(newnodes):
            iR = self.nR - 1
            iP = self.nnodes - self.nP
            iN = self.nR
            logger.info(f' adding node: {iN} between {iR} {iP} from {iR}')
            if self.nnodes - self.current_nnodes > 1:
                stepsize = 1.0 / float(self.nnodes - self.current_nnodes + 1)
            else:
                stepsize = 0.5

            self.nodes[self.nR] = self.add_node(
                self.nodes[iR],
                self.nodes[iP],
                stepsize,
                iN,
                DQMAG_MAX=self.DQMAG_MAX,
                DQMAG_MIN=self.DQMAG_MIN,
                driving_coords=self.driving_coords,
            )

            if self.nodes[self.nR] is None:
                raise RuntimeError('Ran out of space')

            if not self.is_double_ended:
                _ictan, bdist = self.get_tangent(
                    self.nodes[self.nR],
                    None,
                    driving_coords=self.driving_coords,
                )
                self.nodes[self.nR].bdist = bdist

            self.optimizer[self.nR].DMAX = self.optimizer[self.nR - 1].DMAX
            self.current_nnodes += 1
            self.nR += 1
            logger.debug(f' nn={self.current_nnodes},nR={self.nR}')
            self.active[self.nR - 1] = True

    def add_GSM_nodeP(self, newnodes=1):
        """
        Add a node between endpoints on the product side, should only be called inside GSM
        """
        printcool('Adding product node', level=logging.DEBUG)
        if self.current_nnodes + newnodes > self.nnodes:
            raise ValueError('Adding too many nodes, cannot interpolate')

        for _i in range(newnodes):
            n1 = self.nnodes - self.nP
            n2 = self.nnodes - self.nP - 1
            n3 = self.nR - 1
            logger.info(f' adding node: {n2} between {n1} {n3} from {n1}')
            if self.nnodes - self.current_nnodes > 1:
                stepsize = 1.0 / float(self.nnodes - self.current_nnodes + 1)
            else:
                stepsize = 0.5

            self.nodes[-self.nP - 1] = self.add_node(self.nodes[n1], self.nodes[n3], stepsize, n2)
            if self.nodes[-self.nP - 1] is None:
                raise RuntimeError('Ran out of space')

            self.optimizer[n2].DMAX = self.optimizer[n1].DMAX
            self.current_nnodes += 1
            self.nP += 1
            logger.debug(f' nn={self.current_nnodes},nP={self.nP}')
            self.active[-self.nP] = True

            # align center of mass  and rotation
        return

    def ic_reparam_g(self, ic_reparam_steps=4, n0=0, reparam_interior=True):  # see line 3863 of gstring.cpp
        """
        Reparameterize during growth phase
        """

        printcool('Reparametrizing string nodes', level=logging.DEBUG)
        rpmove = np.zeros(self.nnodes)
        rpart = np.zeros(self.nnodes)
        disprms = 0.0

        if self.current_nnodes == self.nnodes:
            return

        for i in range(ic_reparam_steps):
            self.ictan, self.dqmaga = self.get_tangents_growing()
            totaldqmag = np.sum(self.dqmaga[n0 : self.nR - 1]) + np.sum(
                self.dqmaga[self.nnodes - self.nP + 1 : self.nnodes]
            )
            if self.print_level > 0:
                if i == 0:
                    logger.debug(f' totaldqmag (without inner): {totaldqmag:1.2}')
                logger.debug(' spacings dqmaga: ' + ' '.join(f'{self.dqmaga[n]:2.3}' for n in range(self.nnodes)))

            if i == 0:
                if self.current_nnodes != self.nnodes:
                    rpart = np.zeros(self.nnodes)
                    for n in range(n0 + 1, self.nR):
                        rpart[n] = 1.0 / (self.current_nnodes - 2)
                    for n in range(self.nnodes - self.nP, self.nnodes - 1):
                        rpart[n] = 1.0 / (self.current_nnodes - 2)
                else:
                    for n in range(n0 + 1, self.nnodes):
                        rpart[n] = 1.0 / (self.nnodes - 1)
                if self.print_level > 0 and i == 0:
                    logger.debug(' rpart: ' + ' '.join(f'{rpart[n]:1.2}' for n in range(1, self.nnodes - 1)))
            nR0 = self.nR
            nP0 = self.nP

            # TODO CRA 3/2019 why is this here?
            if not reparam_interior and self.nnodes - self.current_nnodes > 2:
                nR0 -= 1
                nP0 -= 1

            deltadq = 0.0
            for n in range(n0 + 1, nR0):
                deltadq = self.dqmaga[n - 1] - totaldqmag * rpart[n]
                rpmove[n] = -deltadq
            for n in range(self.nnodes - nP0, self.nnodes - 1):
                deltadq = self.dqmaga[n + 1] - totaldqmag * rpart[n]
                rpmove[n] = -deltadq

            MAXRE = 1.1

            for n in range(n0 + 1, self.nnodes - 1):
                if abs(rpmove[n]) > MAXRE:
                    rpmove[n] = float(np.sign(rpmove[n]) * MAXRE)

            disprms = float(np.linalg.norm(rpmove[n0 + 1 : self.nnodes - 1]))
            if self.print_level > 0:
                logger.debug(' '.join(f'disp[{n}]: {rpmove[n]:1.2f}' for n in range(n0 + 1, self.nnodes - 1)))
                logger.debug(f' disprms: {disprms:1.3}')

            if disprms < 1e-2:
                break

            move_list = self.make_move_list()
            tan_list = self.make_tan_list()

            if self.mp_cores > 1:
                Vecs = map_method(
                    (
                        (self.nodes[0].coord_obj, 'build_dlc', self.nodes[n].xyz, self.ictan[ntan])
                        for n, ntan in zip(move_list, tan_list)
                        if rpmove[n] < 0
                    ),
                    self.mp_cores,
                )

                i = 0
                for n in move_list:
                    if rpmove[n] < 0:
                        self.nodes[n].coord_basis = Vecs[i]
                        i += 1

                # move the positions
                newXyzs = map_method(
                    (
                        (
                            self.nodes[n].coord_obj,
                            'newCartesian',
                            self.nodes[n].xyz,
                            rpmove[n] * self.nodes[n].constraints[:, 0],
                        )
                        for n in move_list
                        if rpmove[n] < 0
                    ),
                    self.mp_cores,
                )
                i = 0
                for n in move_list:
                    if rpmove[n] < 0:
                        self.nodes[n].xyz = newXyzs[i]
                        i += 1
            else:
                for nmove, ntan in zip(move_list, tan_list):
                    if rpmove[nmove] < 0:
                        logger.debug(f'Moving {nmove} along ictan[{ntan}]')
                        self.nodes[nmove].update_coordinate_basis(constraints=self.ictan[ntan])
                        constraint = self.nodes[nmove].constraints[:, 0]
                        dq0 = rpmove[nmove] * constraint
                        self.nodes[nmove].update_xyz(dq0, verbose=True)

        logger.debug(
            f' spacings (end ic_reparam, steps: {i + 1}/{ic_reparam_steps}): '
            + ' '.join(f'{self.dqmaga[n]:1.2}' for n in range(self.nnodes))
            + f'  disprms: {disprms:1.3}'
        )
