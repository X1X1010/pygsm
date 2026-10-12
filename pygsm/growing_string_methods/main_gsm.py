import logging
from itertools import chain

import numpy as np

from pygsm.molecule.molecule import Molecule
from pygsm.utilities.manage_xyz import xyz_to_np
from pygsm.utilities.output import printcool

from ._shared import map_method
from .growth import GrowthMixin
from .gsm import GSM
from .optimization import OptimizationMixin

logger = logging.getLogger(__name__)

#######################################################################################
############### This class contains the main GSM functions  ###########################
#######################################################################################


class MainGSM(GrowthMixin, OptimizationMixin, GSM):
    """Shared driver for DE-GSM, SE-GSM and SE-Cross: node optimisation, reparametrisation and restarts."""

    #: Node optimisation steps per growth iteration when ``go_gsm`` is called without ``opt_steps``.
    default_growth_opt_steps = 3

    def go_gsm(self, max_iters: int = 50, opt_steps: int | None = None, rtype: int = 2) -> None:
        """Grow the string, then optimise it.

        rtype=2 finds and climbs to the TS, 1 climbs without the exact TS search,
        0 turns off the climbing image and TS search.
        """
        if opt_steps is None:
            opt_steps = self.default_growth_opt_steps
        self.set_V0()

        if not self.isRestarted:
            self._grow_phase(max_iters, opt_steps)

        if self.tscontinue:
            self.optimize_string(max_iter=max_iters, opt_steps=self._string_opt_steps(opt_steps), rtype=rtype)
        else:
            logger.info('Exiting early')
            self.end_early = True

        self.write_string('string')
        logger.info('Finished GSM!')

    def set_V0(self):
        raise NotImplementedError

    def _grow_phase(self, max_iters, opt_steps):
        """Grow the string and reparametrise it (only called when not restarting)."""
        raise NotImplementedError

    def _string_opt_steps(self, opt_steps):
        """Node optimisation steps per iteration in the optimisation phase."""
        return opt_steps

    def refresh_coordinates(self, update_TS=False):
        """
        Refresh the DLC coordinates for the string
        """

        if not self.done_growing:
            # TODO

            if self.mp_cores == 1:
                for n in range(1, self.nnodes - 1):
                    if self.nodes[n] is not None:
                        Vecs = self.newic.coord_obj.build_dlc(self.nodes[n].xyz, self.ictan[n])
                        self.nodes[n].coord_basis = Vecs

            else:
                Vecs = map_method(
                    (
                        (self.newic.coord_obj, 'build_dlc', self.nodes[n].xyz, self.ictan[n])
                        for n in range(1, self.nnodes - 1)
                        if self.nodes[n] is not None
                    ),
                    self.mp_cores,
                )

                i = 0
                for n in range(1, self.nnodes - 1):
                    if self.nodes[n] is not None:
                        self.nodes[n].coord_basis = Vecs[i]
                        i += 1
        else:
            if self.find or self.climb:
                TSnode = self.TSnode
                if self.mp_cores == 1:
                    for n in range(1, self.nnodes - 1):
                        # don't update tsnode coord basis
                        if n != TSnode or (n == TSnode and update_TS):
                            Vecs = self.newic.coord_obj.build_dlc(self.nodes[n].xyz, self.ictan[n])
                            self.nodes[n].coord_basis = Vecs
                else:
                    Vecs = map_method(
                        (
                            (self.newic.coord_obj, 'build_dlc', self.nodes[n].xyz, self.ictan[n])
                            for n in range(1, self.nnodes - 1)
                            if n != TSnode
                        ),
                        self.mp_cores,
                    )
                    for i, n in enumerate(chain(range(1, TSnode), range(TSnode + 1, self.nnodes - 1))):
                        self.nodes[n].coord_basis = Vecs[i]

                    if update_TS:
                        Vec = self.newic.coord_obj.build_dlc(self.nodes[TSnode].xyz, self.ictan[TSnode])
                        self.nodes[TSnode].coord_basis = Vec

            else:
                if self.mp_cores == 1:
                    Vecs = []
                    for n in range(1, self.nnodes - 1):
                        Vecs.append(self.newic.coord_obj.build_dlc(self.nodes[n].xyz, self.ictan[n]))
                elif self.mp_cores > 1:
                    Vecs = map_method(
                        (
                            (self.newic.coord_obj, 'build_dlc', self.nodes[n].xyz, self.ictan[n])
                            for n in range(1, self.nnodes - 1)
                        ),
                        self.mp_cores,
                    )
                for n, node in enumerate(self.nodes[1 : self.nnodes - 1]):
                    node.coord_basis = Vecs[n]

    def optimize_iteration(self, opt_steps):
        """
        Optimize string iteration
        """

        refE = self.nodes[0].energy

        for n in range(self.nnodes):
            if self.nodes[n] and self.active[n]:
                printcool(f'Optimizing node {n}', level=logging.DEBUG)
                opt_type = self.set_opt_type(n)
                osteps = self.mult_steps(n, opt_steps)
                self.optimizer[n].optimize(
                    molecule=self.nodes[n],
                    refE=refE,
                    opt_type=opt_type,
                    opt_steps=osteps,
                    ictan=self.ictan[n],
                    xyzframerate=1,
                )

        # SE-GSM: the last node is not optimised in the optimisation phase; relax it if it is not a minimum
        if not self.is_double_ended and not self.is_crossing_search and self.done_growing:
            fp = self.find_peaks('opting')
            if (
                self.energies[self.nnodes - 1] > self.energies[self.nnodes - 2]
                and fp > 0
                and self.nodes[self.nnodes - 1].gradrms > self.CONV_TOL
            ):
                printcool('Last node is not a minimum, Might need to verify that the last node is a minimum')
                self.optimizer[self.nnodes - 1].optimize(
                    molecule=self.nodes[self.nnodes - 1],
                    refE=refE,
                    opt_type='UNCONSTRAINED',
                    opt_steps=opt_steps,
                    ictan=None,
                )

    # Refactor this code!
    # TODO remove return form_TS hess  3/2021
    # align center of mass  and rotation

    def reparameterize(self, ic_reparam_steps=8, n0=0, nconstraints=1):
        """
        Reparameterize the string
        """
        if self.interp_method == 'DLC':
            self.ic_reparam(
                nodes=self.nodes,
                energies=self.energies,
                climbing=(self.climb or self.find),
                ic_reparam_steps=ic_reparam_steps,
                NUM_CORE=self.mp_cores,
            )
        return

        # TODO old GSM does this here
        # If failed, do exit 1

        # reset pgradrms ?

    # TODO Remove me does not deserve to be a function
    def find_peaks(self, rtype='opting'):
        """
        This doesnt actually calculate peaks, it calculates some other thing
        """
        # rtype 1: growing
        # rtype 2: opting
        # rtype 3: intermediate check
        if rtype not in ['growing', 'opting', 'intermediate']:
            raise RuntimeError

        if rtype == 'growing':
            nnodes = self.nR
        elif rtype == 'opting' or rtype == 'intermediate':
            nnodes = self.nnodes
        else:
            raise ValueError('find peaks bad input')
        alluptol = 0.1
        alluptol2 = 0.5
        allup = True
        diss = False
        energies = self.energies
        for n in range(1, len(energies[:nnodes])):
            if energies[n] + alluptol < energies[n - 1]:
                allup = False
                break

        if (
            energies[nnodes - 1] > 15.0
            and nnodes - 3 > 0
            and (energies[nnodes - 1] - energies[nnodes - 2]) < alluptol2
            and (energies[nnodes - 2] - energies[nnodes - 3]) < alluptol2
            and (energies[nnodes - 3] - energies[nnodes - 4]) < alluptol2
        ):
            logger.debug(' possible dissociative profile')
            diss = True

        logger.debug(f' nnodes  {nnodes}')
        logger.debug(f' all uphill?  {allup}')
        logger.debug(f' dissociative?  {diss}')
        npeaks2 = 0
        minnodes = []
        maxnodes = []
        if energies[1] > energies[0]:
            minnodes.append(0)
        if energies[nnodes - 1] < energies[nnodes - 2]:
            minnodes.append(nnodes - 1)
        for n in range(self.n0, nnodes - 1):
            if energies[n + 1] > energies[n] and energies[n] < energies[n - 1]:
                minnodes.append(n)
            if energies[n + 1] < energies[n] and energies[n] > energies[n - 1]:
                maxnodes.append(n)

        logger.debug(f' min nodes  {minnodes}')
        logger.debug(f' max nodes  {maxnodes}')
        len(maxnodes)
        ediff = 0.5
        PEAK4_EDIFF = 2.0
        if rtype == 'growing':
            ediff = 1.0
        if rtype == 'intermediate':
            ediff = PEAK4_EDIFF

        if rtype == 'growing':
            nmax = np.argmax(energies[: self.nR])
            emax = float(max(energies[: self.nR]))
        else:
            emax = float(max(energies))
            nmax = np.argmax(energies)

        logger.debug(f' emax and nmax in find peaks {emax:3.4f},{nmax} ')

        # check if any node after peak is less than 2 kcal below
        for n in maxnodes:
            diffs = (energies[n] - e > ediff for e in energies[n:nnodes])
            if any(diffs):
                found = n
                npeaks2 += 1
        npeaks = npeaks2
        logger.debug(f' found {npeaks} significant peak(s) TOL {ediff:3.2f}')

        # handle dissociative case
        if rtype == 'intermediate' and npeaks == 1:
            nextmin = 0
            for n in range(found, nnodes - 1):
                if n in minnodes:
                    nextmin = n
                    break
            if nextmin > 0:
                npeaks = 2

        if allup is True and npeaks == 0:
            return -1
        if diss is True and npeaks == 0:
            return -2

        return npeaks

    def print_energies(self, label: str = ' V_profile:', level: int = logging.INFO) -> None:
        """Log the string's relative energies (kcal/mol) on one line."""
        logger.log(level, label + ''.join(f' {float(e):7.3f}' for e in self.energies))

    def get_intermediate(self, noise):
        """
        Check string for intermediates
        noise is a leeway factor for determining intermediate
        """

        energies = self.energies
        potential_min = []
        for i in range(1, (len(energies) - 1)):
            rnoise = 0
            pnoise = 0
            a = 1
            b = 1
            while energies[i - a] >= energies[i]:
                if (energies[i - a] - energies[i]) > rnoise:
                    rnoise = energies[i - a] - energies[i]
                if rnoise > noise:
                    break
                if (i - a) == 0:
                    break
                a += 1

            while energies[i + b] >= energies[i]:
                if (energies[i + b] - energies[i]) > pnoise:
                    pnoise = energies[i + b] - energies[i]
                if pnoise > noise:
                    break
                if (i + b) == len(energies) - 1:
                    break
                b += 1
            if (rnoise > noise) and (pnoise > noise):
                logger.info(f'Potential minimum at image {i}')
                potential_min.append(i)

        return potential_min

    def has_intermediate(self, noise):
        pot_min = self.get_intermediate(noise)
        return len(pot_min) > 0

    def setup_from_geometries(
        self, input_geoms, reparametrize=True, restart_energies=True, start_climb_immediately=False
    ):
        """
        Restart
        input_geoms list of geometries
        reparameterize (boolean) : reparameterize the initial string to make the nodes equidistant
        restart_energies (boolean) : generate the initial energies
        start_climb_immediately (boolean) : set climb to True or False
        """

        printcool('Restarting GSM from geometries')
        self.growth_direction = 0
        nstructs = len(input_geoms)

        if nstructs != self.nnodes:
            logger.debug('need to interpolate')
            raise NotImplementedError
        else:
            geoms = input_geoms

        self.gradrms = [0.0] * nstructs
        self.dE = [1000.0] * nstructs

        self.isRestarted = True
        self.done_growing = True

        # set coordinates from geoms
        self.nodes[0].xyz = xyz_to_np(geoms[0])
        self.nodes[nstructs - 1].xyz = xyz_to_np(geoms[-1])
        for struct in range(1, nstructs - 1):
            self.nodes[struct] = Molecule.copy_from(
                self.nodes[struct - 1], xyz_to_np(geoms[struct]), new_node_id=struct, copy_wavefunction=False
            )
            self.nodes[struct].newHess = 5
            # Turning this off
            # self.nodes[struct].gradrms = np.sqrt(np.dot(self.nodes[struct].gradient,self.nodes
        self.nnodes = self.nR = nstructs

        if start_climb_immediately:
            # should check that this is a climber...
            self.climb = True
        # ALEX CHANGE - rearranged reparameterize and restart_energies 'if' blocks
        if restart_energies:
            self.print_energies()
        if reparametrize:
            printcool('Reparametrizing')
            self.reparameterize(ic_reparam_steps=8)

        self.ictan, self.dqmaga = self.get_tangents(self.nodes)
        self.refresh_coordinates()
        logger.debug(' setting all interior nodes to active')
        for n in range(1, self.nnodes - 1):
            self.active[n] = True
            self.optimizer[n].conv_grms = self.CONV_TOL * 2.5
            self.optimizer[n].options.DMAX = 0.05

        return
