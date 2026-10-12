from __future__ import annotations

import logging

# standard library imports
# third party
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import LbfgsInvHessProduct

from ..coordinate_systems import CartesianCoordinates
from ..utilities import block_matrix, units

# local application imports
from .base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..molecule import Molecule


class IterationData:
    """docstring for iterationData"""

    def __init__(self, alpha, s, y):
        self.alpha = alpha
        self.s_prim = s  # step
        self.y_prim = y  # diff in grad


class LBFGS(BaseOptimizer):
    """Limited-memory BFGS optimizer in delocalized internal coordinates."""

    def __init__(self, options=None, **kwargs):
        super().__init__(options, **kwargs)
        self.k = 0
        self.end = 0

    def optimize(
        self,
        molecule: Molecule,
        refE: float = 0.0,
        opt_type: str = 'UNCONSTRAINED',
        opt_steps: int = 20,
        maxcor=10,
        ictan=None,
        xyzframerate=4,
        verbose=False,
        trajectory: str | None = None,
    ) -> tuple[list, list]:
        # stash/initialize some useful attributes
        logger.debug(f' initial E {molecule.energy - refE:5.4f}')
        geoms = []
        energies = []
        geoms.append(molecule.geometry)
        energies.append(molecule.energy - refE)
        self.check_inputs(molecule, opt_type, ictan)
        nconstraints = self.get_nconstraints(opt_type)

        # form initial coord basis
        constraints = self.get_constraint_vectors(molecule, opt_type, ictan)
        molecule.update_coordinate_basis(constraints=constraints)

        if opt_type == 'SEAM' or opt_type == 'MECI':
            self.opt_cross = True

        # get coordinates
        x = np.copy(molecule.coordinates)
        xyz = np.copy(molecule.xyz)
        num_coords = molecule.num_coordinates - nconstraints - molecule.num_frozen_atoms * 3

        # Evaluate the function value and its gradient.
        fx = molecule.energy
        g = molecule.gradient.copy()

        # project out the constraint
        gc = g.copy()
        for c in molecule.constraints.T:
            gc -= np.dot(gc.T, c[:, np.newaxis]) * c[:, np.newaxis]

        g_prim = block_matrix.dot(molecule.coord_basis, gc)
        molecule.gradrms = np.sqrt(np.dot(gc.T, gc) / num_coords).item()

        # primitive constraint step
        self.cstep_prim = np.zeros_like(g_prim)
        dE = molecule.difference_energy

        # Not checking convergence here anymore ...

        # # reset k in principle k does not have to reset but . . .
        # TRY Turning off Feb 2020
        if opt_type != 'CLIMB':
            self.k = 0
            self.end = 0

        # initialize the iteration data list
        if self.k == 0:
            self.lm = []
            for _i in range(0, maxcor):
                s_prim = np.zeros_like(g_prim)
                y_prim = np.zeros_like(g_prim)
                self.lm.append(IterationData(0.0, s_prim.flatten(), y_prim.flatten()))

        for ostep in range(opt_steps):
            logger.debug(f' On opt step {ostep + 1} ')

            SCALE = self.options.SCALEQN
            if molecule.newHess > 0:
                SCALE = self.options.SCALEQN * molecule.newHess

            if self.k != 0:
                # update vectors s and y:
                # TODO this doesn't work exactly with constraint steps
                self.lm[self.end].s_prim = molecule.coord_obj.Prims.calcDiff(xyz, self.xyzp) - self.cstep_prim.flatten()
                self.lm[self.end].y_prim = g_prim - self.gp_prim

                self.end = (self.end + 1) % maxcor
                bound = min(self.k, maxcor)
                s_prim = np.array([self.lm[i].s_prim.flatten() for i in range(maxcor)])
                y_prim = np.array([self.lm[i].y_prim.flatten() for i in range(maxcor)])
                hess_inv = LbfgsInvHessProduct(s_prim[:bound], y_prim[:bound])
                # compute the negative gradients
                d_prim = -g_prim
                # perform matrix product
                d_prim = hess_inv._matvec(d_prim)
                d_prim = np.reshape(d_prim, (-1, 1)) / SCALE
            else:
                # d: store the negative gradient of the object function on point x.
                d_prim = -g_prim / SCALE
            self.k = self.k + 1

            # form in DLC basis (does nothing if cartesian)
            d = block_matrix.dot(block_matrix.transpose(molecule.coord_basis), d_prim)

            # normalize the direction
            actual_step = np.linalg.norm(d)
            logger.debug(f' actual_step= {actual_step:1.5f}')
            d = d / actual_step  # normalize
            if actual_step > self.DMAX:
                step = self.DMAX
                logger.debug(f' reducing step, new step = {step:1.5f}')
            else:
                step = actual_step

            # store
            xp = x.copy()
            self.xyzp = xyz.copy()
            gp = g.copy()
            self.gp_prim = block_matrix.dot(molecule.coord_basis, gc)
            fxp = fx
            pgradrms = molecule.gradrms

            # => calculate constraint step <= #
            constraint_steps = self.get_constraint_steps(molecule, opt_type, g)
            self.cstep_prim = block_matrix.dot(molecule.coord_basis, constraint_steps)

            # line search
            ls = self.Linesearch(
                nconstraints, x, fx, gc, d, step, xp, constraint_steps, self.linesearch_parameters, molecule, verbose
            )

            # save new values from linesearch
            molecule = ls['molecule']
            step = ls['step']
            x = ls['x']
            fx = ls['fx']
            g = ls['g']

            dEstep = fx - fxp
            dq = x - xp

            # TODO dEpre is missing second order effects or is it?
            dEpre = np.dot(gc.T, dq) * units.KCAL_MOL_PER_AU
            constraint_energy = np.dot(gp.T, constraint_steps) * units.KCAL_MOL_PER_AU
            if opt_type not in ['UNCONSTRAINED', 'ICTAN']:
                logger.debug(f'constraint_energy: {constraint_energy.item():1.4f}')
            dEpre = (dEpre + constraint_energy).item()

            ratio = dEstep / dEpre
            logger.debug(f' dEstep={dEstep:5.4f}')
            logger.debug(f' dEpre={dEpre:5.4f}')
            logger.debug(f' ratio={ratio:5.4f}')

            # revert to the privious point
            if ls['status'] < 0 or (ratio < 0.0 and opt_type != 'CLIMB'):
                if ratio < 0.0:
                    ls['status'] = -1
                x = xp.copy()
                molecule.xyz = self.xyzp.copy()
                g = gp.copy()
                fx = fxp
                ratio = 0.0
                dEstep = 0.0

                logger.warning('[ERROR] the point return to the previous point')
                self.lm = []
                for _i in range(0, maxcor):
                    s_prim = np.zeros_like(g_prim)
                    y_prim = np.zeros_like(g_prim)
                    self.lm.append(IterationData(0.0, s_prim.flatten(), y_prim.flatten()))
                self.k = 0
                self.end = 0
                molecule.newHess = 5
                if self.DMAX <= self.DMIN:
                    logger.debug(' Reached minimum step,exiting')
                    geoms.append(molecule.geometry)
                    energies.append(molecule.energy - refE)
                    break

                self.DMAX = ls['step'] / 2
                if self.DMAX < self.DMIN:
                    self.DMAX = self.DMIN
            else:
                # update molecule xyz
                xyz = molecule.update_xyz(x - xp)

            # if ratio is less than 0.3 than reduce DMAX
            flag = True
            if ratio < 0.3 and ls['status'] == 0:  # and abs(dEpre)>0.05:
                logger.debug(' Reducing DMAX')
                self.DMAX /= 1.5
                if self.DMAX < self.DMIN:
                    self.DMAX = self.DMIN
                if molecule.newHess < 5:
                    molecule.newHess += 1
                flag = False

                if opt_type == 'CLIMB' and self.SCALE_CLIMB < 10.0 and opt_type == 'CLIMB':
                    self.SCALE_CLIMB += 1.0
                    logger.debug(f'SCALING CLIMB BY {self.SCALE_CLIMB}')
            elif ratio > 0.3:
                molecule.newHess -= 1
                if opt_type == 'CLIMB' and self.SCALE_CLIMB > 1.0:
                    self.SCALE_CLIMB -= 1.0

            # project out the constraints
            gc = g.copy()
            for c in molecule.constraints.T:
                gc -= np.dot(gc.T, c[:, np.newaxis]) * c[:, np.newaxis]
            g_prim = block_matrix.dot(molecule.coord_basis, gc)

            dE = molecule.difference_energy
            if dE < 100.0:
                logger.debug(f' difference energy is {dE:5.4f}')
            molecule.gradrms = np.sqrt(np.dot(gc.T, gc) / num_coords).item()

            # control step size  NEW FEB 2020
            if flag:
                if ls['status'] == 0:  # passed
                    dgradrms = molecule.gradrms - pgradrms
                    logger.debug(f'dgradrms  {dgradrms}')
                    if ls['step'] > self.DMAX:
                        if ls['step'] <= self.options.abs_max_step:  # absolute max
                            logger.debug(' Increasing DMAX to {}'.format(ls['step']))
                            self.DMAX = ls['step']
                        else:
                            self.DMAX = self.options.abs_max_step
                    elif ls['step'] < self.DMAX:
                        if ls['step'] >= self.DMIN:  # absolute min
                            logger.debug(' Decreasing DMAX to {}'.format(ls['step']))
                            self.DMAX = ls['step']
                        elif ls['step'] <= self.DMIN:
                            self.DMAX = self.DMIN
                            logger.debug(f' Decreasing DMAX to {self.DMIN}')
                    elif ratio > 0.85 and ratio < 1.1 and actual_step > self.DMAX and dgradrms < -0.00005:
                        logger.debug(' HERE increasing DMAX')
                        self.DMAX *= 1.1
                        if self.options.abs_max_step < self.DMAX:
                            self.DMAX = self.options.abs_max_step
                else:
                    logger.debug('status not zero')

            if ostep % xyzframerate == 0:
                geoms.append(molecule.geometry)
                energies.append(molecule.energy - refE)
                self.write_trajectory(trajectory, geoms, energies)

            if self.options.print_level > 0:
                logger.info(
                    f' Node: {molecule.node_id} Opt step: {ostep + 1} E: {fx - refE:5.4f} predE: {dEpre:5.4f} ratio: {ratio:1.3f} gradrms: {molecule.gradrms:1.5f} ss: {step:1.3f} DMAX: {self.DMAX:1.3f}'
                )

            gmax = float(np.max(np.absolute(gc)))
            disp = float(np.linalg.norm((xyz - self.xyzp).flatten()))
            logger.debug(f' gmax {gmax:5.4f} disp {disp:5.4f} dEstep {dEstep:5.4f} gradrms {molecule.gradrms:5.4f}')
            self.converged = False
            if (
                self.opt_cross
                and abs(dE) < self.conv_dE
                and molecule.gradrms < self.conv_grms
                and abs(gmax) < self.conv_gmax
                and abs(dEstep) < self.conv_Ediff
                and abs(disp) < self.conv_disp
                and ls['status'] == 0
            ):
                # TODO Seam Climb
                if opt_type == 'TS-SEAM':
                    gts = np.dot(g.T, molecule.constraints[:, 0])
                    if abs(gts) < self.conv_grms * 5:
                        self.converged = True
                else:
                    self.converged = True

            elif (
                not self.opt_cross
                and molecule.gradrms < self.conv_grms
                and abs(gmax) < self.conv_gmax
                and abs(dEstep) < self.conv_Ediff
                and abs(disp) < self.conv_disp
            ):
                if opt_type == 'CLIMB':
                    gts = np.dot(g.T, molecule.constraints[:, 0])
                    if abs(gts) < self.conv_grms * 5:
                        self.converged = True
                else:
                    self.converged = True

            if self.converged:
                logger.debug(' converged')
                if ostep % xyzframerate != 0:
                    geoms.append(molecule.geometry)
                    energies.append(molecule.energy - refE)
                    self.write_trajectory(trajectory, geoms, energies)
                break

            # update DLC  --> this changes q, g, Hint
            if not isinstance(molecule.coord_obj, CartesianCoordinates) and opt_type in ('SEAM', 'MECI'):
                constraints = self.get_constraint_vectors(molecule, opt_type, ictan)
                molecule.update_coordinate_basis(constraints=constraints)
                x = np.copy(molecule.coordinates)
                g = molecule.gradient.copy()
                # project out the constraint
                gc = g.copy()
                for c in molecule.constraints.T:
                    gc -= np.dot(gc.T, c[:, np.newaxis]) * c[:, np.newaxis]
                g_prim = block_matrix.dot(molecule.coord_basis, gc)

        return geoms, energies
