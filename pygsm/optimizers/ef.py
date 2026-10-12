from __future__ import annotations

import logging

# standard library imports
# third party
from typing import TYPE_CHECKING

import numpy as np

from ..coords import CartesianCoordinates
from ..utils import block_matrix, units
from .base import BaseOptimizer

# local application imports
from .linesearch import NoLineSearch

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..molecule import Molecule


class EigenvectorFollow(BaseOptimizer):
    """Eigenvector-following (rational function) optimizer with BFGS Hessian updates; supports climbing and TS search.

    Parameters: see :class:`BaseOptimizer`.
    """

    supports_ts_search = True

    def optimize(
        self,
        molecule: Molecule,
        refE: float = 0.0,
        opt_type: str = 'UNCONSTRAINED',
        opt_steps: int = 3,
        ictan=None,
        xyzframerate=4,
        verbose=False,
        trajectory: str | None = None,
    ) -> tuple[list, list]:
        # stash/initialize some useful attributes
        self.check_inputs(molecule, opt_type, ictan)
        nconstraints = self.get_nconstraints(opt_type)

        logger.debug(f' initial E {molecule.energy - refE:5.4f}')
        logger.debug(f' CONV_TOL {self.conv_grms:1.5f}')
        geoms = []
        energies = []
        geoms.append(molecule.geometry)
        energies.append(molecule.energy - refE)
        self.converged = False

        # form initial coord basis
        if opt_type != 'TS':
            constraints = self.get_constraint_vectors(molecule, opt_type, ictan)
            molecule.update_coordinate_basis(constraints=constraints)
            molecule.form_Hessian_in_basis()

        # Evaluate the function value and its gradient.
        fx = molecule.energy
        g = molecule.gradient.copy()
        # project out the constraint
        gc = g.copy()
        for c in molecule.constraints.T:
            gc -= np.dot(gc.T, c[:, np.newaxis]) * c[:, np.newaxis]
        gmax = float(np.max(np.absolute(gc)))

        if self.check_only_grad_converged:
            if molecule.gradrms < self.conv_grms and gmax < self.conv_gmax:
                self.converged = True
                return geoms, energies
            else:
                self.check_only_grad_converged = False

        # for cartesian these are the same
        x = np.copy(molecule.coordinates)
        xyz = np.copy(molecule.xyz)

        if opt_type == 'TS':
            self.linesearch = NoLineSearch
        if opt_type == 'SEAM' or opt_type == 'MECI' or opt_type == 'TS-SEAM':
            self.opt_cross = True

        # TODO are these used? -- n is used for gradrms,linesearch
        if isinstance(molecule.coord_obj, CartesianCoordinates):
            n = molecule.num_coordinates
        else:
            n_actual = molecule.num_coordinates
            n = n_actual - nconstraints
            self.x_prim = np.zeros((molecule.num_primitives, 1), dtype=float)
            self.g_prim = np.zeros((molecule.num_primitives, 1), dtype=float)

        molecule.gradrms = np.sqrt(np.dot(gc.T, gc) / n).item()
        dE = molecule.difference_energy
        update_hess = False

        # ====>  Do opt steps <======= #
        for ostep in range(opt_steps):
            logger.debug(f' On opt step {ostep + 1} for node {molecule.node_id}')

            # update Hess
            if update_hess:
                if opt_type != 'TS':
                    self.update_Hessian(molecule, 'BFGS')
                else:
                    self.update_Hessian(molecule, 'BOFILL')
            update_hess = True

            # => Form eigenvector step <= #
            if isinstance(molecule.coord_obj, CartesianCoordinates):
                raise NotImplementedError
            else:
                if opt_type != 'TS':
                    dq = self.eigenvector_step(molecule, gc)
                else:
                    dq = self.TS_eigenvector_step(molecule, g, ictan)
                    if not self.maxol_good:
                        logger.debug(' Switching to climb! Maxol not good!')
                        nconstraints = 1
                        opt_type = 'CLIMB'

            actual_step = np.linalg.norm(dq)
            dq = dq / actual_step  # normalize
            if actual_step > self.DMAX:
                step = self.DMAX
            else:
                step = actual_step

            # store values
            xp = x.copy()
            gp = g.copy()
            xyzp = xyz.copy()
            fxp = fx
            pgradrms = molecule.gradrms
            if not isinstance(molecule.coord_obj, CartesianCoordinates):
                gp_prim = self.g_prim.copy()

            # => calculate constraint step <= #
            constraint_steps = self.get_constraint_steps(molecule, opt_type, g)

            ls = self.linesearch(
                nconstraints, x, fx, g, dq, step, xp, constraint_steps, self.linesearch_parameters, molecule, verbose
            )

            # get values from linesearch
            molecule = ls['molecule']
            step = ls['step']
            x = ls['x']
            fx = ls['fx']
            g = ls['g']

            if ls['status'] == -2:
                logger.warning('[ERROR] the point return to the privious point')
                x = xp.copy()
                molecule.xyz = xyzp
                g = gp.copy()
                fx = fxp
                ratio = 0.0
                molecule.newHess = 5

            if ls['step'] > self.DMAX:
                if ls['step'] <= self.abs_max_step:  # absolute max
                    logger.debug(' Increasing DMAX to {}'.format(ls['step']))
                    self.DMAX = ls['step']
                else:
                    self.DMAX = self.abs_max_step
            elif ls['step'] < self.DMAX:
                if ls['step'] >= self.DMIN:  # absolute min
                    logger.debug(' Decreasing DMAX to {}'.format(ls['step']))
                    self.DMAX = ls['step']
                elif ls['step'] <= self.DMIN:
                    self.DMAX = self.DMIN
                    logger.debug(f' Decreasing DMAX to {self.DMIN}')

            # calculate predicted value from Hessian, gp is previous constrained gradient
            scaled_dq = dq * step
            dEtemp = np.dot(self.Hessian, scaled_dq)
            dEpre = np.dot(np.transpose(scaled_dq), gc) + 0.5 * np.dot(np.transpose(dEtemp), scaled_dq)
            dEpre *= units.KCAL_MOL_PER_AU
            constraint_energy = np.dot(gp.T, constraint_steps) * units.KCAL_MOL_PER_AU
            dEpre = (dEpre + constraint_energy).item()

            # project out the constraint
            gc = g.copy()
            for c in molecule.constraints.T:
                gc -= np.dot(gc.T, c[:, np.newaxis]) * c[:, np.newaxis]

            # control step size
            dEstep = fx - fxp
            logger.debug(f' dEstep={dEstep:5.4f}')
            ratio = dEstep / dEpre
            molecule.gradrms = np.sqrt(np.dot(gc.T, gc) / n).item()
            if ls['status'] != -2:
                self.step_controller(actual_step, ratio, molecule.gradrms, pgradrms, dEpre, opt_type, dEstep)

            # update molecule xyz
            xyz = molecule.update_xyz(x - xp)
            if ostep % xyzframerate == 0:
                geoms.append(molecule.geometry)
                energies.append(molecule.energy - refE)
                self.write_trajectory(trajectory, geoms, energies)

            # save variables for update Hessian!
            if not isinstance(molecule.coord_obj, CartesianCoordinates):
                # only form g_prim for non-constrained
                self.g_prim = block_matrix.dot(molecule.coord_basis, gc)
                self.dx = x - xp
                self.dg = g - gp

                self.dx_prim_actual = molecule.coord_obj.Prims.calcDiff(xyz, xyzp)
                self.dx_prim_actual = np.reshape(self.dx_prim_actual, (-1, 1))
                self.dx_prim = block_matrix.dot(molecule.coord_basis, scaled_dq)
                self.dg_prim = self.g_prim - gp_prim

            else:
                raise NotImplementedError(' ef not implemented for CART')

            if self.print_level > 0:
                logger.info(
                    f' Node: {molecule.node_id} Opt step: {ostep + 1} E: {fx - refE:5.4f} predE: {dEpre:5.4f} ratio: {ratio:1.3f} gradrms: {molecule.gradrms:1.5f} ss: {step:1.3f} DMAX: {self.DMAX:1.3f}'
                )

            # check for convergence TODO
            fx = molecule.energy
            dE = molecule.difference_energy
            if dE < 1000.0:
                logger.debug(f' difference energy is {dE:5.4f}')
            gmax = float(np.max(np.absolute(gc)))
            disp = float(np.linalg.norm((xyz - xyzp).flatten()))
            xnorm = np.sqrt(np.dot(x.T, x))
            if xnorm < 1.0:
                xnorm = 1.0

            logger.debug(f' gmax {gmax:5.4f} disp {disp:5.4f} Ediff {dEstep:5.4f} gradrms {molecule.gradrms:5.4f}')

            # TODO turn back on conv_DE
            if (
                self.opt_cross
                and abs(dE) < self.conv_dE
                and molecule.gradrms < self.conv_grms
                and abs(gmax) < self.conv_gmax
                and abs(dEstep) < self.conv_Ediff
                and abs(disp) < self.conv_disp
            ):
                if opt_type == 'TS-SEAM':
                    gts = np.dot(g.T, molecule.constraints[:, 0])
                    logger.debug(f' gts {gts:1.4f}')
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
                    if abs(gts) < self.conv_grms * 5.0:
                        self.converged = True
                elif opt_type == 'TS':
                    if self.gtse < self.conv_grms * 5.0:
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
            if not isinstance(molecule.coord_obj, CartesianCoordinates) and opt_type != 'TS':
                constraints = self.get_constraint_vectors(molecule, opt_type, ictan)
                molecule.update_coordinate_basis(constraints=constraints)
                x = np.copy(molecule.coordinates)
                g = molecule.gradient.copy()
                # project out the constraint
                gc = g.copy()
                for c in molecule.constraints.T:
                    gc -= np.dot(gc.T, c[:, np.newaxis]) * c[:, np.newaxis]

        return geoms, energies
