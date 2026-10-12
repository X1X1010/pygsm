from __future__ import annotations

import logging

# standard library imports
# third party
from typing import TYPE_CHECKING

import numpy as np

from ..coords import CartesianCoordinates
from ..utils import block_matrix

# local application imports
from .base import BaseOptimizer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..molecule import Molecule


class BealesCG(BaseOptimizer):
    """Conjugate-gradient optimizer with Beale's restarts."""

    def optimize(
        self,
        molecule: Molecule,
        s0_prim,
        gp_prim,
        refE: float = 0.0,
        opt_type='BEALES_CG',
        opt_steps: int = 3,
        ictan=None,
        xyzframerate=4,
        verbose=False,
        trajectory: str | None = None,
    ) -> tuple[list, list]:
        logger.debug(f' initial E {molecule.energy - refE:5.4f}')

        # stash/initialize some useful attributes
        geoms = []
        energies = []
        geoms.append(molecule.geometry)
        energies.append(molecule.energy - refE)
        self.initial_step = True
        self.disp = 1000.0
        self.Ediff = 1000.0
        self.check_inputs(molecule, opt_type, ictan)
        nconstraints = self.get_nconstraints(opt_type)

        # for cartesian these are the same
        x = np.copy(molecule.coordinates)
        xyz = np.copy(molecule.xyz)

        # number of coordinates
        n = molecule.num_coordinates

        # Evaluate the function value and its gradient.
        fx = molecule.energy
        g = molecule.gradient

        # #if np.all(s0_prim==0.) or s0_prim==None:

        molecule.gradrms = np.sqrt(np.dot(g.T, g) / n).item()
        g_prim = block_matrix.dot(molecule.coord_basis, g)

        update_hess = False

        for ostep in range(opt_steps):
            logger.debug(f' On opt step {ostep + 1} ')

            if update_hess and self.update_hess_in_bg:
                if opt_type != 'TS':
                    self.update_Hessian(molecule, 'BFGS')
                else:
                    raise NotImplementedError
            update_hess = True

            if ostep == 0:
                # IF USE BEALES CG
                if nconstraints < 1:
                    dg = g_prim - gp_prim
                    h = dg / np.dot(s0_prim.T, dg)
                else:  # DONT USE BG
                    h = 0.0 * g_prim
                d_prim = -g_prim + np.dot(h.T, g_prim) * s0_prim  # the search direction
            else:
                dnew = g_prim
                deltanew = np.dot(dnew.T, dnew)
                deltaold = np.dot(gp_prim.T, gp_prim)
                beta = deltanew / deltaold
                d_prim = -g_prim + np.dot(h.T, g_prim) * s0_prim + beta * d_prim

            # form in DLC basis (does nothing if cartesian)
            d = block_matrix.dot(block_matrix.transpose(molecule.coord_basis), d_prim)

            # normalize the direction
            stepsize = np.linalg.norm(d)
            logger.debug(f' stepsize = {stepsize:1.2f}')
            d = d / stepsize  # normalize
            if stepsize > self.DMAX:
                stepsize = self.DMAX
                logger.debug(f' reducing step, new step = {stepsize:1.2f}')

            # store
            xp = x.copy()
            gp = g.copy()
            gp_prim = block_matrix.dot(molecule.coord_basis, gp)
            xyzp = xyz.copy()
            fxp = fx

            # => calculate constraint step <= #
            constraint_steps = self.get_constraint_steps(molecule, opt_type, g)

            # line search
            logger.debug(' Linesearch')
            ls = self.linesearch(
                n, x, fx, g, d, stepsize, xp, constraint_steps, self.linesearch_parameters, molecule, verbose
            )
            logger.debug(' Done linesearch')

            # save new values from linesearch
            molecule = ls['molecule']
            x = ls['x']
            fx = ls['fx']
            g = ls['g']
            step = ls['step']
            if nconstraints > 0:
                g = g - np.dot(g.T, molecule.constraints) * molecule.constraints

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

            # dE
            dEstep = fx - fxp
            logger.debug(f' dEstep={dEstep:5.4f}')

            # revert to the previous point
            if ls['status'] < 0 or dEstep > 0.0:
                x = xp.copy()
                molecule.xyz = xyzp
                g = gp.copy()
                fx = fxp
                logger.warning('[ERROR] the point return to the previous point')
                if nconstraints == 1:
                    return geoms, energies
                else:
                    nconstraints = 1
                    constraints = self.get_constraint_vectors(molecule, opt_type, ictan)
                    molecule.update_coordinate_basis(constraints=constraints)
                    x = np.copy(molecule.coordinates)
                    g = molecule.gradient.copy()
                    # project out the constraint
                    for c in molecule.constraints.T:
                        g -= np.dot(g.T, c[:, np.newaxis]) * c[:, np.newaxis]
                    step = 0.0
                    g_prim = block_matrix.dot(molecule.coord_basis, g)
                    h = 0.0 * g_prim
            else:
                # update molecule xyz
                xyz = molecule.update_xyz(x - xp)
                g_prim = block_matrix.dot(molecule.coord_basis, g)

            if ostep % xyzframerate == 0:
                geoms.append(molecule.geometry)
                energies.append(molecule.energy - refE)
                self.write_trajectory(trajectory, geoms, energies)

            # save variables for update Hessian!
            if not isinstance(molecule.coord_obj, CartesianCoordinates) or self.update_hess_in_bg:
                self.dx_prim = molecule.coord_obj.Prims.calcDiff(xyz, xyzp)
                self.dx_prim = np.reshape(self.dx_prim, (-1, 1))
                self.dg_prim = g_prim - gp_prim
            else:
                raise NotImplementedError(' ef not implemented for CART')

            if self.print_level > 0:
                logger.info(
                    f' Opt step: {ostep + 1} E: {fx - refE:5.4f} gradrms: {molecule.gradrms:1.5f} ss: {step:1.3f} DMAX: {self.DMAX:1.3f}'
                )

            # check for convergence TODO
            molecule.gradrms = np.sqrt(np.dot(g.T, g) / n).item()
            gmax = float(np.max(g))
            disp = float(np.linalg.norm((xyz - xyzp).flatten()))
            logger.debug(f' gmax {gmax:5.4f} disp {disp:5.4f} Ediff {dEstep:5.4f} gradrms {molecule.gradrms:5.4f}')

            self.converged = False
            dE = molecule.difference_energy if self.opt_cross else None
            if (
                self.opt_cross
                and abs(dE) < self.conv_dE
                and molecule.gradrms < self.conv_grms
                and abs(gmax) < self.conv_gmax
                and abs(dEstep) < self.conv_Ediff
                and abs(disp) < self.conv_disp
                and ls['status'] == 0
            ) or (
                not self.opt_cross
                and molecule.gradrms < self.conv_grms
                and abs(gmax) < self.conv_gmax
                and abs(dEstep) < self.conv_Ediff
                and abs(disp) < self.conv_disp
            ):
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
                logger.debug(' updating DLC')
                constraints = self.get_constraint_vectors(molecule, opt_type, ictan)
                molecule.update_coordinate_basis(constraints=constraints)
                x = np.copy(molecule.coordinates)
                fx = molecule.energy
                dE = molecule.difference_energy
                if dE != 1000.0:
                    logger.debug(f' difference energy is {dE:5.4f}')
                g = molecule.gradient.copy()
                if nconstraints > 0:
                    g = g - np.dot(g.T, molecule.constraints) * molecule.constraints
                g_prim = block_matrix.dot(molecule.coord_basis, g)
                logger.debug(' Done update')

        return geoms, energies
