from __future__ import annotations

import logging

# standard library imports
# third party
import numpy as np

from ..coordinate_systems import CartesianCoordinates
from ..utilities import units

# local application imports
from .base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)


class ConjugateGradient(BaseOptimizer):
    """Plain conjugate-gradient optimizer."""

    def optimize(self, molecule, refE=0.0, opt_type='UNCONSTRAINED', opt_steps=3, ictan=None):
        logger.debug(f' initial E {molecule.energy - refE:5.4f}')
        if opt_type == 'TS':
            raise RuntimeError

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

        # form initial coord basis
        constraints = self.get_constraint_vectors(molecule, opt_type, ictan)
        molecule.update_coordinate_basis(constraints=constraints)

        # for cartesian these are the same
        x = np.copy(molecule.coordinates)

        # number of coordinates
        if isinstance(molecule.coord_obj, CartesianCoordinates):
            n = molecule.num_coordinates
        else:
            n_actual = molecule.num_coordinates
            n = n_actual - nconstraints

        # Evaluate the function value and its gradient.
        fx = molecule.energy
        g = molecule.gradient.copy()
        g_prim = np.dot(molecule.coord_basis, g)
        gp = g.copy()
        gp_prim = np.dot(molecule.coord_basis, gp)

        molecule.gradrms = np.sqrt(np.dot(g[nconstraints:].T, g[nconstraints:]) / n).item()
        if molecule.gradrms < self.conv_grms:
            logger.debug(' already at min')
            return geoms, energies

        for ostep in range(opt_steps):
            logger.debug(f' On opt step {ostep + 1} ')

            if self.initial_step is True:
                # d: store the negative gradient of the object function on point x.
                # compute the initial step
                d_prim = -g_prim
                # set initial step to false
                self.initial_step = False
            else:
                # Fletcher-Reeves formula for Beta
                # http://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method
                dnew = -g_prim  #
                deltanew = np.dot(dnew.T, dnew)
                deltaold = np.dot(-gp_prim.T, -gp_prim)
                beta = deltanew / deltaold
                logger.debug(f' beta = {beta:1.2f}')
                d_prim = dnew + beta * d_prim

            # form in DLC basis (does nothing if cartesian)
            d = np.dot(molecule.coord_basis.T, d_prim)

            # normalize the direction
            actual_step = np.linalg.norm(d)
            logger.debug(f' actual_step= {actual_step:1.2f}')
            d = d / actual_step  # normalize
            if actual_step > self.DMAX:
                step = self.DMAX
                logger.debug(f' reducing step, new step = {step:1.2f}')
            else:
                step = actual_step

            # store
            xp = x.copy()
            gp = g.copy()
            gp_prim = np.dot(molecule.coord_basis, gp)
            fxp = fx

            # => calculate constraint step <= #
            constraint_steps = self.get_constraint_steps(molecule, opt_type, g)

            # line search
            logger.debug(' Linesearch')
            ls = self.linesearch(n, x, fx, g, d, step, xp, gp, constraint_steps, self.linesearch_parameters, molecule)
            logger.debug(' Done linesearch')

            # revert to the previous point
            if ls['status'] < 0:
                x = xp.copy()
                logger.warning('[ERROR] the point return to the previous point')
                return ls['status']

            # get values from linesearch
            p_step = step
            step = ls['step']
            x = ls['x']
            fx = ls['fx']
            g = ls['g']
            g_prim = np.dot(molecule.coord_basis, g)

            # control step size
            if step < p_step:
                self.DMAX /= 2.0
            elif step > p_step:
                self.DMAX *= 2.0
            if self.DMAX < 0.01:
                self.DMAX = 0.01
            elif self.DMAX > 0.25:
                self.DMAX = 0.25

            # dE
            dEstep = fx - fxp
            logger.debug(f' dEstep={dEstep:5.4f}')

            # update molecule xyz
            geoms.append(molecule.geometry)
            energies.append(molecule.energy - refE)

            if self.print_level > 0:
                logger.info(
                    f' Opt step: {ostep + 1} E: {fx - refE:5.4f} gradrms: {molecule.gradrms:1.5f} ss: {step:1.3f} DMAX: {self.DMAX:1.3f}'
                )

            gmax = np.max(g) / units.ANGSTROM_TO_AU
            self.disp = np.max(x - xp) / units.ANGSTROM_TO_AU
            self.Ediff = fx - fxp / units.KCAL_MOL_PER_AU
            logger.debug(f' maximum displacement component {self.disp:1.2f} (au)')
            logger.debug(f' maximum gradient component {gmax:1.2f} (au)')

            # check for convergence TODO
            molecule.gradrms = np.sqrt(np.dot(g[nconstraints:].T, g[nconstraints:]) / n).item()
            if molecule.gradrms < self.conv_grms:
                break

            # check if finished
            #    (gmax <= params.conv_gmax and abs(self.Ediff) <= params.conv_Ediff):
            #    #print self.Ediff
            #    #print self.disp

            # update DLC  --> this changes q, g, Hint
            if not isinstance(molecule.coord_obj, CartesianCoordinates):
                constraints = self.get_constraint_vectors(molecule, opt_type, ictan)
                molecule.update_coordinate_basis(constraints=constraints)
                x = np.copy(molecule.coordinates)
                fx = molecule.energy
                dE = molecule.difference_energy
                if dE != 1000.0:
                    logger.debug(f' difference energy is {dE:5.4f}')
                g = molecule.gradient.copy()
                molecule.form_Hessian_in_basis()

        return geoms, energies
