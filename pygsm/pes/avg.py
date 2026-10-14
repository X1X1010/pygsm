# third party
import logging

import numpy as np

# local application imports
from ..utils import units
from .pes import PES

logger = logging.getLogger(__name__)


class AvgPES(PES):
    """Average of two adiabatic surfaces of one level of theory, with derivative coupling (seam/MECI searches)."""

    # TODO can fix this up so it automatically initializes PES1 and PES2?
    def __init__(
        self,
        pes1,
        pes2,
        lot,
    ):
        self.pes1 = pes1.clone(lot=lot)
        self.pes2 = pes2.clone(lot=lot)
        self._dE = 1000.0
        self.lot = lot
        self.lot.do_coupling = True
        self.lot.coupling_states = (pes1.ad_idx, pes2.ad_idx)

    def get_energy(self, xyz):
        if self.pes1.multiplicity == self.pes2.multiplicity:
            assert self.pes2.ad_idx > self.pes1.ad_idx, 'dgrad wrong direction'
        self.dE = self.pes2.get_energy(xyz) - self.pes1.get_energy(xyz)
        return 0.5 * (self.pes1.get_energy(xyz) + self.pes2.get_energy(xyz))

    def get_gradient(self, xyz, frozen_atoms=None):
        return 0.5 * (self.pes1.get_gradient(xyz, frozen_atoms) + self.pes2.get_gradient(xyz, frozen_atoms))

    def get_coupling(self, xyz, frozen_atoms=None):
        assert self.pes1.multiplicity == self.pes2.multiplicity, 'coupling is 0'
        assert self.pes1.ad_idx != self.pes2.ad_idx, 'coupling is 0'
        return np.reshape(
            self.lot.get_coupling(xyz, self.pes1.multiplicity, self.pes1.ad_idx, self.pes2.ad_idx, frozen_atoms),
            (-1, 1),
        )

    def get_dgrad(self, xyz, frozen_atoms=None):
        if self.pes1.multiplicity == self.pes2.multiplicity:
            assert self.pes2.ad_idx > self.pes1.ad_idx, 'dgrad wrong direction'
        return self.pes2.get_gradient(xyz, frozen_atoms) - self.pes1.get_gradient(xyz, frozen_atoms)

    def get_average_gradient(self, xyz, frozen_atoms=None):
        if self.pes1.multiplicity == self.pes2.multiplicity:
            assert self.pes2.ad_idx > self.pes1.ad_idx, 'dgrad wrong direction'
        return 0.5 * (self.pes2.get_gradient(xyz, frozen_atoms) + self.pes1.get_gradient(xyz, frozen_atoms))

    # http://pubs.acs.org/doi/abs/10.1021/acs.jctc.6b00384
    def symmetric_orthogonalization(self, xyz):
        def get_beta(g, h):
            gdoth = np.dot(g.T, h)
            logger.debug(f' g*h = {gdoth:1.4f}')
            dotg = np.dot(g.T, g)
            doth = np.dot(h.T, h)
            arg = (2 * gdoth) / (dotg - doth)
            return 0.5 * np.arctan(arg)

        def calc_pitch(g, h):
            dotg = np.dot(g.T, g)
            doth = np.dot(h.T, h)
            return np.sqrt(0.5 * (dotg + doth))

        def calc_asymmetry(g, h):
            dotg = np.dot(g.T, g)
            doth = np.dot(h.T, h)
            return (dotg - doth) / (dotg + doth)

        # get vectors
        dgrad = 0.5 * self.get_dgrad(xyz)
        dvec = self.get_coupling(xyz)
        sab = self.get_average_gradient(xyz)
        logger.debug(f' dE is {self.dE:5.4f}')

        # non-adiabatic coupling is derivative coupling times dE
        nac = dvec * self.dE / units.KCAL_MOL_PER_AU
        beta = get_beta(dgrad, nac)
        logger.debug(f' beta = {beta:1.6f}')

        # rotate nac and dgrad to be orthonormal
        save_nac = nac.copy()
        nac = nac * np.cos(beta) - dgrad * np.sin(beta)
        dgrad = dgrad * np.cos(beta) - save_nac * np.sin(beta)

        norm_nac = np.linalg.norm(nac)
        norm_dg = np.linalg.norm(dgrad)

        pitch = calc_pitch(dgrad, nac)
        asymmetry = calc_asymmetry(dgrad, nac)
        if asymmetry < 0.0:
            # swap nac and dgrad
            dgrad_copy = dgrad.copy()
            dgrad = nac.copy()
            nac = dgrad_copy
            norm_nac = np.linalg.norm(nac)
            norm_dg = np.linalg.norm(dgrad)
            asymmetry = calc_asymmetry(dgrad, nac)

        dotx = np.dot(sab.T, dgrad) / norm_dg
        doty = np.dot(sab.T, nac) / norm_nac
        sab_x = dotx / pitch
        sab_y = doty / pitch

        # flip direction of dgrad,nac
        logger.debug(f' sab_x {sab_x:1.2f} sab_y {sab_y:1.2f}')
        if sab_x < 0.0 and sab_y > 0.0:
            logger.debug(' flipping dgrad')
            dgrad = -dgrad
            dotx = np.dot(sab.T, dgrad) / norm_dg
            sab_x = dotx / pitch
        elif sab_x > 0.0 and sab_y < 0.0:
            logger.debug(' flipping nac')
            nac = -nac
            doty = np.dot(sab.T, nac) / norm_nac
            sab_y = doty / pitch

        sigma = np.sqrt(sab_x * sab_x + sab_y * sab_y)
        theta_s = np.arctan(sab_y / sab_x)
        logger.debug(
            f' pitch  {pitch:1.5f}, asymmetry {asymmetry:1.5f}, sigma {sigma:1.5f} theta_s {theta_s:1.5f} sx {sab_x:1.2f} sy {sab_y:1.2f}'
        )

        P = ((sigma**2.0) / (1.0 - asymmetry**2.0)) * (1.0 - asymmetry * np.cos(2 * theta_s))
        B = ((sigma**2) / (4 * asymmetry**2)) ** (1.0 / 3) * (
            ((1.0 + asymmetry) * (np.cos(theta_s)) ** 2) ** (1.0 / 3)
            + ((1 - asymmetry) * (np.sin(theta_s)) ** 2) ** (1.0 / 3)
        )
        logger.debug(f' P {P:2.5f} B {B:2.5f}')

        if P < 1.0:
            logger.debug(' Your conical intersection is peaked')
        elif P > 1.0:
            logger.debug(' Your conical intersection is sloped')
        if B < 1.0:
            logger.debug(' Your conical intersection is bifurcating')
        if B > 1.0:
            logger.debug(' Your conical intersection is single path')

        return {
            'x': dgrad / norm_dg,
            'y': nac / norm_nac,
            'pitch': pitch,
            'asymmetry': asymmetry,
            'sigma': sigma,
            'theta_s': theta_s,
            'sab_x': sab_x,
            'sab_y': sab_y,
        }
