# third party
import numpy as np

# local application imports
from ..utilities import units
from .pes import PES


class Avg_PES(PES):
    """Avg potential energy surface calculators"""

    # TODO can fix this up so it automatically initializes PES1 and PES2?
    def __init__(
        self,
        PES1,
        PES2,
        lot,
    ):
        self.options = PES1.options
        self.PES1 = PES(PES1.options.replace(lot=lot))
        self.PES2 = PES(PES2.options.replace(lot=lot))
        self._dE = 1000.0
        self.lot = lot
        self.lot.do_coupling = True
        self.lot.coupling_states = (PES1.ad_idx, PES2.ad_idx)

    @classmethod
    def create_pes_from(cls, PES, options=None, copy_wavefunction=True):
        lot = type(PES.lot).copy(PES.lot, options or {}, copy_wavefunction)
        return cls(PES.PES1, PES.PES2, lot)

    def get_energy(self, xyz):
        if self.PES1.multiplicity == self.PES2.multiplicity:
            assert self.PES2.ad_idx > self.PES1.ad_idx, 'dgrad wrong direction'
        self.dE = self.PES2.get_energy(xyz) - self.PES1.get_energy(xyz)
        return 0.5 * (self.PES1.get_energy(xyz) + self.PES2.get_energy(xyz))

    def get_gradient(self, xyz, frozen_atoms=None):
        return 0.5 * (self.PES1.get_gradient(xyz, frozen_atoms) + self.PES2.get_gradient(xyz, frozen_atoms))

    def get_coupling(self, xyz, frozen_atoms=None):
        assert self.PES1.multiplicity == self.PES2.multiplicity, 'coupling is 0'
        assert self.PES1.ad_idx != self.PES2.ad_idx, 'coupling is 0'
        return np.reshape(
            self.lot.get_coupling(xyz, self.PES1.multiplicity, self.PES1.ad_idx, self.PES2.ad_idx, frozen_atoms),
            (-1, 1),
        )

    def get_dgrad(self, xyz, frozen_atoms=None):
        if self.PES1.multiplicity == self.PES2.multiplicity:
            assert self.PES2.ad_idx > self.PES1.ad_idx, 'dgrad wrong direction'
        return self.PES2.get_gradient(xyz, frozen_atoms) - self.PES1.get_gradient(xyz, frozen_atoms)

    def get_average_gradient(self, xyz, frozen_atoms=None):
        if self.PES1.multiplicity == self.PES2.multiplicity:
            assert self.PES2.ad_idx > self.PES1.ad_idx, 'dgrad wrong direction'
        return 0.5 * (self.PES2.get_gradient(xyz, frozen_atoms) + self.PES1.get_gradient(xyz, frozen_atoms))

    # http://pubs.acs.org/doi/abs/10.1021/acs.jctc.6b00384
    def symmetric_orthogonalization(self, xyz):
        def get_beta(g, h):
            gdoth = np.dot(g.T, h)
            print(f' g*h = {gdoth:1.4f}')
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
        print(f' dE is {self.dE:5.4f}')

        # non-adiabatic coupling is derivative coupling times dE
        nac = dvec * self.dE / units.KCAL_MOL_PER_AU
        beta = get_beta(dgrad, nac)
        print(f' beta = {beta:1.6f}')

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
        print(f' sab_x {sab_x:1.2f} sab_y {sab_y:1.2f}')
        if sab_x < 0.0 and sab_y > 0.0:
            print(' flipping dgrad')
            dgrad = -dgrad
            dotx = np.dot(sab.T, dgrad) / norm_dg
            sab_x = dotx / pitch
        elif sab_x > 0.0 and sab_y < 0.0:
            print(' flipping nac')
            nac = -nac
            doty = np.dot(sab.T, nac) / norm_nac
            sab_y = doty / pitch

        sigma = np.sqrt(sab_x * sab_x + sab_y * sab_y)
        theta_s = np.arctan(sab_y / sab_x)
        print(
            f' pitch  {pitch:1.5f}, asymmetry {asymmetry:1.5f}, sigma {sigma:1.5f} theta_s {theta_s:1.5f} sx {sab_x:1.2f} sy {sab_y:1.2f}'
        )

        P = ((sigma**2.0) / (1.0 - asymmetry**2.0)) * (1.0 - asymmetry * np.cos(2 * theta_s))
        B = ((sigma**2) / (4 * asymmetry**2)) ** (1.0 / 3) * (
            ((1.0 + asymmetry) * (np.cos(theta_s)) ** 2) ** (1.0 / 3)
            + ((1 - asymmetry) * (np.sin(theta_s)) ** 2) ** (1.0 / 3)
        )
        print(f' P {P:2.5f} B {B:2.5f}')

        if P < 1.0:
            print(' Your conical intersection is peaked')
        elif P > 1.0:
            print(' Your conical intersection is sloped')
        if B < 1.0:
            print(' Your conical intersection is bifurcating')
        if B > 1.0:
            print(' Your conical intersection is single path')

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
