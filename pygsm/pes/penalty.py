# third party
# local application imports
import logging

from ..utils import units
from .pes import PES

logger = logging.getLogger(__name__)


class PenaltyPES(PES):
    """Average of two surfaces plus a penalty on their energy gap (Levine-Coe-Martinez), for crossing searches."""

    def __init__(
        self,
        pes1,
        pes2,
        lot,
        sigma=1.0,
        alpha=0.02 * units.KCAL_MOL_PER_AU,
    ):
        self.pes1 = pes1.clone(lot=lot)
        self.pes2 = pes2.clone(lot=lot)
        self.lot = lot
        self.alpha = alpha
        self.dE = 1000.0
        self.sigma = sigma
        logger.debug(
            f' PES1 multiplicity: {self.pes1.multiplicity} PES2 multiplicity: {self.pes2.multiplicity} sigma: {self.sigma}'
        )

    def get_energy(self, geom):
        E1 = self.pes1.get_energy(geom)
        E2 = self.pes2.get_energy(geom)

        avgE = 0.5 * (E1 + E2)
        self.dE = E2 - E1
        # TODO what to do if PES2 is or goes lower than PES1?
        G = (self.dE * self.dE) / (abs(self.dE) + self.alpha)
        return avgE + self.sigma * G

    def get_gradient(self, geom, frozen_atoms=None):
        self.grad1 = self.pes1.get_gradient(geom, frozen_atoms)
        self.grad2 = self.pes2.get_gradient(geom, frozen_atoms)
        avg_grad = 0.5 * (self.grad1 + self.grad2)
        dgrad = self.grad2 - self.grad1
        if self.dE < 0:
            dgrad *= -1
        factor = (
            self.sigma * ((self.dE * self.dE) + 2.0 * self.alpha * abs(self.dE)) / ((abs(self.dE) + self.alpha) ** 2)
        )
        return avg_grad + factor * dgrad

    def get_avg_gradient(self, xyz, frozen_atoms=None):
        return 0.5 * (self.pes1.get_gradient(xyz, frozen_atoms) + self.pes2.get_gradient(xyz, frozen_atoms))
