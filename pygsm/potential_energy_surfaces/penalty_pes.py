import sys

# third party
# local application imports
from ..utilities import units
from .pes import PES


class Penalty_PES(PES):
    """ penalty potential energy surface calculators """

    def __init__(self,
                 PES1,
                 PES2,
                 lot,
                 sigma=1.0,
                 alpha=0.02*units.KCAL_MOL_PER_AU,
                 ):
        self.PES1 = PES(PES1.options.copy().set_values({
            "lot": lot,
        }))
        self.PES2 = PES(PES2.options.copy().set_values({
            "lot": lot,
        }))
        self.lot = lot
        self.alpha = alpha
        self.dE = 1000.
        self.sigma = sigma
        print(f' PES1 multiplicity: {self.PES1.multiplicity} PES2 multiplicity: {self.PES2.multiplicity} sigma: {self.sigma}')

    @classmethod
    def create_pes_from(cls, PES, options=None, copy_wavefunction=True):
        lot = type(PES.lot).copy(PES.lot, options or {}, copy_wavefunction)
        return cls(PES.PES1, PES.PES2, lot, PES.sigma, PES.alpha)

    def get_energy(self, geom):
        E1 = self.PES1.get_energy(geom)
        E2 = self.PES2.get_energy(geom)

        avgE = 0.5*(E1+E2)
        self.dE = E2-E1
        # TODO what to do if PES2 is or goes lower than PES1?
        G = (self.dE*self.dE)/(abs(self.dE) + self.alpha)
        sys.stdout.flush()
        return avgE+self.sigma*G

    def get_gradient(self, geom, frozen_atoms=None):
        self.grad1 = self.PES1.get_gradient(geom, frozen_atoms)
        self.grad2 = self.PES2.get_gradient(geom, frozen_atoms)
        avg_grad = 0.5*(self.grad1 + self.grad2)
        dgrad = self.grad2 - self.grad1
        if self.dE < 0:
            dgrad *= -1
        factor = self.sigma*((self.dE*self.dE) + 2.*self.alpha*abs(self.dE))/((abs(self.dE) + self.alpha)**2)
        return avg_grad + factor*dgrad

    def get_avg_gradient(self, xyz, frozen_atoms=None):
        return 0.5*(self.PES1.get_gradient(xyz, frozen_atoms) + self.PES2.get_gradient(xyz, frozen_atoms))
