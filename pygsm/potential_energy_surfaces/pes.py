# third party
import numpy as np

# local application imports
from ..utilities import elements, options, units

ELEMENT_TABLE = elements.ElementData()


class PES:
    """ PES object """

    @staticmethod
    def default_options():

        if hasattr(PES, '_default_options'):
            return PES._default_options.copy()
        opt = options.Options()

        opt.add_option(
            key='lot',
            value=None,
            required=True,
            doc='Level of theory object')

        opt.add_option(
            key='ad_idx',
            value=0,
            required=True,
            doc='adiabatic index')

        opt.add_option(
            key='multiplicity',
            value=1,
            required=True,
            doc='multiplicity')

        opt.add_option(
            key="FORCE",
            value=None,
            required=False,
            doc='Apply a constant force between atoms in units of AU, e.g. [(1,2,0.1214)]. Negative is tensile, positive is compresive',
        )

        opt.add_option(
            key='RESTRAINTS',
            value=None,
            required=False,
            doc='Translational harmonic constraint'
        )

        opt.add_option(
            key='mass',
            value=None,
            required=False,
            doc='Mass is sometimes required'
        )

        PES._default_options = opt
        return PES._default_options.copy()

    @classmethod
    def from_options(cls, **kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return cls(cls.default_options().set_values(kwargs))

    # TODO make kwargs
    @classmethod
    def create_pes_from(cls, PES, options=None, copy_wavefunction=True):
        lot = type(PES.lot).copy(PES.lot, options or {}, copy_wavefunction)

        return cls(PES.options.copy().set_values({
            "lot": lot,
        }))

    def __init__(self,
                 options,
                 ):
        """ Constructor """
        self.options = options

        self.lot = self.options['lot']
        self.ad_idx = self.options['ad_idx']
        self.multiplicity = self.options['multiplicity']
        self.FORCE = self.options['FORCE']
        self.RESTRAINTS = self.options['RESTRAINTS']
        self._dE = 1000.

    @property
    def dE(self):
        return self._dE

    @dE.setter
    def dE(self, value):
        self._dE = value

    @property
    def energy(self):
        return self.get_energy(self.lot.currentCoords)


    def get_energy(self, xyz):
        fdE = 0.
        if self.FORCE is not None:
            for i in self.FORCE:
                force = i[2]
                diff = (xyz[i[0]] - xyz[i[1]])
                d = np.linalg.norm(diff)*units.ANGSTROM_TO_AU  # AU
                fdE += force*d*units.KCAL_MOL_PER_AU
        kdE = 0.
        if self.RESTRAINTS is not None:
            for i in self.RESTRAINTS:
                a = i[0]
                force = i[1]   # In kcal/mol/Ang^2?
                kdE += 0.5*force*(xyz[a] - self.reference_xyz[a])**2
        return self.lot.get_energy(xyz, self.multiplicity, self.ad_idx) + fdE + kdE   # Kcal/mol

    def get_finite_difference_hessian(self, coords, qm_region=None):
        ''' Calculate Finite Differnce Hessian

        Params:
            coords ((natoms,3) np.ndarray - system coordinates  (x,y,z)
            qm_region list of QM atoms in a QMMM simulation to obtain environment perturbed Hessian with the size of the QM region

        Returns:
            Hessian (N1,N1) np.ndarray

        '''
        if qm_region is None:
            hess = np.zeros((len(coords)*3, len(coords)*3))
            n1_region = [x for x in range(3*len(coords))]
        else:
            n1_region = []
            for n in qm_region:
                for j in range(3):
                    n1_region.append(n*3+j)
            print('n1_region')
            print(n1_region)
            hess = np.zeros((len(n1_region), len(n1_region)))
        print("hess shape", hess.shape)

        def gen_row(n):
            vec = np.zeros(coords.shape[0]*3)
            vec[n] = 1.
            return vec

        n_actual = 0
        for n in range(len(coords)*3):
            if n in n1_region:
                print("on hessian product ", n)
                row = gen_row(n)
                ans = self.get_finite_difference_hessian_product(coords, row)
                hess[n_actual] = np.squeeze(ans[n1_region])
                n_actual += 1
        return hess

    def get_finite_difference_hessian_product(self, coords, direction, FD_STEP_LENGTH=0.001):

        # format the direction
        direction = direction/np.linalg.norm(direction)
        direction = direction.reshape((len(coords), 3))

        # fd step
        fdstep = direction*FD_STEP_LENGTH
        fwd_coords = coords+fdstep
        bwd_coords = coords-fdstep

        # calculate grad fwd and bwd in a.u. (Bohr/Ha)
        grad_fwd = self.get_gradient(fwd_coords)/units.ANGSTROM_TO_AU
        grad_bwd = self.get_gradient(bwd_coords)/units.ANGSTROM_TO_AU

        return (grad_fwd-grad_bwd)/(FD_STEP_LENGTH*2)

    def get_gradient(self, xyz, frozen_atoms=None):

        grad = self.lot.get_gradient(xyz, self.multiplicity, self.ad_idx, frozen_atoms=frozen_atoms)
        if self.FORCE is not None:
            for i in self.FORCE:
                atoms = [i[0], i[1]]
                force = i[2]
                diff = (xyz[i[0]] - xyz[i[1]])
                d = np.linalg.norm(diff)*units.ANGSTROM_TO_AU  # Bohr

                # Constant force
                # grad=\nabla E + FORCE
                t = (force/d/2.)  # Hartree/bohr
                t *= units.ANGSTROM_TO_AU   # Ha/bohr * bohr/ang = Ha/ang
                sign = 1
                for a in atoms:
                    grad[a] += sign*t*diff.T
                    sign *= -1
        if self.RESTRAINTS is not None:
            for i in self.RESTRAINTS:
                a = i[0]
                force = i[1]
                grad[a] += force*(xyz[a] - self.reference_xyz[a])

        grad = np.reshape(grad, (-1, 1))
        return grad  # Ha/ang
