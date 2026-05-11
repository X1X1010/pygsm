import numpy as np
from .internal_coordinates import InternalCoordinates
from .primitive_internals import PrimitiveInternalCoordinates
from .slots import CartesianX, CartesianY, CartesianZ


class CartesianCoordinates(InternalCoordinates):
    """
    Cartesian coordinate system, written as a kind of internal coordinate class.  
    This one does not support constraints, because that requires adding some 
    primitive internal coordinates.
    """
    def __init__(self, options):
        super(CartesianCoordinates, self).__init__(options)
        self.Internals = []
        self.cPrims = []
        self.cVals = []
        self.atoms = options['atoms']
        self.natoms = len(self.atoms)
        top_settings={'make_primitives': False}
        self.Prims = PrimitiveInternalCoordinates(options.copy().set_values({'extra_kwargs':top_settings}))

        for i in range(self.natoms):
            self.Prims.add(CartesianX(i, w=1.0))
            self.Prims.add(CartesianY(i, w=1.0))
            self.Prims.add(CartesianZ(i, w=1.0))
        #if 'constraints' in kwargs and kwargs['constraints'] is not None:
        #    raise RuntimeError('Do not use constraints with Cartesian coordinates')

        self.Vecs = np.eye(3 * self.natoms)

    def guess_hessian(self, xyz: np.ndarray):
        return 0.5 * np.eye(xyz.size)

    def calcGrad(self, xyz: np.ndarray, gradx: np.ndarray):
        return gradx

    def newCartesian(self, xyz: np.ndarray, dq: np.ndarray, verbose: bool = True):
        return xyz + dq.reshape(-1, 3)
    
    def calculate(self, coords: np.ndarray):
        return coords
