# standard library imports
from time import time

import numpy as np
from numpy.linalg import multi_dot

from pygsm.utilities import block_matrix, math_utils, output

# local application imports
from .internal_coordinates import InternalCoordinates
from .primitive_internals import PrimitiveInternalCoordinates
from .slots import RotationA, RotationB, RotationC


class DelocalizedInternalCoordinates(InternalCoordinates):
    def __init__(self, options=None, **kwargs):
        super().__init__(options, **kwargs)
        options = self.options

        # Cache  some useful attributes
        self.atoms = options.atoms
        self.natoms = len(self.atoms)

        # The DLC contains an instance of primitive internal coordinates.
        if self.options.primitives is None:
            print(' making primitives ')
            t1 = time()
            self.Prims = PrimitiveInternalCoordinates(options.replace())
            dt = time() - t1
            print(f' Time to make prims {dt:.3f}')
            self.options.primitives = self.Prims
        else:
            print(' setting primitives from options!')
            t0 = time()
            self.Prims = PrimitiveInternalCoordinates.copy(self.options.primitives)

            print(f' num of primitives {len(self.Prims.Internals)}')
            dt = time() - t0
            print(f' Time to copy prims {dt:.3f}')
            self.Prims.clearCache()

        xyz = options.xyz
        xyz = xyz.flatten()
        connect = options.connect
        addcart = options.addcart
        addtr = options.addtr
        if addtr:
            if connect:
                raise RuntimeError(
                    " Intermolecular displacements are defined by translation and rotations! \
                                    Don't add connect!"
                )
        elif addcart:
            if connect:
                raise RuntimeError(
                    " Intermolecular displacements are defined by cartesians! \
                                    Don't add connect!"
                )
        else:
            pass

        self.build_dlc(xyz)

    def clearCache(self):
        super().clearCache()
        self.Prims.clearCache()

    def __repr__(self):
        return self.Prims.__repr__()

    def update(self, other):
        return self.Prims.update(other.Prims)

    def join(self, other):
        return self.Prims.join(other.Prims)

    def copy(self, xyz):
        return type(self)(self.options.replace(xyz=xyz))

    def addConstraint(self, cPrim, cVal, xyz):
        self.Prims.addConstraint(cPrim, cVal, xyz)

    def getConstraints_from(self, other):
        self.Prims.getConstraints_from(other.Prims)

    def haveConstraints(self):
        return len(self.Prims.cPrims) > 0

    def getConstraintViolation(self, xyz):
        return self.Prims.getConstraintViolation(xyz)

    def printConstraints(self, xyz, thre=1e-5):
        self.Prims.printConstraints(xyz, thre=thre)

    def getConstraintTargetVals(self):
        return self.Prims.getConstraintTargetVals()

    def augmentGH(self, xyz, G, H):
        """
        Add extra dimensions to the gradient and Hessian corresponding to the constrained degrees of freedom.
        The Hessian becomes:  H  c
                              cT 0
        where the elements of cT are the first derivatives of the constraint function
        (typically a single primitive minus a constant) with respect to the DLCs.

        Since we picked a DLC to represent the constraint (cProj), we only set one element
        in each row of cT to be nonzero. Because cProj = a_i * Prim_i + a_j * Prim_j, we have
        d(Prim_c)/d(cProj) = 1.0/a_c where "c" is the index of the primitive being constrained.

        The extended elements of the Gradient are equal to the constraint violation.

        Parameters
        ----------
        xyz : np.ndarray
            Flat array containing Cartesian coordinates in atomic units
        G : np.ndarray
            Flat array containing internal coordinate gradient
        H : np.ndarray
            Square array containing internal coordinate Hessian

        Returns
        -------
        GC : np.ndarray
            Flat array containing gradient extended by constraint violations
        HC : np.ndarray
            Square matrix extended by partial derivatives d(Prim)/d(cProj)
        """
        # Number of internals (elements of G)
        ni = len(G)
        # Number of constraints
        nc = len(self.Prims.cPrims)
        # Total dimension
        nt = ni + nc
        # Lower block of the augmented Hessian
        cT = np.zeros((nc, ni), dtype=float)
        c0 = np.zeros(nc, dtype=float)
        for ic, c in enumerate(self.Prims.cPrims):
            # Look up the index of the primitive that is being constrained
            iPrim = self.Prims.Internals.index(c)
            # The DLC corresponding to the constrained primitive (a.k.a. cProj) is self.Vecs[self.cDLC[ic]].
            # For a differential change in the DLC, the primitive that we are constraining changes by:
            cT[ic, self.cDLC[ic]] = 1.0 / self.Vecs[iPrim, self.cDLC[ic]]
            # Calculate the further change needed in this constrained variable
            c0[ic] = self.Prims.cVals[ic] - c.value(xyz)
            if c.isPeriodic:
                Plus2Pi = c0[ic] + 2 * np.pi
                Minus2Pi = c0[ic] - 2 * np.pi
                if np.abs(c0[ic]) > np.abs(Plus2Pi):
                    c0[ic] = Plus2Pi
                if np.abs(c0[ic]) > np.abs(Minus2Pi):
                    c0[ic] = Minus2Pi
        # Construct augmented Hessian
        HC = np.zeros((nt, nt), dtype=float)
        HC[0:ni, 0:ni] = H[:, :]
        HC[ni:nt, 0:ni] = cT[:, :]
        HC[0:ni, ni:nt] = cT.T[:, :]
        # Construct augmented gradient
        GC = np.zeros(nt, dtype=float)
        GC[0:ni] = G[:]
        GC[ni:nt] = -c0[:]
        return GC, HC

    def applyConstraints(self, xyz):
        """
        Pass in Cartesian coordinates and return new coordinates that satisfy the constraints exactly.
        This is not used in the current constrained optimization code that uses Lagrange multipliers instead.
        """
        xyz1 = xyz.copy()
        niter = 0
        dQ0 = None
        while True:
            dQ = np.zeros(len(self.Internals), dtype=float)
            for ic, c in enumerate(self.Prims.cPrims):
                # Look up the index of the primitive that is being constrained
                iPrim = self.Prims.Internals.index(c)
                # Look up the index of the DLC that corresponds to the constraint
                iDLC = self.cDLC[ic]
                # Calculate the further change needed in this constrained variable
                dQ[iDLC] = (self.Prims.cVals[ic] - c.value(xyz1)) / self.Vecs[iPrim, iDLC]
                if c.isPeriodic:
                    Plus2Pi = dQ[iDLC] + 2 * np.pi
                    Minus2Pi = dQ[iDLC] - 2 * np.pi
                    if np.abs(dQ[iDLC]) > np.abs(Plus2Pi):
                        dQ[iDLC] = Plus2Pi
                    if np.abs(dQ[iDLC]) > np.abs(Minus2Pi):
                        dQ[iDLC] = Minus2Pi
            xyz2 = self.newCartesian(xyz1, dQ, verbose=False)
            if np.linalg.norm(dQ) < 1e-6:
                return xyz2
            if niter > 1 and np.linalg.norm(dQ) > np.linalg.norm(dQ0):
                return xyz1
            xyz1 = xyz2.copy()
            niter += 1
            dQ0 = dQ.copy()

    def newCartesian_withConstraint(self, xyz, dQ, thre=0.1, verbose=False):
        xyz2 = self.newCartesian(xyz, dQ, verbose)
        constraintSmall = len(self.Prims.cPrims) > 0
        for ic, c in enumerate(self.Prims.cPrims):
            w = c.w if type(c) in [RotationA, RotationB, RotationC] else 1.0
            current = c.value(xyz) / w
            reference = self.Prims.cVals[ic] / w
            diff = current - reference
            if np.abs(diff - 2 * np.pi) < np.abs(diff):
                diff -= 2 * np.pi
            if np.abs(diff + 2 * np.pi) < np.abs(diff):
                diff += 2 * np.pi
            if np.abs(diff) > thre:
                constraintSmall = False
        if constraintSmall:
            xyz2 = self.applyConstraints(xyz2)
        return xyz2

    def wilsonB(self, xyz):
        Bp = self.Prims.wilsonB(xyz)
        return block_matrix.dot(block_matrix.transpose(self.Vecs), Bp)

    def calcGradProj(self, xyz, gradx):
        """
        Project out the components of the internal coordinate gradient along the
        constrained degrees of freedom. This is used to calculate the convergence
        criteria for constrained optimizations.

        Parameters
        ----------
        xyz : np.ndarray
            Flat array containing Cartesian coordinates in atomic units
        gradx : np.ndarray
            Flat array containing gradient in Cartesian coordinates

        """
        if len(self.Prims.cPrims) == 0:
            return gradx
        self.calculate(xyz)
        Ginv = self.GInverse(xyz)
        Bmat = self.wilsonB(xyz)
        # Internal coordinate gradient
        Gq = multi_dot([Ginv, Bmat, gradx.T])
        Gqc = np.array(Gq).flatten()
        # Remove the directions that are along the DLCs that we are constraining
        for i in self.cDLC:
            Gqc[i] = 0.0
        Gxc = multi_dot([Bmat.T, Gqc.T]).flatten()
        return Gxc

    def build_dlc(self, xyz, C=None):
        """
        Build the delocalized internal coordinates (DLCs) which are linear
        combinations of the primitive internal coordinates. Each DLC is stored
        as a column in self.Vecs.

        In short, each DLC is an eigenvector of the G-matrix, and the number of
        nonzero eigenvalues of G should be equal to 3*N.

        After creating the DLCs, we construct special ones corresponding to primitive
        coordinates that are constrained (cProj).  These are placed in the front (i.e. left)
        of the list of DLCs, and then we perform a Gram-Schmidt orthogonalization.

        This function is called at the end of __init__ after the coordinate system is already
        specified (including which primitives are constraints).

        Parameters
        ----------
        xyz     : np.ndarray
                  Flat array containing Cartesian coordinates in atomic units
        C       : np.ndarray
                Float array containing difference in primitive coordinates
        """

        output.click()
        G = self.Prims.GMatrix(xyz)  # in primitive coords
        time_G = output.click()

        tmpvecs = []
        for A in G.matlist:
            L, Q = np.linalg.eigh(A)
            LargeVals = 0
            LargeIdx = []
            for ival, value in enumerate(L):
                if np.abs(value) > 1e-6:
                    LargeVals += 1
                    LargeIdx.append(ival)
            tmpvecs.append(Q[:, LargeIdx])

        self.Vecs = block_matrix(tmpvecs)

        time_eig = output.click()
        print(f' Timings: Build G: {time_G:.3f} Eig: {time_eig:.3f}')

        self.Internals = [f'DLC {i + 1}' for i in range(self.Vecs.shape[1])]

        # Vecs has number of rows equal to the number of primitives, and
        # number of columns equal to the number of delocalized internal coordinates.
        if self.haveConstraints():
            self.form_cVec_from_cPrims()

        if C is not None:
            # orthogonalize
            if (C[:] == 0.0).all():
                raise RuntimeError
            Cn = math_utils.orthogonalize(C)

            # transform C into basis of DLC
            # CRA 3/2019 NOT SURE WHY THIS IS DONE
            # couldn't Cn just be used?

            cVecs = block_matrix.dot(block_matrix.dot(self.Vecs, block_matrix.transpose(self.Vecs)), Cn)

            # normalize C_U
            try:
                cVecs = math_utils.orthogonalize(cVecs)
            except Exception as error:
                raise RuntimeError(f'error forming cVec:\n{cVecs}') from error

            # project constraints into vectors
            self.Vecs = block_matrix.project_constraint(self.Vecs, cVecs)
        return self.Vecs

    def build_dlc_conjugate(self, xyz, C=None):
        """
        Build the delocalized internal coordinates (DLCs) which are linear
        combinations of the primitive internal coordinates. Each DLC is stored
        as a column in self.Vecs.

        In short, each DLC is an eigenvector of the G-matrix, and the number of
        nonzero eigenvalues of G should be equal to 3*N.

        After creating the DLCs, we construct special ones corresponding to primitive
        coordinates that are constrained (cProj).  These are placed in the front (i.e. left)
        of the list of DLCs, and then we perform a Gram-Schmidt orthogonalization.

        This function is called at the end of __init__ after the coordinate system is already
        specified (including which primitives are constraints).

        Parameters
        ----------
        xyz     : np.ndarray
                  Flat array containing Cartesian coordinates in atomic units
        C       : np.ndarray
                Float array containing difference in primitive coordinates
        """

        print(' starting to build G prim')
        output.click()
        G = self.Prims.GMatrix(xyz)  # in primitive coords
        time_G = output.click()
        print(f' Timings: Build G: {time_G:.3f} ')

        tmpvecs = []
        for A in G.matlist:
            L, Q = np.linalg.eigh(A)
            LargeVals = 0
            LargeIdx = []
            for ival, value in enumerate(L):
                if np.abs(value) > 1e-6:
                    LargeVals += 1
                    LargeIdx.append(ival)
            tmpvecs.append(Q[:, LargeIdx])
        self.Vecs = block_matrix(tmpvecs)
        output.click()

        self.Internals = [f'DLC {i + 1}' for i in range(len(LargeIdx))]

        # Vecs has number of rows equal to the number of primitives, and
        # number of columns equal to the number of delocalized internal coordinates.

        # TODO use block diagonal
        if C is not None:
            # orthogonalize
            if (C[:] == 0.0).all():
                raise RuntimeError
            G = block_matrix.full_matrix(self.Prims.GMatrix(xyz))
            Cn = math_utils.conjugate_orthogonalize(C, G)

            # transform C into basis of DLC
            # CRA 3/2019 NOT SURE WHY THIS IS DONE
            # couldn't Cn just be used?
            cVecs = block_matrix.dot(block_matrix.dot(self.Vecs, block_matrix.transpose(self.Vecs)), Cn)

            # normalize C_U
            try:
                cVecs = math_utils.conjugate_orthogonalize(cVecs, G)
            except Exception as error:
                raise RuntimeError(f'error forming cVec:\n{cVecs}') from error

            # project constraints into vectors
            self.Vecs = block_matrix.project_conjugate_constraint(self.Vecs, cVecs, G)

        return

        # # LPW: Coefficients of DLC's are in each column and DLCs corresponding to constraints should basically be like (0 1 0 0 0 ..)
        # # Expect to see a diagonal matrix here
        # # Expect to see "large" eigenvalues here (no less than 0.1 ideally)

    def form_cVec_from_cPrims(self):
        """forms the constraint vector from self.cPrim -- not used in GSM"""
        # CRA 3/2019 warning:
        # I'm not sure how this works!!!
        # multiple constraints appears to be problematic!!!
        self.cDLC = [i for i in range(len(self.Prims.cPrims))]
        for _ic, c in enumerate(self.Prims.cPrims):
            # Look up the index of the primitive that is being constrained
            iPrim = self.Prims.Internals.index(c)
            # Pick a row out of the eigenvector space. This is a linear combination of the DLCs.
            cVec = self.Vecs[iPrim, :]
            cVec = np.array(cVec)
            cVec /= np.linalg.norm(cVec)
            # This is a "new DLC" that corresponds to the primitive that we are constraining
            cProj = np.dot(self.Vecs, cVec.T)
            cProj /= np.linalg.norm(cProj)
        V = cProj.reshape(-1, 1)
        return V

    def __eq__(self, other):
        return self.Prims == other.Prims

    def __ne__(self, other):
        return not self.__eq__(other)

    def largeRots(self):
        """Determine whether a molecule has rotated by an amount larger than some threshold (hardcoded in Prims.largeRots())."""
        return self.Prims.largeRots()

    def calcDiff(self, coord1, coord2):
        """Calculate difference in internal coordinates, accounting for changes in 2*pi of angles."""
        PMDiff = self.Prims.calcDiff(coord1, coord2)
        Answer = block_matrix.dot(block_matrix.transpose(self.Vecs), PMDiff)
        return np.array(Answer).flatten()

    def calculate(self, coords):
        """Calculate the DLCs given the Cartesian coordinates."""
        PrimVals = self.Prims.calculate(coords)
        Answer = block_matrix.dot(block_matrix.transpose(self.Vecs), PrimVals)
        # print np.dot(np.array(self.Vecs[0,:]).flatten(), np.array(Answer).flatten())
        # print PrimVals[0]
        return np.array(Answer).flatten()

    # overwritting the parent internalcoordinates GMatrix
    # which is an elegant way to use the derivatives
    # but there is a more efficient way to compute G
    # using the block diagonal properties of G and V
    def GMatrix(self, xyz):
        tmpvecs = []
        Gp = self.Prims.GMatrix(xyz)
        Vt = block_matrix.transpose(self.Vecs)
        for vt, G, v in zip(Vt.matlist, Gp.matlist, self.Vecs.matlist):
            tmpvecs.append(np.dot(np.dot(vt, G), v))
        return block_matrix(tmpvecs)

    def MW_GMatrix(self, xyz, mass):
        tmpvecs = []
        s3a = 0
        Bp = self.Prims.wilsonB(xyz)
        Vt = block_matrix.transpose(self.Vecs)

        s3a = 0
        for vt, b, v in zip(Vt.matlist, Bp.matlist, self.Vecs.matlist):
            e3a = s3a + b.shape[1]
            tmpvecs.append(np.linalg.multi_dot([vt, b / mass[s3a:e3a], b.T, v]))
            s3a = e3a
        return block_matrix(tmpvecs)

    def derivatives(self, coords):
        """Obtain the change of the DLCs with respect to the Cartesian coordinates."""
        PrimDers = self.Prims.derivatives(coords)
        Answer = np.zeros((self.Vecs.shape[1], PrimDers.shape[1], PrimDers.shape[2]), dtype=float)

        # block matrix tensor dot
        count = 0
        for block in self.Vecs.matlist:
            for i in range(block.shape[1]):
                for j in range(block.shape[0]):
                    Answer[count, :, :] += block[j, i] * PrimDers[j, :, :]
                count += 1

        return Answer

    def second_derivatives(self, coords):
        """Obtain the second derivatives of the DLCs with respect to the Cartesian coordinates."""
        PrimDers = self.Prims.second_derivatives(coords)

        # block matrix tensor dot
        Answer = np.zeros((self.Vecs.shape[1], coords.shape[0], 3, coords.shape[0], 3))
        count = 0
        for block in self.Vecs.matlist:
            for i in range(block.shape[1]):
                for j in range(block.shape[0]):
                    Answer[count, :, :, :, :] += block[j, i] * PrimDers[j, :, :, :, :]
                count += 1

        return Answer

    def GInverse(self, xyz):
        return self.GInverse_EIG(xyz)

    # TODO this needs to be fixed
    def GInverse_EIG(self, xyz):
        xyz = xyz.reshape(-1, 3)
        G = self.GMatrix(xyz)
        tmpGi = [np.linalg.inv(g) for g in G.matlist]
        return block_matrix(tmpGi)

    def repr_diff(self, other):
        return self.Prims.repr_diff(other.Prims)

    def guess_hessian(self, coords):
        """Build the guess Hessian, consisting of a diagonal matrix
        in the primitive space and changed to the basis of DLCs."""
        Hprim = self.Prims.guess_hessian(coords)
        return multi_dot([self.Vecs.T, Hprim, self.Vecs])

    def resetRotations(self, xyz):
        """Reset the reference geometries for calculating the orientational variables."""
        self.Prims.resetRotations(xyz)
