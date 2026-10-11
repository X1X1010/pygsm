#!/usr/bin/env python

# standard library imports
import time

# third party
from collections import OrderedDict
import numpy as np

from pygsm.utilities import elements, options, output, block_matrix

ELEMENT_TABLE = elements.ElementData()

CacheWarning = False


class InternalCoordinates(object):

    @staticmethod
    def default_options():
        ''' InternalCoordinates default options.'''

        if hasattr(InternalCoordinates, '_default_options'):
            return InternalCoordinates._default_options.copy()
        opt = options.Options()

        opt.add_option(
            key="xyz",
            required=True,
            doc='cartesian coordinates in angstrom'
        )

        opt.add_option(
            key='atoms',
            required=True,
                doc='atom element named tuples/dictionary must be of type list[elements].'
        )

        opt.add_option(
            key='connect',
            value=False,
            allowed_types=[bool],
            doc="Connect the fragments/residues together with a minimum spanning bond,\
                    use for DLC, Don't use for TRIC, or HDLC.",
        )

        opt.add_option(
            key='addcart',
            value=False,
            allowed_types=[bool],
            doc="Add cartesian coordinates\
                    use to form HDLC ,Don't use for TRIC, DLC.",
        )

        opt.add_option(
            key='addtr',
            value=False,
            allowed_types=[bool],
            doc="Add translation and rotation coordinates\
                    use for TRIC.",
        )

        opt.add_option(
            key='constraints',
            value=None,
            allowed_types=[list],
            doc='A list of Distance,Angle,Torsion constraints (see slots.py),\
                    This is only useful if doing a constrained geometry optimization\
                    since GSM will handle the constraint automatically.'
        )
        opt.add_option(
            key='cVals',
            value=None,
            allowed_types=[list],
            doc='List of Distance,Angle,Torsion constraints values'
        )

        opt.add_option(
            key='form_topology',
            value=True,
            doc='A lazy argument for forming the topology on the fly, dont use this',
        )

        opt.add_option(
            key='primitives',
            value=None,
            doc='This is a Primitive internal coordinates object -- can be used instead \
                        of creating new primitive object'
        )

        opt.add_option(
            key='topology',
            value=None,
            doc='This is the molecule topology, used for building primitives'
        )

        opt.add_option(
            key='print_level',
            value=1,
            required=False,
            allowed_types=[int],
            doc='0-- no printing, 1-- printing')

        InternalCoordinates._default_options = opt
        return InternalCoordinates._default_options.copy()

    @classmethod
    def from_options(cls, **kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return cls(cls.default_options().set_values(kwargs))

    def __init__(self,
                 options
                 ):

        self.options = options
        self.stored_wilsonB = OrderedDict()

    def addConstraint(self, cPrim, cVal):
        raise NotImplementedError("Constraints not supported with Cartesian coordinates")

    def haveConstraints(self):
        raise NotImplementedError("Constraints not supported with Cartesian coordinates")

    def augmentGH(self, xyz, G, H):
        raise NotImplementedError("Constraints not supported with Cartesian coordinates")

    def calcGradProj(self, xyz, gradx):
        raise NotImplementedError("Constraints not supported with Cartesian coordinates")

    def clearCache(self):
        self.stored_wilsonB = OrderedDict()

    def wilsonB(self, xyz):
        """
        Given Cartesian coordinates xyz, return the Wilson B-matrix
        given by dq_i/dx_j where x is flattened (i.e. x1, y1, z1, x2, y2, z2)
        """
        global CacheWarning
        t0 = time.time()
        xhash = hash(xyz.tostring())
        time.time() - t0
        if xhash in self.stored_wilsonB:
            ans = self.stored_wilsonB[xhash]
            return ans
        WilsonB = []
        Der = self.derivatives(xyz)
        for i in range(Der.shape[0]):
            WilsonB.append(Der[i].flatten())
        self.stored_wilsonB[xhash] = np.array(WilsonB)
        if len(self.stored_wilsonB) > 1000 and not CacheWarning:
            output.logger.warning("\x1b[91mWarning: more than 100 B-matrices stored, memory leaks likely\x1b[0m")
            CacheWarning = True
        ans = np.array(WilsonB)
        return ans

    def GMatrix(self, xyz, u=None):
        """
        Given Cartesian coordinates xyz, return the G-matrix
        given by G = BuBt where u is an arbitrary matrix (default to identity)
        """
        Bmat = self.wilsonB(xyz)

        if u is None:
            BuBt = np.dot(Bmat, Bmat.T)
        else:
            BuBt = np.dot(Bmat, np.dot(u, Bmat.T))
        return BuBt

    def GInverse_EIG(self, xyz):
        xyz = xyz.reshape(-1, 3)
        G = self.GMatrix(xyz)
        Gi = np.linalg.inv(G)
        # print "G-time: %.3f Inv-time: %.3f" % (time_G, time_inv)
        return Gi

    def checkFiniteDifference(self, xyz):
        xyz = xyz.reshape(-1, 3)
        Analytical = self.derivatives(xyz)
        FiniteDifference = np.zeros_like(Analytical)
        h = 1e-5
        for i in range(xyz.shape[0]):
            for j in range(3):
                x1 = xyz.copy()
                x2 = xyz.copy()
                x1[i, j] += h
                x2[i, j] -= h
                PMDiff = self.calcDiff(x1, x2)
                FiniteDifference[:, i, j] = PMDiff/(2*h)
        for i in range(Analytical.shape[0]):
            output.logger.info(f"IC {i}/{Analytical.shape[0]} : {self.Internals[i]}")
            lines = [""]
            maxerr = 0.0
            for j in range(Analytical.shape[1]):
                lines.append("Atom %s" % (j+1))
                for k in range(Analytical.shape[2]):
                    error = Analytical[i, j, k] - FiniteDifference[i, j, k]
                    if np.abs(error) > 1e-5:
                        color = "\x1b[91m"
                    else:
                        color = "\x1b[92m"
                    lines.append("{} {: .5e} {: .5e} {}{: .5e}\x1b[0m".format("xyz"[k], Analytical[i, j, k], FiniteDifference[i, j, k], color, Analytical[i, j, k] - FiniteDifference[i, j, k]))
                    if maxerr < np.abs(error):
                        maxerr = np.abs(error)
            if maxerr > 1e-5:
                output.logger.info('\n'.join(lines))
            else:
                output.logger.info(f"Max Error = {maxerr:.5e}")
        output.logger.info("Finite-difference Finished")

    def checkFiniteDifferenceHess(self, xyz):
        xyz = xyz.reshape(-1, 3)
        Analytical = self.second_derivatives(xyz)
        FiniteDifference = np.zeros_like(Analytical)
        h = 1e-4
        verbose = False
        output.logger.info("-=# Now checking second derivatives of internal coordinates w/r.t. Cartesians #=-\n")
        for j in range(xyz.shape[0]):
            for m in range(3):
                for k in range(xyz.shape[0]):
                    for n in range(3):
                        x1 = xyz.copy()
                        x2 = xyz.copy()
                        x3 = xyz.copy()
                        x4 = xyz.copy()
                        x1[j, m] += h
                        x1[k, n] += h  # (+, +)
                        x2[j, m] += h
                        x2[k, n] -= h  # (+, -)
                        x3[j, m] -= h
                        x3[k, n] += h  # (-, +)
                        x4[j, m] -= h
                        x4[k, n] -= h  # (-, -)
                        PMDiff1 = self.calcDiff(x1, x2)
                        PMDiff2 = self.calcDiff(x4, x3)
                        FiniteDifference[:, j, m, k, n] += (PMDiff1+PMDiff2)/(4*h**2)
        for i in range(Analytical.shape[0]):
            title = f"{f'IC {i+1}/{Analytical.shape[0]}':>20} : {str(self.Internals[i]):>20}"
            lines = [title]
            if verbose:
                print(title+'\n')
            maxerr = 0.0
            numerr = 0
            for j in range(Analytical.shape[1]):
                for m in range(Analytical.shape[2]):
                    for k in range(Analytical.shape[3]):
                        for n in range(Analytical.shape[4]):
                            ana = Analytical[i, j, m, k, n]
                            fin = FiniteDifference[i, j, m, k, n]
                            error = ana - fin
                            message = "Atom {} {} {} {} a: {: 12.5e} n: {: 12.5e} e: {: 12.5e} {}".format(j+1, 'xyz'[m], k+1, 'xyz'[n], ana, fin,
                                                                                                error, 'X' if np.abs(error) > 1e-5 else '')
                            if np.abs(error) > 1e-5:
                                numerr += 1
                            if (ana != 0.0 or fin != 0.0) and verbose:
                                print(message+'\n')
                            lines.append(message)
                            if maxerr < np.abs(error):
                                maxerr = np.abs(error)
            if maxerr > 1e-5 and not verbose:
                print('\n'.join(lines)+'\n')
            print(f"{title} : Max Error = {maxerr: 12.5e} ({numerr} above threshold)\n")
        print("Finite-difference Finished\n")
        return FiniteDifference

    def calcGrad(self, xyz, gradx, frozen_atoms=None):
        Ginv = self.GInverse(xyz)
        Bmat = self.wilsonB(xyz)

        # Internal coordinate gradient
        return block_matrix.dot(Ginv, block_matrix.dot(Bmat, gradx))

    def calcHess(self, xyz, gradx, hessx):
        """
        Compute the internal coordinate Hessian. 
        Expects Cartesian coordinates to be provided in a.u.
        """

        self.calculate(xyz)
        Ginv = self.GInverse(xyz)
        Ginv = block_matrix.full_matrix(Ginv)
        Bmat = self.wilsonB(xyz)
        Bmat = block_matrix.full_matrix(Bmat)

        BptGq = self.calcCg(xyz,gradx)
        Hx_BptGq = hessx - BptGq 

        Hq = np.einsum('ps,sm,mn,nr,rq', Ginv, Bmat, Hx_BptGq, Bmat.T, Ginv, optimize=True)
        return Hq

    def readCache(self, xyz, dQ):
        if not hasattr(self, 'stored_xyz'):
            return None
        if np.linalg.norm(self.stored_xyz - xyz) < 1e-10:
            if np.linalg.norm(self.stored_dQ - dQ) < 1e-10:
                return self.stored_newxyz
        return None

    def writeCache(self, xyz, dQ, newxyz):
        self.stored_xyz = xyz.copy()
        self.stored_dQ = dQ.copy()
        self.stored_newxyz = newxyz.copy()
  
    def newCartesian(self, xyz, dQ, frozen_atoms=None, verbose=True):
        cached = self.readCache(xyz, dQ)
        if cached is not None:
            # print "Returning cached result"
            return cached
        xyz1 = xyz.copy()
        dQ1 = dQ.flatten()
        # Iterate until convergence:
        microiter = 0
        ndqs = []
        ndqt = 100.
        rmsds = []
        self.bork = False
        # Damping factor
        damp = 1.0
        
        # Function to exit from loop
        def finish(microiter, rmsdt, ndqt, xyzsave, xyz_iter1):
            if ndqt > 1e-1:
                if verbose:
                    output.logger.info(f" Failed to obtain coordinates after {microiter} microiterations (rmsd = {rmsdt:.3e} |dQ| = {ndqt:.3e})\n")
                self.bork = True
                self.writeCache(xyz, dQ, xyz_iter1)
                return xyzsave.reshape((-1, 3))
            elif ndqt > 1e-3:
                if verbose:
                    output.logger.info(f" Approximate coordinates obtained after {microiter} microiterations (rmsd = {rmsdt:.3e} |dQ| = {ndqt:.3e})\n")
            else:
                if verbose:
                    output.logger.info(f" Cartesian coordinates obtained after {microiter} microiterations (rmsd = {rmsdt:.3e} |dQ| = {ndqt:.3e})\n")
            self.writeCache(xyz, dQ, xyzsave)
            return xyzsave.reshape((-1, 3))

        fail_counter = 0
        while True:
            microiter += 1
            Bmat = self.wilsonB(xyz1)
            Ginv = self.GInverse(xyz1)

            # Get new Cartesian coordinates
            dxyz = damp*block_matrix.dot(block_matrix.transpose(Bmat), block_matrix.dot(Ginv, dQ1))

            if frozen_atoms is not None:
                for a in [3*i for i in frozen_atoms]:
                    dxyz[a:a+3] = 0.

            xyz2 = xyz1 + dxyz.reshape((-1, 3))
            if microiter == 1:
                xyzsave = xyz2.copy()
                xyz_iter1 = xyz2.copy()
            # Calculate the actual change in internal coordinates
            dQ_actual = self.calcDiff(xyz2, xyz1)
            rmsd = np.sqrt(np.mean((np.array(xyz2-xyz1).flatten())**2))
            ndq = np.linalg.norm(dQ1-dQ_actual)
            if len(ndqs) > 0:
                if ndq > ndqt:
                    if verbose:
                        output.logger.info(f" Iter: {microiter} Err-dQ (Best) = {ndq:.5e} ({ndqt:.5e}) RMSD: {rmsd:.5e} Damp: {damp:.5e} (Bad)\n")
                    damp /= 2
                    fail_counter += 1
                else:
                    if verbose:
                        output.logger.info(f" Iter: {microiter} Err-dQ (Best) = {ndq:.5e} ({ndqt:.5e}) RMSD: {rmsd:.5e} Damp: {damp:.5e} (Good)\n")
                    fail_counter = 0
                    damp = min(damp*1.2, 1.0)
                    rmsdt = rmsd
                    ndqt = ndq
                    xyzsave = xyz2.copy()
            else:
                if verbose:
                    output.logger.info(f" Iter: {microiter} Err-dQ = {ndq:.5e} RMSD: {rmsd:.5e} Damp: {damp:.5e}\n")
                rmsdt = rmsd
                ndqt = ndq
            ndqs.append(ndq)
            rmsds.append(rmsd)
            # Check convergence / fail criteria
            if rmsd < 1e-6 or ndq < 1e-6:
                return finish(microiter, rmsdt, ndqt, xyzsave, xyz_iter1)
            if fail_counter >= 5:
                return finish(microiter, rmsdt, ndqt, xyzsave, xyz_iter1)
            if microiter == 50:
                return finish(microiter, rmsdt, ndqt, xyzsave, xyz_iter1)
            # Figure out the further change needed
            dQ1 = dQ1 - dQ_actual
            xyz1 = xyz2.copy()
