#!/usr/bin/env python

# standard library imports

import logging
from collections import OrderedDict
from typing import Any

# third party
import numpy as np

from pygsm.utils import block_matrix, elements
from pygsm.utils.config import Configurable

logger = logging.getLogger(__name__)

ELEMENT_TABLE = elements.ElementData()

# Warn once per process when a B-matrix cache grows suspiciously large.
_CACHE_WARNING_ISSUED = False
_CACHE_WARNING_SIZE = 1000


class InternalCoordinates(Configurable):
    """Base class of the coordinate systems: B-matrix (cached), G-matrix and Cartesian back-transformation.

    Parameters
    ----------
    xyz : Cartesian coordinates in Angstrom, shape (natoms, 3).
    atoms : ``elements.Element`` of each atom.
    connect : connect fragments with minimum-spanning bonds (DLC); not for TRIC or HDLC.
    addcart : add Cartesian coordinates (HDLC); not for TRIC or DLC.
    addtr : add translation and rotation coordinates (TRIC).
    form_topology : build the topology and primitives on construction.
    primitives : ``PrimitiveInternalCoordinates`` to reuse instead of building new ones.
    topology : molecular graph used to build the primitives (built from ``xyz`` when not given).
    """

    def __init__(
        self,
        xyz: np.ndarray,
        atoms: list,
        *,
        connect: bool = False,
        addcart: bool = False,
        addtr: bool = False,
        form_topology: bool = True,
        primitives: Any = None,
        topology: Any = None,
    ):
        self.xyz = xyz
        self.atoms = atoms
        self.connect = connect
        self.addcart = addcart
        self.addtr = addtr
        self.form_topology = form_topology
        self.primitives = primitives
        self.topology = topology
        self.stored_wilsonB = OrderedDict()

    def addConstraint(self, cPrim, cVal):
        raise NotImplementedError('Constraints not supported with Cartesian coordinates')

    def haveConstraints(self):
        raise NotImplementedError('Constraints not supported with Cartesian coordinates')

    def augmentGH(self, xyz, G, H):
        raise NotImplementedError('Constraints not supported with Cartesian coordinates')

    def calcGradProj(self, xyz, gradx):
        raise NotImplementedError('Constraints not supported with Cartesian coordinates')

    def clearCache(self):
        self.stored_wilsonB = OrderedDict()

    def wilsonB(self, xyz):
        """
        Given Cartesian coordinates xyz, return the Wilson B-matrix
        given by dq_i/dx_j where x is flattened (i.e. x1, y1, z1, x2, y2, z2)
        """
        xhash = hash(xyz.tobytes())
        if xhash in self.stored_wilsonB:
            return self.stored_wilsonB[xhash]
        Der = self.derivatives(xyz)
        WilsonB = [Der[i].flatten() for i in range(Der.shape[0])]
        self._store_wilsonB(xhash, np.array(WilsonB))
        return np.array(WilsonB)

    def _store_wilsonB(self, xhash, Bmat):
        """Cache a B-matrix under the hash of its geometry, warning once if the cache grows large."""
        global _CACHE_WARNING_ISSUED
        self.stored_wilsonB[xhash] = Bmat
        if len(self.stored_wilsonB) > _CACHE_WARNING_SIZE and not _CACHE_WARNING_ISSUED:
            logger.warning(
                f'\x1b[91mWarning: more than {_CACHE_WARNING_SIZE} B-matrices stored, memory leaks likely\x1b[0m'
            )
            _CACHE_WARNING_ISSUED = True

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
                FiniteDifference[:, i, j] = PMDiff / (2 * h)
        for i in range(Analytical.shape[0]):
            logger.info(f'IC {i}/{Analytical.shape[0]} : {self.Internals[i]}')
            lines = ['']
            maxerr = 0.0
            for j in range(Analytical.shape[1]):
                lines.append(f'Atom {j + 1}')
                for k in range(Analytical.shape[2]):
                    error = Analytical[i, j, k] - FiniteDifference[i, j, k]
                    if np.abs(error) > 1e-5:
                        color = '\x1b[91m'
                    else:
                        color = '\x1b[92m'
                    lines.append(
                        '{} {: .5e} {: .5e} {}{: .5e}\x1b[0m'.format(
                            'xyz'[k],
                            Analytical[i, j, k],
                            FiniteDifference[i, j, k],
                            color,
                            Analytical[i, j, k] - FiniteDifference[i, j, k],
                        )
                    )
                    if maxerr < np.abs(error):
                        maxerr = np.abs(error)
            if maxerr > 1e-5:
                logger.info('\n'.join(lines))
            else:
                logger.info(f'Max Error = {maxerr:.5e}')
        logger.info('Finite-difference Finished')

    def checkFiniteDifferenceHess(self, xyz):
        xyz = xyz.reshape(-1, 3)
        Analytical = self.second_derivatives(xyz)
        FiniteDifference = np.zeros_like(Analytical)
        h = 1e-4
        verbose = False
        logger.info('-=# Now checking second derivatives of internal coordinates w/r.t. Cartesians #=-')
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
                        FiniteDifference[:, j, m, k, n] += (PMDiff1 + PMDiff2) / (4 * h**2)
        for i in range(Analytical.shape[0]):
            title = f'{f"IC {i + 1}/{Analytical.shape[0]}":>20} : {self.Internals[i]!s:>20}'
            lines = [title]
            if verbose:
                logger.debug(title + '')
            maxerr = 0.0
            numerr = 0
            for j in range(Analytical.shape[1]):
                for m in range(Analytical.shape[2]):
                    for k in range(Analytical.shape[3]):
                        for n in range(Analytical.shape[4]):
                            ana = Analytical[i, j, m, k, n]
                            fin = FiniteDifference[i, j, m, k, n]
                            error = ana - fin
                            message = 'Atom {} {} {} {} a: {: 12.5e} n: {: 12.5e} e: {: 12.5e} {}'.format(
                                j + 1, 'xyz'[m], k + 1, 'xyz'[n], ana, fin, error, 'X' if np.abs(error) > 1e-5 else ''
                            )
                            if np.abs(error) > 1e-5:
                                numerr += 1
                            if (ana != 0.0 or fin != 0.0) and verbose:
                                logger.debug(message + '')
                            lines.append(message)
                            if maxerr < np.abs(error):
                                maxerr = np.abs(error)
            if maxerr > 1e-5 and not verbose:
                logger.debug('\n'.join(lines))
            logger.warning(f'{title} : Max Error = {maxerr: 12.5e} ({numerr} above threshold)')
        logger.debug('Finite-difference Finished')
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

        BptGq = self.calcCg(xyz, gradx)
        Hx_BptGq = hessx - BptGq

        Hq = np.einsum('ps,sm,mn,nr,rq', Ginv, Bmat, Hx_BptGq, Bmat.T, Ginv, optimize=True)
        return Hq

    def readCache(self, xyz, dQ):
        if not hasattr(self, 'stored_xyz'):
            return None
        if np.linalg.norm(self.stored_xyz - xyz) < 1e-10 and np.linalg.norm(self.stored_dQ - dQ) < 1e-10:
            return self.stored_newxyz
        return None

    def writeCache(self, xyz, dQ, newxyz):
        self.stored_xyz = xyz.copy()
        self.stored_dQ = dQ.copy()
        self.stored_newxyz = newxyz.copy()

    def newCartesian(self, xyz, dQ, frozen_atoms=None, verbose=True):
        cached = self.readCache(xyz, dQ)
        if cached is not None:
            return cached
        xyz1 = xyz.copy()
        dQ1 = dQ.flatten()
        # Iterate until convergence:
        microiter = 0
        ndqs = []
        ndqt = 100.0
        rmsds = []
        self.bork = False
        # Damping factor
        damp = 1.0

        # Function to exit from loop
        def finish(microiter, rmsdt, ndqt, xyzsave, xyz_iter1):
            if ndqt > 1e-1:
                if verbose:
                    logger.warning(
                        f' Failed to obtain coordinates after {microiter} microiterations (rmsd = {rmsdt:.3e} |dQ| = {ndqt:.3e})'
                    )
                self.bork = True
                self.writeCache(xyz, dQ, xyz_iter1)
                return xyzsave.reshape((-1, 3))
            elif ndqt > 1e-3:
                if verbose:
                    logger.debug(
                        f' Approximate coordinates obtained after {microiter} microiterations (rmsd = {rmsdt:.3e} |dQ| = {ndqt:.3e})'
                    )
            else:
                if verbose:
                    logger.debug(
                        f' Cartesian coordinates obtained after {microiter} microiterations (rmsd = {rmsdt:.3e} |dQ| = {ndqt:.3e})'
                    )
            self.writeCache(xyz, dQ, xyzsave)
            return xyzsave.reshape((-1, 3))

        fail_counter = 0
        while True:
            microiter += 1
            Bmat = self.wilsonB(xyz1)
            Ginv = self.GInverse(xyz1)

            # Get new Cartesian coordinates
            dxyz = damp * block_matrix.dot(block_matrix.transpose(Bmat), block_matrix.dot(Ginv, dQ1))

            if frozen_atoms is not None:
                for a in [3 * i for i in frozen_atoms]:
                    dxyz[a : a + 3] = 0.0

            xyz2 = xyz1 + dxyz.reshape((-1, 3))
            if microiter == 1:
                xyzsave = xyz2.copy()
                xyz_iter1 = xyz2.copy()
            # Calculate the actual change in internal coordinates
            dQ_actual = self.calcDiff(xyz2, xyz1)
            rmsd = np.sqrt(np.mean((np.array(xyz2 - xyz1).flatten()) ** 2))
            ndq = np.linalg.norm(dQ1 - dQ_actual)
            if len(ndqs) > 0:
                if ndq > ndqt:
                    if verbose:
                        logger.debug(
                            f' Iter: {microiter} Err-dQ (Best) = {ndq:.5e} ({ndqt:.5e}) RMSD: {rmsd:.5e} Damp: {damp:.5e} (Bad)'
                        )
                    damp /= 2
                    fail_counter += 1
                else:
                    if verbose:
                        logger.debug(
                            f' Iter: {microiter} Err-dQ (Best) = {ndq:.5e} ({ndqt:.5e}) RMSD: {rmsd:.5e} Damp: {damp:.5e} (Good)'
                        )
                    fail_counter = 0
                    damp = min(damp * 1.2, 1.0)
                    rmsdt = rmsd
                    ndqt = ndq
                    xyzsave = xyz2.copy()
            else:
                if verbose:
                    logger.debug(f' Iter: {microiter} Err-dQ = {ndq:.5e} RMSD: {rmsd:.5e} Damp: {damp:.5e}')
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
