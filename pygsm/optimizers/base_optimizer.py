import logging

# third party
import numpy as np

from ..coordinate_systems import CartesianCoordinates, DelocalizedInternalCoordinates
from ..utilities import block_matrix, manage_xyz, units
from ..utilities.config import Configurable, check_choice

# local application imports
from ._linesearch import NoLineSearch, backtrack

logger = logging.getLogger(__name__)


def sorted_eigh(mat, asc=False):
    """
    Return eigenvalues and eigenvectors of a symmetric matrix
    in descending order and associated eigenvectors.

    This is just a convenience function to get eigenvectors
    in descending or ascending order as desired.
    """
    L, Q = np.linalg.eigh(mat)
    if asc:
        idx = L.argsort()
    else:
        idx = L.argsort()[::-1]
    L = L[idx]
    Q = Q[:, idx]
    return L, Q


def force_positive_definite(H):
    """
    Force all eigenvalues to be positive.
    """
    # Sorted eigenvalues and corresponding eigenvectors of the Hessian
    Hvals, Hvecs = sorted_eigh(H, asc=True)
    Hs = np.zeros_like(H)
    for i in range(H.shape[0]):
        if Hvals[i] > 0:
            Hs += Hvals[i] * np.outer(Hvecs[:, i], Hvecs[:, i])
        else:
            Hs -= Hvals[i] * np.outer(Hvecs[:, i], Hvecs[:, i])
    return Hs


# TODO Add primitive constraint e.g. a list of internal coordinates to be left basically frozen throughout optimization
class BaseOptimizer(Configurable):
    """Shared machinery of the optimizers: step-size control, Hessian updates and constraint handling.

    Parameters
    ----------
    OPTTHRESH : gradrms convergence threshold (also available as ``conv_grms``).
    conv_Ediff : energy-change convergence threshold.
    conv_dE : state energy-difference convergence threshold.
    conv_gmax : max-gradient convergence threshold.
    conv_disp : max atomic displacement threshold.
    opt_cross : optimise a state crossing (seam/MECI searches).
    DMAX : initial step size; adapted to the performance of the optimisation.
    DMIN : minimum step size.
    abs_max_step : absolute max step; a hard cutoff, because the IC -> Cartesian
        back-transformation becomes pathological for large steps.
    ftol : strong Armijo condition: required relative energy decrease.
    SCALEQN : scale of the quasi-Newton step.
    SCALE_CLIMB : scale of the climbing-image step.
    update_hess_in_bg : keep updating the Hessian in the background (non-BFGS optimizers).
    SCALEW : weight scale (unused by the current optimizers).
    Linesearch : ``'NoLineSearch'`` or ``'backtrack'``.
    MAXAD : max atomic displacement used by the step control.
    print_level : diagnostic detail at DEBUG log level: 0 less, 1 more, 2 everything.
    HESS_TANG_TOL_TS : Hessian/tangent overlap tolerance for the TS node.
    """

    #: Whether the optimizer can take eigenvector-following TS steps (the exact TS search, go_gsm rtype=2).
    supports_ts_search = False

    def __init__(
        self,
        *,
        OPTTHRESH: float = 0.0005,
        conv_Ediff: float = 100.0,
        conv_dE: float = 1.0,
        conv_gmax: float = 100.0,
        conv_disp: float = 100.0,
        opt_cross: bool = False,
        DMAX: float = 0.1,
        DMIN: float = 0.0001,
        abs_max_step: float = 0.4,
        ftol: float = 0.0001,
        SCALEQN: float = 1,
        SCALE_CLIMB: float = 1.0,
        update_hess_in_bg: bool = True,
        SCALEW: float = 1.0,
        Linesearch: str = 'NoLineSearch',
        MAXAD: float = 0.075,
        print_level: int = 1,
        HESS_TANG_TOL_TS: float = 0.35,
    ):
        check_choice('Linesearch', Linesearch, ('NoLineSearch', 'backtrack'))
        self.OPTTHRESH = OPTTHRESH
        self.conv_Ediff = conv_Ediff
        self.conv_dE = conv_dE
        self.conv_gmax = conv_gmax
        self.conv_disp = conv_disp
        self.opt_cross = opt_cross
        self.DMAX = DMAX
        self.DMIN = DMIN
        self.abs_max_step = abs_max_step
        self.ftol = ftol
        self.SCALEQN = SCALEQN
        self.SCALE_CLIMB = SCALE_CLIMB
        self.update_hess_in_bg = update_hess_in_bg
        self.SCALEW = SCALEW
        self.Linesearch = Linesearch
        self.MAXAD = MAXAD
        self.print_level = print_level
        self.HESS_TANG_TOL_TS = HESS_TANG_TOL_TS
        self.linesearch = backtrack if Linesearch == 'backtrack' else NoLineSearch

        # TS node properties
        self.nneg = 0  # number of negative eigenvalues

        # Hessian
        self.Hint = None
        self.dx = 0.0
        self.dg = 0.0
        self.maxol_good = True
        self.gtse = 100.0

        # additional parameters needed by linesearch
        self.linesearch_parameters = {
            'epsilon': 1e-5,
            'ftol': ftol,  # 1e-4,
            'wolfe': 0.9,
            'max_linesearch': 3,
            'min_step': self.DMIN,
            'max_step': abs_max_step,
        }

        # Converged
        self.converged = False
        self.check_only_grad_converged = False

        return

    @property
    def conv_grms(self):
        """gradrms convergence threshold (the ``OPTTHRESH`` parameter)."""
        return self.OPTTHRESH

    @conv_grms.setter
    def conv_grms(self, value):
        self.OPTTHRESH = value

    @staticmethod
    def write_trajectory(trajectory, geoms, energies):
        """Write the optimisation frames to ``trajectory``; does nothing when it is None."""
        if trajectory is not None:
            manage_xyz.write_xyzs_w_comments(trajectory, geoms, energies, scale=1.0)

    def get_nconstraints(self, opt_type):
        if opt_type in ['ICTAN', 'CLIMB']:
            nconstraints = 1
        elif opt_type in ['MECI']:
            nconstraints = 2
        elif opt_type in ['SEAM', 'TS-SEAM']:
            nconstraints = 3
        else:
            nconstraints = 0
        return nconstraints

    def check_inputs(self, molecule, opt_type, ictan):
        if opt_type in ['MECI', 'SEAM', 'TS-SEAM']:
            assert molecule.PES.lot.do_coupling is True, 'Turn do_coupling on.'
        if opt_type in ['UCONSTRAINED']:
            assert ictan is None
        if opt_type in ['ICTAN', 'CLIMB', 'TS', 'SEAM', 'TS-SEAM', 'BEALES_CG'] and ictan.any() is None:
            raise RuntimeError('Need ictan')

    def set_lambda1(self, opt_type, eigen, maxoln=None):
        if opt_type == 'TS':
            leig = eigen[1]  # ! this is eigen[0] if update_ic_eigen() ### also diff values
            if maxoln != 0:
                leig = eigen[0]
            if leig < 0.0 and maxoln == 0:
                lambda1 = -leig
            else:
                lambda1 = 0.01
        else:
            leig = eigen[0]
            if leig < 0:
                lambda1 = -leig + 0.015
            else:
                lambda1 = 0.005
        if abs(lambda1) < 0.005:
            lambda1 = 0.005

        return lambda1

    def get_constraint_vectors(self, molecule, opt_type, ictan=None):
        if opt_type == 'UNCONSTRAINED':
            constraints = None
        elif opt_type == 'ICTAN' or opt_type == 'CLIMB' or opt_type == 'BEALES_CG':
            constraints = ictan
        elif opt_type == 'MECI':
            logger.debug('MECI')
            dgrad_U = block_matrix.dot(molecule.coord_basis, molecule.difference_gradient)
            dvec_U = block_matrix.dot(molecule.coord_basis, molecule.derivative_coupling)
            constraints = np.hstack((dgrad_U, dvec_U))
        elif opt_type == 'SEAM' or opt_type == 'TS-SEAM':
            dgrad_U = block_matrix.dot(molecule.coord_basis, molecule.difference_gradient)
            dvec_U = block_matrix.dot(molecule.coord_basis, molecule.derivative_coupling)
            constraints = np.hstack((ictan, dgrad_U, dvec_U))
        else:
            raise NotImplementedError
        return constraints

    def get_constraint_steps(self, molecule, opt_type, g):
        n = len(g)
        # TODO Raise Error for CartesianCoordinates

        # TODO 4/24/2019 block matrix/distributed constraints
        constraint_steps = np.zeros((n, 1))

        # 6/5 climb works with block matrix distributed constraints
        # => ictan climb
        if opt_type == 'CLIMB':
            gts = np.dot(g.T, molecule.constraints[:, 0])
            max_step = 0.05 / self.SCALE_CLIMB
            if gts > np.abs(max_step):
                gts = np.sign(gts) * max_step
            logger.debug(f' gts {gts.item():.4f}')
            constraint_steps = gts * molecule.constraints[:, 0]
            constraint_steps = constraint_steps[:, np.newaxis]
        # => MECI
        elif opt_type == 'MECI':
            dq = self.dgrad_step(molecule)
            constraint_steps[:, 0] = dq * molecule.constraints[:, 0]

        elif opt_type == 'SEAM':
            dq = self.dgrad_step(molecule)
            constraint_steps[:, 0] = dq * molecule.constraints[:, 1]
        # => seam climb
        elif opt_type == 'TS-SEAM':
            gts = np.dot(g.T, molecule.constraints[:, 0])

            # climbing step
            max_step = 0.05 / self.SCALE_CLIMB
            if gts > np.abs(max_step):
                gts = np.sign(gts) * max_step
            logger.debug(f' gts {gts.item():.4f}')
            constraint_steps = gts * molecule.constraints[:, 0]
            constraint_steps = constraint_steps[:, np.newaxis]

            # to CI step
            dq = self.dgrad_step(molecule)
            constraint_steps[:, 0] += dq * molecule.constraints[:, 1]

        return constraint_steps

    def dgrad_step(self, molecule):
        """takes a linear step along dgrad"""

        norm_dg = np.linalg.norm(molecule.difference_gradient)
        if self.print_level > 0:
            logger.debug(f' norm_dg is {norm_dg:1.4f}')
            logger.debug(f' dE is {molecule.difference_energy:1.4f}')

        dq = -molecule.difference_energy / units.KCAL_MOL_PER_AU / norm_dg
        if dq < self.DMAX / 5:
            dq = -self.DMAX / 5
        if dq < -0.075:
            dq = -0.075

        return dq

    def step_controller(self, step, ratio, gradrms, pgradrms, dEpre, opt_type, dE_iter):
        # => step controller controls DMAX/DMIN <= #

        if opt_type in ['TS', 'CLIMB']:
            if ratio < 0.0 and abs(dEpre) > 0.05:
                logger.debug('sign problem, decreasing DMAX')
                self.DMAX /= 1.35
            elif ratio < 0.75 or ratio > 1.5:  # and abs(dEpre)>0.05:
                if self.print_level > 0:
                    logger.debug(' decreasing DMAX')
                if step < self.DMAX:
                    self.DMAX = step / 1.1
                else:
                    self.DMAX = self.DMAX / 1.2

            elif ratio > 0.85 and ratio < 1.3:
                if gradrms > (pgradrms + 0.0005):
                    logger.debug(' decreasing DMAX, gradrms increased')
                    self.DMAX -= self.DMAX / 10.0
                elif gradrms < (pgradrms + 0.0005):
                    if self.DMAX < 0.05:
                        logger.debug(' increased DMAX, gradrms decreased')
                        logger.debug(gradrms)
                        logger.debug(pgradrms)
                        logger.debug(' increasing DMAX')
                        self.DMAX = self.DMAX * 1.1
                    elif gradrms < (pgradrms - 0.0005) and ratio > 0.9 and ratio < 1.1:
                        self.DMAX = self.DMAX * 1.1

            if self.DMAX > 0.25:
                self.DMAX = 0.25
        else:
            if dE_iter > 0.001 and opt_type in ['UNCONSTRAINED', 'ICTAN']:
                if self.print_level > 0:
                    logger.debug(' decreasing DMAX')
                if step < self.DMAX:
                    self.DMAX = step / 1.5
                else:
                    self.DMAX = self.DMAX / 1.5
            elif (ratio < 0.25 or ratio > 1.5) and abs(dEpre) > 0.05:
                if self.print_level > 0:
                    logger.debug(' decreasing DMAX')
                if step < self.DMAX:
                    self.DMAX = step / 1.1
                else:
                    self.DMAX = self.DMAX / 1.2
            elif ratio > 0.75 and ratio < 1.25 and step > self.DMAX and gradrms < (pgradrms * 1.35):
                if self.print_level > 0:
                    logger.debug(' increasing DMAX')
                self.DMAX = self.DMAX * 1.1 + 0.01
            if self.DMAX > 0.25:
                self.DMAX = 0.25

        if self.DMAX < self.DMIN:
            self.DMAX = self.DMIN

    def eigenvector_step(self, molecule, g):
        SCALE = self.SCALEQN
        if molecule.newHess > 0:
            SCALE = self.SCALEQN * molecule.newHess
        if self.SCALEQN > 10.0:
            SCALE = 10.0

        if self.print_level > 1:
            logger.debug(f'new_hess {molecule.newHess}')

        if self.print_level > 1:
            logger.debug('constraints')
            logger.debug(molecule.constraints.T)

        P = np.eye(len(molecule.constraints), dtype=float)
        for c in molecule.constraints.T:
            P -= np.outer(c[:, np.newaxis], c[:, np.newaxis].T)
        self.Hessian = np.dot(np.dot(P, molecule.Hessian), P)

        e, v_temp = np.linalg.eigh(self.Hessian)
        gqe = np.dot(v_temp.T, g)
        lambda1 = self.set_lambda1('NOT-TS', e)

        if self.print_level > 1:
            logger.debug(f' eigenvalues  {e}')
        if self.print_level > 1:
            logger.debug(f' eigenvectors  {v_temp}')

        if self.print_level > 1:
            logger.debug(f' g  {g.T}')

        if self.print_level > 1:
            logger.debug(f' gqe  {gqe.T}')

        dqe0 = -gqe.flatten() / (e + lambda1) / SCALE
        dqe0 = [np.sign(i) * self.MAXAD if abs(i) > self.MAXAD else i for i in dqe0]
        dqe0 = np.asarray(dqe0)

        # => Convert step back to DLC basis <= #
        dq = np.dot(v_temp, dqe0)
        dq = [np.sign(i) * self.MAXAD if abs(i) > self.MAXAD else i for i in dq]
        dq = np.asarray(dq)

        dq = np.reshape(dq, (-1, 1))
        for c in molecule.constraints.T:
            dq -= np.dot(c[:, np.newaxis].T, dq) * c[:, np.newaxis]

        if self.print_level > 1:
            logger.debug(f' dq  {dq.T}')
        return np.reshape(dq, (-1, 1))

    # need to modify this only for the DLC region
    def TS_eigenvector_step(self, molecule, g, ictan):
        """
        Takes an eigenvector step using the Bofill updated Hessian ~1 negative eigenvalue in the
        direction of the reaction path.

        """
        SCALE = self.SCALEQN
        if molecule.newHess > 0:
            SCALE = self.SCALEQN * molecule.newHess
        if self.SCALEQN > 10.0:
            SCALE = 10.0

        # constraint vector
        norm = np.linalg.norm(ictan)
        C = ictan / norm
        Vecs = molecule.coord_basis
        Cn = block_matrix.dot(block_matrix.dot(Vecs, block_matrix.transpose(Vecs)), C)
        norm = np.linalg.norm(Cn)
        Cn = Cn / norm

        # => get eigensolution of Hessian <=
        self.Hessian = molecule.Hessian.copy()
        eigen, tmph = np.linalg.eigh(self.Hessian)  # nicd,nicd
        tmph = tmph.T

        # TODO nneg should be self and checked
        self.nneg = sum(1 for e in eigen if e < -0.01)

        # => Overlap metric <= #
        overlap = np.dot(block_matrix.dot(tmph, block_matrix.transpose(Vecs)), Cn)

        logger.debug(f' overlap {overlap[:4].T}')
        logger.debug(f' nneg {self.nneg}')
        # Max overlap metrics
        path_overlap, maxoln = self.maxol_w_Hess(overlap[0:4])
        logger.debug(f' t/ol {maxoln}: {path_overlap:3.2f}')

        # => set lamda1 scale factor <=#
        lambda1 = self.set_lambda1('TS', eigen, maxoln)

        self.maxol_good = True
        if path_overlap < self.HESS_TANG_TOL_TS:
            self.maxol_good = False

        if self.maxol_good:
            # => grad in eigenvector basis <= #
            gqe = np.dot(tmph, g)
            path_overlap_e_g = gqe[maxoln]
            logger.debug(f' gtse: {path_overlap_e_g[0]:1.4f} ')
            # save gtse in memory ...
            self.gtse = abs(path_overlap_e_g[0])
            # => calculate eigenvector step <=#
            dqe0 = np.zeros((molecule.num_coordinates, 1))
            for i in range(molecule.num_coordinates):
                if i != maxoln:
                    dqe0[i] = -gqe[i] / (abs(eigen[i]) + lambda1) / SCALE
            lambda0 = 0.0025
            dqe0[maxoln] = gqe[maxoln] / (abs(eigen[maxoln]) + lambda0) / SCALE

            # => Convert step back to DLC basis <= #
            dq = np.dot(tmph.T, dqe0)  # should it be transposed?
            dq = [np.sign(i) * self.MAXAD if abs(i) > self.MAXAD else i for i in dq]
            dq = np.asarray(dq)

            dq = np.reshape(dq, (-1, 1))
        else:
            # => if overlap is small use Cn as Constraint <= #
            molecule.update_coordinate_basis(ictan)
            g = molecule.gradient
            logger.debug(molecule.constraints.T)
            molecule.form_Hessian_in_basis()
            dq = self.eigenvector_step(molecule, g)

        return dq

    def maxol_w_Hess(self, overlap):
        # Max overlap metrics
        absoverlap = np.abs(overlap)
        path_overlap = np.max(absoverlap)
        path_overlap_n = np.argmax(absoverlap)
        return path_overlap, path_overlap_n

    def update_Hessian(self, molecule, mode='BFGS'):
        """
        mode 1 is BFGS, mode 2 is BOFILL
        """
        assert mode == 'BFGS' or mode == 'BOFILL', 'no update implemented with that mode'
        # do this even if mode==BOFILL
        change = self.update_bfgs(molecule)

        if isinstance(molecule.coord_obj, DelocalizedInternalCoordinates):
            molecule.update_Primitive_Hessian(change=change)
            if self.print_level > 1:
                logger.debug('change')
                logger.debug(change)
                logger.debug(' updated primitive internals Hessian')
                logger.debug(molecule.Primitive_Hessian)
            if mode == 'BFGS':
                molecule.form_Hessian_in_basis()
            if mode == 'BOFILL':
                change = self.update_bofill(molecule)
                molecule.update_Hessian(change)
        molecule.newHess -= 1

        return change

    def update_bfgs(self, molecule):
        if not isinstance(molecule.coord_obj, CartesianCoordinates):
            return self.update_bfgsp(molecule)
        else:
            raise NotImplementedError

    def update_bfgsp(self, molecule):
        if self.print_level > 1:
            logger.debug('In update bfgsp')
            logger.debug(f'dx_prim  {self.dx_prim.T}')
            logger.debug(f'dg_prim  {self.dg_prim.T}')

        Hdx = np.dot(molecule.Primitive_Hessian, self.dx_prim)
        dxHdx = np.dot(np.transpose(self.dx_prim), Hdx)
        dgdg = np.outer(self.dg_prim, self.dg_prim)
        dgtdx = np.dot(np.transpose(self.dg_prim), self.dx_prim)
        change = np.zeros_like(molecule.Primitive_Hessian)

        if self.print_level > 1:
            logger.debug(f'Hdx {Hdx.T}')
            logger.debug(f'dgtdx: {dgtdx.item():1.8f} dxHdx: {dxHdx.item():1.8f}')
            logger.debug(f'dgdg\n{dgdg}')

        if dgtdx > 0.0:
            if dgtdx < 0.001:
                dgtdx = 0.001
            change += dgdg / dgtdx
        if dxHdx > 0.0:
            if dxHdx < 0.001:
                dxHdx = 0.001
            change -= np.outer(Hdx, Hdx) / dxHdx

        return change

    def update_bofill(self, molecule):
        logger.debug(' in update bofill')

        G = np.copy(molecule.Hessian)  # nicd,nicd
        Gdx = np.dot(G, self.dx)  # (nicd,nicd)(nicd,1) = (nicd,1)
        dgmGdx = self.dg - Gdx  # (nicd,1)

        # MS
        dgmGdxtdx = np.dot(dgmGdx.T, self.dx)  # (1,nicd)(nicd,1)
        Gms = np.outer(dgmGdx, dgmGdx) / dgmGdxtdx

        # PSB
        dxdx = np.outer(self.dx, self.dx)
        dxtdx = np.dot(self.dx.T, self.dx)
        dxtdg = np.dot(self.dx.T, self.dg)
        dxtGdx = np.dot(self.dx.T, Gdx)
        dxtdx2 = dxtdx * dxtdx
        dxtdgmdxtGdx = dxtdg - dxtGdx
        Gpsb = np.outer(dgmGdx, self.dx) / dxtdx + np.outer(self.dx, dgmGdx) / dxtdx - dxtdgmdxtGdx * dxdx / dxtdx2

        # Bofill mixing
        dxtE = np.dot(self.dx.T, dgmGdx)  # (1,nicd)(nicd,1)
        EtE = np.dot(dgmGdx.T, dgmGdx)  # E is dgmGdx
        phi = 1.0 - dxtE * dxtE / (dxtdx * EtE)

        change = (1.0 - phi) * Gms + phi * Gpsb
        return change
