import math
import os
from collections import Counter
from copy import copy
from dataclasses import dataclass
from itertools import chain
from typing import Any

import numpy as np

from pygsm.coordinate_systems import Angle, Dihedral, Distance, OutOfPlane
from pygsm.molecule import Molecule
from pygsm.potential_energy_surfaces import PenaltyPES
from pygsm.utilities import output
from pygsm.utilities.config import Config, option
from pygsm.utilities.manage_xyz import write_molden_geoms

from ._shared import map_method

#######################################################################################
#### This class contains the main constructor, object properties and staticmethods ####
#######################################################################################


# TODO interpolate is still sloppy. It shouldn't create a new molecule node itself
# but should create the xyz. GSM should create the new molecule based off that xyz.
# TODO nconstraints in ic_reparam and write_iters is irrelevant


@dataclass(kw_only=True)
class GSMConfig(Config):
    reactant: Any = option(doc='Molecule for the reactant (initial) structure')
    product: Any = option(None, doc='Molecule for the product structure (double-ended methods only)')
    nnodes: int = option(1, doc='number of string nodes')
    optimizer: Any = option(
        doc='optimizer used for every node (e.g. eigenvector_follow); GSM adjusts most of its settings itself'
    )
    driving_coords: list = option(
        default_factory=list, doc='driving coordinates for single-ended methods, e.g. [("ADD", 1, 11)]; atoms 1-indexed'
    )
    CONV_TOL: float = option(0.0005, doc='node gradrms convergence threshold')
    CONV_gmax: float = option(0.001, doc='node max-gradient convergence threshold')
    CONV_Ediff: float = option(0.1, doc='node energy-change convergence threshold')
    CONV_dE: float = option(0.5, doc='state energy-difference convergence threshold')
    ADD_NODE_TOL: float = option(0.1, doc='gradrms below which the frontier node is converged enough to add the next')
    growth_direction: int = option(
        0, doc='0: grow from both ends, 1: from the reactant, 2: from the product', choices=(0, 1, 2)
    )
    DQMAG_MAX: float = option(0.8, doc='max step along the tangent for single-ended methods')
    DQMAG_MIN: float = option(0.2, doc='min step along the tangent for single-ended methods')
    print_level: int = option(1)
    xyz_writer: Any = option(write_molden_geoms, doc='function used to write string trajectories')
    mp_cores: int = option(1, doc='processes used for parallel node updates (use with caution)')
    BDIST_RATIO: float = option(
        0.5, doc='single-ended growth is complete once bdist drops below (1 - BDIST_RATIO) of its initial value'
    )
    ID: int = option(0, doc='string id used in file names')
    interp_method: str = option('DLC', doc='interpolation/reparametrisation method', choices=('Geodesic', 'DLC'))
    noise: float = option(100.0, doc='energy noise used when checking for intermediates')
    scratch_dir: str = option('scratch', doc='directory for per-node optimisation output and string trajectories')


class GSM:
    """String of nodes (Molecules) with tangents, interpolation and reparametrisation; base of the GSM drivers."""

    config_class = GSMConfig

    #: True for strings that start from both reactant and product (DE-GSM).
    is_double_ended = False
    #: True for the single-ended crossing search (SE-Cross), which optimises on a penalty PES.
    is_crossing_search = False

    def __init__(self, options=None, **kwargs):
        self.options = self.config_class.build(options, **kwargs)
        options = self.options

        self.scratch_dir = self.options.scratch_dir

        # Cache attributes
        self.nnodes: int = self.options.nnodes
        self.nodes: list[Molecule] = [None for _ in range(self.nnodes)]
        self.nodes[0] = self.options.reactant
        self.nodes[-1] = self.options.product
        self.driving_coords = self.options.driving_coords
        self.growth_direction = self.options.growth_direction
        self.isRestarted = False
        self.DQMAG_MAX = self.options.DQMAG_MAX
        self.DQMAG_MIN = self.options.DQMAG_MIN
        self.BDIST_RATIO = self.options.BDIST_RATIO
        self.ID = self.options.ID
        self.optimizer = []
        self.interp_method = self.options.interp_method
        self.CONV_TOL = self.options.CONV_TOL
        self.noise = self.options.noise
        self.mp_cores = self.options.mp_cores
        self.xyz_writer = self.options.xyz_writer

        optimizer = options.optimizer
        for _count in range(self.nnodes):
            self.optimizer.append(optimizer.__class__(optimizer.options.replace()))
        self.print_level = options.print_level

        # Set initial values
        self.current_nnodes = 2
        self.nR = 1
        self.nP = 1
        self.climb = False
        self.find = False
        self.ts_exsteps = 3  # multiplier for ts node
        self.n0 = 1  # something to do with added nodes? "first node along current block"
        self.end_early = False
        self.tscontinue = True  # whether to continue with TS opt or not
        self.found_ts = False
        self.rn3m6 = np.sqrt(3.0 * self.nodes[0].natoms - 6.0)
        self.gaddmax = self.options.ADD_NODE_TOL  # self.options.ADD_NODE_TOL/self.rn3m6;
        print(' gaddmax:', self.gaddmax)
        self.ictan = [None] * self.nnodes
        self.active = [False] * self.nnodes
        self.climber = False  # is this string a climber?
        self.finder = False  # is this string a finder?
        self.done_growing = False
        self.nclimb = 0
        self.nhessreset = 10  # are these used??? TODO
        self.hessrcount = 0  # are these used?!  TODO
        self.hess_counter = 0  # it is probably good to reset the hessian
        self.newclimbscale = 2.0
        self.TS_E_0 = None
        self.dE_iter = 100.0  # change in max TS node

        self.nopt_intermediate = 0  # might be a duplicate of endearly_counter
        self.flag_intermediate = False
        self.endearly_counter = 0  # Find the intermediate x time
        self.pot_min = []
        self.ran_out = False  # if it ran out of iterations
        self._cached_energies = None
        self._cached_energy_signature = None

        self.newic = Molecule.copy_from(self.nodes[0])  # newic object is used for coordinate transformations

    def node_scratch_path(self, n):
        """Scratch directory for node ``n`` of this string (used for optimisation trajectories)."""
        return os.path.join(os.getcwd(), self.scratch_dir, f'{self.ID:03d}', str(n))

    @property
    def TSnode(self):
        """
        The current node with maximum energy
        """
        # Treat GSM with penalty a little different since penalty will increase energy based on energy
        # differences, which might not be great for Climbing Image
        if not self.is_crossing_search and isinstance(self.nodes[0].PES, PenaltyPES):
            energies = np.asarray([0.0] * self.nnodes)
            for i, node in enumerate(self.nodes):
                if node is not None:
                    energies[i] = (node.PES.PES1.energy + node.PES.PES2.energy) / 2.0
            return int(np.argmax(energies))
        else:
            # make sure TS is not zero or last node
            return int(np.argmax(self.energies[1 : self.nnodes - 1]) + 1)

    @property
    def emax(self):
        return self.energies[self.TSnode]

    @property
    def npeaks(self):
        """ """
        minnodes = []
        maxnodes = []
        energies = self.energies
        if energies[1] > energies[0]:
            minnodes.append(0)
        if energies[self.nnodes - 1] < energies[self.nnodes - 2]:
            minnodes.append(self.nnodes - 1)
        for n in range(self.n0, self.nnodes - 1):
            if energies[n + 1] > energies[n] and energies[n] < energies[n - 1]:
                minnodes.append(n)
            if energies[n + 1] < energies[n] and energies[n] > energies[n - 1]:
                maxnodes.append(n)

        return len(maxnodes)

    @property
    def energies(self):
        """
        Energies of string
        """
        signature = tuple(None if node is None else (id(node), node.state_token) for node in self.nodes)
        if signature != self._cached_energy_signature:
            reference_energy = self.nodes[0].energy
            energies = []
            for node in self.nodes:
                if node is not None:
                    energies.append(node.energy - reference_energy)
            self._cached_energies = energies
            self._cached_energy_signature = signature
        return self._cached_energies

    @energies.setter
    def energies(self, list_of_E):
        """
        setter for energies
        """
        self.E = list_of_E

    @property
    def geometries(self):
        geoms = []
        for ico in self.nodes:
            if ico is not None:
                geoms.append(ico.geometry)
        return geoms

    @property
    def gradrmss(self):
        self._gradrmss = []
        for ico in self.nodes:
            if ico is not None:
                gradrms = ico.gradrms
                self._gradrmss.append(gradrms)
        return self._gradrmss

    @property
    def dEs(self):
        self._dEs = []
        for ico in self.nodes:
            if ico is not None:
                dE = ico.difference_energy if isinstance(ico.difference_energy, float) else ico.difference_energy.item()
                self._dEs.append(dE)
        return self._dEs

    @property
    def ictan(self):
        return self._ictan

    @ictan.setter
    def ictan(self, value):
        self._ictan = value

    @property
    def dqmaga(self):
        return self._dqmaga

    @dqmaga.setter
    def dqmaga(self, value):
        self._dqmaga = value

    @staticmethod
    def add_node(nodeR, nodeP, stepsize, node_id, **kwargs):
        """
        Add a node between  nodeR and nodeP or if nodeP is none use driving coordinate to add new node
        """

        # get driving coord
        driving_coords = kwargs.get('driving_coords')
        DQMAG_MAX = kwargs.get('DQMAG_MAX', 0.8)
        DQMAG_MIN = kwargs.get('DQMAG_MIN', 0.2)

        if nodeP is None:
            if driving_coords is None:
                raise RuntimeError("You didn't supply a driving coordinate and product node is None!")

            BDISTMIN = 0.05
            ictan, bdist = GSM.get_tangent(nodeR, None, driving_coords=driving_coords)

            if bdist < BDISTMIN:
                print(f'bdist too small {bdist:.3f}')
                return None
            new_node = Molecule.copy_from(nodeR, new_node_id=node_id)
            new_node.update_coordinate_basis(constraints=ictan)
            constraint = new_node.constraints[:, 0]
            sign = -1.0

            dqmag_scale = 1.5
            minmax = DQMAG_MAX - DQMAG_MIN
            a = bdist / dqmag_scale
            if a > 1.0:
                a = 1.0
            dqmag = sign * (DQMAG_MIN + minmax * a)
            if dqmag > DQMAG_MAX:
                dqmag = DQMAG_MAX
            print(f' dqmag: {dqmag:4.3f} from bdist: {bdist:4.3f}')

            dq0 = dqmag * constraint
            print(f' dq0[constraint]: {dqmag:1.3f}')

            new_node.update_xyz(dq0)
            new_node.bdist = bdist

        else:
            ictan, _ = GSM.get_tangent(nodeR, nodeP)
            nodeR.update_coordinate_basis(constraints=ictan)
            constraint = nodeR.constraints[:, 0]
            dqmag = np.linalg.norm(ictan)
            print(f' dqmag: {dqmag:1.3f}')
            sign = 1.0
            dqmag *= sign * stepsize
            print(f' scaled dqmag: {dqmag:1.3f}')

            dq0 = dqmag * constraint
            old_xyz = nodeR.xyz.copy()
            new_xyz = nodeR.coord_obj.newCartesian(old_xyz, dq0)
            new_node = Molecule.copy_from(MoleculeA=nodeR, xyz=new_xyz, new_node_id=node_id)

        return new_node

    @staticmethod
    def get_tangent_xyz(xyz1, xyz2, prim_coords):
        PMDiff = np.zeros(len(prim_coords))
        for k, prim in enumerate(prim_coords):
            if type(prim) is Distance:
                PMDiff[k] = 2.5 * prim.calcDiff(xyz2, xyz1)
            else:
                PMDiff[k] = prim.calcDiff(xyz2, xyz1)
        return np.reshape(PMDiff, (-1, 1))

    @staticmethod
    def get_tangent(node1, node2, print_level=1, **kwargs):
        """
        Get internal coordinate tangent between two nodes, assumes they have unique IDs
        """

        if node2 is not None and node1.node_id != node2.node_id:
            print(f' getting tangent from between {node2.node_id} {node1.node_id} pointing towards {node2.node_id}')

            PMDiff = np.zeros(node2.num_primitives)
            for k, prim in enumerate(node2.primitive_internal_coordinates):
                if type(prim) is Distance:
                    PMDiff[k] = 2.5 * prim.calcDiff(node2.xyz, node1.xyz)
                else:
                    PMDiff[k] = prim.calcDiff(node2.xyz, node1.xyz)

            return np.reshape(PMDiff, (-1, 1)), None
        else:
            print(' getting tangent from node ', node1.node_id)

            driving_coords = kwargs.get('driving_coords')
            assert driving_coords is not None, ' Driving coord is None!'

            c = Counter(elem[0] for elem in driving_coords)
            c['ADD']
            c['BREAK']
            c['nangles']
            c['ntorsions']

            ictan = np.zeros((node1.num_primitives, 1), dtype=float)
            bdist = 0.0
            atoms = node1.atoms
            xyz = node1.xyz.copy()

            for i in driving_coords:
                if 'ADD' in i:
                    # order indices to avoid duplicate bonds
                    if i[1] < i[2]:
                        index = [i[1] - 1, i[2] - 1]
                    else:
                        index = [i[2] - 1, i[1] - 1]

                    bond = Distance(index[0], index[1])
                    prim_idx = node1.coord_obj.Prims.dof_index(index, 'Distance')
                    if len(i) == 3:
                        # TODO why not just use the covalent radii?
                        d0 = (atoms[index[0]].vdw_radius + atoms[index[1]].vdw_radius) / 2.8
                    elif len(i) == 4:
                        d0 = i[3]
                    current_d = bond.value(xyz)

                    # TODO don't set tangent if value is too small
                    ictan[prim_idx] = -1 * (d0 - current_d)
                    # => calc bdist <=
                    if current_d > d0:
                        bdist += np.dot(ictan[prim_idx], ictan[prim_idx])
                    if print_level > 0:
                        print(
                            f' bond {(i[1], i[2])} target (less than): {d0:4.3f} current d: {current_d:4.3f} diff: {np.asarray(ictan[prim_idx]).item():4.3f} '
                        )

                elif 'BREAK' in i:
                    # order indices to avoid duplicate bonds
                    if i[1] < i[2]:
                        index = [i[1] - 1, i[2] - 1]
                    else:
                        index = [i[2] - 1, i[1] - 1]
                    bond = Distance(index[0], index[1])
                    prim_idx = node1.coord_obj.Prims.dof_index(index, 'Distance')
                    if len(i) == 3:
                        d0 = atoms[index[0]].vdw_radius + atoms[index[1]].vdw_radius
                    elif len(i) == 4:
                        d0 = i[3]

                    current_d = bond.value(xyz)
                    ictan[prim_idx] = -1 * (d0 - current_d)

                    # => calc bdist <=
                    if current_d < d0:
                        bdist += np.dot(ictan[prim_idx], ictan[prim_idx])

                    if print_level > 0:
                        print(
                            f' bond {(i[1], i[2])} target (greater than): {d0:4.3f}, current d: {current_d:4.3f} diff: {np.asarray(ictan[prim_idx]).item():4.3f} '
                        )
                elif 'ANGLE' in i:
                    if i[1] < i[3]:
                        index = [i[1] - 1, i[2] - 1, i[3] - 1]
                    else:
                        index = [i[3] - 1, i[2] - 1, i[1] - 1]
                    angle = Angle(index[0], index[1], index[2])
                    prim_idx = node1.coord_obj.Prims.dof_index(index, 'Angle')
                    anglet = i[4]
                    ang_value = angle.value(xyz)
                    ang_diff = anglet * np.pi / 180.0 - ang_value
                    if print_level > 0:
                        print(f' anglev: {ang_value:4.3f} align to {anglet:4.3f} diff(rad): {ang_diff:4.3f}')
                    ictan[prim_idx] = -ang_diff
                    # TODO need to come up with an adist
                elif 'TORSION' in i:
                    if i[1] < i[4]:
                        index = [i[1] - 1, i[2] - 1, i[3] - 1, i[4] - 1]
                    else:
                        index = [i[4] - 1, i[3] - 1, i[2] - 1, i[1] - 1]
                    torsion = Dihedral(index[0], index[1], index[2], index[3])
                    prim_idx = node1.coord_obj.Prims.dof_index(index, 'Dihedral')
                    tort = i[5]
                    torv = torsion.value(xyz)
                    tor_diff = tort - torv * 180.0 / np.pi
                    if tor_diff > 180.0:
                        tor_diff -= 360.0
                    elif tor_diff < -180.0:
                        tor_diff += 360.0
                    ictan[prim_idx] = -tor_diff * np.pi / 180.0

                    if tor_diff * np.pi / 180.0 > 0.1 or tor_diff * np.pi / 180.0 < 0.1:
                        bdist += np.dot(ictan[prim_idx], ictan[prim_idx])
                    if print_level > 0:
                        print(
                            f' current torv: {torv * 180.0 / np.pi:4.3f} align to {tort:4.3f} diff(deg): {tor_diff:4.3f}'
                        )

                elif 'OOP' in i:
                    index = [i[1] - 1, i[2] - 1, i[3] - 1, i[4] - 1]
                    oop = OutOfPlane(index[0], index[1], index[2], index[3])
                    prim_idx = node1.coord_obj.Prims.dof_index(index, 'OutOfPlane')
                    oopt = i[5]
                    oopv = oop.value(xyz)
                    oop_diff = oopt - oopv * 180.0 / np.pi
                    if oop_diff > 180.0:
                        oop_diff -= 360.0
                    elif oop_diff < -180.0:
                        oop_diff += 360.0
                    ictan[prim_idx] = -oop_diff * np.pi / 180.0

                    if oop_diff * np.pi / 180.0 > 0.1 or oop_diff * np.pi / 180.0 < 0.1:
                        bdist += np.dot(ictan[prim_idx], ictan[prim_idx])
                    if print_level > 0:
                        print(
                            f' current oopv: {oopv * 180.0 / np.pi:4.3f} align to {oopt:4.3f} diff(deg): {oop_diff:4.3f}'
                        )

            bdist = np.sqrt(bdist)
            if np.all(ictan == 0.0):
                raise RuntimeError(' All elements are zero')
            return ictan, bdist

    @staticmethod
    def get_tangents(nodes, n0=0, print_level=0):
        """
        Get the normalized internal coordinate tangents and magnitudes between all nodes
        """
        nnodes = len(nodes)
        dqmaga = [0.0] * nnodes
        ictan = [[]] * nnodes

        for n in range(n0 + 1, nnodes):
            assert nodes[n] is not None, 'n is bad'
            assert nodes[n - 1] is not None, 'n-1 is bad'
            ictan[n] = GSM.get_tangent_xyz(nodes[n - 1].xyz, nodes[n].xyz, nodes[0].primitive_internal_coordinates)

            dqmaga[n] = 0.0
            dqmaga[n] = np.linalg.norm(ictan[n])

            ictan[n] /= dqmaga[n]

            # NOTE:
            # vanilla GSM has a strange metric for distance
            # no longer following 7/1/2020
            # just a fancy way to get the normalized tangent vector
            if dqmaga[n] < 0.0:
                raise RuntimeError

        # TEMPORORARY parallel idea

        if print_level > 1:
            print('------------printing ictan[:]-------------')
            for n in range(n0 + 1, nnodes):
                print(f'ictan[{n}]')
                print(ictan[n].T)
        if print_level > 0:
            print('------------printing dqmaga---------------')
            for n in range(n0 + 1, nnodes):
                print(f' {dqmaga[n]:5.4}', end='')
                if (n) % 5 == 0:
                    print()
            print()
        return ictan, dqmaga

    @staticmethod
    def get_three_way_tangents(nodes, energies, find=True, n0=0, print_level=0):
        """
        Calculates internal coordinate tangent with a three-way tangent at TS node
        """
        nnodes = len(nodes)
        ictan = [[]] * nnodes
        dqmaga = [0.0] * nnodes
        TSnode = np.argmax(energies)  # allow for the possibility of TS node to be endpoints?

        last_node_max = TSnode == nnodes - 1
        first_node_max = TSnode == 0
        if first_node_max or last_node_max:
            print('*********** This will cause a range error in the following for loop *********')
            print('** Setting the middle of the string to be TS node to get proper directions **')
            TSnode = nnodes // 2

        for n in range(n0, nnodes):
            do3 = False
            print('getting tan[{' + str(n) + '}]')
            if n < TSnode:
                # The order is very important here
                # the way it should be ;(
                intic_n = n + 1
                newic_n = n

                # old way

            elif n > TSnode:
                # The order is very important here
                intic_n = n
                newic_n = n - 1
            else:
                do3 = True
                newic_n = n
                intic_n = n + 1
                int2ic_n = n - 1

            if do3:
                if first_node_max or last_node_max:
                    t1, _ = GSM.get_tangent(nodes[intic_n], nodes[newic_n])
                    t2, _ = GSM.get_tangent(nodes[newic_n], nodes[int2ic_n])
                    print(' done 3 way tangent')
                    ictan0 = t1 + t2
                else:
                    f1 = 0.0
                    dE1 = abs(energies[n + 1] - energies[n])
                    dE2 = abs(energies[n] - energies[n - 1])
                    dEmax = max(dE1, dE2)
                    dEmin = min(dE1, dE2)
                    if energies[n + 1] > energies[n - 1]:
                        f1 = dEmax / (dEmax + dEmin + 0.00000001)
                    else:
                        f1 = 1 - dEmax / (dEmax + dEmin + 0.00000001)

                    print(f' 3 way tangent ({n}): f1:{f1:3.2}')

                    t1, _ = GSM.get_tangent(nodes[intic_n], nodes[newic_n])
                    t2, _ = GSM.get_tangent(nodes[newic_n], nodes[int2ic_n])
                    print(' done 3 way tangent')
                    ictan0 = f1 * t1 + (1.0 - f1) * t2
            else:
                ictan0, _ = GSM.get_tangent(nodes[newic_n], nodes[intic_n])

            ictan[n] = ictan0 / np.linalg.norm(ictan0)
            dqmaga[n] = np.linalg.norm(ictan0)

        return ictan, dqmaga

    @staticmethod
    def ic_reparam(nodes, energies, climbing=False, ic_reparam_steps=8, print_level=1, NUM_CORE=1, MAXRE=0.25):
        """
        Reparameterizes the string using Delocalizedin internal coordinatesusing three-way tangents at the TS node
        Only pushes nodes outwards during reparameterization because otherwise too many things change.
            Be careful, however, if the path is allup or alldown then this can cause
        Parameters
        ----------
        nodes : list of molecule objects
        energies : list of energies in kcal/mol
        ic_reparam_steps : int max number of reparameterization steps
        print_level : int verbosity
        """
        output.printcool('reparametrizing string nodes')

        nnodes = len(nodes)
        rpart = np.zeros(nnodes)
        for n in range(1, nnodes):
            rpart[n] = 1.0 / (nnodes - 1)
        deltadqs = np.zeros(nnodes)
        TSnode = np.argmax(energies)
        disprms = 100
        if ((TSnode == nnodes - 1) or (TSnode == 0)) and climbing:
            raise RuntimeError(" TS node shouldn't be the first or last node")

        ideal_progress_gained = np.zeros(nnodes)
        if climbing:
            for n in range(1, TSnode):
                ideal_progress_gained[n] = 1.0 / (TSnode)
            for n in range(TSnode + 1, nnodes):
                ideal_progress_gained[n] = 1.0 / (nnodes - TSnode - 1)
            ideal_progress_gained[TSnode] = 0.0
        else:
            for n in range(1, nnodes):
                ideal_progress_gained[n] = 1.0 / (nnodes - 1)

        for i in range(ic_reparam_steps):
            ictan, dqmaga = GSM.get_tangents(nodes)
            totaldqmag = np.sum(dqmaga)

            if climbing:
                progress = np.zeros(nnodes)
                progress_gained = np.zeros(nnodes)
                h1dqmag = np.sum(dqmaga[: TSnode + 1])
                h2dqmag = np.sum(dqmaga[TSnode + 1 : nnodes])
                if print_level > 0:
                    print(f' h1dqmag, h2dqmag: {h1dqmag:3.2f} {h2dqmag:3.2f}')
                progress_gained[:TSnode] = dqmaga[:TSnode] / h1dqmag
                progress_gained[TSnode + 1 :] = dqmaga[TSnode + 1 :] / h2dqmag
                progress[:TSnode] = np.cumsum(progress_gained[:TSnode])
                progress[TSnode:] = np.cumsum(progress_gained[TSnode:])
            else:
                progress = np.cumsum(dqmaga) / totaldqmag
                progress_gained = dqmaga / totaldqmag

            if i == 0:
                orig_dqmaga = copy(dqmaga)
                orig_progress_gained = copy(progress_gained)

            if climbing:
                difference = np.zeros(nnodes)
                for n in range(TSnode):
                    difference[n] = ideal_progress_gained[n] - progress_gained[n]
                    deltadqs[n] = difference[n] * h1dqmag
                for n in range(TSnode + 1, nnodes):
                    difference[n] = ideal_progress_gained[n] - progress_gained[n]
                    deltadqs[n] = difference[n] * h2dqmag
            else:
                difference = ideal_progress_gained - progress_gained
                deltadqs = difference * totaldqmag

            if print_level > 1:
                print(' ideal progress gained per step', end=' ')
                for n in range(nnodes):
                    print(f' step [{n}]: {ideal_progress_gained[n]:1.3f}', end=' ')
                print()
                print(' path progress                 ', end=' ')
                for n in range(nnodes):
                    print(f' step [{n}]: {progress_gained[n]:1.3f}', end=' ')
                print()
                print(' difference                    ', end=' ')
                for n in range(nnodes):
                    print(f' step [{n}]: {difference[n]:1.3f}', end=' ')
                print()
                print(' deltadqs                      ', end=' ')
                for n in range(nnodes):
                    print(f' step [{n}]: {deltadqs[n]:1.3f}', end=' ')
                print()

            disprms = np.linalg.norm(deltadqs) / np.sqrt(nnodes - 1)
            print(f' disprms: {disprms:1.3}\n')

            if disprms < 0.02:
                break

            # Move nodes
            if climbing:
                deltadqs[TSnode - 2] -= deltadqs[TSnode - 1]
                deltadqs[nnodes - 2] -= deltadqs[nnodes - 1]
                for n in range(1, nnodes - 1):
                    if abs(deltadqs[n]) > MAXRE:
                        deltadqs[n] = np.sign(deltadqs[n]) * MAXRE
                for n in range(TSnode - 1):
                    deltadqs[n + 1] += deltadqs[n]
                for n in range(TSnode + 1, nnodes - 2):
                    deltadqs[n + 1] += deltadqs[n]
                for n in range(nnodes):
                    if abs(deltadqs[n]) > MAXRE:
                        deltadqs[n] = np.sign(deltadqs[n]) * MAXRE

                if NUM_CORE > 1:
                    # 5/14/2021 TS node this up?!
                    tans = [
                        ictan[n] if deltadqs[n] < 0 else ictan[n + 1]
                        for n in chain(range(1, TSnode), range(TSnode + 1, nnodes - 1))
                    ]  # + [ ictan[n] if deltadqs[n]<0 else ictan[n+1] for n in range(TSnode+1,nnodes-1)]
                    Vecs = map_method(
                        (
                            (nodes[0].coord_obj, 'build_dlc', node.xyz, tan)
                            for node, tan in zip(nodes[1:TSnode] + nodes[TSnode + 1 : nnodes - 1], tans)
                        ),
                        NUM_CORE,
                    )
                    for n, node in enumerate(nodes[1:TSnode] + nodes[TSnode + 1 : nnodes - 1]):
                        node.coord_basis = Vecs[n]

                    # move the positions
                    dqs = [
                        deltadqs[n] * nodes[n].constraints[:, 0]
                        for n in chain(range(1, TSnode), range(TSnode + 1, nnodes - 1))
                    ]
                    newXyzs = map_method(
                        (
                            (node.coord_obj, 'newCartesian', node.xyz, dq)
                            for node, dq in zip(nodes[1:TSnode] + nodes[TSnode + 1 : nnodes - 1], dqs)
                        ),
                        NUM_CORE,
                    )
                    for n, node in enumerate(nodes[1:TSnode] + nodes[TSnode + 1 : nnodes - 1]):
                        node.xyz = newXyzs[n]
                else:
                    for n in chain(range(1, TSnode), range(TSnode + 1, nnodes - 1)):
                        if deltadqs[n] < 0:
                            print(f' Moving node {n} along tan[{n}] this much {deltadqs[n]}')
                            nodes[n].update_coordinate_basis(ictan[n])
                            constraint = nodes[n].constraints[:, 0]
                            dq = deltadqs[n] * constraint
                            nodes[n].update_xyz(dq, verbose=(print_level > 1))
                        elif deltadqs[n] > 0:
                            print(f' Moving node {n} along tan[{n + 1}] this much {deltadqs[n]}')
                            nodes[n].update_coordinate_basis(ictan[n + 1])
                            constraint = nodes[n].constraints[:, 0]
                            dq = deltadqs[n] * constraint
                            nodes[n].update_xyz(dq, verbose=(print_level > 1))
            else:
                # e.g 11-2 = 9, deltadq[9] -= deltadqs[10]
                deltadqs[nnodes - 2] -= deltadqs[nnodes - 1]
                for n in range(1, nnodes - 1):
                    if abs(deltadqs[n]) > MAXRE:
                        deltadqs[n] = np.sign(deltadqs[n]) * MAXRE
                for n in range(1, nnodes - 2):
                    deltadqs[n + 1] += deltadqs[n]
                for n in range(1, nnodes - 1):
                    if abs(deltadqs[n]) > MAXRE:
                        deltadqs[n] = np.sign(deltadqs[n]) * MAXRE

                if NUM_CORE > 1:
                    # Update the coordinate basis
                    tans = [ictan[n] if deltadqs[n] < 0 else ictan[n + 1] for n in range(1, nnodes - 1)]
                    Vecs = map_method(
                        (
                            (nodes[0].coord_obj, 'build_dlc', node.xyz, tan)
                            for node, tan in zip(nodes[1 : nnodes - 1], tans)
                        ),
                        NUM_CORE,
                    )
                    for n, node in enumerate(nodes[1 : nnodes - 1]):
                        node.coord_basis = Vecs[n]
                    # move the positions
                    dqs = [deltadqs[n] * nodes[n].constraints[:, 0] for n in range(1, nnodes - 1)]
                    newXyzs = map_method(
                        (
                            (node.coord_obj, 'newCartesian', node.xyz, dq)
                            for node, dq in zip(nodes[1 : nnodes - 1], dqs)
                        ),
                        NUM_CORE,
                    )
                    for n, node in enumerate(nodes[1 : nnodes - 1]):
                        node.xyz = newXyzs[n]
                else:
                    for n in range(1, nnodes - 1):
                        if deltadqs[n] < 0:
                            print(f' Moving node {n} along tan[{n}] this much {deltadqs[n]}')
                            nodes[n].update_coordinate_basis(ictan[n])
                            constraint = nodes[n].constraints[:, 0]
                            dq = deltadqs[n] * constraint
                            nodes[n].update_xyz(dq, verbose=(print_level > 1))
                        elif deltadqs[n] > 0:
                            print(f' Moving node {n} along tan[{n + 1}] this much {deltadqs[n]}')
                            nodes[n].update_coordinate_basis(ictan[n + 1])
                            constraint = nodes[n].constraints[:, 0]
                            dq = deltadqs[n] * constraint
                            nodes[n].update_xyz(dq, verbose=(print_level > 1))

        if climbing:
            ictan, dqmaga = GSM.get_tangents(nodes)
            h1dqmag = np.sum(dqmaga[: TSnode + 1])
            h2dqmag = np.sum(dqmaga[TSnode + 1 : nnodes])
            if print_level > 0:
                print(f' h1dqmag, h2dqmag: {h1dqmag:3.2f} {h2dqmag:3.2f}')
            progress_gained[:TSnode] = dqmaga[:TSnode] / h1dqmag
            progress_gained[TSnode + 1 :] = dqmaga[TSnode + 1 :] / h2dqmag
            progress[:TSnode] = np.cumsum(progress_gained[:TSnode])
            progress[TSnode:] = np.cumsum(progress_gained[TSnode:])
        else:
            ictan, dqmaga = GSM.get_tangents(nodes)
            totaldqmag = np.sum(dqmaga)
            progress = np.cumsum(dqmaga) / totaldqmag
            progress_gained = dqmaga / totaldqmag
        print()
        if print_level > 0:
            print(' ideal progress gained per step', end=' ')
            for n in range(nnodes):
                print(f' step [{n}]: {ideal_progress_gained[n]:1.3f}', end=' ')
            print()
            print(' original path progress        ', end=' ')
            for n in range(nnodes):
                print(f' step [{n}]: {orig_progress_gained[n]:1.3f}', end=' ')
            print()
            print(' reparameterized path progress ', end=' ')
            for n in range(nnodes):
                print(f' step [{n}]: {progress_gained[n]:1.3f}', end=' ')
            print()

        print(' spacings (begin ic_reparam, steps', end=' ')
        for n in range(nnodes):
            print(f' {orig_dqmaga[n]:1.2}', end=' ')
        print()
        print(f' spacings (end ic_reparam, steps: {i + 1}/{ic_reparam_steps}):', end=' ')
        for n in range(nnodes):
            print(f' {dqmaga[n]:1.2}', end=' ')
        print(f'\n  disprms: {disprms:1.3}')

        return

    # TODO move to string utils or delete altogether

    @staticmethod
    def calc_optimization_metrics(nodes: list[Molecule]):
        nnodes = len(nodes)
        rn3m6 = math.sqrt(3 * nodes[0].natoms - 6)
        totalgrad, gradrms, sum_gradrms = 0.0, 0.0, 0.0
        for i, ico in enumerate(nodes[1 : nnodes - 1]):
            if ico is None:
                continue
            ico_gradrms = ico.gradrms
            print(f' node: {i:02d} gradrms: {ico_gradrms:.6f}', end='')
            if i % 5 == 0:
                print()
            totalgrad += ico_gradrms * rn3m6
            gradrms += ico_gradrms**2
            sum_gradrms += ico_gradrms
        print()
        gradrms = math.sqrt(gradrms / (nnodes - 2))
        return totalgrad, gradrms, sum_gradrms
