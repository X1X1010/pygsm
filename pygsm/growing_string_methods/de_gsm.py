from pygsm.utilities import nifty
from .main_gsm import MainGSM


class DE_GSM(MainGSM):

    def __init__(
            self,
            options,
    ):

        super(DE_GSM, self).__init__(options)

        print(" Assuming primitives are union!")
        print(" number of primitives is", self.nodes[0].num_primitives)

    # TODO Change rtype to a more meaningful argument name
    def go_gsm(self, max_iters=50, opt_steps=3, rtype=2):
        """
        rtype=2 Find and Climb TS,
        1 Climb with no exact find,
        0 turning of climbing image and TS search
        """
        self.set_V0()

        if not self.isRestarted:
            if self.growth_direction == 0:
                self.add_GSM_nodes(2)
            elif self.growth_direction == 1:
                self.add_GSM_nodeR(1)
            elif self.growth_direction == 2:
                self.add_GSM_nodeP(1)

            # Grow String
            self.grow_string(max_iters=max_iters, max_opt_steps=opt_steps)
            nifty.printcool("Done Growing the String!!!")
            self.done_growing = True

            # nifty.printcool("initial ic_reparam")
            self.reparameterize()
            # self.xyz_writer('grown_string_{:03}.xyz'.format(self.ID), self.geometries, self.energies, self.gradrmss, self.dEs)

        if self.tscontinue:
            self.optimize_string(max_iter=max_iters, opt_steps=opt_steps, rtype=rtype)
        else:
            print("Exiting early")
            self.end_early = True

        print("Finished GSM!")


    def add_GSM_nodes(self, newnodes=1):
        if self.current_nnodes+newnodes > self.nnodes:
            print("Adding too many nodes, cannot add_GSM_node")
        sign = -1
        for i in range(newnodes):
            sign *= -1
            if sign == 1:
                self.add_GSM_nodeR()
            else:
                self.add_GSM_nodeP()

    def set_active(self, nR, nP):
        # print(" Here is active:",self.active)
        if nR != nP and self.growth_direction == 0:
            print((" setting active nodes to %i and %i" % (nR, nP)))
        elif self.growth_direction == 1:
            print((" setting active node to %i " % nR))
        elif self.growth_direction == 2:
            print((" setting active node to %i " % nP))
        else:
            print((" setting active node to %i " % nR))

        for i in range(self.nnodes):
            if self.nodes[i] is not None:
                self.optimizer[i].conv_grms = self.CONV_TOL*2.
        self.optimizer[nR].conv_grms = self.options['ADD_NODE_TOL']
        self.optimizer[nP].conv_grms = self.options['ADD_NODE_TOL']
        print(" conv_tol of node %d is %.4f" % (nR, self.optimizer[nR].conv_grms))
        print(" conv_tol of node %d is %.4f" % (nP, self.optimizer[nP].conv_grms))
        self.active[nR] = True
        self.active[nP] = True
        if self.growth_direction == 1:
            self.active[nP] = False
        if self.growth_direction == 2:
            self.active[nR] = False
        # print(" Here is new active:",self.active)

    def check_if_grown(self):
        '''
        Check if the string is grown
        Returns True if grown 
        '''

        return self.current_nnodes == self.nnodes

    def grow_nodes(self):
        '''
        Grow nodes
        '''

        if self.nodes[self.nR-1].gradrms < self.gaddmax and self.growth_direction != 2:
            if self.nodes[self.nR] is None:
                self.add_GSM_nodeR()
                print(" getting energy for node %d: %5.4f" % (self.nR-1, self.nodes[self.nR-1].energy - self.nodes[0].V0))
        if self.nodes[self.nnodes-self.nP].gradrms < self.gaddmax and self.growth_direction != 1:
            if self.nodes[-self.nP-1] is None:
                self.add_GSM_nodeP()
                print(" getting energy for node %d: %5.4f" % (self.nnodes-self.nP, self.nodes[-self.nP].energy - self.nodes[0].V0))
        return

    def make_tan_list(self):
        ncurrent, nlist = self.make_difference_node_list()
        param_list = []
        for n in range(ncurrent-2):
            if nlist[2*n] not in param_list:
                param_list.append(nlist[2*n])
        return param_list

    def make_move_list(self):
        ncurrent, nlist = self.make_difference_node_list()
        param_list = []
        for n in range(ncurrent):
            if nlist[2*n+1] not in param_list:
                param_list.append(nlist[2*n+1])
        return param_list

    def make_difference_node_list(self):
        '''
        Returns ncurrent and a list of indices that can be iterated over to produce
        tangents for the string pathway.
        '''
        # TODO: THis can probably be done more succinctly using a list of tuples
        ncurrent = 0
        nlist = [0]*(2*self.nnodes)
        for n in range(self.nR-1):
            nlist[2*ncurrent] = n
            nlist[2*ncurrent+1] = n+1
            ncurrent += 1

        for n in range(self.nnodes-self.nP+1, self.nnodes):
            nlist[2*ncurrent] = n
            nlist[2*ncurrent+1] = n-1
            ncurrent += 1

        nlist[2*ncurrent] = self.nR - 1
        nlist[2*ncurrent+1] = self.nnodes - self.nP

        if False:
            nlist[2*ncurrent+1] = self.nR - 2  # for isMAP_SE

        # TODO is this actually used?
        # if self.nR == 0: nlist[2*ncurrent] += 1
        # if self.nP == 0: nlist[2*ncurrent+1] -= 1
        ncurrent += 1
        nlist[2*ncurrent] = self.nnodes - self.nP
        nlist[2*ncurrent+1] = self.nR-1
        # #TODO is this actually used?
        # if self.nR == 0: nlist[2*ncurrent+1] += 1
        # if self.nP == 0: nlist[2*ncurrent] -= 1
        ncurrent += 1

        return ncurrent, nlist

    def set_V0(self):
        self.nodes[0].V0 = self.nodes[0].energy

        # TODO should be actual gradient
        self.nodes[0].gradrms = 0.
        if self.growth_direction != 1:
            self.nodes[-1].gradrms = 0.
            print(" Energy of the end points are %4.3f, %4.3f" % (self.nodes[0].energy, self.nodes[-1].energy))
            print(" relative E %4.3f, %4.3f" % (0.0, self.nodes[-1].energy-self.nodes[0].energy))
        else:
            print(" Energy of end points are %4.3f " % self.nodes[0].energy)
            # self.nodes[-1].energy = self.nodes[0].energy
            # self.nodes[-1].gradrms = 0.
