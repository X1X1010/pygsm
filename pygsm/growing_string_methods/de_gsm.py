from pygsm.utilities import output
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
            output.printcool("Done Growing the String!!!")
            self.done_growing = True

            self.reparameterize()

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
        if nR != nP and self.growth_direction == 0:
            print((f" setting active nodes to {nR} and {nP}"))
        elif self.growth_direction == 1:
            print((f" setting active node to {nR} "))
        elif self.growth_direction == 2:
            print((f" setting active node to {nP} "))
        else:
            print((f" setting active node to {nR} "))

        for i in range(self.nnodes):
            if self.nodes[i] is not None:
                self.optimizer[i].conv_grms = self.CONV_TOL*2.
        self.optimizer[nR].conv_grms = self.options['ADD_NODE_TOL']
        self.optimizer[nP].conv_grms = self.options['ADD_NODE_TOL']
        print(f" conv_tol of node {nR} is {self.optimizer[nR].conv_grms:.4f}")
        print(f" conv_tol of node {nP} is {self.optimizer[nP].conv_grms:.4f}")
        self.active[nR] = True
        self.active[nP] = True
        if self.growth_direction == 1:
            self.active[nP] = False
        if self.growth_direction == 2:
            self.active[nR] = False

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
                print(f" getting energy for node {self.nR-1}: {self.nodes[self.nR-1].energy - self.nodes[0].V0:5.4f}")
        if self.nodes[self.nnodes-self.nP].gradrms < self.gaddmax and self.growth_direction != 1:
            if self.nodes[-self.nP-1] is None:
                self.add_GSM_nodeP()
                print(f" getting energy for node {self.nnodes-self.nP}: {self.nodes[-self.nP].energy - self.nodes[0].V0:5.4f}")
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
        ncurrent += 1
        nlist[2*ncurrent] = self.nnodes - self.nP
        nlist[2*ncurrent+1] = self.nR-1
        # #TODO is this actually used?
        ncurrent += 1

        return ncurrent, nlist

    def set_V0(self):
        self.nodes[0].V0 = self.nodes[0].energy

        # TODO should be actual gradient
        self.nodes[0].gradrms = 0.
        if self.growth_direction != 1:
            self.nodes[-1].gradrms = 0.
            print(f" Energy of the end points are {self.nodes[0].energy:4.3f}, {self.nodes[-1].energy:4.3f}")
            print(f" relative E {0.0:4.3f}, {self.nodes[-1].energy-self.nodes[0].energy:4.3f}")
        else:
            print(f" Energy of end points are {self.nodes[0].energy:4.3f} ")
