# standard library imports
import sys
import os
import re
import subprocess

# third party
import numpy as np

# local application imports
from ..utilities import manage_xyz
from .base_lot import Lot


class Molpro(Lot):

    def __init__(self, options):
        super(Molpro, self).__init__(options)

        self.file_options.set_active('basis', '6-31g', str, '')
        self.file_options.set_active('closed', None, int, '')
        self.file_options.set_active('occ', None, int, '')
        self.file_options.set_active('n_electrons', None, int, '')
        self.file_options.set_active('memory', 800, int, '')
        self.file_options.set_active('n_states', 2, int, '')

        # set all active values to self for easy access
        for key in self.file_options.ActiveOptions:
            setattr(self, key, self.file_options.ActiveOptions[key])

        if self.has_nelectrons is False:
            for i in self.states:
                self.get_nelec(self.geom, i[0])

    def write_input_file(self, tempfilename):
        # TODO gopro needs a number
        tempfile = open(tempfilename, 'w')
        tempfile.write(f' file,2,mp_{self.ID:04d}_{self.node_id:04d}\n')
        tempfile.write(f' memory,{self.memory},m\n')
        tempfile.write(' symmetry,nosym\n')
        tempfile.write(' orient,noorient\n\n')
        tempfile.write(' geometry={\n')
        for coord in geom:
            for i in coord:
                tempfile.write(' '+str(i))
            tempfile.write('\n')
        tempfile.write('}\n\n')

        # get states
        singlets = self.search_tuple(self.states, 1)
        len_singlets = len(singlets)
        # len_E_singlets = singlets[-1][1] + 1  # final adiabat +1 because 0 indexed
        triplets = self.search_tuple(self.states, 3)
        len_triplets = len(triplets)

        tempfile.write(f' basis={self.basis}\n\n')

        # do singlets
        if len_singlets != 0:
            tempfile.write(' {multi\n')
            tempfile.write(' direct\n')
            tempfile.write(f' closed,{self.closed}\n')
            tempfile.write(f' occ,{self.occ}\n')
            tempfile.write(f' wf,{self.n_electrons},1,0\n')
            # this can be made the len of singlets
            tempfile.write(f' state,{self.n_states}\n')

            for state in self.gradient_states:
                s = state[1]
                print("running grad state ", s)
                grad_name = "510"+str(s)+".1"
                tempfile.write(f' CPMCSCF,GRAD,{s+1}.1,record={grad_name}\n')

            # TODO this can only do coupling if states is 2, want to generalize to 3 states
            if self.coupling_states:
                tempfile.write(f'CPMCSCF,NACM,{self.coupling_states[0]}.1,{self.coupling_states[1]}.1,record=5200.1\n')
            tempfile.write(' }\n')

            for state in self.gradient_states:
                s = state[1]
                grad_name = "510"+str(s)+".1"
                tempfile.write(f'Force;SAMC,{grad_name};varsav\n')
            if self.coupling_states:
                tempfile.write('Force;SAMC,5200.1;varsav\n')

        if len_triplets != 0:
            tempfile.write(' {multi\n')
            nclosed = self.nocc
            nocc = nclosed+self.nactive
            tempfile.write(f' closed,{nclosed}\n')
            tempfile.write(f' occ,{nocc}\n')
            tempfile.write(f' wf,{self.n_electrons},1,2\n')
            # nstates = len(self.states)
            tempfile.write(f' state,{len_triplets}\n')

            for state in triplets:
                s = state[1]
                grad_name = "511"+str(s)+".1"
                tempfile.write(f' CPMCSCF,GRAD,{s+1}.1,record={grad_name}\n')
            tempfile.write(' }\n')

            for s in triplets:
                s = state[1]
                grad_name = "511"+str(s)+".1"
                tempfile.write(f'Force;SAMC,{grad_name};varsav\n')

        tempfile.close()

    # designed to do multiple multiplicities at once... maybe not such a good idea, that feature is currently broken
    # TODO
    def runall(self, geom, runtype=None):

        self.Gradients = {}
        self.Energies = {}
        self.Couplings = {}
        tempfilename = f'scratch/{self.node_id}/gopro.com'
        try:
            scratch = os.environ['SLURM_LOCAL_SCRATCH']
            args = ['-W', 'scratch', '-n', str(self.nproc), tempfilename, '-d', scratch]
        except:
            args = [os.environ['MOLPRO_OPTIONS'], '-n', str(self.nproc), tempfilename]

        print(args)

        # WRite input file
        self.write_input_file(tempfilename)

        # Run Process
        command = ['molpro']
        command.extend(args)
        output = subprocess.Popen(command, stdout=open(f'scratch/{self.node_id}/gopro.out', 'w'), stderr=subprocess.PIPE).communicate()
        print(output[0])

        # Parse
        self.parse()
        self.write_E_to_file()

        self.hasRanForCurrentCoords = True
        return

    def parse(self):
        # Now read the output
        tempfileout = f'scratch/{self.node_id}/gopro.out'
        pattern = re.compile(r'MCSCF STATE \d.1 Energy \s* ([-+]?[0-9]*\.?[0-9]+)')
        tmp = []
        for i, line in enumerate(open(tempfileout)):
            for match in re.finditer(pattern, line):
                tmp.append(float(match.group(1)))

        for state, E in zip(self.states, tmp):
            self._Energies[state] = self.Energy(E, 'Hartree')

        tmpgrada = []
        tmpgrad = []
        tmpcoup = []
        with open(tempfileout, "r") as f:
            for line in f:
                if line.startswith("GRADIENT FOR STATE", 7):  # will work for SA-MC and RSPT2 HF
                    for i in range(3):
                        next(f)
                    for i in range(len(geom)):
                        findline = next(f, '').strip()
                        mobj = re.match(r'^\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$', findline)
                        tmpgrad.append([
                            float(mobj.group(2)),
                            float(mobj.group(3)),
                            float(mobj.group(4)),
                        ])
                    tmpgrada.append(tmpgrad)
                    tmpgrad = []
                if line.startswith(" SA-MC NACME FOR STATES"):
                    for i in range(3):
                        next(f)
                    for i in range(len(geom)):
                        findline = next(f, '').strip()
                        mobj = re.match(r'^\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$', findline)
                        tmpcoup.append([
                            float(mobj.group(2)),
                            float(mobj.group(3)),
                            float(mobj.group(4)),
                        ])
                    self.Couplings[self.coupling_states] = self.Coupling(np.asarray(tmpcoup), "Hartree/Bohr")
        for state, grad in zip(self.states, tmpgrada):
            self._Gradients[state] = self.Gradient(np.asarray(grad), "Hartree/Bohr")

    @classmethod
    def copy(cls, lot, options, copy_wavefunction=True):
        """ create a copy of this lot object"""
        # print(" creating copy, new node id =",node_id)
        # print(" old node id = ",self.node_id)
        node_id = options.get('node_id', 1)
        if node_id != lot.node_id and copy_wavefunction:
            cmd = f"cp scratch/mp_{lot.ID:04d}_{lot.node_id:04d} scratch/mp_{lot.ID:04d}_{node_id:04d}"
            print(cmd)
            os.system(cmd)
        return cls(lot.options.copy().set_values(options))


if __name__ == '__main__':
    filepath = "../../data/ethylene.xyz"
    molpro = Molpro.from_options(states=[(1, 0), (1, 1)], fnm=filepath, lot_inp_file='../../data/ethylene_molpro.com', coupling_states=(0, 1))
    geom = manage_xyz.read_xyz(filepath)
    xyz = manage_xyz.xyz_to_np(geom)
    print(molpro.get_energy(xyz, 1, 0))
    print(molpro.get_gradient(xyz, 1, 0))
    print(molpro.get_coupling(xyz, 1, 0, 1))
