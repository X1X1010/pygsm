# standard library imports
# local application imports
import sys
import os
import re

# third party
import numpy as np

from ..utilities import manage_xyz
from .base_lot import Lot
from .file_options import File_Options


class BAGEL(Lot):

    def __init__(self, options):
        super(BAGEL, self).__init__(options)

        print(f" making folder scratch/{self.node_id}")
        os.system(f'mkdir -p scratch/{self.node_id}')

        # regular options
        self.file_options.set_active('basis', '6-31g', str, '')
        self.file_options.set_active('df_basis', 'svp-jkfit', str, '')
        self.file_options.set_active('maxiter', 200, int, '')

        # CASSCF
        self.file_options.set_active('casscf', None, bool, '')
        self.file_options.set_active('nstate', None, int, '')
        self.file_options.set_active('nact', None, int, '')
        self.file_options.set_active('nclosed', None, int, '')

        # CASPT2 options
        self.file_options.set_active('caspt2', None, bool, '', depend=(
            self.file_options.casscf), msg='casscf must be True')
        self.file_options.set_active(
            'ms', True, bool, '', depend=(self.file_options.caspt2), msg='')
        self.file_options.set_active(
            'xms', True, bool, '', depend=(self.file_options.caspt2), msg='')
        self.file_options.set_active(
            'sssr', False, bool, '', depend=(self.file_options.caspt2), msg='')
        self.file_options.set_active(
            'shift', 0.3, float, '', depend=(self.file_options.caspt2), msg='')
        self.file_options.set_active(
            'frozen', True, bool, '', depend=(self.file_options.caspt2), msg='')

        keys_to_del = []
        for key, value in self.file_options.ActiveOptions.items():
            if value is None:
                keys_to_del.append(key)
        for key in keys_to_del:
            self.file_options.deactivate(key)

        if "casscf" in self.file_options.ActiveOptions:
            for key in ['nstate', 'nact', 'nclosed']:
                if key not in self.file_options.ActiveOptions:
                    raise RuntimeError

        keys_to_del = []
        if "caspt2" not in self.file_options.ActiveOptions:
            for key in ['ms', 'xms', 'sssr', 'shift', 'frozen']:
                print('deactivating key ', key)
                keys_to_del.append(key)
        for key in keys_to_del:
            self.file_options.deactivate(key)

        guess_file = f'scratch/{self.node_id}/orbs'
        self.file_options.set_active('load_ref', guess_file, str, doc='guess',
                                     depend=(os.path.isfile(guess_file+'.archive')), msg='ref does not exist or not needed, deactivating for now...')

        if self.node_id == 0:
            for line in self.file_options.record():
                print(line)

        # set all active values to self for easy access
        for key in self.file_options.ActiveOptions:
            setattr(self, key, self.file_options.ActiveOptions[key])

    @classmethod
    def copy(cls, lot, options, copy_wavefunction=True):

        # Get node id for new struct
        node_id = options.get('node_id', 1)
        print(f" making folder scratch/{node_id}")
        os.system(f'mkdir -p scratch/{node_id}')

        file_options = File_Options.copy(lot.file_options)
        options['file_options'] = file_options

        if node_id != lot.node_id and copy_wavefunction:
            old_path = f'scratch/{lot.node_id}/orbs.archive'
            new_path = f'scratch/{node_id}/orbs.archive'
            cmd = 'cp -r ' + old_path + ' ' + new_path
            print(f" copying scr files\n {cmd}")
            os.system(cmd)
            os.system('wait')
        return cls(lot.options.copy().set_values(options))

    def write_input(self, geom, runtype='gradient'):
        # filenames
        inpfilename = f'scratch/{self.node_id}/bagel.json'
        inpfile = open(inpfilename, 'w')

        # Header
        inpfile.write('{ "bagel" : [\n\n')

        # molecule section
        inpfile.write(('{\n'
                       ' "title" : "molecule",\n'))

        # basis set
        inpfile.write(f' "basis" : "{self.basis}",\n')
        inpfile.write(f' "df_basis" : "{self.df_basis}",\n')
        inpfile.write('"angstrom" : true,\n')

        # write geometry
        inpfile.write(' "geometry" : [\n')
        for atom in geom[:-1]:
            inpfile.write(f' {{ "atom" : "{atom[0]}", "xyz" : [ {atom[1]:14.6f}, {atom[2]:14.6f}, {atom[3]:14.6f} ] }},\n')
        inpfile.write(f' {{ "atom" : "{geom[-1][0]}", "xyz" : [ {geom[-1][1]:14.6f}, {geom[-1][2]:14.6f}, {geom[-1][3]:14.6f} ] }}\n')
        inpfile.write(']\n')
        inpfile.write('},\n')

        # Load reference
        if 'load_ref' in self.file_options.ActiveOptions:
            inpfile.write(('{ \n'
                           ' "title" : "load_ref", \n'
                           f' "file" : "scratch/{self.node_id}/orbs", \n'
                           ' "continue_geom" : false \n},\n'))

        if "casscf" in self.file_options.ActiveOptions:
            inpfile.write(('{\n'
                           ' "title" : "casscf", \n'
                           f' "nstate" : {self.nstate}, \n'
                           f' "nact" : {self.nact}, \n'
                           f' "nclosed" : {self.nclosed}, \n'
                           f' "maxiter": {self.maxiter} \n'
                           '  },\n\n'))
        else:
            print(" only casscf implemented now")
            raise NotImplementedError

        if runtype == "gradient":
            inpfile.write(('{\n'
                           '  "title" : "forces",\n'))

            inpfile.write('     "grads" : [\n')
            num_states = len(self.gradient_states)
            for i, state in enumerate(self.gradient_states):
                if i+1 == num_states:
                    inpfile.write(
                        f'     {{ "title" : "force", "target" : {state[1]} }}\n')
                else:
                    inpfile.write(
                        f'     {{ "title" : "force", "target" : {state[1]} }},\n')
            inpfile.write(' \n],\n')
        elif runtype == "gh":
            inpfile.write(('{\n'
                           '  "title" : "forces",\n'))

            inpfile.write('     "grads" : [\n')
            num_states = len(self.gradient_states)
            for i, state in enumerate(self.gradient_states):
                inpfile.write(
                    f'     {{ "title" : "force", "target" : {state[1]} }},\n')

            inpfile.write(f'     {{ "title"  : "nacme", "target1" : {self.coupling_states[0]}, "tartet2": {self.coupling_states[1]}, "nacmtype" : "full" }}\n')
            inpfile.write(' ],\n')

        if "caspt2" in self.file_options.ActiveOptions and (runtype == 'gh' or runtype == 'gradient'):
            inpfile.write((' "method" : [ {\n'
                           '  "title" : "caspt2",\n'
                           '  "smith" : {\n'
                           '    "method" : "caspt2",\n'
                           f'    "ms" : "{self.ms}",\n'
                           f'    "xms" : "{self.xms}",\n'
                           f'    "sssr" : "{self.sssr}",\n'
                           f'    "shift" : {self.shift},\n'
                           f'    "frozen" : "{self.frozen}"\n'
                           '  },\n'
                           f'  "nstate" : {self.nstate},\n'
                           f'  "nact" : {self.nact},\n'
                           f'  "nclosed" : {self.nclosed},\n'
                           '  "charge" : 0\n'
                           '  } ]\n'
                           '},\n'))
        elif runtype == 'energy':
            pass
        else:
            print("only caspt2 gradient implemented")
            raise NotImplementedError

        inpfile.write(('{\n'
                       '"title" : "save_ref",\n'
                       f'"file" : "scratch/{self.node_id}/orbs"\n'
                       '}\n'))

        inpfile.write(']}')
        inpfile.close()

    def parse(self, geom, runtype='gradient'):
        self.Gradients = {}
        self.Energies = {}
        self.Couplings = {}

        # Parse the output for Energies
        tempfileout = f'scratch/{self.node_id}/output.dat'
        tmp = []
        if "caspt2" in self.file_options.ActiveOptions:
            pattern = re.compile(
                r'\s. MS-CASPT2 energy : state  \d \s* ([-+]?[0-9]*\.?[0-9]+)')
            for line in open(tempfileout):
                for match in re.finditer(pattern, line):
                    tmp.append(float(match.group(1)))
        else:
            print("CASSCF not currently enabled")
            for i in self.states:
                tmp.append(0.0)

        for E, state in zip(tmp, self.states):
            self._Energies[state] = self.Energy(E, 'Hartree')

        # Save to file
        self.write_E_to_file()

        # Parse the output for Gradients
        if runtype == "gradient" or runtype == 'gh':
            tmpgrada = []
            tmpgrad = []
            tmpcoup = []
            count = 1
            with open(tempfileout, "r") as f:
                for line in f:
                    if line.startswith("* Nuclear energy gradient", 2) and count < 3:
                        next(f)
                        for i in range((len(geom)*4)):
                            findline = next(f, '').strip()
                            if findline.startswith("o Atom"):
                                pass
                            else:
                                mobj = re.match(
                                    r'. \s* ([-+]?[0-9]*\.?[0-9]+)', findline)
                                tmpgrad.append(float(mobj.group(1)))
                        tmpgrada.append(np.asarray(tmpgrad))
                        tmpgrad = []
                        count += 1
                    elif line.startswith("* Nuclear energy gradient", 2) and count == 3 and runtype == 'gh':
                        next(f)
                        for i in range((len(geom)*4)):
                            findline = next(f, '').strip()
                            if findline.startswith("o Atom"):
                                pass
                            else:
                                mobj = re.match(
                                    r'. \s* ([-+]?[0-9]*\.?[0-9]+)', findline)
                                tmpcoup.append(float(mobj.group(1)))
                        self.Couplings[self.coupling_states] = self.Coupling(
                            np.asarray(tmpcoup), 'Hartree/Bohr')
                        break

            for tup, tmpgrad in zip(self.gradient_states, tmpgrada):
                self._Gradients[tup] = self.Gradient(tmpgrad, 'Hartree/Bohr')

    def run(self, geom, runtype='gradient'):
        # first write the file, run it, and the read the output

        inpfilename = f'scratch/{self.node_id}/bagel.json'
        outfilename = f'scratch/{self.node_id}/output.dat'

        # Write
        self.write_input(geom, runtype)

        # Run BAGEL
        # RUN THE CALCULATION ###
        cmd = f"BAGEL {inpfilename} > {outfilename}"
        os.system(cmd)
        os.system('wait')

        # Parse the output for Energies
        self.parse(geom, runtype)

        # Turn on guess for calculations after running
        if 'load_ref' not in self.file_options.ActiveOptions:
            self.file_options.set_active(
                'load_ref', f'scratch/{self.node_id}/orbs', str, '')
        return

    def runall(self, geom, runtype=None):
        ''' calculate all states with BAGEL '''

        if (not self.gradient_states and not self.coupling_states) or runtype == 'energy':
            print(" only calculating energies")
            # TODO what about multiple multiplicities?
            tup = self.states[0]
            self.run(geom, 'energy')
            # make grada all None
            for tup in self.states:
                self._Gradients[tup] = self.Gradients(None, None)
        elif self.gradient_states and self.coupling_states or runtype == 'gh':
            self.run(geom, 'gh')
        elif self.gradient_states and not self.coupling_states or runtype == 'gradient':
            self.run(geom, 'gradient')
        else:
            raise RuntimeError

        self.hasRanForCurrentCoords = True
        return


if __name__ == "__main__":
    # ,units.ANGSTROM_TO_AU)
    geom = manage_xyz.read_xyz('../../data/ethylene.xyz')
    B = BAGEL.from_options(states=[(1, 0), (1, 1)], gradient_states=[(
        1, 0), (1, 1)], coupling_states=(0, 1), geom=geom, lot_inp_file='bagel.txt', node_id=0)
    coords = manage_xyz.xyz_to_np(geom)
    E0 = B.get_energy(coords, 1, 0)
    E1 = B.get_energy(coords, 1, 1)
    g0 = B.get_gradient(coords, 1, 0)
    g1 = B.get_gradient(coords, 1, 1)
    c = B.get_coupling(coords, 1, 0, 1)
    print(E0, E1)
    print(g0.T)
    print(g1.T)
    print(c.T)

    # for line in B.file_options.record():
    #    print(line)

    # for key,value in B.file_options.ActiveOptions.items():
    #    print(key,value)
