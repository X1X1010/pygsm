# standard library imports
import subprocess
import sys
import os

# third party
import numpy as np

# local application imports
from .base_lot import Lot


class QChem(Lot):
    def __init__(self, options):
        super(QChem, self).__init__(options)

        qcscratch = os.environ['QCSCRATCH']
        for state in self.states:
            tempfolder = qcscratch + f'/string_{self.ID:03d}/{self.node_id}.{state[0]}/'
            print(f" making temp folder {tempfolder}")
            os.system(f'mkdir -p {tempfolder}')

        copy_input_file = os.getcwd() + "/QChem_input.txt"
        print(copy_input_file)
        self.write_preamble(self.geom, self.states[0][0], copy_input_file)
        print(f" making folder scratch/{self.node_id}")
        os.system(f'mkdir -p scratch/{self.node_id}')

    def write_preamble(self, geom, multiplicity, tempfilename, jobtype='FORCE'):

        tempfile = open(tempfilename, 'w')
        if not self.lot_inp_file:
            tempfile.write(' $rem\n')
            tempfile.write(f' JOBTYPE {jobtype}\n')
            tempfile.write(f' EXCHANGE {self.functional}\n')
            tempfile.write(' SCF_ALGORITHM rca_diis\n')
            tempfile.write(' SCF_MAX_CYCLES 300\n')
            tempfile.write(f' BASIS {self.basis}\n')
            # tempfile.write(' ECP LANL2DZ \n')
            tempfile.write(' WAVEFUNCTION_ANALYSIS FALSE\n')
            tempfile.write(' GEOM_OPT_MAX_CYCLES 300\n')
            tempfile.write('scf_convergence 6\n')
            tempfile.write(' SYM_IGNORE TRUE\n')
            tempfile.write(' SYMMETRY   FALSE\n')
            tempfile.write('molden_format true\n')
            tempfile.write(' $end\n')
            tempfile.write('\n')
            tempfile.write('$molecule\n')
        else:
            with open(self.lot_inp_file) as lot_inp:
                lot_inp_lines = lot_inp.readlines()
            for line in lot_inp_lines:
                tempfile.write(line)

        tempfile.write(f'{self.charge} {multiplicity}\n')
        if os.path.isfile("link.txt"):
            with open("link.txt") as link:
                link_lines = link.readlines()
            tmp_geom = [list(i) for i in geom]
            for i, coord in enumerate(tmp_geom):
                coord.append(link_lines[i].rstrip('\n'))
                for i in coord:
                    tempfile.write(str(i)+' ')
                tempfile.write('\n')
        else:
            for coord in geom:
                for i in coord:
                    tempfile.write(str(i)+' ')
                tempfile.write('\n')
        tempfile.write('$end')
        tempfile.close()

    def run(self, geom, multiplicity, ad_idx, runtype='gradient'):

        assert ad_idx == 0, "pyGSM Q-Chem doesn't currently support ad_idx!=0"

        qcscratch = os.environ['QCSCRATCH']
        tempfilename = qcscratch + f'/string_{self.ID:03d}/{self.node_id}.{multiplicity}/tempQCinp'

        if self.calc_grad and runtype != "energy":
            self.write_preamble(geom, multiplicity, tempfilename)
        else:
            self.write_preamble(geom, multiplicity, tempfilename, jobtype='SP')

        cmd = ['qchem']
        args = ['-nt', str(self.nproc),
                '-save',
                tempfilename,
                f'{tempfilename}.qchem.out',
                f'string_{self.ID:03d}/{self.node_id}.{multiplicity}'
                ]
        cmd.extend(args)

        #  Run the process
        subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0]

        self.parse(qcscratch, multiplicity)

        return

    def parse(self, qcscratch, multiplicity):
        # PARSE OUTPUT #
        if self.calc_grad:
            efilepath = qcscratch + f'/string_{self.ID:03d}/{self.node_id}.{multiplicity}/GRAD'
            with open(efilepath) as efile:
                elines = efile.readlines()

            temp = 0
            for lines in elines:
                if temp == 1:
                    # defaulting to the ground-state
                    self._Energies[(multiplicity, 0)] = self.Energy(float(lines.split()[0]), 'Hartree')
                    break
                if "$" in lines:
                    temp += 1

            with open(efilepath) as efile:
                gradlines = efile.readlines()
            temp = 0
            tmp = []
            for lines in gradlines:
                if '$' in lines:
                    temp += 1
                elif temp == 2:
                    tmpline = lines.split()
                    tmp.append([float(i) for i in tmpline])
                elif temp == 3:
                    break
            self._Gradients[(multiplicity, 0)] = self.Gradient(np.asarray(tmp), 'Hartree/Bohr')
        else:
            raise NotImplementedError
        self.write_E_to_file()

    @classmethod
    def copy(cls, lot, options, copy_wavefunction=True):
        base = os.environ['QCSCRATCH']
        node_id = options.get('node_id', 1)

        if node_id != lot.node_id:  # and copy_wavefunction: # other theories are more sensitive than qchem -- commenting out
            for state in lot.states:
                multiplicity = state[0]
                efilepath_old = base + f'/string_{lot.ID:03d}/{lot.node_id}.{multiplicity}'
                efilepath_new = base + f'/string_{lot.ID:03d}/{node_id}.{multiplicity}'
                cmd = 'cp -r ' + efilepath_old + ' ' + efilepath_new
                print(f" copying QCSCRATCH files\n {cmd}")
                os.system(cmd)
        return cls(lot.options.copy().set_values(options))
