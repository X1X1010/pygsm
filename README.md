pygsm
==============================
Reaction path and photochemistry tool: double-ended, single-ended and
crossing-search growing string methods (DE-GSM, SE-GSM, SE-Cross).

This is a modernised fork of [ZimmermanGroup/pyGSM](https://github.com/ZimmermanGroup/pyGSM).
The [upstream documentation](https://zimmermangroup.github.io/pyGSM/) still describes the method,
but the Python API here differs (see below).

## Install
Python 3.10 or newer. From the repository root:
```
pip install -e ".[ase,xtb]"
```
Extras: `ase` (any ASE calculator as the level of theory), `xtb` (GFN-xTB via tblite),
`plot` (string energy plot in the CLI), `test`, and `dev` (ruff, pre-commit, coverage).

## Command line
The `gsm` command runs DE-GSM, SE-GSM or SE-Cross on an xyz file:
```
# DE-GSM: reaction.xyz holds the reactant and product frames
gsm --xyzfile reaction.xyz -m DE_GSM -p xTB

# SE-GSM: drive the reactant along the coordinates in isomers.txt
gsm --xyzfile reactant.xyz -m SE_GSM -i isomers.txt -p xTB

# any ASE calculator as the level of theory
gsm --xyzfile reaction.xyz -m DE_GSM -p ase \
    --ase-class tblite.ase.TBLite --ase-kwargs '{"method": "GFN2-xTB"}'

# SE-Cross: singlet/triplet crossing (MECI/MESX) on a penalty surface
gsm --xyzfile reactant.xyz -m SE_Cross -i isomers.txt -p xTB --multiplicity 1 3 --adiabatic-index 0 0

# restart from the string written by an earlier run
gsm --xyzfile reaction.xyz -m DE_GSM -p xTB --restart-file string_000.xyz
```
Options use `--kebab-case`; `-m`, `-p`, `-n` and `-i` are short for `--mode`, `--package`,
`--num-nodes` and `--isomers`. The old single-dash spellings (`-num_nodes`, `-CONV_TOL`, ...) still
work but print a deprecation warning. `--multiplicity` and `--adiabatic-index` must list the same
number of states; `Avg_PES`, `Penalty_PES` and `SE_Cross` require at least two.
`gsm --help` lists every option; the main ones are:

| option | default | meaning |
|---|---|---|
| **input** | | |
| `--xyzfile FILE` | required | xyz file: reactant and product frames (DE-GSM) or the reactant (SE modes) |
| `-m`, `--mode` | required | `DE_GSM`, `SE_GSM` or `SE_Cross` |
| `-i`, `--isomers FILE` | | driving coordinates (SE modes; format below) |
| `--charge N` | 0 | total charge |
| `--multiplicity M ...` | 1 | spin multiplicity of each state |
| `--adiabatic-index I ...` | 0 | adiabatic index of each state (0-based) |
| `--pes-type` | `PES` | `PES`, `Avg_PES` or `Penalty_PES` (two states; SE-Cross always uses `Penalty_PES`) |
| `--sigma S` | 1.0 | strength of the energy-gap penalty for `Penalty_PES` |
| **level of theory** | | |
| `-p`, `--package` | `xTB` | `xTB` (GFN-xTB via tblite) or `ase` |
| `--xtb-hamiltonian` | `GFN2-xTB` | `GFN2-xTB`, `GFN1-xTB` or `IPEA1-xTB` |
| `--xtb-accuracy`, `--xtb-electronic-temperature` | 1.0, 300 K | tblite settings |
| `--solvent NAME` | none | ALPB implicit solvent (xTB only) |
| `--ase-class PATH`, `--ase-kwargs JSON` | | ASE calculator class and its keyword arguments (`-p ase`) |
| **string** | | |
| `-n`, `--num-nodes N` | 9 DE-GSM, 20 SE | number of nodes |
| `--coordinate-type` | `TRIC` | `TRIC`, `DLC` or `HDLC` |
| `--max-gsm-iters N` | 100 | growth and optimisation iterations |
| `--max-opt-steps N` | 3 DE-GSM, 20 SE | node optimisation steps per iteration |
| `--conv-tol X` | 0.0005 | node gradrms convergence |
| `--add-node-tol X` | 0.01 | gradrms at which the frontier node is good enough to add the next one |
| `--conv-gmax`, `--conv-ediff`, `--conv-de` | 100, 100, 1.0 | max-gradient, energy-change and state-gap convergence |
| `--growth-direction` | 0 | DE-GSM: 0 grow from both ends, 1 from the reactant, 2 from the product |
| `--dqmag-max`, `--bdist-ratio` | 0.8, 0.5 | SE growth: max step along the driving coordinates; when growth is complete |
| `--reactant-geom-fixed`, `--product-geom-fixed` | off | don't pre-optimise the end points |
| `--mp-cores N` | 1 | processes for parallel node updates |
| **TS search** | | |
| *(default)* | | climbing image, then the exact TS search (needs `eigenvector_follow`; other optimizers fall back to climbing only) |
| `--only-climb` | | climbing image only |
| `--no-climb` | | neither |
| `--optimize-meci`, `--optimize-mesx` | | crossing searches (with `--pes-type Penalty_PES`) |
| **optimizer** | | |
| `--optimizer` | `eigenvector_follow` | `eigenvector_follow`, `lbfgs` (large systems), `conjugate_gradient` or `beales_cg` |
| `--linesearch` | `NoLineSearch` | `NoLineSearch` or `backtrack` |
| `--dmax X` | 0.1 | initial step size |
| **restart** | | |
| `--restart-file FILE` | | string to start from instead of growing one (e.g. `string_000.xyz`) |
| `--reparametrize` | off | redistribute the restart string's nodes evenly first |
| `--start-climb-immediately` | off | start climbing straight away on restart |
| **output** | | |
| `--id N` | 0 | string id used in the output file names |
| `--log-level` | `INFO` | `DEBUG` adds the per-step diagnostics; `WARNING` shows problems only |
| `--xyz-output-format` | `molden` | `molden` (energies, gradients) or `multixyz` |
| `--only-drive` | off | only build the interpolated string (no optimisation) and write `interpolated.xyz` |
| `--dont-analyze-ics` | off | don't write `IC_data_<ID>.txt` |

When `--conv-gmax`, `--conv-ediff` and `--conv-de` are omitted, SE-GSM and SE-Cross retain
string thresholds of 0.001, 0.1 and 0.5 alongside the node optimizer defaults listed above.
Explicit values apply to both the string and its node optimizers.

`--restraint-file` holds atoms near their input positions with harmonic restraints (one
`atom k` per line, k in kcal/mol/A^2) and `--force-file` applies constant forces between atom pairs
(one `atom1 atom2 force` per line, in au). These and the other less common options (frozen/hybrid/
fragment index files, `--bonds-file`, print levels) are described in `gsm --help`; atom indices in
all these files are 0-based.

**Driving coordinates** (`-i`): one per line, atoms numbered from 1 as in the xyz file:
`ADD a b [d]` forms the a-b bond, `BREAK a b [d]` breaks it (target distance `d` in Angstrom,
optional; the default comes from van der Waals radii), and `ANGLE a b c deg`, `TORSION a b c d deg`
and `OOP a b c d deg` drive an angle, dihedral or out-of-plane angle to a target in degrees.
The file has no comment syntax. For example, to form two bonds:
```
ADD 4 12
ADD 1 11
```

**Output**: a run writes, to the working directory (`<ID>` is `--id` as 3 digits, default `000`):

| file | content |
|---|---|
| `string_<ID>.xyz` | the string (Molden format), updated every iteration; restart with `--restart-file` |
| `grown_string_<ID>.xyz` | the string at the end of growth (SE-GSM, SE-Cross) |
| `TSnode_<ID>.xyz` / `meci_<ID>.xyz` | highest-energy node / MECI |
| `IC_data_<ID>.txt` | internal coordinates of the minima and TS |
| `string_<ID>.png` | energy profile (needs the `plot` extra) |

If DE-GSM cannot finish growing the string within `--max-gsm-iters`, `gsm` stops with an error
(exit status 1) and the partial string stays in `string_<ID>.xyz`.

## Python API
```python
from pygsm.gsm import DEGSM
from pygsm.lot.ase import ASELoT
from pygsm.molecule import Molecule
from pygsm.optimizers import EigenvectorFollow
from pygsm.pes import PES
from pygsm.utils import output

output.setup_logging()                      # pygsm logs; it prints nothing unless you ask
lot = ASELoT(calculator=calc, geom=geom)    # any ASE calculator
pes = PES(lot=lot, ad_idx=0, multiplicity=1)
reactant = Molecule(pes, coords, geom=geom, form_hessian=True)
product = Molecule.copy_from(reactant, xyz=product_xyz, new_node_id=10)
gsm = DEGSM(reactant=reactant, product=product, nnodes=11, optimizer=EigenvectorFollow())
gsm.go_gsm(max_iters=100, opt_steps=3, rtype=2)
gsm.energies, gsm.geometries, gsm.TSnode     # results
```
Every class takes its settings as constructor arguments (see the class docstrings) and
keeps them as attributes of the same name; `optimizer.clone(dmax=0.2)`, `pes.clone(lot=...)`
or `lot.clone(node_id=3)` copy an object with some settings changed. The library writes no
files unless asked: pass `output_dir=...` to a GSM for the string checkpoint, or
`trajectory=...` to `optimizer.optimize`. Complete scripts, including building the
delocalised internal coordinates, are in [`examples/`](examples).

## Tests
```
OMP_NUM_THREADS=1 pytest                            # unit + golden numerical regression tests
OMP_NUM_THREADS=1 python -m tests.bench --compare   # wall-time regression check
```
The golden tests compare full GSM runs against the baselines in `tests/golden/baselines`.
Only re-record them (`python -m tests.golden.make_golden`) in a commit that intentionally changes numerics.
