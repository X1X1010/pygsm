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
```
gsm --xyzfile reaction.xyz --mode DE_GSM --package xTB --num-nodes 11
gsm --xyzfile reactant.xyz -m SE_GSM -i isomers.txt -p ase \
    --ase-class tblite.ase.TBLite --ase-kwargs '{"method": "GFN2-xTB"}'
```
`gsm --help` lists all options; `-m`, `-p`, `-n` and `-i` are short for `--mode`, `--package`,
`--num-nodes` and `--isomers`. The old single-dash spellings (`-num_nodes`, `-CONV_TOL`, ...) still
work but are deprecated. A run writes, to the working directory (`<ID>` is `--id`, default `000`):

| file | content |
|---|---|
| `string_<ID>.xyz` | the string (Molden format), updated every iteration; restart with `--restart-file` |
| `grown_string_<ID>.xyz` | the string at the end of growth (SE-GSM, SE-Cross) |
| `TSnode_<ID>.xyz` / `meci_<ID>.xyz` | highest-energy node / MECI |
| `IC_data_<ID>.txt` | internal coordinates of the minima and TS |
| `string_<ID>.png` | energy profile (needs the `plot` extra) |

`--log-level DEBUG` adds the per-step diagnostics to the log.

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
reactant = Molecule(geom=geom, PES=pes, coord_obj=coords, Form_Hessian=True)
product = Molecule.copy_from(reactant, xyz=product_xyz, new_node_id=10)
gsm = DEGSM(reactant=reactant, product=product, nnodes=11, optimizer=EigenvectorFollow())
gsm.go_gsm(max_iters=100, opt_steps=3, rtype=2)
gsm.energies, gsm.geometries, gsm.TSnode     # results
```
Every class takes its settings as constructor arguments (see the class docstrings) and
keeps them as attributes of the same name; `optimizer.clone(DMAX=0.2)`, `pes.clone(lot=...)`
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
