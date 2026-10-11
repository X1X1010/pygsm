pygsm
==============================
Reaction path and photochemistry tool


## Documentation
See https://zimmermangroup.github.io/pyGSM/


## Install instructions
`pygsm` can be installed by setting up a python environment (e.g. venv, conda), cloning the GitHub repository, and installing with pip. From the root directory in the git repository, this would look like:
```
pip install -e .
```

Optional extras: `ase` (ASE calculators), `xtb` (GFN-xTB via tblite), `plot` (string energy plot in the CLI),
`test` and `dev` (ruff, pre-commit). For development:
```
pip install -e ".[dev]"
```

## Tests
```
OMP_NUM_THREADS=1 pytest                       # unit + golden numerical regression tests
OMP_NUM_THREADS=1 python -m tests.bench --compare   # wall-time regression check
```
The golden tests compare full GSM runs against the baselines in `tests/golden/baselines`.
Only re-record them (`python -m tests.golden.make_golden`) in a commit that intentionally changes numerics.
