import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from pygsm.cli import parse_arguments

DATA_DIR = Path(__file__).resolve().parent / 'data'
REPO_ROOT = Path(__file__).resolve().parents[1]


def test_parse_arguments_handles_cli_defaults():
    config = parse_arguments(
        verbose=False,
        argv=[
            '--xyzfile',
            str(DATA_DIR / 'ethylene.xyz'),
            '--mode',
            'SE_GSM',
            '--package',
            'ase',
            '--ase-class',
            'ase.calculators.morse.MorsePotential',
            '--nproc',
            '4',
        ],
    )

    assert config['EST_Package'] == 'ase'
    assert config['gsm_type'] == 'SE_GSM'
    assert config['nproc'] == 4


def test_short_options_and_legacy_single_dash_spellings():
    base = [
        '--xyzfile',
        str(DATA_DIR / 'ethylene.xyz'),
        '-p',
        'ase',
        '--ase-class',
        'ase.calculators.morse.MorsePotential',
    ]
    short = parse_arguments(verbose=False, argv=[*base, '-m', 'SE_GSM', '-n', '7', '-i', 'isomers.txt'])
    assert (short['gsm_type'], short['num_nodes'], short['isomers_file']) == ('SE_GSM', 7, 'isomers.txt')

    with pytest.warns(FutureWarning, match='-num_nodes -> --num-nodes'):
        legacy = parse_arguments(
            verbose=False, argv=[*base, '-mode', 'SE_GSM', '-num_nodes', '7', '-CONV_TOL=0.001', '-ID', '3']
        )
    assert (legacy['gsm_type'], legacy['num_nodes'], legacy['CONV_TOL'], legacy['ID']) == ('SE_GSM', 7, 0.001, 3)


def test_python_module_cli_help_smoke():
    result = subprocess.run(
        [sys.executable, '-m', 'pygsm.cli', '--help'],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert 'Reaction path transition state and photochemistry tool' in result.stdout


def test_console_script_help_smoke():
    gsm = shutil.which('gsm', path=str(Path(sys.executable).parent))
    if gsm is None:
        pytest.skip('gsm console script is not installed')
    result = subprocess.run(
        [gsm, '--help'],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert 'Reaction path transition state and photochemistry tool' in result.stdout
