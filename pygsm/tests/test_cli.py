from pathlib import Path
import subprocess
import sys

from pygsm.cli import parse_arguments


DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_parse_arguments_handles_cli_defaults():
    config = parse_arguments(
        verbose=False,
        argv=[
            '-xyzfile',
            str(DATA_DIR / 'ethylene.xyz'),
            '-mode',
            'SE_GSM',
            '-package',
            'ase',
            '--ase-class',
            'ase.calculators.morse.MorsePotential',
            '-nproc',
            '4',
        ],
    )

    assert config['EST_Package'] == 'ase'
    assert config['gsm_type'] == 'SE_GSM'
    assert config['nproc'] == 4


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


def test_bin_gsm_help_smoke():
    result = subprocess.run(
        [sys.executable, 'bin/gsm', '--help'],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert 'Reaction path transition state and photochemistry tool' in result.stdout
