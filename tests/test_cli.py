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


@pytest.mark.parametrize(('mode', 'nodes'), [('DE_GSM', 9), ('SE_GSM', 20), ('SE_Cross', 20)])
def test_num_nodes_default_depends_on_mode(mode, nodes):
    states = ['--multiplicity', '1', '3', '--adiabatic-index', '0', '0'] if mode == 'SE_Cross' else []
    config = parse_arguments(verbose=False, argv=['--xyzfile', str(DATA_DIR / 'ethylene.xyz'), '-m', mode, *states])
    assert config['num_nodes'] == nodes


@pytest.mark.parametrize(
    'options',
    [
        ['--multiplicity'],
        ['--adiabatic-index'],
        ['--multiplicity', '1', '3'],
        ['--pes-type', 'Avg_PES'],
        ['--pes-type', 'Penalty_PES'],
        ['--mode', 'SE_Cross'],
    ],
)
def test_invalid_state_lists_are_cli_errors(options, capsys):
    with pytest.raises(SystemExit) as error:
        parse_arguments(verbose=False, argv=['--xyzfile', str(DATA_DIR / 'ethylene.xyz'), '--mode', 'DE_GSM', *options])
    assert error.value.code == 2
    assert 'error:' in capsys.readouterr().err


def test_se_cross_resolves_penalty_surface_before_building_lot():
    config = parse_arguments(
        verbose=False,
        argv=[
            '--xyzfile', str(DATA_DIR / 'ethylene.xyz'), '--mode', 'SE_Cross',
            '--multiplicity', '1', '3', '--adiabatic-index', '0', '0',
        ],
    )  # fmt: skip
    assert config['PES_type'] == 'Penalty_PES'


def test_dont_analyze_ics_flag():
    argv = ['--xyzfile', str(DATA_DIR / 'ethylene.xyz'), '-m', 'DE_GSM']
    assert parse_arguments(verbose=False, argv=argv)['analyze_ICs'] is True
    assert parse_arguments(verbose=False, argv=[*argv, '--dont-analyze-ics'])['analyze_ICs'] is False


@pytest.mark.parametrize('analyze', [True, False])
def test_post_processing_ic_data(tmp_path, monkeypatch, analyze):
    from types import SimpleNamespace

    from pygsm.cli import post_processing

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('pygsm._cli.runtime.plot', lambda **kwargs: None)
    energies = [0.0, 5.0, 12.0, 3.0, -4.0]
    nodes = [
        SimpleNamespace(
            primitive_internal_coordinates=['bond 1-2', 'angle 1-2-3'],
            primitive_internal_values=[1.0 + n, 2.0 + n],
            energy=e,
        )
        for n, e in enumerate(energies)
    ]
    gsm = SimpleNamespace(ID=0, nodes=nodes, energies=energies, TSnode=2)
    post_processing(gsm, analyze_ICs=analyze, have_TS=True)

    path = tmp_path / 'IC_data_000.txt'
    assert path.exists() is analyze
    if analyze:
        lines = path.read_text().splitlines()
        assert lines[0].startswith('Internals')
        assert lines[1:] == ['bond 1-2\t1.0\t3.0\t5.0', 'angle 1-2-3\t2.0\t4.0\t6.0']


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
