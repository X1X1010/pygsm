"""Argument parsing for the ``gsm`` command."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import textwrap
import warnings
from collections.abc import Sequence

from pygsm.utils import output

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Reaction path transition state and photochemistry tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Example of use:
            --------------------------------
            gsm --mode DE_GSM --xyzfile reaction.xyz --package xTB --id 1
            """
        ),
    )
    parser.add_argument('--xyzfile', help='XYZ file containing reactant and, if DE-GSM, product.', required=True)
    parser.add_argument('-i', '--isomers', help='driving coordinate file', type=str, required=False)
    parser.add_argument(
        '-m',
        '--mode',
        help='growing string method',
        choices=['DE_GSM', 'SE_GSM', 'SE_Cross'],
        type=str,
        required=True,
    )
    parser.add_argument(
        '--only-drive',
        action='store_true',
        help='only build the interpolated string (no optimisation); writes interpolated.xyz',
    )
    parser.add_argument(
        '-p',
        '--package',
        default='xTB',
        type=str,
        help='Level of theory: GFN-xTB via tblite, or any ASE calculator (see --ase-class) (default: %(default)s)',
        choices=['xTB', 'ase'],
    )
    parser.add_argument(
        '--id', default=0, type=int, help='string identification number (default: %(default)s)', required=False
    )
    parser.add_argument(
        '-n',
        '--num-nodes',
        type=int,
        default=None,
        help='number of nodes for string (defaults: 9 DE-GSM, 20 SE-GSM and SE-Cross)',
        required=False,
    )
    parser.add_argument(
        '--pes-type',
        type=str,
        default='PES',
        help='potential energy surface; Avg_PES and Penalty_PES need two states (default: %(default)s)',
        choices=['PES', 'Avg_PES', 'Penalty_PES'],
    )
    parser.add_argument(
        '--adiabatic-index',
        nargs='+',
        type=int,
        default=[0],
        help='adiabatic index of each state, 0-based (default: %(default)s)',
        required=False,
    )
    parser.add_argument(
        '--multiplicity',
        nargs='+',
        type=int,
        default=[1],
        help='spin multiplicity of each state (default: %(default)s)',
    )
    parser.add_argument(
        '--force-file',
        type=str,
        default=None,
        help='file of constant forces, one "atom1 atom2 force" per line (0-based atoms, force in au; negative is tensile, positive compressive)',
    )
    parser.add_argument(
        '--restraint-file',
        type=str,
        default=None,
        help='file of harmonic position restraints, one "atom k" per line (0-based atom, k in kcal/mol/A^2) '
        'holding the atom near its input position',
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='eigenvector_follow',
        choices=['eigenvector_follow', 'lbfgs', 'conjugate_gradient', 'beales_cg'],
        help='node optimizer; only eigenvector_follow does the exact TS search, lbfgs suits large systems '
        '(default: %(default)s)',
    )
    parser.add_argument(
        '--opt-print-level',
        type=int,
        default=1,
        help='Printout for optimization. 2 prints everything in opt.',
        required=False,
    )
    parser.add_argument(
        '--gsm-print-level',
        type=int,
        default=1,
        help='GSM diagnostic detail at --log-level DEBUG: 0 less, 1 more, 2 everything',
        required=False,
    )
    parser.add_argument(
        '--log-level',
        type=str.upper,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING'],
        help='Log verbosity: DEBUG adds the per-step diagnostics (default: %(default)s)',
    )
    parser.add_argument(
        '--xtb-hamiltonian',
        type=str,
        default='GFN2-xTB',
        help='xTB hamiltonian',
        choices=['GFN2-xTB', 'GFN1-xTB', 'IPEA1-xTB'],
        required=False,
    )
    parser.add_argument('--xtb-accuracy', type=float, default=1.0, help='xTB accuracy', required=False)
    parser.add_argument(
        '--xtb-electronic-temperature', type=float, default=300.0, help='xTB electronic temperature', required=False
    )
    parser.add_argument(
        '--xyz-output-format',
        type=str,
        default='molden',
        choices=['molden', 'multixyz'],
        help='format of the string files (default: %(default)s)',
    )
    parser.add_argument('--solvent', type=str, help='Solvent to use (xTB calculations only)', required=False)
    parser.add_argument(
        '--linesearch',
        type=str,
        default='NoLineSearch',
        help='line search of the node optimizer (default: %(default)s)',
        choices=['NoLineSearch', 'backtrack'],
    )
    parser.add_argument(
        '--coordinate-type',
        type=str,
        default='TRIC',
        help='Coordinate system (default %(default)s)',
        choices=['TRIC', 'DLC', 'HDLC'],
    )
    parser.add_argument(
        '--add-node-tol',
        type=float,
        default=0.01,
        help='gradrms at which the frontier node is converged enough to add the next one (default: %(default)s)',
        required=False,
    )
    parser.add_argument(
        '--dqmag-max',
        type=float,
        default=0.8,
        help='Maximum step size in single-ended mode (default: %(default)s)',
        required=False,
    )
    parser.add_argument(
        '--bdist-ratio',
        type=float,
        default=0.5,
        help='Reaction completion convergence in SE modes (default: %(default)s)',
    )
    parser.add_argument(
        '--conv-tol',
        type=float,
        default=0.0005,
        help='Convergence tolerance for optimizing nodes (default: %(default)s)',
        required=False,
    )
    parser.add_argument(
        '--growth-direction',
        type=int,
        default=0,
        help='Direction adding new nodes (default: %(default)s)',
        choices=[0, 1, 2],
    )
    parser.add_argument(
        '--reactant-geom-fixed', action='store_true', help='Fix reactant geometry i.e. do not pre-optimize'
    )
    parser.add_argument(
        '--product-geom-fixed', action='store_true', help='Fix product geometry i.e. do not pre-optimize'
    )
    parser.add_argument(
        '--nproc',
        type=int,
        default=1,
        help='Processors for calculation. Python will detect OMP_NUM_THREADS, only use this if you want to force the number of processors',
    )
    parser.add_argument('--charge', type=int, default=0, help='Total system charge (default: %(default)s)')
    parser.add_argument(
        '--max-gsm-iters', type=int, default=100, help='The maximum number of GSM cycles (default: %(default)s)'
    )
    parser.add_argument(
        '--max-opt-steps',
        type=int,
        help='The maximum number of node optimizations per GSM cycle (defaults: 3 DE-GSM, 20 SE-GSM)',
    )
    parser.add_argument('--only-climb', action='store_true', help='Only use climbing image to optimize TS')
    parser.add_argument('--no-climb', action='store_true', help="Don't climb to the TS")
    parser.add_argument('--optimize-mesx', action='store_true', help='optimize to the MESX')
    parser.add_argument('--optimize-meci', action='store_true', help='optimize to the MECI')
    parser.add_argument('--restart-file', help='string file to restart from, e.g. string_000.xyz', type=str)
    parser.add_argument(
        '--mp-cores',
        type=int,
        default=1,
        help='Use python multiprocessing to parallelize jobs on a single compute node. Set OMP_NUM_THREADS, ncpus accordingly.',
    )
    parser.add_argument(
        '--dont-analyze-ics',
        dest='analyze_ics',
        action='store_false',
        help="don't write IC_data_<ID>.txt (internal coordinates of the minima and the TS)",
    )
    parser.add_argument(
        '--hybrid-coord-idx-file',
        type=str,
        default=None,
        help='A filename containing a list of  indices to use in hybrid coordinates. 0-Based indexed',
    )
    parser.add_argument(
        '--frozen-coord-idx-file',
        type=str,
        default=None,
        help='A filename containing a list of  indices to be frozen. 0-Based indexed',
    )
    parser.add_argument(
        '--conv-ediff',
        type=float,
        help='energy-change convergence (defaults: 100 for nodes, 0.1 for the SE string)',
    )
    parser.add_argument(
        '--conv-de',
        type=float,
        help='state energy-gap convergence (defaults: 1.0 for nodes, 0.5 for the SE string)',
    )
    parser.add_argument(
        '--conv-gmax', type=float, help='max-gradient convergence (defaults: 100 for nodes, 0.001 for the SE string)'
    )
    parser.add_argument('--dmax', default=0.1, type=float, help='initial optimizer step size (default: %(default)s)')
    parser.add_argument(
        '--sigma', default=1.0, type=float, help='The strength of the difference energy penalty in Penalty_PES'
    )
    parser.add_argument(
        '--prim-idx-file', type=str, help='A filename containing a list of indices to define fragments. 0-Based indexed'
    )
    parser.add_argument('--reparametrize', action='store_true', help='Reparametrize restart string equally along path')
    parser.add_argument(
        '--interp-method',
        default='DLC',
        choices=['DLC'],
        help='interpolation and reparametrisation method (only DLC is implemented)',
    )
    parser.add_argument('--bonds-file', type=str, help='A file which contains the bond indices (0-based)')
    parser.add_argument(
        '--start-climb-immediately', action='store_true', help='Start climbing immediately when restarting.'
    )

    group_ase = parser.add_argument_group('ASE', 'ASE calculator options')
    group_ase.add_argument(
        '--ase-class', type=str, help='ASE calculator import path, eg. "ase.calculators.lj.LennardJones"'
    )
    group_ase.add_argument(
        '--ase-kwargs',
        type=str,
        help='ASE calculator keyword args, as JSON dictionary, eg. {"param_filename":"path/to/file.xml"}',
    )
    return parser


def _translate_legacy_options(argv: Sequence[str], parser: argparse.ArgumentParser) -> list[str]:
    """Rewrite old single-dash spellings (``-num_nodes``, ``-CONV_TOL=1e-3``) to the long options.

    pyGSM used single-dash multi-letter options; they keep working, with a
    FutureWarning, until they are removed in a future release.
    """
    long_options = {opt for action in parser._actions for opt in action.option_strings if opt.startswith('--')}
    translated, legacy = [], []
    for token in argv:
        name, eq, value = token.partition('=')
        if len(name) > 2 and name.startswith('-') and not name.startswith('--'):
            candidate = '--' + name[1:].lower().replace('_', '-')
            if candidate in long_options:
                legacy.append(f'{name} -> {candidate}')
                token = candidate + eq + value
        translated.append(token)
    if legacy:
        warnings.warn(
            'single-dash long options are deprecated and will be removed; use ' + ', '.join(legacy),
            FutureWarning,
            stacklevel=3,
        )
    return translated


def _resolve_nproc(requested_nproc: int, verbose: bool) -> int:
    if requested_nproc > 1:
        if verbose:
            logger.info(f'forcing number of processors to be {requested_nproc}!!!')
        return requested_nproc

    try:
        nproc = int(os.environ['OMP_NUM_THREADS'])
    except (KeyError, TypeError, ValueError):
        nproc = 1

    if verbose:
        logger.info(f' Using {nproc} processors')
    return nproc


def _default_num_nodes(gsm_type: str, num_nodes: int | None) -> int:
    if num_nodes is not None:
        return num_nodes
    if gsm_type == 'DE_GSM':
        return 9
    return 20


def _print_banner() -> None:
    msg = r"""
    __        __   _                            _
    \ \      / /__| | ___ ___  _ __ ___   ___  | |_ ___
     \ \ /\ / / _ \ |/ __/ _ \| '_ ` _ \ / _ \ | __/ _ \
      \ V  V /  __/ | (_| (_) | | | | | |  __/ | || (_) |
       \_/\_/ \___|_|\___\___/|_| |_| |_|\___|  \__\___/
                                    ____ ____  __  __
                       _ __  _   _ / ___/ ___||  \/  |
                      | '_ \| | | | |  _\___ \| |\/| |
                      | |_) | |_| | |_| |___) | |  | |
                      | .__/ \__, |\____|____/|_|  |_|
                      |_|    |___/
#==========================================================================#
#| If this code has benefited your research, please support us by citing: |#
#|                                                                        |#
#| Aldaz, C.; Kammeraad J. A.; Zimmerman P. M. "Discovery of conical      |#
#| intersection mediated photochemistry with growing string methods",     |#
#| Phys. Chem. Chem. Phys., 2018, 20, 27394                               |#
#| http://dx.doi.org/10.1039/c8cp04703k                                   |#
#|                                                                        |#
#| Wang, L.-P.; Song, C.C. (2016) "Geometry optimization made simple with |#
#| translation and rotation coordinates", J. Chem, Phys. 144, 214108.     |#
#| http://dx.doi.org/10.1063/1.4952956                                    |#
#==========================================================================#
    """
    logger.info(msg)


def parse_arguments(verbose: bool = True, argv: Sequence[str] | None = None):
    parser = _build_parser()
    args = parser.parse_args(_translate_legacy_options(sys.argv[1:] if argv is None else argv, parser))
    if args.mode == 'SE_Cross':
        args.pes_type = 'Penalty_PES'
    if len(args.multiplicity) != len(args.adiabatic_index):
        parser.error('--multiplicity and --adiabatic-index must have the same number of states')
    if args.pes_type != 'PES' and len(args.multiplicity) < 2:
        parser.error(f'{args.pes_type} requires at least two states (--multiplicity and --adiabatic-index)')
    if (args.optimize_mesx or args.optimize_meci) and args.pes_type != 'Penalty_PES':
        parser.error('--optimize-mesx and --optimize-meci require --pes-type Penalty_PES')
    output.setup_logging(args.log_level)

    if verbose:
        _print_banner()

    nproc = _resolve_nproc(args.nproc, verbose=verbose)
    inpfileq = {
        'xyzfile': args.xyzfile,
        'EST_Package': args.package,
        'reactant_geom_fixed': args.reactant_geom_fixed,
        'nproc': nproc,
        'states': None,
        'xTB_Hamiltonian': args.xtb_hamiltonian,
        'xTB_accuracy': args.xtb_accuracy,
        'xTB_electronic_temperature': args.xtb_electronic_temperature,
        'solvent': args.solvent,
        'PES_type': args.pes_type,
        'adiabatic_index': args.adiabatic_index,
        'multiplicity': args.multiplicity,
        'charge': args.charge,
        'FORCE_FILE': args.force_file,
        'RESTRAINT_FILE': args.restraint_file,
        'FORCE': None,
        'RESTRAINTS': None,
        'optimizer': args.optimizer,
        'opt_print_level': args.opt_print_level,
        'linesearch': args.linesearch,
        'DMAX': args.dmax,
        'xyz_output_format': args.xyz_output_format,
        'coordinate_type': args.coordinate_type,
        'hybrid_coord_idx_file': args.hybrid_coord_idx_file,
        'frozen_coord_idx_file': args.frozen_coord_idx_file,
        'prim_idx_file': args.prim_idx_file,
        'gsm_type': args.mode,
        'num_nodes': _default_num_nodes(args.mode, args.num_nodes),
        'isomers_file': args.isomers,
        'ADD_NODE_TOL': args.add_node_tol,
        'CONV_TOL': args.conv_tol,
        'BDIST_RATIO': args.bdist_ratio,
        'DQMAG_MAX': args.dqmag_max,
        'growth_direction': args.growth_direction,
        'ID': args.id,
        'product_geom_fixed': args.product_geom_fixed,
        'gsm_print_level': args.gsm_print_level,
        'max_gsm_iters': args.max_gsm_iters,
        'max_opt_steps': args.max_opt_steps,
        'sigma': args.sigma,
        'only_climb': args.only_climb,
        'restart_file': args.restart_file,
        'no_climb': args.no_climb,
        'optimize_mesx': args.optimize_mesx,
        'optimize_meci': args.optimize_meci,
        'bonds_file': args.bonds_file,
        'mp_cores': args.mp_cores,
        'interp_method': args.interp_method,
        'only_drive': args.only_drive,
        'reparametrize': args.reparametrize,
        'analyze_ICs': args.analyze_ics,
        'start_climb_immediately': args.start_climb_immediately,
        'ase_class': args.ase_class,
        'ase_kwargs': args.ase_kwargs,
    }
    # SE strings historically use tighter thresholds than their node optimizers.
    # Preserve those defaults, but apply an explicit CLI setting to both layers.
    for name, value, node_default, se_default in (
        ('gmax', args.conv_gmax, 100.0, 0.001),
        ('Ediff', args.conv_ediff, 100.0, 0.1),
        ('dE', args.conv_de, 1.0, 0.5),
    ):
        string_default = node_default if args.mode == 'DE_GSM' else se_default
        inpfileq[f'conv_{name}'] = node_default if value is None else value
        inpfileq[f'CONV_{name}'] = string_default if value is None else value

    if verbose:
        output.printcool_dictionary(inpfileq, title='Parsed GSM Keys : Values')

    if inpfileq['charge'] != 0:
        logger.warning(
            'Warning: charge is not implemented for all level of theories. Make sure this is correct for your package.'
        )

    return inpfileq
