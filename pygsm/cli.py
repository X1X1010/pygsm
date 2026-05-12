"""Console entrypoints and script compatibility helpers for `pygsm`."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_loader
from importlib.machinery import SourceFileLoader
from pathlib import Path


def _load_bin_gsm():
    script_path = Path(__file__).resolve().parent.parent / 'bin' / 'gsm'
    loader = SourceFileLoader('pygsm._bin_gsm', str(script_path))
    spec = spec_from_loader(loader.name, loader)
    if spec is None:
        raise ImportError(f'Could not load CLI module from {script_path}')
    module = module_from_spec(spec)
    loader.exec_module(module)
    return module


_BIN_GSM = _load_bin_gsm()

cleanup_scratch = _BIN_GSM.cleanup_scratch
main = _BIN_GSM.main
parse_arguments = _BIN_GSM.parse_arguments
post_processing = _BIN_GSM.post_processing
