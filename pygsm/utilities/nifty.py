"""Deprecated alias for :mod:`pygsm.utilities.output`.

Kept only so existing callers of ``nifty.printcool`` keep working; import from
:mod:`pygsm.utilities.output` instead.
"""

from .output import click, commadash, logger, printcool, printcool_dictionary

__all__ = ['click', 'commadash', 'logger', 'printcool', 'printcool_dictionary']
