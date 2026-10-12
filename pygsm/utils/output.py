"""Console output helpers: the package logger, banner printing and timers.

Library code logs through module loggers under the ``pygsm`` logger and never
prints; :func:`setup_logging` sends that output to stdout or a file.
``printcool``/``printcool_dictionary`` originate from ForceBalance/geomeTRIC
(Lee-Ping Wang).
"""

from __future__ import annotations

import itertools
import logging
import sys
import time
from collections import OrderedDict

import numpy as np

#: Parent logger of every pygsm module logger (``logging.getLogger(__name__)``).
logger = logging.getLogger('pygsm')
logger.addHandler(logging.NullHandler())


class _StdoutHandler(logging.StreamHandler):
    """Stream handler bound to whatever ``sys.stdout`` is at emit time, so redirection works."""

    def __init__(self):
        super().__init__(sys.stdout)

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value):
        pass


_handler: logging.Handler | None = None


def setup_logging(level: int | str = logging.INFO, filename: str | None = None) -> logging.Handler:
    """Print pygsm log messages, unadorned, to stdout (or append them to ``filename``).

    pygsm only logs; it never configures logging on import. Applications call this
    (the ``gsm`` CLI does) or attach their own handlers to the ``pygsm`` logger.
    ``logging.DEBUG`` shows the detailed per-step diagnostics. Calling it again
    replaces the handler installed by the previous call.
    """
    global _handler
    if _handler is not None:
        logger.removeHandler(_handler)
        _handler.close()
    _handler = logging.FileHandler(filename) if filename is not None else _StdoutHandler()
    _handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(_handler)
    logger.setLevel(level)
    return _handler


def printcool(text: str, level: int = logging.INFO, minwidth: int = 50) -> str:
    """Log ``text`` (one or more lines) centred in a banner box; return a matching bottom bar."""
    lines = text.split('\n')
    width = max(minwidth, *(len(line) for line in lines))
    bar = '#' + '=' * (width + 6) + '#'
    logger.log(level, '\n'.join([bar, *(f'#|  {line:^{width}}  |#' for line in lines), bar]))
    return '-' * (width + 8)


def printcool_dictionary(Dict, title: str = 'Dictionary Keys : Values', keywidth: int = 25, topwidth: int = 50) -> None:
    """Log the non-``None`` entries of a mapping as a key/value table under a ``printcool`` banner.

    Keys are sorted unless ``Dict`` is an ``OrderedDict``.
    """
    if Dict is None:
        return
    bar = printcool(title, minwidth=topwidth)
    keys = list(Dict) if isinstance(Dict, OrderedDict) else sorted(Dict)
    logger.info('\n'.join(f'{key!s:<{keywidth}} {Dict[key]!s} ' for key in keys if Dict[key] is not None))
    logger.info(bar)


def _encode(values):
    return [[len(list(group)), name] for name, group in itertools.groupby(values)]


def _segments(encoded):
    begins = np.array([sum(k[0] for k in encoded[:j]) for j, item in enumerate(encoded) if item[1] == 1])
    lens = np.array([item[0] for item in encoded if item[1] == 1])
    return [(i, i + j) for i, j in zip(begins, lens, strict=True)]


def commadash(values):
    """Format 0-based indices as 1-based ranges, e.g. ``[26, 27, 28, 99]`` -> ``'27-29,100'``."""
    L = sorted(values)
    if len(L) == 0:
        return '(empty)'
    L.append(L[-1] + 1)
    mask = [i in L for i in range(L[-1])]
    return ','.join(f'{i[0] + 1}-{i[1]}' if (i[1] - 1 > i[0]) else f'{i[0] + 1}' for i in _segments(_encode(mask)))


def click():
    """Stopwatch: return the seconds elapsed since the previous call."""
    ans = time.time() - click.t0
    click.t0 = time.time()
    return ans


click.t0 = time.time()
