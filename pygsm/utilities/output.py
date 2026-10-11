"""Console output helpers: the package logger, banner printing and timers.

``printcool``/``printcool_dictionary`` originate from ForceBalance/geomeTRIC
(Lee-Ping Wang).
"""

from __future__ import annotations

import itertools
import logging
import re
import sys
import time
from collections import OrderedDict

import numpy as np


class RawStreamHandler(logging.StreamHandler):
    """Write log messages to stdout verbatim (no level/name prefix, no added newline)."""

    def __init__(self, stream=sys.stdout):
        super().__init__(stream)

    def emit(self, record):
        self.stream.write(record.getMessage())
        self.flush()


logger = logging.getLogger('NiftyLogger')
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(RawStreamHandler())


def printcool(text, sym='#', bold=False, color=2, ansi=None, bottom='-', minwidth=50, center=True, sym2='='):
    """Print ``text`` centred in an ANSI-coloured box and return a matching bottom bar.

    ``color`` is the ANSI colour code (1 red, 2 green, 3 yellow, 4 blue, 5 magenta,
    6 cyan, 7 white; 0 or None for no colour); ``ansi`` overrides it with a raw
    escape code. ``center`` may be a list with one flag per line.
    """

    def newlen(line):
        return len(re.sub(r'\x1b\[[0-9;]*m', '', line))

    lines = text.split('\n')
    width = max(minwidth, max(newlen(line) for line in lines))
    bar = sym + sym2 * (width + 6) + sym
    logger.info('\r' + bar + '\n')
    for ln, line in enumerate(lines):
        c1 = center[ln] if isinstance(center, list) else center
        padleft = ' ' * int((width - newlen(line)) / 2) if c1 else ''
        padright = ' ' * (width - newlen(line) - len(padleft))
        if ansi is not None:
            logger.info(f'{sym}| \x1b[{ansi}m{padleft} ' + line + f' {padright}\x1b[0m |{sym}\n')
        elif color == 0 and bold:
            logger.info(f'{sym}| \x1b[1m{padleft} ' + line + f' {padright}\x1b[0m |{sym}\n')
        elif not color:
            logger.info(f'{sym}| {padleft} ' + line + f' {padright} |{sym}\n')
        else:
            logger.info(f'{sym}| \x1b[{"1;" if bold else ""}9{color}m{padleft} ' + line + f' {padright}\x1b[0m |{sym}\n')
    logger.info(bar + '\n')
    return bottom * (width + 8) + '\n'


def printcool_dictionary(Dict, title='Dictionary Keys : Values', bold=False, color=2, keywidth=25, topwidth=50, center=True, leftpad=0):
    """Print the non-``None`` entries of a mapping as a key/value table under a ``printcool`` banner.

    Keys are sorted unless ``Dict`` is an ``OrderedDict``.
    """
    if Dict is None:
        return
    bar = printcool(title, bold=bold, color=color, minwidth=topwidth, center=center)
    keys = list(Dict) if isinstance(Dict, OrderedDict) else sorted(Dict)
    logger.info('\n'.join(' ' * leftpad + f'{key!s:<{keywidth}} {Dict[key]!s} ' for key in keys if Dict[key] is not None))
    logger.info(f'\n{bar}')


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
