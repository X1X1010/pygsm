"""Public CLI entrypoints for ``pygsm``."""

from __future__ import annotations

from ._cli import main, parse_arguments, post_processing

__all__ = [
    'main',
    'parse_arguments',
    'post_processing',
]


if __name__ == '__main__':
    raise SystemExit(main())
