"""Typed, validated option containers for pygsm objects.

Each configurable class owns a ``@dataclass(kw_only=True)`` subclass of
:class:`Config`. Fields carry their defaults and documentation; allowed values
are declared with :func:`option` and checked when the config is created or
replaced. Copies are shallow (field values are shared), matching how pygsm
objects share level-of-theory, coordinate and molecule objects.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from typing import Any, TypeVar

C = TypeVar('C', bound='Config')


def option(
    default: Any = dataclasses.MISSING, *, doc: str = '', choices=None, default_factory: Any = dataclasses.MISSING
):
    """Declare a config field with documentation and, optionally, the allowed values."""
    metadata = {'doc': doc}
    if choices is not None:
        metadata['choices'] = tuple(choices)
    if default_factory is not dataclasses.MISSING:
        return field(default_factory=default_factory, metadata=metadata)
    return field(default=default, metadata=metadata)


@dataclass(kw_only=True)
class Config:
    def __post_init__(self):
        for f in fields(self):
            choices = f.metadata.get('choices')
            if choices is not None and getattr(self, f.name) not in choices:
                raise ValueError(
                    f'{type(self).__name__}.{f.name} must be one of {choices!r}, got {getattr(self, f.name)!r}'
                )

    @classmethod
    def build(cls: type[C], config: C | None = None, **changes) -> C:
        """Return ``config`` with ``changes`` applied, or a new config from ``changes``."""
        if config is None:
            return cls(**changes)
        return config.replace(**changes) if changes else config

    def replace(self: C, **changes) -> C:
        """Shallow copy with some fields changed (validated)."""
        return dataclasses.replace(self, **changes)

    def as_kwargs(self) -> dict[str, Any]:
        """Field values as a dict, without copying the values."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
