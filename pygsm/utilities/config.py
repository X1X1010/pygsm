"""Constructor-parameter helpers shared by the configurable pygsm classes.

Configurable classes take their settings as keyword arguments of ``__init__``
and store each one in an attribute of the same name (the scikit-learn
convention). :meth:`Configurable.get_params` reads them back from the
signature, so :meth:`Configurable.clone` can build a new object with the same
(current) settings without the defaults being written down twice.
"""

from __future__ import annotations

import inspect
from typing import Any, TypeVar

T = TypeVar('T', bound='Configurable')

_PARAMETER_KINDS = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


def check_choice(name: str, value: Any, choices) -> None:
    """Raise ``ValueError`` unless ``value`` is one of ``choices``."""
    if value not in choices:
        raise ValueError(f'{name} must be one of {tuple(choices)!r}, got {value!r}')


class Configurable:
    """Mixin for classes whose ``__init__`` arguments are stored as attributes of the same name."""

    @classmethod
    def param_names(cls) -> list[str]:
        """Names of the constructor parameters, including those of base-class constructors."""
        names: list[str] = []
        for klass in reversed(cls.__mro__):
            init = klass.__dict__.get('__init__')
            if init is None:
                continue
            for parameter in list(inspect.signature(init).parameters.values())[1:]:
                if parameter.kind in _PARAMETER_KINDS and parameter.name not in names:
                    names.append(parameter.name)
        return names

    def get_params(self) -> dict[str, Any]:
        """The current value of every constructor parameter."""
        return {name: getattr(self, name) for name in self.param_names()}

    def clone(self: T, **changes) -> T:
        """A new object built with this one's current parameters, some of them changed."""
        params = self.get_params()
        unknown = set(changes) - set(params)
        if unknown:
            raise TypeError(f'{type(self).__name__} has no parameter(s) {sorted(unknown)}')
        return type(self)(**{**params, **changes})
