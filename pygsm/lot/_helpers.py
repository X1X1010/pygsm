"""Internal helpers for level-of-theory state bookkeeping."""

from __future__ import annotations


def normalize_states(states):
    """Fill missing lower states for each multiplicity without dropping other spin states."""
    highest = {}
    for multiplicity, index in states:
        highest[multiplicity] = max(highest.get(multiplicity, 0), index)
    normalized = [(mult, index) for mult, highest_index in highest.items() for index in range(highest_index + 1)]
    if set(normalized) == set(states):
        return states, False
    return normalized, True


def infer_gradient_states(states, gradient_states, calc_grad):
    if gradient_states is not None:
        return gradient_states
    return list(states) if calc_grad else []
