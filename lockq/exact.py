"""Exact likelihood of the complete unordered multiset of observed batches.

Separate, small NumPy reference implementation of the two- and three-state
likelihood formula. It is not the original Torch training pipeline.
"""
from collections import Counter, defaultdict
from math import factorial

import numpy as np


def _transition_matrices(transition, time_steps, states):
    arr = np.asarray(transition, dtype=float)
    if arr.shape == (states, states):
        return np.broadcast_to(arr, (time_steps - 1, states, states))
    if arr.shape == (time_steps - 1, states, states):
        return arr
    raise ValueError("Invalid transition shape")


def occupancy_distribution(initial, transition, time_steps):
    """P(hidden-state occupation counts), summed over hidden paths."""
    initial = np.asarray(initial, dtype=float)
    states = len(initial)
    if states not in (2, 3) or time_steps < 1:
        raise ValueError("Expected two or three states and at least one batch")
    trans = _transition_matrices(transition, time_steps, states)
    joint = {}
    for state, prob in enumerate(initial):
        counts = tuple(int(i == state) for i in range(states))
        joint[state, counts] = float(prob)
    for t in range(time_steps - 1):
        next_joint = defaultdict(float)
        for (old, counts), prob in joint.items():
            for new in range(states):
                updated = list(counts)
                updated[new] += 1
                next_joint[new, tuple(updated)] += prob * trans[t, old, new]
        joint = next_joint
    result = defaultdict(float)
    for (_, counts), prob in joint.items():
        result[counts] += prob
    return dict(result)


def ordered_likelihood(batches, emissions, initial, transition):
    """Standard forward likelihood for a batch order."""
    if not batches:
        raise ValueError("Empty batch record")
    initial = np.asarray(initial, dtype=float)
    trans = _transition_matrices(transition, len(batches), len(initial))
    state = initial * np.asarray(emissions[batches[0]], dtype=float)
    for t in range(1, len(batches)):
        state = (state @ trans[t - 1]) * np.asarray(emissions[batches[t]], dtype=float)
    return float(state.sum())


def unordered_likelihood(batches, emissions, initial, transition):
    """Exact probability of an unordered batch multiset, including duplicates."""
    if not batches:
        raise ValueError("Empty batch record")
    states = len(initial)
    occupation = occupancy_distribution(initial, transition, len(batches))
    coefficients = {tuple([0] * states): 1.0}
    for batch in batches:
        emission = np.asarray(emissions[batch], dtype=float)
        if emission.shape != (states,):
            raise ValueError("Invalid emission vector")
        next_coefficients = defaultdict(float)
        for counts, weight in coefficients.items():
            for state in range(states):
                updated = list(counts)
                updated[state] += 1
                next_coefficients[tuple(updated)] += weight * emission[state]
        coefficients = next_coefficients
    duplicates = 1
    for multiplicity in Counter(batches).values():
        duplicates *= factorial(multiplicity)
    total = 0.0
    for counts, value in coefficients.items():
        total += (
            occupation.get(counts, 0.0)
            * value
            * np.prod([factorial(n) for n in counts])
        )
    return float(total / duplicates)
