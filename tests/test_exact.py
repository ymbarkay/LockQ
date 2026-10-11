"""Tests of the exact unordered multiset likelihood."""
from itertools import permutations
from math import isclose

import numpy as np
import pytest

from lockq.exact import occupancy_distribution, ordered_likelihood, unordered_likelihood


@pytest.mark.parametrize("states", [2, 3])
@pytest.mark.parametrize("batches", [(1, 0, 1), (0, 2, 1, 2), (1, 1, 1)])
def test_exact_matches_all_distinct_permutations(states, batches):
    rng = np.random.default_rng(400 + states + len(batches))
    initial = rng.dirichlet(np.ones(states))
    transitions = rng.dirichlet(np.ones(states), size=states)
    emissions = {b: rng.random(states) * 0.3 for b in set(batches)}
    expected = sum(
        ordered_likelihood(list(p), emissions, initial, transitions)
        for p in set(permutations(batches))
    )
    observed = unordered_likelihood(list(batches), emissions, initial, transitions)
    assert isclose(expected, observed, rel_tol=1e-12, abs_tol=1e-14)
    assert isclose(
        sum(occupancy_distribution(initial, transitions, len(batches)).values()),
        1.0, abs_tol=1e-12
    )


def test_unordered_is_permutation_invariant():
    initial = [0.3, 0.7]
    transitions = [[0.95, 0.05], [0.1, 0.9]]
    emissions = {1: [0.1, 0.4], 2: [0.7, 0.2], 3: [0.2, 0.2]}
    a = unordered_likelihood([1, 2, 3], emissions, initial, transitions)
    b = unordered_likelihood([3, 1, 2], emissions, initial, transitions)
    assert a == pytest.approx(b, abs=1e-15)
    oa = ordered_likelihood([1, 2, 3], emissions, initial, transitions)
    ob = ordered_likelihood([3, 1, 2], emissions, initial, transitions)
    assert abs(oa - ob) > 1e-8


def test_exchangeable_model():
    initial = [0.4, 0.6]
    transitions = [[0.4, 0.6], [0.4, 0.6]]
    emissions = {1: [0.1, 0.8], 2: [0.5, 0.2], 3: [0.3, 0.5]}
    a = ordered_likelihood([1, 2, 3], emissions, initial, transitions)
    b = ordered_likelihood([3, 1, 2], emissions, initial, transitions)
    assert a == pytest.approx(b, abs=1e-15)
