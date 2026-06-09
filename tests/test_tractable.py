"""The constructive side of the barrier: Weitz certificates and structure."""

import random

import numpy as np
import pytest

from worldkernel import order_parameter, ring_of_cliques, transfer_marginals, weitz_interval
from worldkernel.barrier import bp_marginals, exact_marginals, random_regular


# ---- Route 1: certified intervals --------------------------------------------

def test_weitz_interval_always_contains_exact():
    """The certificate is unconditional: at every depth, below and above the
    threshold, the interval contains the exact marginal."""
    rng = random.Random(11)
    for d in (3, 7):
        adj = random_regular(16, d, rng)
        ex = exact_marginals(adj, 16, 1.0)
        for v in (0, 5, 11):
            for depth in (2, 4, 6):
                lo, hi = weitz_interval(adj, v, 1.0, depth)
                assert lo - 1e-12 <= ex[v] <= hi + 1e-12


def test_weitz_converges_to_exact_below_threshold():
    rng = random.Random(11)
    adj = random_regular(16, 3, rng)
    ex = exact_marginals(adj, 16, 1.0)
    lo, hi = weitz_interval(adj, 0, 1.0, depth=14)
    assert hi - lo < 1e-4
    assert abs(0.5 * (lo + hi) - ex[0]) < 1e-4


def test_weitz_contraction_collapses_above_threshold():
    """Same machinery, wildly different certificate quality across d_c:
    the decay rate tracks the order parameter (d-1)*eta."""
    rng = random.Random(11)
    below = random_regular(16, 3, rng)  # (d-1)eta = 0.64
    above = random_regular(16, 7, rng)  # (d-1)eta = 1.22
    w_below = np.diff(weitz_interval(below, 0, 1.0, depth=8))[0]
    w_above = np.diff(weitz_interval(above, 0, 1.0, depth=5))[0]
    assert w_below < 0.005
    assert w_above > 0.2
    assert w_above / w_below > 20


def test_weitz_width_monotone_in_depth():
    rng = random.Random(11)
    adj = random_regular(16, 4, rng)
    widths = [np.diff(weitz_interval(adj, 0, 1.0, L))[0] for L in (2, 4, 6, 8, 10)]
    assert all(a >= b - 1e-12 for a, b in zip(widths, widths[1:]))


# ---- Route 2: structure beats degree ------------------------------------------

def test_transfer_matrix_matches_enumeration():
    """Exact validation: the O(m s^3) transfer matrix reproduces brute-force
    enumeration to machine precision."""
    for m, s in ((3, 4), (4, 3), (3, 5)):
        adj = ring_of_cliques(m, s)
        ex = exact_marginals(adj, m * s, 1.0)
        tm = transfer_marginals(m, s, 1.0)
        assert np.max(np.abs(tm - ex)) < 1e-10


def test_ring_degrees_above_critical():
    """The demonstration graph sits far above the Sly-Sun critical degree."""
    s = 9
    adj = ring_of_cliques(40, s)
    degs = sorted({len(a) for a in adj})
    assert degs == [s - 1, s]  # interior s-1, port vertices s
    assert order_parameter(s - 1, 1.0) > 1.3  # deep in the 'hard' regime


def test_structure_beats_degree_at_scale():
    """n=360, degree 8-9 (above d_c): the transfer matrix is exact and instant
    where enumeration has 2^360 states; BP, the generic message-passing answer,
    is measurably wrong on the same instance."""
    m, s = 40, 9
    tm = transfer_marginals(m, s, 1.0)
    assert tm.shape == (360,)
    assert np.all((tm > 0) & (tm < 1))
    # at most one occupied vertex per clique: states sum below 1 per clique
    assert np.all(tm.reshape(m, s).sum(axis=1) < 1.0)
    bp = bp_marginals(ring_of_cliques(m, s), m * s, 1.0)
    assert float(np.mean(np.abs(bp - tm))) > 0.03


def test_ring_of_cliques_validation():
    with pytest.raises(ValueError):
        ring_of_cliques(2, 5)
    with pytest.raises(ValueError):
        ring_of_cliques(5, 1)
