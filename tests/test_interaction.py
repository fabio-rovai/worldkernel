"""Bounded interaction rank: long-range queries through a low-rank channel."""

import numpy as np
import pytest

from worldkernel.interaction import (
    _enum_pair,
    interaction_rank,
    ring_clique_marginal,
    ring_clique_pair,
    treewidth_cost,
)
from worldkernel.tractable import transfer_marginals


def test_long_range_pair_matches_enumeration():
    for m, s, ga, va, gb, vb in [(3, 4, 0, 2, 2, 3), (4, 3, 0, 0, 2, 1),
                                 (4, 4, 0, 3, 2, 2), (5, 3, 0, 0, 3, 2)]:
        mine = ring_clique_pair(m, s, ga, va, gb, vb)
        assert mine == pytest.approx(_enum_pair(m, s, ga, va, gb, vb), abs=1e-9)


def test_marginal_matches_transfer_engine():
    m, s = 4, 5
    tm = transfer_marginals(m, s)
    for v in range(s):
        assert ring_clique_marginal(m, s, 0, v) == pytest.approx(float(tm[v]), abs=1e-9)


def test_adjacent_ports_cannot_co_occur():
    # vertex 0 (exit port) of clique 0 and vertex 1 (entry port) of clique 1
    # are joined by the inter-clique edge: the pair off-diagonal is 0
    assert ring_clique_pair(4, 4, 0, 0, 1, 1) == pytest.approx(0.0, abs=1e-12)


def test_interaction_rank_below_width_cost():
    for s in (10, 20, 30):
        assert interaction_rank(s) == s + 1          # linear
        assert treewidth_cost(s) == 2**s             # exponential
        assert interaction_rank(s) < treewidth_cost(s)


def test_no_overflow_at_large_m_and_valid_probability():
    # rings far too large to enumerate or to run the width engine on
    for m, s in [(1000, 20), (5000, 30)]:
        v = ring_clique_pair(m, s, 0, 2, m // 2, 3)
        assert np.isfinite(v)
        assert 0.0 <= v <= 1.0


def test_marginal_is_a_valid_probability_at_scale():
    v = ring_clique_marginal(2000, 25, 0, 0)
    assert np.isfinite(v) and 0.0 <= v <= 1.0
