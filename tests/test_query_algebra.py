"""The query closure algebra: global functionals exact above the threshold."""

import numpy as np
import pytest

from worldkernel import CouplingKernel
from worldkernel.query_algebra import (
    expected_occupancy,
    linear_query,
    pairwise_coherence,
    ratio_query,
)
from worldkernel.query_class import occupation_pattern_prob
from worldkernel.tractable import ring_of_cliques


def _enum_moments(adj, lam=1.0):
    n = len(adj)
    mask = [0] * n
    for i in range(n):
        for j in adj[i]:
            mask[i] |= 1 << j
    Z = 0.0
    d = np.zeros(n)
    M = np.zeros((n, n))
    for s in range(1 << n):
        ok = True
        t = s
        while t:
            i = (t & -t).bit_length() - 1
            if mask[i] & s:
                ok = False
                break
            t &= t - 1
        if not ok:
            continue
        w = lam ** bin(s).count("1")
        Z += w
        occ = [v for v in range(n) if (s >> v) & 1]
        for a in occ:
            d[a] += w
            for b in occ:
                M[a, b] += w
    return d / Z, M / Z


def test_expected_occupancy_global_matches_enum():
    adj = ring_of_cliques(3, 6)  # degree 6, above d_c
    d, _ = _enum_moments(adj)
    eo = expected_occupancy(adj)
    assert eo.n_terms == len(adj)            # global: one term per vertex
    assert eo.value == pytest.approx(d.sum(), abs=1e-9)


def test_global_coherence_exact_above_threshold_and_matches_kernel():
    adj = ring_of_cliques(3, 7)  # degree 7, above d_c
    d, M = _enum_moments(adj)
    n = len(adj)
    coh = pairwise_coherence(adj)
    truth = sum(M[i, j] for i in range(n) for j in range(i + 1, n))
    assert coh.value == pytest.approx(truth, abs=1e-9)
    assert coh.n_terms == n * (n - 1) // 2   # all pairs: a global query
    # cross-check against the kernel's own coherence on the exact moment matrix
    assert coh.value == pytest.approx(CouplingKernel(M).pairwise_coherence(), abs=1e-9)


def test_closure_under_linear_combination():
    adj = ring_of_cliques(3, 6)
    # a hand-built linear functional: 2*P(Y_0=1) - 3*P(Y_1=1,Y_2=1)
    terms = [(2.0, {0: 1}), (-3.0, {1: 1, 2: 1})]
    lq = linear_query(adj, terms)
    manual = (2.0 * occupation_pattern_prob(adj, {0: 1}).value
              - 3.0 * occupation_pattern_prob(adj, {1: 1, 2: 1}).value)
    assert lq.value == pytest.approx(manual, abs=1e-12)


def test_closure_under_ratio():
    adj = ring_of_cliques(3, 6)
    r = ratio_query(adj, {0: 1, 17: 0}, {0: 1})
    num = occupation_pattern_prob(adj, {0: 1, 17: 0}).value
    den = occupation_pattern_prob(adj, {0: 1}).value
    assert r.value == pytest.approx(num / den, abs=1e-12)


def test_coherence_cost_is_polynomial_not_count():
    adj = ring_of_cliques(4, 6)  # n=24, |C| in the thousands
    coh = pairwise_coherence(adj)
    n = len(adj)
    assert coh.n_terms == n * (n - 1) // 2   # O(n^2) terms, not |C|
    assert coh.width <= 7                     # bounded width => poly per term
