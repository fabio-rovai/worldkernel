"""The tractable query class: k-local counterfactuals on bounded-width worlds."""

import pytest

from worldkernel.query_class import (
    coupling_rank,
    necessity_from_couplings,
    occupation_pattern_prob,
    pairwise_offdiagonal,
)
from worldkernel.tractable import ring_of_cliques


def _enum(adj, pattern, lam=1.0):
    n = len(adj)
    mask = [0] * n
    for i in range(n):
        for j in adj[i]:
            mask[i] |= 1 << j
    z = num = 0.0
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
        z += w
        if all(((s >> v) & 1) == b for v, b in pattern.items()):
            num += w
    return num / z


def test_pairwise_offdiagonal_above_threshold_matches_enum():
    adj = ring_of_cliques(3, 6)  # degree 6, above d_c
    v = pairwise_offdiagonal(adj, 0, 17)
    assert v.coupling_rank == 4
    assert v.value == pytest.approx(_enum(adj, {0: 1, 17: 1}), abs=1e-9)


def test_klocal_hierarchy_matches_enum():
    adj = ring_of_cliques(3, 7)
    for pat in ({0: 1}, {0: 1, 20: 1}, {0: 1, 1: 0, 20: 1}):
        v = occupation_pattern_prob(adj, pat)
        assert v.value == pytest.approx(_enum(adj, pat), abs=1e-9)
        assert v.coupling_rank == 2 ** len(pat)


def test_infeasible_pattern_is_zero():
    adj = ring_of_cliques(3, 6)
    # two vertices in the same clique cannot both be occupied
    clique0 = [0] + sorted(adj[0])[:1]
    v = occupation_pattern_prob(adj, {clique0[0]: 1, clique0[1]: 1})
    assert v.value == 0.0


def test_coupling_rank_decoupled_from_count():
    assert coupling_rank(1) == 2
    assert coupling_rank(2) == 4
    assert coupling_rank(3) == 8
    # the rank depends only on query locality k, never on |C| or n


def test_necessity_functional_is_klocal_and_correct():
    adj = ring_of_cliques(3, 6)
    pn = necessity_from_couplings(adj, treat=0, ctrl=17)
    p_t1 = _enum(adj, {0: 1})
    p_t1_c0 = _enum(adj, {0: 1, 17: 0})
    assert pn.value == pytest.approx(p_t1_c0 / p_t1, abs=1e-9)
    assert pn.coupling_rank == 4
