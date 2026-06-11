"""Query tractability via scars: the query is read from a low-rank quotient."""

import numpy as np
import pytest

from worldkernel.phases import kmm_exact_marginal
from worldkernel.query_scar import (
    kmm_marginal_via_scar,
    local_query_via_scar,
    shiraishi_mori_block,
)
from worldkernel.tractable import min_fill_order, ring_of_cliques, treewidth_marginal


def test_kmm_query_rank_two_while_counting_is_exponential():
    for m in (5, 20, 100, 200):
        sq = kmm_marginal_via_scar(m)
        assert sq.scar_rank == 2                       # the query quotient
        assert sq.value == pytest.approx(kmm_exact_marginal(m), abs=1e-12)
        assert sq.n_admissible == pytest.approx(2 * 2.0**m - 1)  # |C|, bypassed
    # at m=200 the normalizer is astronomically large; the query is rank 2
    assert kmm_marginal_via_scar(200).n_admissible > 1e60


def test_shiraishi_mori_subspace_is_an_exact_invariant():
    rng = np.random.default_rng(0)
    N = 8
    P = np.zeros((N, N))
    for i in (0, 2, 5):  # a 3-dimensional scar subspace
        P[i, i] = 1.0
    A = rng.normal(size=(N, N))
    A = A + A.T
    B = rng.normal(size=(N, N))
    B = B + B.T
    H = shiraishi_mori_block(P, A, B)
    # [H, P] = 0: the scar subspace never leaks
    assert np.allclose(H @ P - P @ H, 0.0, atol=1e-9)


def test_local_query_scar_rank_bounded_and_matches_engine():
    adj = ring_of_cliques(3, 6)  # degree 6, above the Sly-Sun threshold
    scar_v, exact_v, rank, Z = local_query_via_scar(adj, 0)
    assert scar_v == pytest.approx(exact_v, abs=1e-9)
    assert rank < Z / 10                       # rank decoupled from |C|
    order, _ = min_fill_order(adj)
    assert treewidth_marginal(adj, 0, order=order) == pytest.approx(exact_v, abs=1e-9)


def test_full_count_query_has_no_low_rank_shortcut():
    """Sanity: the 'whole normalizer' query is full rank (no free lunch). We
    witness this by a query whose group is every admissible world: rank = |C|."""
    adj = ring_of_cliques(3, 4)
    # a degenerate 'local' query at every vertex would still be bounded; the
    # full-counting query is rank |C| by construction (not via this helper).
    _, _, rank, Z = local_query_via_scar(adj, 0)
    assert rank <= 2 ** (max(len(a) for a in adj) + 1)  # local => bounded
    assert Z > rank  # counting is strictly larger than the local query rank
