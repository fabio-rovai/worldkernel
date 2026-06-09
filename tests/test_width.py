"""The ontology-to-width bridge: exact computation governed by width."""

import random

import numpy as np
import pytest

from worldkernel import order_parameter, ring_of_cliques, transfer_marginals
from worldkernel.barrier import exact_marginals, random_regular
from worldkernel.tractable import (
    disjointness_graph,
    hardcore_z,
    min_fill_order,
    treewidth_marginal,
)


def test_min_fill_width_on_known_graphs():
    # a path has width 1
    path = [set() for _ in range(6)]
    for i in range(5):
        path[i].add(i + 1)
        path[i + 1].add(i)
    assert min_fill_order(path)[1] == 1
    # a clique K5 has width 4
    k5 = [set(range(5)) - {i} for i in range(5)]
    assert min_fill_order(k5)[1] == 4
    # ring of cliques: width stays near the clique size
    _, w = min_fill_order(ring_of_cliques(6, 4))
    assert w <= 5


def test_variable_elimination_matches_enumeration():
    rng = random.Random(11)
    adj = random_regular(12, 3, rng)
    ex = exact_marginals(adj, 12, 1.0)
    ve = np.array([treewidth_marginal(adj, v) for v in range(12)])
    assert np.max(np.abs(ve - ex)) < 1e-10


def test_variable_elimination_matches_transfer_matrix():
    m, s = 4, 4
    tm = transfer_marginals(m, s)
    adj = ring_of_cliques(m, s)
    order, _ = min_fill_order(adj)
    ve = np.array([treewidth_marginal(adj, v, order=order) for v in range(m * s)])
    assert np.max(np.abs(ve - tm)) < 1e-10


def test_hardcore_z_counts_independent_sets():
    # triangle at lam=1: independent sets are {}, {0}, {1}, {2} -> Z = 4
    tri = [{1, 2}, {0, 2}, {0, 1}]
    assert hardcore_z(tri, 1.0) == pytest.approx(4.0)
    # clamping the occupied vertex selects exactly its singleton set
    assert hardcore_z(tri, 1.0, clamp={0: 1}) == pytest.approx(1.0)


def test_taxonomy_width_tracks_branching_not_size():
    """The ontology bridge: 584 classes, degree above d_c, width ~ branching.
    Exact off-diagonal computation stays polynomial because disjointness
    constraints are local."""
    adj = disjointness_graph(branching=8, depth=3)
    n = len(adj)
    assert n == 8 + 64 + 512
    max_deg = max(len(a) for a in adj)
    assert order_parameter(max_deg, 1.0) > 1.0  # 'hard' regime by degree
    order, width = min_fill_order(adj)
    assert width <= 12  # near the branching factor, far below n
    p = treewidth_marginal(adj, 0, order=order)
    # top-level classes: 8 mutually disjoint siblings plus the empty option,
    # weakly perturbed by subtree weights: near-uniform over 9 options
    assert 0.05 < p < 0.2


def test_taxonomy_marginal_matches_enumeration_small():
    adj = disjointness_graph(branching=3, depth=2)  # 12 classes
    n = len(adj)
    ex = exact_marginals(adj, n, 1.0)
    ve = np.array([treewidth_marginal(adj, v) for v in range(n)])
    assert np.max(np.abs(ve - ex)) < 1e-10
