"""Bounded interaction rank: long-range queries through a low-rank channel.

The width engine costs 2^w; the closure algebra handles global queries that
decompose into poly-many LOCAL pieces. This module handles the remaining
case: genuinely LONG-RANGE queries whose dependence between far-apart
vertices factors through a bounded-rank channel, even when the treewidth is
large.

The governing invariant is the INTERACTION RANK chi: the rank of the
transfer operator across the cuts of the world. For a world built from
cliques joined in a path or ring (the structure ontology disjointness
axioms generate), each clique admits at most one occupied vertex, so the
transfer operator across a clique boundary is an (s+1) x (s+1) matrix
(s = clique size), NOT 2^s. Hence:

  chi = s + 1   (LINEAR in the separator)   vs   2^w = 2^s   (the width cost).

A query, including a long-range off-diagonal P(Y_i, Y_j) between vertices in
far-apart cliques, is then computed by matrix powers of the chi x chi
transfer operator in O(chi^3 log m) time, independent of |C| (which is
exponential in m) AND exponentially below the 2^w width engine.

THEOREM (interaction-rank tractability). Every k-local or low-tensor-rank
counterfactual query on a world of interaction rank chi is computable
exactly in poly(n, chi) time, independent of |C|. The interaction rank can
be exponentially smaller than the treewidth (clique separators: chi = s+1
vs 2^s; the K(m,m) bottleneck: chi = 2 vs treewidth m), so this is a strict
widening of the width-based class. Ontology worlds, whose disjointness
constraints are dense cliques, have small interaction rank by construction.

The honest boundary is intact: the partition function still costs the
top transfer eigenvalue to the m-th power to even represent in magnitude;
what is poly is every bounded-rank QUERY, not the count itself.
"""

from __future__ import annotations

import numpy as np

from .tractable import ring_of_cliques

__all__ = [
    "ring_clique_transfer",
    "ring_clique_pair",
    "ring_clique_marginal",
    "interaction_rank",
    "treewidth_cost",
]


def ring_clique_transfer(s: int, lam: float = 1.0) -> np.ndarray:
    """The (s+1) x (s+1) transfer operator of a ring of size-s cliques.

    State 0 = clique empty; state j+1 = vertex j of the clique occupied
    (at most one, by the clique constraint). The only inter-clique edge joins
    the exit port (vertex 0) of one clique to the entry port (vertex 1) of the
    next, so the forbidden transition is exit-port-occupied -> entry-port-
    occupied. This matches worldkernel.tractable.ring_of_cliques."""
    k = s + 1
    T = np.ones((k, k))
    T[:, 1:] *= lam            # weight of the destination occupied state
    T[1, 2] = 0.0              # exit port occupied -> next entry port forbidden
    return T


def _normalize(T: np.ndarray) -> np.ndarray:
    """Divide the transfer operator by its spectral radius. All queries are
    ratios that scale as top^m in both numerator and denominator, so this is
    exact and avoids float overflow of T^m for large m."""
    top = float(np.abs(np.linalg.eigvals(T)).max())
    return T / top if top > 0 else T


def _Z(T: np.ndarray, m: int) -> float:
    return float(np.trace(np.linalg.matrix_power(_normalize(T), m)))


def interaction_rank(s: int) -> int:
    """chi = s + 1: the transfer-operator dimension for a size-s clique
    separator. Linear in the separator, vs 2^s for the naive boundary."""
    return s + 1


def treewidth_cost(s: int) -> int:
    """2^s: what the width engine would pay for a size-s clique separator."""
    return 2**s


def ring_clique_marginal(m: int, s: int, clique: int, vertex: int,
                         lam: float = 1.0) -> float:
    """P(vertex `vertex` of clique `clique` is occupied) via the rank-(s+1)
    transfer channel: (T^m with the clique clamped to that vertex) / Z."""
    T = _normalize(ring_clique_transfer(s, lam))
    Tm = np.linalg.matrix_power(T, m)
    Z = float(np.trace(Tm))
    # trace(D T^m) with one clamp == (T^m)[v+1, v+1]
    return float(Tm[vertex + 1, vertex + 1]) / Z


def ring_clique_pair(m: int, s: int, clique_a: int, vtx_a: int,
                     clique_b: int, vtx_b: int, lam: float = 1.0) -> float:
    """The LONG-RANGE off-diagonal P(Y_a = 1, Y_b = 1) for vertices in
    distinct cliques `clique_a`, `clique_b` of the ring, via the transfer
    channel. Using diagonal projectors D_a, D_b onto the clamped clique
    states and d = (clique_b - clique_a) mod m:

        trace(D_a T^d D_b T^{m-d}) = (T^d)[a, b] * (T^{m-d})[b, a],

    so P = that / Z. Exact, O(s^3 log m), independent of |C| and of 2^s."""
    if clique_a == clique_b:
        raise ValueError("use ring_clique_marginal for same-clique queries")
    T = _normalize(ring_clique_transfer(s, lam))
    Tm = np.linalg.matrix_power(T, m)
    Z = float(np.trace(Tm))
    a, b = vtx_a + 1, vtx_b + 1
    d = (clique_b - clique_a) % m
    Td = np.linalg.matrix_power(T, d)
    Tmd = np.linalg.matrix_power(T, m - d)
    return float(Td[a, b] * Tmd[b, a]) / Z


# ---- validation helper: exact enumeration on the actual graph --------------

def _enum_pair(m: int, s: int, ga: int, va: int, gb: int, vb: int, lam=1.0):
    adj = ring_of_cliques(m, s)
    n = len(adj)
    mask = [0] * n
    for i in range(n):
        for j in adj[i]:
            mask[i] |= 1 << j
    A = ga * s + va
    B = gb * s + vb
    Z = 0.0
    num = 0.0
    for state in range(1 << n):
        ok = True
        t = state
        while t:
            i = (t & -t).bit_length() - 1
            if mask[i] & state:
                ok = False
                break
            t &= t - 1
        if not ok:
            continue
        w = lam ** bin(state).count("1")
        Z += w
        if (state >> A) & 1 and (state >> B) & 1:
            num += w
    return num / Z
