"""The tractable query class: k-local counterfactuals on bounded-width worlds.

The query-scar result showed a single occupation marginal is read from a
low-rank quotient. This widens the win to the queries the kernel actually
asks, and pins the exact tractable class.

A counterfactual query supported on a vertex set Q (the off-diagonal
P(Y_i, Y_j), a higher moment, a probability of necessity, a fraction harmed)
is a functional of the hard-core measure restricted to Q. Its COUPLING RANK
is the number of joint occupation patterns of Q the query distinguishes, at
most 2^|Q|. Each pattern's weight is a clamped partition-function ratio,
computable by variable elimination in time O(n * 2^{w+1}) where w is the
min-fill width of the world's constraint graph. Therefore:

  THEOREM (tractable query class). Every k-local counterfactual query
  (|Q| <= k) on a world of treewidth w is computable exactly in
  O(2^k * n * 2^{w+1}) time, independent of |C| (the partition function).
  For bounded k and bounded w this is polynomial, even when |C| is
  exponential and counting is Sly-Sun hard above the degree threshold.

The coupling rank (<= 2^k) is the scar dimension of the query; width is the
per-pattern cost. The class {bounded-k local queries on bounded-width
worlds} is exactly the kernel's working regime: pairwise off-diagonals
(k=2), three-world nested counterfactuals (k=3), and their functionals.
"""

from __future__ import annotations

from dataclasses import dataclass

from .tractable import hardcore_z, min_fill_order

__all__ = [
    "QueryVerdict",
    "occupation_pattern_prob",
    "pairwise_offdiagonal",
    "coupling_rank",
    "necessity_from_couplings",
]


@dataclass
class QueryVerdict:
    value: float
    coupling_rank: int   # patterns the query reads (scar dimension), <= 2^k
    width: int           # per-pattern elimination cost exponent
    exact: bool = True

    def __repr__(self) -> str:
        return (f"QueryVerdict({self.value:.6f}, rank={self.coupling_rank}, "
                f"width={self.width})")


def occupation_pattern_prob(adj, pattern: dict[int, int], lam: float = 1.0):
    """P(the listed vertices take the given 0/1 occupations) under the
    hard-core measure, by variable elimination. pattern maps vertex -> bit.

    Clamping occupied vertices may conflict with the constraints (two adjacent
    vertices both occupied) -> probability 0. Cost: one elimination per the
    clamped and unclamped Z, poly in the min-fill width."""
    # feasibility: no two clamped-occupied vertices adjacent
    occ = [v for v, b in pattern.items() if b == 1]
    for a in occ:
        for c in occ:
            if a != c and c in adj[a]:
                return QueryVerdict(0.0, 2 ** len(pattern), 0)
    order, width = min_fill_order(adj)
    z = hardcore_z(adj, lam, order=order)
    z_clamped = hardcore_z(adj, lam, clamp=dict(pattern), order=order)
    return QueryVerdict(z_clamped / z, 2 ** len(pattern), width)


def pairwise_offdiagonal(adj, i: int, j: int, lam: float = 1.0) -> QueryVerdict:
    """The kernel's core object: the off-diagonal coupling P(Y_i=1, Y_j=1)
    of two worlds, computed in poly(width) regardless of |C|. Coupling
    rank 4 (the joint occupation patterns of {i, j})."""
    v = occupation_pattern_prob(adj, {i: 1, j: 1}, lam)
    return QueryVerdict(v.value, coupling_rank=4, width=v.width)


def coupling_rank(support_size: int) -> int:
    """The scar dimension of a query on `support_size` vertices: at most
    2^support_size joint occupation patterns. Bounded k => bounded rank,
    independent of n and |C|."""
    return 2**support_size


def necessity_from_couplings(adj, treat: int, ctrl: int, lam: float = 1.0):
    """Probability-of-necessity-style functional of the pairwise off-diagonal,
    showing a downstream counterfactual is still k-local (k=2) hence tractable.

    PN-analogue = P(Y_ctrl=0 | Y_treat=1) = P(treat=1, ctrl=0) / P(treat=1),
    read from the same rank-4 coupling, poly(width)."""
    p_t1 = occupation_pattern_prob(adj, {treat: 1}, lam).value
    if p_t1 <= 0:
        return QueryVerdict(float("nan"), 4, 0)
    p_t1_c0 = occupation_pattern_prob(adj, {treat: 1, ctrl: 0}, lam)
    return QueryVerdict(p_t1_c0.value / p_t1, coupling_rank=4, width=p_t1_c0.width)
