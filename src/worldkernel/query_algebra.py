"""The query closure algebra: global counterfactuals from local pieces.

The k-local theorem (query_class) makes a query supported on <= k vertices
tractable. This widens the class to GLOBAL queries, the ones that touch every
vertex, by the observation that coupling rank is governed by the query's
STRUCTURE, not its support size. Two structural mechanisms:

  LINEAR DECOMPOSITION. A query that is a poly-size linear combination of
  k-local terms, Q = sum_S c_S f_S with each f_S a k-local pattern
  probability, is tractable: evaluate each term by variable elimination in
  poly(width) and sum. The pairwise coherence Q = sum_{i<j} P(Y_i=1, Y_j=1),
  the expected occupancy E[sum_i Y_i], and every bounded-order moment are of
  this form. They touch all n vertices yet cost only O(n^k * poly(width)),
  independent of |C|.

  CLOSURE. The tractable query class is closed under (i) poly-size linear
  combination, (ii) ratios (conditional / probability-of-necessity style),
  and (iii) products of bounded order. So a whole algebra of global
  counterfactual functionals inherits tractability from the local pieces.

THEOREM (closure of the tractable class). If Q_1, ..., Q_r are tractable
counterfactual queries (each k-local, on a treewidth-w world) and r =
poly(n), then any poly-size arithmetic combination of them, sum c_a Q_a,
products of bounded degree, and ratios Q_a / Q_b, is computable exactly in
poly(n, 2^w) time, independent of |C|. This is closed under the operations
that build the kernel's reported quantities: aggregate coherence, average
causal effects, probabilities of necessity and sufficiency, fractions
harmed, and natural (in)direct effects assembled from them.

The honest boundary is unchanged: the partition function itself is the one
functional NOT of this form (its rank is |C|), so the algebra characterizes
the tractable frontier; it does not cross it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .query_class import occupation_pattern_prob
from .tractable import min_fill_order

__all__ = [
    "AlgebraVerdict",
    "expected_occupancy",
    "pairwise_coherence",
    "linear_query",
    "ratio_query",
]


@dataclass
class AlgebraVerdict:
    value: float
    n_terms: int      # poly-size: the number of local pieces summed
    width: int        # per-piece elimination cost exponent
    exact: bool = True

    def __repr__(self) -> str:
        return (f"AlgebraVerdict({self.value:.6f}, terms={self.n_terms}, "
                f"width={self.width})")


def expected_occupancy(adj, lam: float = 1.0) -> AlgebraVerdict:
    """E[sum_i Y_i] = sum_i P(Y_i=1): a GLOBAL query (all vertices) that is a
    linear sum of n marginals, each poly(width)."""
    n = len(adj)
    _, w = min_fill_order(adj)
    total = sum(occupation_pattern_prob(adj, {i: 1}, lam).value for i in range(n))
    return AlgebraVerdict(total, n_terms=n, width=w)


def pairwise_coherence(adj, lam: float = 1.0) -> AlgebraVerdict:
    """Q = sum_{i<j} P(Y_i=1, Y_j=1), the kernel's aggregate off-diagonal
    coherence. A GLOBAL query over all O(n^2) pairs, each a rank-4 local
    coupling computed by variable elimination: exact in O(n^2 * poly(width)),
    independent of |C|. (Where structure is absent the kernel falls back to
    the PSD / Frechet bounds; here, on a bounded-width world, it is exact.)"""
    n = len(adj)
    _, w = min_fill_order(adj)
    total = 0.0
    npairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += occupation_pattern_prob(adj, {i: 1, j: 1}, lam).value
            npairs += 1
    return AlgebraVerdict(total, n_terms=npairs, width=w)


def linear_query(adj, terms, lam: float = 1.0) -> AlgebraVerdict:
    """A general linear functional sum_a c_a * P(pattern_a). `terms` is a list
    of (coefficient, pattern dict). Tractable for poly-many terms."""
    _, w = min_fill_order(adj)
    total = sum(c * occupation_pattern_prob(adj, pat, lam).value for c, pat in terms)
    return AlgebraVerdict(float(total), n_terms=len(terms), width=w)


def ratio_query(adj, num_pattern, den_pattern, lam: float = 1.0) -> AlgebraVerdict:
    """A conditional / probability-of-necessity functional
    P(num_pattern) / P(den_pattern): closure under ratios. Both numerator and
    denominator are k-local, so the ratio is tractable."""
    _, w = min_fill_order(adj)
    den = occupation_pattern_prob(adj, den_pattern, lam).value
    if den <= 0:
        return AlgebraVerdict(float("nan"), 2, w)
    num = occupation_pattern_prob(adj, num_pattern, lam).value
    return AlgebraVerdict(num / den, n_terms=2, width=w)
