"""The constructive side of the barrier: certified intervals and structure.

The Sly-Sun theorem forbids one thing only: a general efficient algorithm for
the off-diagonal aggregate above the threshold (that would give NP = RP).
It does not forbid either of the two routes implemented here.

ROUTE 1: certified intervals in polynomial time, everywhere.
``weitz_interval`` runs the hard-core ratio recursion on the self-avoiding-walk
tree (Weitz 2006), truncated at a chosen depth, with INTERVAL arithmetic at the
truncation: every unknown boundary ratio lies in [0, lam], so propagating the
interval through the (monotone) recursion yields rigorous upper and lower
bounds on the true marginal at every depth. Below the threshold the interval
contracts geometrically (correlation decay); above it the contraction stalls.
The barrier stops being a wall you hit blind and becomes a quantity the
algorithm certifies about its own answer.

ROUTE 2: structure beats degree.
Sly-Sun is a worst-case statement about DEGREE; the tractability parameter for
exact computation is WIDTH. ``ring_of_cliques`` builds constraint graphs whose
internal degree is far above d_c yet whose treewidth is a small constant, and
``transfer_marginals`` computes their hard-core marginals EXACTLY in
O(m * s^2) time by transfer matrix, at sizes where brute-force enumeration is
astronomically dead and where belief propagation is wrong. Worlds whose
constraints come from structured ontologies live in this class by design.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "weitz_interval",
    "ring_of_cliques",
    "transfer_marginals",
]


# ---- Route 1: certified marginal intervals (Weitz SAW tree) -----------------

def weitz_interval(adj, v: int, lam: float = 1.0, depth: int = 8) -> tuple[float, float]:
    """Certified bounds on the hard-core occupation marginal of vertex v.

    Weitz (2006): the marginal of v on G equals the marginal at the root of
    the self-avoiding-walk tree of G, where a walk that revisits a vertex w
    becomes a leaf PINNED occupied or unoccupied according to whether the
    cycle-closing edge is larger or smaller than the walk's exit edge in the
    ordering around w. This function runs the occupation-ratio recursion
    R_u = lam * prod_w 1/(1 + R_w) on that tree, truncated at ``depth``;
    truncation leaves contribute the full ratio interval [0, lam], which
    contains every possible continuation, so the returned interval rigorously
    contains the true marginal at EVERY depth. The recursion is antitone in
    each child ratio, so a child's upper bound feeds the parent's lower bound
    and vice versa.

    Below the Sly-Sun threshold the interval contracts geometrically in depth
    (correlation decay) and converges to the exact marginal; above it the
    contraction stalls. The certificate is unconditional either way.
    """
    order = {
        u: {w: i for i, w in enumerate(sorted(adj[u]))} for u in range(len(adj))
    }

    def ratio(u: int, parent: int, exit_at: dict[int, int], d: int) -> tuple[float, float]:
        if d == 0:
            return 0.0, lam
        lo_prod = hi_prod = 1.0
        for w in sorted(adj[u]):
            if w == parent:
                continue
            if w in exit_at:  # walk closes a cycle at w: pinned leaf
                if order[w][u] > order[w][exit_at[w]]:
                    return 0.0, 0.0  # leaf occupied: factor 0 kills R_u exactly
                continue  # leaf unoccupied: factor 1
            exit_at[u] = w
            r_lo, r_hi = ratio(w, u, exit_at, d - 1)
            del exit_at[u]
            lo_prod *= 1.0 / (1.0 + r_hi)
            hi_prod *= 1.0 / (1.0 + r_lo)
        return lam * lo_prod, lam * hi_prod

    r_lo, r_hi = ratio(v, -1, {}, depth)
    return r_lo / (1.0 + r_lo), r_hi / (1.0 + r_hi)


# ---- Route 2: high degree, bounded width -------------------------------------

def ring_of_cliques(m: int, s: int):
    """Constraint graph: a ring of m cliques of size s.

    Internal degree is s-1 (s for the two port vertices), so for s >= 7 the
    graph sits far above the Sly-Sun critical degree, yet its treewidth is the
    constant s: exactly the regime where structure rescues computation.
    Vertex j of clique i is i*s + j; vertex 0 is the exit port (edge to vertex
    1 of the next clique around the ring).
    """
    if m < 3 or s < 2:
        raise ValueError("need m >= 3 cliques of size s >= 2")
    n = m * s
    adj = [set() for _ in range(n)]
    for i in range(m):
        base = i * s
        for a in range(s):
            for b in range(a + 1, s):
                adj[base + a].add(base + b)
                adj[base + b].add(base + a)
        nxt = ((i + 1) % m) * s + 1
        adj[base].add(nxt)
        adj[nxt].add(base)
    return adj


def transfer_marginals(m: int, s: int, lam: float = 1.0) -> np.ndarray:
    """EXACT hard-core occupation marginals on ring_of_cliques(m, s).

    A clique holds at most one occupied vertex, so its state space is
    {empty, vertex 0 occupied, ..., vertex s-1 occupied}: s+1 states. The only
    inter-clique constraint forbids (exit port of clique i) and (entry port of
    clique i+1) both occupied. Z = trace(T^m) for the (s+1) x (s+1) transfer
    matrix T, and per-vertex marginals are diagonal elements of T^m over Z.
    O(m * s^3) arithmetic total: polynomial, at any degree.
    """
    k = s + 1  # state 0 = clique empty, state j+1 = vertex j occupied
    T = np.ones((k, k))
    T[:, 1:] *= lam  # weight of the destination state
    T[1, 2] = 0.0  # exit port occupied -> next entry port occupied forbidden
    Tm = np.linalg.matrix_power(T, m)
    Z = np.trace(Tm)
    # P(clique in state sigma) = (T^m)[sigma, sigma] / Z, same for every clique
    # by rotational symmetry; vertex j of any clique is occupied in state j+1.
    p_state = np.diag(Tm) / Z
    return np.tile(p_state[1:], m)
