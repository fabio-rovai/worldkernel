"""The Sly-Sun backdoor collapse: the hardness parameter is not degree.

Theorem (backdoor collapse). Let G be an admissibility graph and B a set of
vertices with max degree of G - B at most 5 (a UNIQUENESS BACKDOOR). Every
independent set decomposes by its trace sigma on B, and conditioning on
sigma deletes B and the neighbourhood of the occupied part, leaving a
residual graph below the Sly-Sun threshold. Hence

    Z_G = sum over independent sigma in G[B] of  lam^|sigma| * Z_residual(sigma)

and Z plus every restricted occupation marginal is computable in
2^|B| * poly(n): FIXED-PARAMETER TRACTABLE in the backdoor size b(G),
even when the global degree is far above 6. Conditioning only deletes
vertices, so the same B stays a backdoor under every adaptive restriction.

The honest counterpart: minimizing |B| is itself a hard vertex-deletion
problem, and on worst-case Sly-Sun instances b(G) is necessarily Omega(n)
(otherwise NP = RP). The escape is the SEARCH/VERIFY separation: the search
for B may be heuristic, but the certificate (max degree of G - B at most 5)
is checked in linear time, and once it is in hand the computation is honest.

Scars, in the paper's language, are exactly this: a scar set is a deletion-
to-uniqueness certificate, not just an SDP-tightening constraint.

Residual computation here uses the package's exact variable elimination
(certified by the width of the residual); the theorem's general-purpose
engine is Weitz's FPTAS at degree 5, which `weitz_interval` provides in
certified-interval form when a residual is too wide for elimination.
"""

from __future__ import annotations

import itertools

from .tractable import hardcore_z, min_fill_order

__all__ = [
    "find_backdoor",
    "verify_backdoor",
    "backdoor_z",
    "backdoor_marginal",
    "hub_world",
]


def verify_backdoor(adj, B: set[int], max_degree: int = 5) -> bool:
    """The linear-time certificate: max degree of G - B at most max_degree."""
    B = set(B)
    for v in range(len(adj)):
        if v in B:
            continue
        if len([w for w in adj[v] if w not in B]) > max_degree:
            return False
    return True


def find_backdoor(adj, max_degree: int = 5) -> set[int]:
    """Greedy peel: repeatedly move the highest-residual-degree vertex into B.

    Heuristic by design (minimum backdoor is a hard vertex-deletion problem);
    the returned set always satisfies the certificate, which is what the
    runtime bound needs. Better heuristics (Bethe instability, Fiedler
    concentration) slot in here without touching the theorem."""
    n = len(adj)
    alive_deg = {v: len(adj[v]) for v in range(n)}
    B: set[int] = set()
    while True:
        v = max(alive_deg, key=alive_deg.get)
        if alive_deg[v] <= max_degree:
            break
        B.add(v)
        del alive_deg[v]
        for w in adj[v]:
            if w in alive_deg:
                alive_deg[w] -= 1
    assert verify_backdoor(adj, B, max_degree)
    return B


def _independent_subsets(adj, B: list[int]):
    """All independent subsets of G[B] (the 2^|B| backdoor assignments)."""
    for bits in itertools.product((0, 1), repeat=len(B)):
        sigma = [v for v, b in zip(B, bits) if b]
        ok = all(w not in adj[u] for i, u in enumerate(sigma) for w in sigma[i + 1:])
        if ok:
            yield set(sigma)


def _residual(adj, drop: set[int]):
    """Induced subgraph on V minus drop, with the old-to-new vertex map."""
    keep = [v for v in range(len(adj)) if v not in drop]
    idx = {v: i for i, v in enumerate(keep)}
    sub = [set(idx[w] for w in adj[v] if w in idx) for v in keep]
    return sub, idx


def backdoor_z(adj, B: set[int] | None = None, lam: float = 1.0) -> float:
    """Exact partition function by backdoor conditioning.

    Z = sum over independent sigma of lam^|sigma| * Z(G minus B minus N(sigma)).
    Residuals are evaluated by exact variable elimination; cost is
    2^|B| times the residual elimination cost."""
    if B is None:
        B = find_backdoor(adj)
    Bl = sorted(B)
    z = 0.0
    for sigma in _independent_subsets(adj, Bl):
        drop = set(Bl) | {w for u in sigma for w in adj[u]}
        sub, _ = _residual(adj, drop)
        order, _w = min_fill_order(sub)
        z += (lam ** len(sigma)) * hardcore_z(sub, lam, order=order)
    return z


def backdoor_marginal(
    adj, v: int, B: set[int] | None = None, lam: float = 1.0
) -> float:
    """Exact occupation marginal of v via the same decomposition:
    P(v occupied) = Z(v pinned occupied) / Z, both by backdoor conditioning."""
    if B is None:
        B = find_backdoor(adj)
    z = backdoor_z(adj, B, lam)
    # pin v occupied: drop v and its neighbourhood, weight lam
    Bl = sorted(B)
    z_occ = 0.0
    for sigma in _independent_subsets(adj, Bl):
        if v in {w for u in sigma for w in adj[u]}:
            continue  # sigma forbids v
        if v in sigma:
            drop = set(Bl) | {w for u in sigma for w in adj[u]}
            sub, _ = _residual(adj, drop)
            z_occ += (lam ** len(sigma)) * hardcore_z(sub, lam)
            continue
        drop = set(Bl) | {w for u in sigma for w in adj[u]} | {v} | set(adj[v])
        if v in Bl and v not in sigma:
            continue  # v pinned empty by this assignment
        sub, _ = _residual(adj, drop)
        z_occ += lam * (lam ** len(sigma)) * hardcore_z(sub, lam)
    return z_occ / z


def hub_world(n_base: int, d_base: int, n_hubs: int, hub_degree: int, seed: int = 11):
    """A world whose global degree breaks the Sly-Sun threshold while its
    backdoor is tiny: a d_base-regular bulk (below threshold) plus n_hubs
    high-degree hub constraints wired across it. b(G) <= n_hubs by
    construction; the degree-based diagnosis says 'hard', the backdoor
    says 'two bits of conditioning away from easy'."""
    import random as _random

    from .barrier import random_regular

    rng = _random.Random(seed)
    base = random_regular(n_base, d_base, rng)
    n = n_base + n_hubs
    adj = [set(x) for x in base] + [set() for _ in range(n_hubs)]
    for h in range(n_base, n):
        targets = rng.sample(range(n_base), hub_degree)
        for t in targets:
            adj[h].add(t)
            adj[t].add(h)
    return adj
