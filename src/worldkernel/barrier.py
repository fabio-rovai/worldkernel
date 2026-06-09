"""The counting barrier: where the off-diagonal stops being computable.

The aggregate counterfactual over many units with mutual-exclusion constraints
is a distribution over INDEPENDENT SETS of a constraint graph (the hard-core
model): admissible joint "necessity" patterns are exactly the independent sets.
Computing that off-diagonal aggregate reduces to hard-core marginals.

Belief propagation computes those marginals correctly below the Sly-Sun
threshold (correlation decay, Weitz 2006) and fails above it (Sly-Sun 2014).
As the constraint-graph degree d crosses d_c, the off-diagonal aggregate
becomes inaccessible to tractable computation. The order parameter is

  (d - 1) * eta,  eta = 1 - u,  where u solves  u + lam * u^d = 1,

which crosses 1 at d_c (d_c = 5.141 at unit fugacity). At finite n the BP
error rises monotonically with the order parameter rather than jumping: the
sharp transition is the asymptotic statement.

This module provides the order parameter, the critical degree, and the
finite-n instruments (exact enumeration and BP marginals on random regular
graphs) used to produce the keystone figure.
"""

from __future__ import annotations

import random

import numpy as np

__all__ = [
    "fixed_point_u",
    "order_parameter",
    "d_critical",
    "random_regular",
    "exact_marginals",
    "bp_marginals",
]


# ---- the Sly-Sun order parameter --------------------------------------------

def fixed_point_u(d: float, lam: float = 1.0) -> float:
    """Cavity vacancy fixed point: the unique u in (0,1) with u + lam*u^d = 1.

    The naive iteration oscillates above threshold, so bisect."""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        u = 0.5 * (lo + hi)
        if u + lam * u**d - 1.0 < 0.0:
            lo = u
        else:
            hi = u
    return 0.5 * (lo + hi)


def order_parameter(d: float, lam: float = 1.0) -> float:
    """(d-1) * eta: the BP recursion's contraction factor at the fixed point.

    Below 1: correlation decay, the off-diagonal aggregate is tractable.
    Above 1: the Sly-Sun hard regime."""
    return (d - 1.0) * (1.0 - fixed_point_u(d, lam))


def d_critical(lam: float = 1.0) -> float:
    """The degree at which the order parameter crosses 1 (5.141 at lam=1)."""
    lo, hi = 2.0, 10.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if order_parameter(mid, lam) < 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---- finite-n instruments ----------------------------------------------------

def random_regular(n: int, d: int, rng: random.Random, swaps_mult: int = 25):
    """Random d-regular graph: circulant base plus double-edge swaps."""
    adj: list[set[int]] = [set() for _ in range(n)]
    for i in range(n):
        for k in range(1, d // 2 + 1):
            adj[i].add((i + k) % n)
            adj[i].add((i - k) % n)
        if d % 2 == 1:  # n must be even: antipodal edge
            adj[i].add((i + n // 2) % n)
    edges = [(a, b) for a in range(n) for b in adj[a] if a < b]
    for _ in range(swaps_mult * len(edges)):
        (a, b), (c, e) = rng.choice(edges), rng.choice(edges)
        if len({a, b, c, e}) < 4:
            continue
        if e in adj[a] or c in adj[b]:  # would create a multi-edge
            continue
        adj[a].discard(b)
        adj[b].discard(a)
        adj[c].discard(e)
        adj[e].discard(c)
        adj[a].add(e)
        adj[e].add(a)
        adj[c].add(b)
        adj[b].add(c)
        edges = [(x, y) for x in range(n) for y in adj[x] if x < y]
    return adj


def exact_marginals(adj, n: int, lam: float = 1.0) -> np.ndarray:
    """Exact hard-core occupation marginals by enumeration over 2^n subsets."""
    mask = [0] * n
    for i in range(n):
        for j in adj[i]:
            mask[i] |= 1 << j
    incl = np.zeros(n)
    Z = 0.0
    for S in range(1 << n):
        ok = True
        s = S
        while s:
            i = (s & -s).bit_length() - 1
            if mask[i] & S:
                ok = False
                break
            s &= s - 1
        if ok:
            w = lam ** bin(S).count("1")
            Z += w
            b = S
            while b:
                i = (b & -b).bit_length() - 1
                incl[i] += w
                b &= b - 1
    return incl / Z


def bp_marginals(adj, n: int, lam: float = 1.0, iters: int = 400, damp: float = 0.5) -> np.ndarray:
    """Belief-propagation hard-core marginals (cavity occupation messages)."""
    h = {(i, j): 0.3 for i in range(n) for j in adj[i]}
    for _ in range(iters):
        new = {}
        for i in range(n):
            for j in adj[i]:
                prod = 1.0
                for k in adj[i]:
                    if k != j:
                        prod *= 1.0 - h[(k, i)]
                val = lam * prod / (1.0 + lam * prod)
                new[(i, j)] = damp * h[(i, j)] + (1 - damp) * val
        h = new
    p = np.zeros(n)
    for i in range(n):
        prod = 1.0
        for k in adj[i]:
            prod *= 1.0 - h[(k, i)]
        p[i] = lam * prod / (1.0 + lam * prod)
    return p
