"""Proof-carrying kernels: hard to compute is not hard to verify.

The third escape from the counting barrier changes the computational model.
The hard-core normalizer is a hypercube sum of a low-degree polynomial:

    Z = sum over x in {0,1}^n of  prod over edges (u,v) of (1 - x_u x_v),

with per-variable degree equal to the vertex degree. That is exactly the
shape the SUM-CHECK protocol (Lund-Fortnow-Karloff-Nisan) verifies: a prover
(who may spend exponential effort, be an offline solver, a market, or a
physical oracle) claims Z and runs n rounds of univariate polynomials; the
verifier checks each round with two evaluations and one random field element,
and finishes with a single direct evaluation of the product at a random
point: polynomial time, soundness error at most n * max_degree / p over F_p.

Architecture consequence: a world model can refuse to COMPUTE off-diagonal
normalizers above the barrier and still safely USE claimed values, because
false claims are rejected with overwhelming probability. The kernel entry
becomes value + certificate.

This module implements the protocol concretely for the hard-core polynomial
at unit fugacity over F_p with p = 2^61 - 1: an honest (exponential-time)
prover for instances small enough to test, and a polynomial-time verifier
that is the part an agent would actually run.
"""

from __future__ import annotations

import random as _random

import numpy as np

__all__ = ["P", "SumcheckProver", "verify_z", "hardcore_eval"]

P = (1 << 61) - 1  # a Mersenne prime


def hardcore_eval(adj, point: list[int]) -> int:
    """Evaluate g(x) = prod over edges (1 - x_u x_v) at a field point.
    Polynomial time: one multiplication per edge. This is the verifier's
    only direct contact with the instance."""
    val = 1
    n = len(adj)
    seen = set()
    for u in range(n):
        for v in adj[u]:
            e = (min(u, v), max(u, v))
            if e in seen:
                continue
            seen.add(e)
            val = (val * (1 - point[u] * point[v])) % P
    return val % P


class SumcheckProver:
    """Honest prover: exponential in n (it really does the sum), which is
    the point: the EFFORT lives here, the TRUST lives in the verifier."""

    def __init__(self, adj) -> None:
        self.adj = adj
        self.n = len(adj)

    def claimed_z(self) -> int:
        total = 0
        for bits in range(1 << self.n):
            x = [(bits >> i) & 1 for i in range(self.n)]
            total += hardcore_eval(self.adj, x)
        return total % P

    def round_poly(self, prefix: list[int], var: int) -> list[int]:
        """The round polynomial g_i(X) = sum over the remaining Boolean
        variables of g(prefix, X, suffix), returned as evaluations at
        X = 0, 1, ..., deg (enough to interpolate; deg = degree of var)."""
        deg = len(self.adj[var])
        rest = self.n - var - 1
        evals = []
        for xval in range(deg + 1):
            s = 0
            for bits in range(1 << rest):
                suffix = [(bits >> i) & 1 for i in range(rest)]
                point = prefix + [xval] + suffix
                s = (s + hardcore_eval(self.adj, point)) % P
            evals.append(s)
        return evals


def _interp_eval(evals: list[int], x: int) -> int:
    """Evaluate the unique poly through (i, evals[i]) at x, over F_p
    (Lagrange interpolation on the points 0..deg)."""
    k = len(evals)
    total = 0
    for i in range(k):
        num, den = 1, 1
        for j in range(k):
            if j == i:
                continue
            num = (num * (x - j)) % P
            den = (den * (i - j)) % P
        total = (total + evals[i] * num * pow(den, P - 2, P)) % P
    return total % P


def _poly_sum01(evals: list[int]) -> int:
    return (evals[0] + (evals[1] if len(evals) > 1 else evals[0])) % P


def verify_z(adj, claimed_z: int, prover: SumcheckProver, seed: int = 11) -> bool:
    """The polynomial-time verifier. Drives the rounds, draws random field
    challenges, and accepts iff every round is consistent and the final
    direct evaluation of g matches. Soundness: a false claimed_z survives
    with probability at most n * max_degree / p (about 1e-17 here)."""
    rng = _random.Random(seed)
    n = len(adj)
    claim = claimed_z % P
    prefix: list[int] = []
    for var in range(n):
        evals = prover.round_poly(prefix, var)
        if len(evals) != len(adj[var]) + 1 or any(not 0 <= e < P for e in evals):
            return False
        if _poly_sum01(evals) != claim:
            return False  # the round polynomial does not carry the claim
        r = rng.randrange(P)
        claim = _interp_eval(evals, r)
        prefix.append(r)
    return claim == hardcore_eval(adj, prefix)
