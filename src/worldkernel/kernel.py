"""The coupling kernel of admissible possible worlds.

A world model, in this framework, is not a predictor. It is a positive
semidefinite (PSD) coupling kernel K(T, T') over admissible possible worlds:

  - the DIAGONAL K(T, T) holds what prediction recovers: the marginal law of
    each potential world (rung-1 observational and rung-2 interventional data),
  - the OFF-DIAGONAL K(T, T') holds the cross-world coupling between potential
    outcomes: the quantity every genuine counterfactual reads, and the quantity
    no amount of rung-1/2 data identifies.

For k binary potential outcomes v = (Y_0, ..., Y_{k-1}) the kernel is the
second-moment matrix M = E[v v^T]:

  M_ii = P(Y_i = 1)          (identified by a k-arm randomized trial)
  M_ij = P(Y_i = 1, Y_j = 1) (cross-world coupling, unidentified from marginals)

This module provides the kernel object and three nested bounds on any
counterfactual functional of the second moments, given only the diagonal:

  frechet_interval  -- per-entry Frechet-Hoeffding box, polynomial time,
  psd_interval      -- adds the PSD constraint (an SDP over a k x k matrix),
                       polynomial time and strictly tighter than Frechet,
  exact_interval    -- the tight identified set via an LP over all 2^k
                       response types; exact but exponential in k.

The ordering Frechet >= PSD >= exact (as sets) is the practical content of the
kernel's PSD structure: real partial-identifying information that survives at
scales where the exact response-type polytope is computationally dead.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import linprog

__all__ = [
    "CouplingKernel",
    "frechet_interval",
    "psd_interval",
    "exact_interval",
]


class CouplingKernel:
    """PSD second-moment matrix over k binary potential worlds."""

    def __init__(self, matrix) -> None:
        M = np.asarray(matrix, dtype=float)
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError("kernel must be a square matrix")
        if not np.allclose(M, M.T, atol=1e-9):
            raise ValueError("kernel must be symmetric")
        self.M = M

    # -- structure ----------------------------------------------------------
    @property
    def k(self) -> int:
        return self.M.shape[0]

    @property
    def diagonal(self) -> np.ndarray:
        """Marginals P(Y_i = 1): the part rung-1/2 data identify."""
        return np.diag(self.M).copy()

    def is_psd(self, tol: float = 1e-9) -> bool:
        return bool(np.linalg.eigvalsh(self.M).min() >= -tol)

    def admissible(self, tol: float = 1e-9) -> bool:
        """PSD and every off-diagonal inside its Frechet-Hoeffding box."""
        if not self.is_psd(tol):
            return False
        d = self.diagonal
        for i in range(self.k):
            if not (-tol <= d[i] <= 1.0 + tol):
                return False
            for j in range(i + 1, self.k):
                lo = max(0.0, d[i] + d[j] - 1.0)
                hi = min(d[i], d[j])
                if not (lo - tol <= self.M[i, j] <= hi + tol):
                    return False
        return True

    # -- queries -------------------------------------------------------------
    def pairwise_coherence(self) -> float:
        """Q = sum_{i<j} P(Y_i=1, Y_j=1).

        The expected number of arm pairs jointly successful for the same unit:
        a genuine cross-world aggregate, linear in the off-diagonal.
        """
        iu = np.triu_indices(self.k, k=1)
        return float(self.M[iu].sum())


# ---- bounds on Q given only the diagonal ------------------------------------

def frechet_interval(diag) -> tuple[float, float]:
    """Box bound on Q: each coupling optimized independently in its
    Frechet-Hoeffding interval. Polynomial time; the bound marginals alone give."""
    d = np.asarray(diag, dtype=float)
    k = len(d)
    lo = hi = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            lo += max(0.0, d[i] + d[j] - 1.0)
            hi += min(d[i], d[j])
    return lo, hi


def psd_interval(diag) -> tuple[float, float]:
    """SDP bound on Q: Frechet boxes plus the kernel constraint M >= 0.

    Polynomial time (k x k SDP) and a valid outer bound on the exact identified
    set; strictly tighter than Frechet on a nontrivial fraction of instances.
    Requires cvxpy (install with the ``sdp`` extra)."""
    try:
        import cvxpy as cp
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "psd_interval requires cvxpy: pip install 'worldkernel[sdp]'"
        ) from exc

    d = np.asarray(diag, dtype=float)
    k = len(d)

    def solve(sense: str) -> float:
        M = cp.Variable((k, k), symmetric=True)
        cons = [M >> 0]
        for i in range(k):
            cons.append(M[i, i] == d[i])
            for j in range(i + 1, k):
                cons.append(M[i, j] >= max(0.0, d[i] + d[j] - 1.0))
                cons.append(M[i, j] <= min(d[i], d[j]))
        Q = sum(M[i, j] for i in range(k) for j in range(i + 1, k))
        prob = cp.Problem(cp.Maximize(Q) if sense == "max" else cp.Minimize(Q), cons)
        prob.solve(solver=cp.CLARABEL)
        return float(prob.value)

    return solve("min"), solve("max")


def exact_interval(diag) -> tuple[float, float]:
    """Tight identified set for Q: LP over all 2^k response types matching the
    marginals. Exact, but 2^k variables; use only for modest k."""
    d = np.asarray(diag, dtype=float)
    k = len(d)
    types = np.array(list(itertools.product([0, 1], repeat=k)))
    n = len(types)
    cobj = np.array(
        [
            sum(1 for i in range(k) for j in range(i + 1, k) if t[i] and t[j])
            for t in types
        ],
        dtype=float,
    )
    A_eq = [np.ones(n)]
    b_eq = [1.0]
    for i in range(k):
        A_eq.append((types[:, i] == 1).astype(float))
        b_eq.append(d[i])
    A_eq, b_eq = np.array(A_eq), np.array(b_eq)
    bounds = [(0, 1)] * n
    lo = linprog(cobj, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    hi = linprog(-cobj, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    return float(lo.fun), float(-hi.fun)
