"""The off-diagonal at scale: mediation and the Natural Direct Effect.

Structure X -> M -> Y, all binary. The counterfactual of interest is the
Natural Direct Effect

  NDE = P(Y_{X=1, M=M_0} = 1) - P(Y_{X=0, M=M_0} = 1):

set X = 1 for the outcome while holding the mediator at the value it would take
under X = 0. Y_{1, M_0} is a nested, cross-world counterfactual: it reads the
mediator from the X=0 world and the outcome from the X=1 world. No single
intervention reaches it.

This module fixes EVERYTHING an experiment can measure (rungs 1 and 2):

  P(M = 1 | do X = x)
  P(M = m, Y = 1 | do X = x)     the in-world (M, Y) joint under each arm
  P(Y = 1 | do X = x, do M = m)  every controlled direct outcome

and computes the identified interval of the NDE by linear programming over the
64-atom response-type polytope. A non-zero interval width IS the off-diagonal
freedom: the cross-world coupling between M_0 and the Y response, the kernel
entry K(T0, T1). With the seeded reference distribution the interval spans
zero: the same experimental record is consistent with the direct effect being
harmful or helpful, and only the off-diagonal coupling decides the sign.

The counting barrier is concrete here: the response-type space grows
64 -> 4096 -> ~4.2M atoms for 1 -> 2 -> 3 mediators.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

__all__ = [
    "ATOMS",
    "m_val",
    "y_val",
    "rung12_constraints",
    "nde_vector",
    "nde_interval",
    "rung12_summary",
    "random_reference",
    "atom_count",
]

# Response-type atoms for X -> M -> Y, all binary.
# M-type iM in 0..3:  M(x) = bit_x of iM.
# Y-type iY in 0..15: Y(x, m) = bit_(2x + m) of iY.
ATOMS: list[tuple[int, int]] = [(iM, iY) for iM in range(4) for iY in range(16)]


def m_val(iM: int, x: int) -> int:
    return (iM >> x) & 1


def y_val(iY: int, x: int, m: int) -> int:
    return (iY >> (2 * x + m)) & 1


def rung12_constraints(p0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Constraint matrix and values for every rung-1/2 functional of p0."""
    rows: list[np.ndarray] = []
    b: list[float] = []

    def add(coef_fn) -> None:
        c = np.array([coef_fn(iM, iY) for (iM, iY) in ATOMS], dtype=float)
        rows.append(c)
        b.append(float(c @ p0))

    add(lambda iM, iY: 1.0)  # normalization
    for x in (0, 1):  # P(M=1 | do X=x)
        add(lambda iM, iY, x=x: 1.0 if m_val(iM, x) == 1 else 0.0)
    for x in (0, 1):  # in-world (M, Y) joint per arm
        for m in (0, 1):
            add(
                lambda iM, iY, x=x, m=m: 1.0
                if (m_val(iM, x) == m and y_val(iY, x, m_val(iM, x)) == 1)
                else 0.0
            )
    for x in (0, 1):  # controlled direct outcomes
        for m in (0, 1):
            add(lambda iM, iY, x=x, m=m: 1.0 if y_val(iY, x, m) == 1 else 0.0)

    return np.array(rows), np.array(b)


def nde_vector() -> np.ndarray:
    """NDE = E[Y(1, M(0)) - Y(0, M(0))] as a linear functional on the atoms."""
    return np.array(
        [y_val(iY, 1, m_val(iM, 0)) - y_val(iY, 0, m_val(iM, 0)) for (iM, iY) in ATOMS],
        dtype=float,
    )


def nde_interval(
    p0: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Identified NDE interval given the full rung-1/2 record of p0.

    Returns (lo, hi, p_lo, p_hi): the endpoints and the two response-type
    distributions achieving them. Both endpoint models reproduce the same
    rung-1/2 record; they differ only in the off-diagonal coupling."""
    A, b = rung12_constraints(p0)
    nde = nde_vector()
    n = len(ATOMS)
    lo = linprog(nde, A_eq=A, b_eq=b, bounds=[(0, None)] * n, method="highs")
    hi = linprog(-nde, A_eq=A, b_eq=b, bounds=[(0, None)] * n, method="highs")
    return float(nde @ lo.x), float(nde @ hi.x), lo.x, hi.x


def rung12_summary(p0: np.ndarray) -> dict[str, float]:
    """The experimental record: every rung-1/2 quantity the LP fixes."""

    def val(coef_fn) -> float:
        c = np.array([coef_fn(iM, iY) for (iM, iY) in ATOMS], dtype=float)
        return float(c @ p0)

    out: dict[str, float] = {}
    for x in (0, 1):
        out[f"P(M=1|do X={x})"] = val(lambda iM, iY, x=x: m_val(iM, x) == 1)
        out[f"P(Y=1|do X={x})"] = val(
            lambda iM, iY, x=x: y_val(iY, x, m_val(iM, x)) == 1
        )
        for m in (0, 1):
            out[f"P(Y=1|do X={x},do M={m})"] = val(
                lambda iM, iY, x=x, m=m: y_val(iY, x, m) == 1
            )
    return out


def random_reference(seed: int = 0) -> np.ndarray:
    """Seeded reference distribution over the 64 atoms (independent M/Y types).

    seed=0 reproduces the paper's verified instance: NDE interval
    approximately [-0.381, +0.187], width 0.568, spanning zero."""
    rng = np.random.default_rng(seed)
    pm = rng.dirichlet(np.ones(4) * 2.0)
    py = rng.dirichlet(np.ones(16) * 1.0)
    p0 = np.array([pm[iM] * py[iY] for (iM, iY) in ATOMS])
    return p0 / p0.sum()


def atom_count(n_mediators: int) -> int:
    """Size of the response-type space for a chain with n mediators.

    64 -> 4096 -> ~4.2M for 1 -> 2 -> 3 mediators: the counting barrier."""
    m_types = 4**n_mediators
    y_inputs = 2 ** (n_mediators + 1)  # Y reads X and all n mediators
    return m_types * 2**y_inputs
