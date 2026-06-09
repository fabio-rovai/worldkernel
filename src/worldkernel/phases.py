"""Phase-quotient kernel access: factor the hardness out, do not fight it.

Above the Sly-Sun threshold the obstruction is phase coexistence: the
admissible-world measure mu is a mixture

    mu = sum_a alpha_a * mu_a   (+ a residual of mass at most rho)

of phases inside which correlation decay holds. Any bounded query f is then
AFFINE in the phase weights:

    E_mu[f] = sum_a alpha_a * E_{mu_a}[f],

so once the within-phase values are computed (each phase is tractable), the
only unknown is the weight vector alpha, a LOW-DIMENSIONAL object. Evidence,
ontology constraints and moment conditions carve a convex feasible set A for
alpha; minimizing and maximizing the affine query over A (an LP) gives a
certified interval, collapsing to a point when the weights are identified.
Full kernel reconstruction stays NP-hard; the QUERY-INDUCED QUOTIENT is
polynomial whenever the phase rank is.

The rigorously certified instance shipped here is the paper's own gadget:
the complete bipartite world K_{m,m} at fugacity lam. Its degree m is
arbitrarily far above the threshold and its world count 2(1+lam)^m - 1 is
astronomically beyond enumeration, yet its phase rank is exactly 2: in any
admissible world at most one side is occupied, so conditioning on the
occupied side makes the other side empty and the occupied side a PRODUCT
measure (correlation decay restored, trivially). The quotient computes any
occupation query at m = 200 in microseconds, exactly.

The honest caveats, kept honest: finding a certified phase decomposition in
general is itself hard (the glass regime has exponential phase rank, and
there the right output is the interval with the unidentified object named:
the cross-phase weights). This module is the affine machinery plus the
certified rank-2 instance, not a universal phase finder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

__all__ = ["PhaseQuotient", "kmm_quotient", "kmm_exact_marginal"]


@dataclass
class PhaseQuotient:
    """Certified phase decomposition seen by a query class.

    phase_values[a] = E_{mu_a}[f] for the query at hand (each computed inside
    a tractable phase); weight_box[a] = identified interval for alpha_a;
    residual_mass = rho, the mass outside the listed phases."""

    phase_values: list[float]
    weight_box: list[tuple[float, float]]
    residual_mass: float = 0.0

    def interval(self) -> tuple[float, float]:
        """Certified interval for E_mu[f]: LP over the feasible weights
        (box intersected with the simplex), padded by the residual mass."""
        r = len(self.phase_values)
        if r != len(self.weight_box):
            raise ValueError("one weight interval per phase")
        c = np.array(self.phase_values, dtype=float)
        bounds = [(max(0.0, lo), min(1.0, hi)) for lo, hi in self.weight_box]
        a_eq = np.ones((1, r))
        b_eq = np.array([1.0 - self.residual_mass])
        lo = linprog(c, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
        hi = linprog(-c, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if not (lo.success and hi.success):
            raise ValueError("infeasible weight constraints")
        # the residual carries mass rho with f in [0, 1]: it can only add
        return float(c @ lo.x), float(-hi.fun) + self.residual_mass


def kmm_quotient(m: int, lam: float = 1.0, side_evidence: tuple[float, float] | None = None):
    """The rank-2 quotient of K_{m,m}: exact, at any m.

    Phases: L (right side empty; left side an independent product measure)
    and R (symmetric). The empty world sits in both; we resolve the overlap
    by assigning it proportionally, which reproduces the exact mixture.
    Returns (PhaseQuotient for the query 'a fixed left vertex is occupied',
    exact weights). ``side_evidence`` optionally constrains alpha_L to an
    interval (partial identification of the phase weight)."""
    if m < 1:
        raise ValueError("m >= 1")
    # Z = 2(1+lam)^m - 1; phase masses (with the empty world split evenly)
    zp = (1.0 + lam) ** m
    z = 2.0 * zp - 1.0
    alpha_l = (zp - 0.5) / z
    # within phase L, the left side is product Bernoulli(lam/(1+lam)) over
    # subsets of the left side (empty world included with its split half):
    # E_{mu_L}[v occupied] for v on the left:
    v_l = (lam / (1.0 + lam)) * zp / (zp - 0.5)
    v_r = 0.0  # in phase R the left vertex is never occupied
    if side_evidence is None:
        box = [(alpha_l, alpha_l), (1.0 - alpha_l, 1.0 - alpha_l)]
    else:
        lo, hi = side_evidence
        box = [(lo, hi), (1.0 - hi, 1.0 - lo)]
    return PhaseQuotient([v_l, v_r], box), alpha_l


def kmm_exact_marginal(m: int, lam: float = 1.0) -> float:
    """Ground truth, closed form: P(a fixed left vertex is occupied) =
    lam (1+lam)^{m-1} / (2(1+lam)^m - 1)."""
    return lam * (1.0 + lam) ** (m - 1) / (2.0 * (1.0 + lam) ** m - 1.0)
