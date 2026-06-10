"""Breaking the binary ceiling: continuous-outcome counterfactual couplings.

Frontier world models live in continuous state. This module lifts the
kernel's identity to continuous outcomes: a randomized experiment identifies
the two marginal laws F0 of Y_0 and F1 of Y_1 (the diagonal); the joint law
of (Y_0, Y_1), equivalently the COPULA coupling them, is the off-diagonal
and is unidentified. Everything rung-3 asks about the individual treatment
effect Delta = Y_1 - Y_0 is therefore interval-identified, and the sharp
intervals are classical:

  MAKAROV BOUNDS on the effect distribution. For every delta,
    P(Delta <= delta) is sharply bounded by
      lower  L(delta) = sup_y max(F1(y) - F0(y - delta), 0)
      upper  U(delta) = 1 + inf_y min(F1(y) - F0(y - delta), 0).
  These are pointwise sharp over all couplings of (F0, F1).

  FRECHET-HOEFFDING coupling extremes. The comonotone coupling (rank-sorted
  matching: optimal transport for every convex cost) and the antimonotone
  coupling (reverse matching) bound every supermodular functional:
  E[c(Y_0, Y_1)] for supermodular c is MAXIMIZED comonotone and MINIMIZED
  antimonotone (and the reverse for submodular costs such as |Y_1 - Y_0|).

  QUANTILE-LEVEL IDENTIFICATION. Quantiles of Delta inherit interval bounds
  by inverting the Makarov envelope; the quantile-treatment-effect curve
  F1^{-1}(u) - F0^{-1}(u) is the comonotone point inside them, a coupling
  CHOICE, not a fact.

Everything operates on empirical samples (what an audit of a neural world
model actually has) with plain numpy. The point estimates every standard
pipeline reports (independence or comonotone couplings) are single points
inside these intervals; the intervals are what the data say.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "makarov_bounds",
    "prob_benefit_bounds",
    "effect_quantile_bounds",
    "expected_effect",
    "supermodular_extremes",
    "abs_effect_bounds",
    "comonotone_qte",
]


def _ecdf(samples: np.ndarray):
    s = np.sort(np.asarray(samples, dtype=float))

    def F(y):
        return np.searchsorted(s, y, side="right") / len(s)

    return F, s


def makarov_bounds(
    y0: np.ndarray, y1: np.ndarray, delta: float
) -> tuple[float, float]:
    """Sharp bounds on P(Y_1 - Y_0 <= delta) over all couplings.

    Computed on the empirical CDFs; the sup/inf over y is attained on the
    pooled grid of sample points and their delta-shifts."""
    F0, s0 = _ecdf(y0)
    F1, s1 = _ecdf(y1)
    grid = np.unique(np.concatenate([s1, s0 + delta]))
    f1 = F1(grid)
    f0 = F0(grid - delta)
    lower = float(np.max(np.maximum(f1 - f0, 0.0)))
    upper = float(1.0 + np.min(np.minimum(f1 - f0, 0.0)))
    return lower, upper


def prob_benefit_bounds(y0: np.ndarray, y1: np.ndarray) -> tuple[float, float]:
    """Bounds on P(Y_1 > Y_0), the continuous probability of benefit:
    1 - Makarov bounds at delta = 0 (with the complementary envelope)."""
    lo_cdf, hi_cdf = makarov_bounds(y0, y1, 0.0)
    return 1.0 - hi_cdf, 1.0 - lo_cdf


def effect_quantile_bounds(
    y0: np.ndarray, y1: np.ndarray, u: float, grid_size: int = 512
) -> tuple[float, float]:
    """Interval for the u-quantile of Delta = Y_1 - Y_0: invert the Makarov
    envelope on a grid spanning the possible effect range."""
    if not 0.0 < u < 1.0:
        raise ValueError("u in (0, 1)")
    y0 = np.asarray(y0, float)
    y1 = np.asarray(y1, float)
    lo_d = float(y1.min() - y0.max())
    hi_d = float(y1.max() - y0.min())
    deltas = np.linspace(lo_d, hi_d, grid_size)
    lowers = np.array([makarov_bounds(y0, y1, d)[0] for d in deltas])
    uppers = np.array([makarov_bounds(y0, y1, d)[1] for d in deltas])
    # quantile lower bound: smallest delta with UPPER cdf >= u
    q_lo = float(deltas[np.argmax(uppers >= u)])
    # quantile upper bound: smallest delta with LOWER cdf >= u
    idx = np.argmax(lowers >= u)
    q_hi = float(deltas[idx]) if lowers[idx] >= u else hi_d
    return q_lo, q_hi


def expected_effect(y0: np.ndarray, y1: np.ndarray) -> float:
    """E[Delta] = E[Y_1] - E[Y_0]: point-identified (coupling-free), the
    continuous ACE."""
    return float(np.mean(y1) - np.mean(y0))


def supermodular_extremes(y0, y1, cost) -> tuple[float, float]:
    """(min, max) of E[c(Y_0, Y_1)] over all couplings, for supermodular c:
    antimonotone (reverse-sorted) attains the min, comonotone (sorted) the
    max. For submodular costs swap the reading. Requires equal sample sizes
    (resample beforehand otherwise)."""
    a = np.sort(np.asarray(y0, float))
    b = np.sort(np.asarray(y1, float))
    if len(a) != len(b):
        raise ValueError("equal sample sizes required; resample first")
    hi = float(np.mean(cost(a, b)))  # comonotone
    lo = float(np.mean(cost(a, b[::-1])))  # antimonotone
    return (lo, hi) if lo <= hi else (hi, lo)


def abs_effect_bounds(y0, y1) -> tuple[float, float]:
    """Bounds on E|Y_1 - Y_0|: the minimum is the comonotone matching (the
    1-Wasserstein distance between the marginals), the maximum the
    antimonotone one. |y1 - y0| is submodular, so the roles flip relative
    to supermodular_extremes."""
    a = np.sort(np.asarray(y0, float))
    b = np.sort(np.asarray(y1, float))
    if len(a) != len(b):
        raise ValueError("equal sample sizes required; resample first")
    lo = float(np.mean(np.abs(b - a)))  # comonotone = W1(F0, F1)
    hi = float(np.mean(np.abs(b[::-1] - a)))  # antimonotone
    return lo, hi


def comonotone_qte(y0, y1, u: float) -> float:
    """The quantile treatment effect F1^{-1}(u) - F0^{-1}(u): the point the
    comonotone coupling CHOICE pins inside the Makarov quantile interval.
    Reporting it as 'the' individual effect quantile is an assumption."""
    return float(
        np.quantile(np.asarray(y1, float), u) - np.quantile(np.asarray(y0, float), u)
    )
