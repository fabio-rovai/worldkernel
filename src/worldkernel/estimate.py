"""Estimation: the kernel meets finite data.

A trial reports counts, not marginals. This module turns per-arm counts into
identified intervals with a SIMULTANEOUS coverage guarantee: Wilson score
intervals per arm, Bonferroni-combined, then the identified-set bound
functions evaluated over the confidence box. Because every two-world bound
(PN, PNS, harmed, helped) is monotone in each marginal on the box, the
extremes sit at corners, so corner evaluation is exact for the box.

The output interval has the conformal reading: with probability at least
``coverage`` over the sampling, the reported interval contains the entire
true identified set, hence the true rung-3 value. Sampling uncertainty and
identification uncertainty are both inside, and the part that never shrinks
with n is the identification core.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .witness import frechet_harmed_bounds, frechet_pn_bounds

__all__ = ["wilson", "EstimatedInterval", "pn_bounds_from_counts",
           "harmed_bounds_from_counts", "ace_from_counts"]


def wilson(k: int, n: int, alpha: float) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    from scipy.stats import norm

    z = norm.ppf(1.0 - alpha / 2.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class EstimatedInterval:
    lo: float
    hi: float
    identified_lo: float  # the plug-in identified set (no sampling widening)
    identified_hi: float
    coverage: float

    @property
    def sampling_inflation(self) -> float:
        """How much of the reported width is sampling, not identification."""
        return (self.hi - self.lo) - (self.identified_hi - self.identified_lo)


def _corner_bounds(bound_fn, n0, k0, n1, k1, coverage) -> EstimatedInterval:
    alpha = (1.0 - coverage) / 2.0  # Bonferroni across the two arms
    r0_lo, r0_hi = wilson(k0, n0, alpha)
    r1_lo, r1_hi = wilson(k1, n1, alpha)
    los, his = [], []
    for r0 in (r0_lo, r0_hi):
        for r1 in (r1_lo, r1_hi):
            lo, hi = bound_fn(r0, r1)
            los.append(lo)
            his.append(hi)
    plo, phi = bound_fn(k0 / n0, k1 / n1)
    return EstimatedInterval(
        lo=float(min(los)), hi=float(max(his)),
        identified_lo=float(plo), identified_hi=float(phi),
        coverage=coverage,
    )


def pn_bounds_from_counts(
    n0: int, k0: int, n1: int, k1: int, coverage: float = 0.95
) -> EstimatedInterval:
    """Probability-of-necessity interval from trial counts, with simultaneous
    coverage at least ``coverage`` over the sampling."""
    return _corner_bounds(frechet_pn_bounds, n0, k0, n1, k1, coverage)


def harmed_bounds_from_counts(
    n0: int, k0: int, n1: int, k1: int, coverage: float = 0.95
) -> EstimatedInterval:
    """Fraction-harmed interval from trial counts, same guarantee."""
    return _corner_bounds(frechet_harmed_bounds, n0, k0, n1, k1, coverage)


def ace_from_counts(
    n0: int, k0: int, n1: int, k1: int, coverage: float = 0.95
) -> EstimatedInterval:
    """The ACE (point-identified): the interval here is sampling-only."""

    def ace(r0: float, r1: float) -> tuple[float, float]:
        return r1 - r0, r1 - r0

    return _corner_bounds(ace, n0, k0, n1, k1, coverage)
