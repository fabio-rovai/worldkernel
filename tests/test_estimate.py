"""Estimation: corner-evaluated confidence boxes with simultaneous coverage."""

import numpy as np
import pytest

from worldkernel.estimate import (
    ace_from_counts,
    harmed_bounds_from_counts,
    pn_bounds_from_counts,
    wilson,
)
from worldkernel.witness import frechet_pn_bounds


def test_wilson_basics():
    lo, hi = wilson(50, 100, 0.05)
    assert lo < 0.5 < hi
    assert wilson(0, 0, 0.05) == (0.0, 1.0)
    # tighter with more data
    lo2, hi2 = wilson(500, 1000, 0.05)
    assert (hi2 - lo2) < (hi - lo)


def test_estimated_interval_contains_identified_set():
    est = pn_bounds_from_counts(260, 168, 185, 140)  # the NSW counts
    plo, phi = frechet_pn_bounds(168 / 260, 140 / 185)
    assert est.lo <= plo and phi <= est.hi
    assert est.sampling_inflation > 0
    assert est.identified_lo == pytest.approx(plo)


def test_sampling_inflation_shrinks_with_n_identification_does_not():
    small = pn_bounds_from_counts(100, 65, 100, 75)
    big = pn_bounds_from_counts(10000, 6500, 10000, 7500)
    assert big.sampling_inflation < small.sampling_inflation / 3
    # the identified core is the same law, so it does not shrink
    assert big.identified_hi - big.identified_lo == pytest.approx(
        small.identified_hi - small.identified_lo, abs=1e-9
    )


def test_simulated_coverage_holds():
    """Nominal 90% simultaneous coverage: the reported interval contains the
    TRUE identified set (hence the true PN) in at least ~90% of resamples."""
    rng = np.random.default_rng(11)
    r0, r1, n = 0.55, 0.70, 400
    true_lo, true_hi = frechet_pn_bounds(r0, r1)
    hits = 0
    sims = 400
    for _ in range(sims):
        k0 = rng.binomial(n, r0)
        k1 = rng.binomial(n, r1)
        est = pn_bounds_from_counts(n, k0, n, k1, coverage=0.90)
        if est.lo <= true_lo and true_hi <= est.hi:
            hits += 1
    assert hits / sims >= 0.88  # >= nominal minus simulation noise


def test_ace_is_sampling_only():
    est = ace_from_counts(1000, 600, 1000, 700)
    assert est.identified_lo == pytest.approx(est.identified_hi)
    assert est.lo < 0.1 < est.hi


def test_harmed_counts_interval():
    est = harmed_bounds_from_counts(9715, 8806, 9720, 8848)  # IST
    assert est.lo == pytest.approx(0.0, abs=1e-9)
    assert 0.08 < est.hi < 0.12
