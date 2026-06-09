"""The kernel on real public trial data (offline: derived count tables).

Counts below were computed from the public datasets by
experiments/public_trials.py on 2026-06-09 and are embedded here so CI does
not need network access. Provenance:

  NSW   Lalonde / Dehejia-Wahba experimental sample (mixtape nsw_mixtape.dta),
        Y = employed in 1978 (re78 > 0):
        control n=260, employed=168; treated n=185, employed=140.
  IST   International Stroke Trial (IST_corrected.csv, Sandercock et al.),
        Y = alive at 14 days (ID14 == 0):
        no-aspirin n=9715, alive=8806; aspirin n=9720, alive=8848.
  STAR  Tennessee STAR kindergarten (AER::STAR via Rdatasets), Y = reading
        score >= pooled median (433): regular n=2006, above=979;
        small n=1739, above=995; regular+aide n=2044, above=983.
"""

import numpy as np
import pytest

from worldkernel import (
    CouplingKernel,
    TwoWorldKernel,
    exact_interval,
    frechet_harmed_bounds,
    frechet_interval,
    frechet_pn_bounds,
)

NSW = dict(n0=260, k0=168, n1=185, k1=140)
IST = dict(n0=9715, k0=8806, n1=9720, k1=8848)
STAR = dict(regular=(2006, 979), small=(1739, 995), aide=(2044, 983))


def rates(c):
    return c["k0"] / c["n0"], c["k1"] / c["n1"]


def test_nsw_diagonal_and_intervals():
    r0, r1 = rates(NSW)
    assert r0 == pytest.approx(0.6462, abs=1e-4)
    assert r1 == pytest.approx(0.7568, abs=1e-4)
    lo, hi = frechet_pn_bounds(r0, r1)
    # positive ACE (+0.111) yet PN only identified to a ~32-point interval
    assert lo == pytest.approx(0.146, abs=2e-3)
    assert hi == pytest.approx(0.467, abs=2e-3)
    # both canonical couplings are admissible kernels inside it
    for p11 in (min(r0, r1), r0 * r1):
        k = TwoWorldKernel(r0, r1, p11)
        assert k.admissible()
        assert lo - 1e-9 <= k.pn() <= hi + 1e-9


def test_ist_off_diagonal_freedom_dominates_sampling_error():
    """The headline: at n=19,435 sampling error is tiny, the rung-3 interval
    is not. More data cannot close it; only an off-diagonal assumption can."""
    r0, r1 = rates(IST)
    lo, hi = frechet_pn_bounds(r0, r1)
    width = hi - lo
    assert width > 0.09  # ~10 PN points of off-diagonal freedom
    # binomial standard error on each arm's marginal, propagated crudely
    se = np.sqrt(r0 * (1 - r0) / IST["n0"]) + np.sqrt(r1 * (1 - r1) / IST["n1"])
    assert width > 10 * se  # identification gap >> sampling noise
    # monotonicity (aspirin never kills a would-have-survived patient)
    # collapses fraction-harmed to zero and PN to the lower endpoint
    mono = TwoWorldKernel(r0, r1, p11=min(r0, r1))
    assert mono.harmed() == pytest.approx(0.0, abs=1e-12)
    assert mono.pn() == pytest.approx(lo, abs=1e-12)


def test_ist_harmed_interval_contains_zero_but_not_only_zero():
    r0, r1 = rates(IST)
    h_lo, h_hi = frechet_harmed_bounds(r0, r1)
    assert h_lo == pytest.approx(0.0, abs=1e-12)  # ACE > 0, so harm not forced
    assert h_hi > 0.08  # but up to ~9% harmed is consistent with the data


def test_helped_minus_harmed_is_the_ace():
    """The ACE identity: helped - harmed = r1 - r0 for EVERY coupling.
    Rungs 1-2 pin the difference; the off-diagonal sets the two terms."""
    r0, r1 = rates(NSW)
    lo_p11, hi_p11 = max(0.0, r0 + r1 - 1.0), min(r0, r1)
    for p11 in np.linspace(lo_p11, hi_p11, 7):
        k = TwoWorldKernel(r0, r1, p11)
        assert k.helped() - k.harmed() == pytest.approx(r1 - r0)


def test_star_three_arm_kernel():
    d = [k / n for (n, k) in STAR.values()]
    # exact identified set sits inside the Frechet box
    fl, fh = frechet_interval(d)
    el, eh = exact_interval(d)
    assert fl - 1e-9 <= el <= eh <= fh + 1e-9
    # the independence-coupling kernel is an admissible point
    M = np.outer(d, d)
    np.fill_diagonal(M, d)
    assert CouplingKernel(M).admissible()
    # small-class effect: +8.4 points over regular on the diagonal
    assert d[1] - d[0] == pytest.approx(0.084, abs=2e-3)
