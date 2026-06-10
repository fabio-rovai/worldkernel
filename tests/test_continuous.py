"""Continuous outcomes: Makarov bounds, coupling extremes, quantiles."""

import numpy as np
import pytest

from worldkernel.continuous import (
    abs_effect_bounds,
    comonotone_qte,
    effect_quantile_bounds,
    expected_effect,
    makarov_bounds,
    prob_benefit_bounds,
    supermodular_extremes,
)

RNG = np.random.default_rng(11)
N = 4000


@pytest.fixture(scope="module")
def shifted_normals():
    """Y_0 ~ N(0,1), Y_1 ~ N(0.5, 1): same shape, shifted by tau = 0.5."""
    return RNG.normal(0, 1, N), RNG.normal(0.5, 1, N)


def test_expected_effect_is_coupling_free(shifted_normals):
    y0, y1 = shifted_normals
    assert expected_effect(y0, y1) == pytest.approx(0.5, abs=0.06)


def test_makarov_bounds_contain_known_couplings(shifted_normals):
    """Simulate (Y_0, Y_1) under three explicit couplings with the SAME
    marginals; every conditional effect cdf must sit inside Makarov."""
    y0, y1 = shifted_normals
    z = RNG.normal(0, 1, N)
    couplings = {
        "comonotone": (z, z + 0.5),
        "independent": (RNG.normal(0, 1, N), RNG.normal(0.5, 1, N)),
        "antimonotone": (z, -z + 0.5),
    }
    for delta in (-0.5, 0.0, 0.5, 1.0):
        lo, hi = makarov_bounds(y0, y1, delta)
        assert 0.0 <= lo <= hi <= 1.0
        for name, (a, b) in couplings.items():
            truth = float(np.mean(b - a <= delta))
            assert lo - 0.05 <= truth <= hi + 0.05, (name, delta)


def test_makarov_width_is_real_off_diagonal_freedom(shifted_normals):
    """At delta = tau the effect cdf ranges from ~0 (comonotone: constant
    effect, P(Delta <= tau) jumps there) to large: a wide identified set."""
    y0, y1 = shifted_normals
    lo, hi = makarov_bounds(y0, y1, 0.5)
    assert hi - lo > 0.5


def test_prob_benefit(shifted_normals):
    y0, y1 = shifted_normals
    lo, hi = prob_benefit_bounds(y0, y1)
    # comonotone: benefit is certain (constant +0.5 shift) -> hi near 1
    assert hi > 0.95
    # antimonotone coupling gives P(benefit) ~ P(z < 0.25) ~ 0.6 > lo
    assert lo < 0.45  # the data alone cannot certify majority benefit
    ind = float(np.mean(RNG.normal(0.5, 1, N)[:, None] > RNG.normal(0, 1, N)[None, :100]))
    assert lo - 0.05 <= ind <= hi + 0.05


def test_quantile_bounds_contain_comonotone_point(shifted_normals):
    y0, y1 = shifted_normals
    for u in (0.25, 0.5, 0.75):
        q_lo, q_hi = effect_quantile_bounds(y0, y1, u)
        assert q_lo <= comonotone_qte(y0, y1, u) + 0.05
        assert comonotone_qte(y0, y1, u) - 0.05 <= q_hi
        assert q_hi - q_lo > 0.3  # genuinely unidentified


def test_shift_alternative_comonotone_qte(shifted_normals):
    """Equal-shape shift: the comonotone QTE is the constant shift at every
    quantile, while the identified interval stays wide: choice vs fact."""
    y0, y1 = shifted_normals
    for u in (0.3, 0.5, 0.7):
        assert comonotone_qte(y0, y1, u) == pytest.approx(0.5, abs=0.1)


def test_abs_effect_bounds(shifted_normals):
    y0, y1 = shifted_normals
    lo, hi = abs_effect_bounds(y0, y1)
    # comonotone floor = W1 distance = |shift| for equal shapes
    assert lo == pytest.approx(0.5, abs=0.06)
    # antimonotone ceiling ~ E|2Z - 0.5| for Z std normal: ~ 1.63
    assert hi == pytest.approx(1.63, abs=0.1)
    assert lo < hi


def test_supermodular_extremes_product_cost(shifted_normals):
    y0, y1 = shifted_normals
    lo, hi = supermodular_extremes(y0, y1, lambda a, b: a * b)
    # E[Y0 Y1] in [-1, 1] for these marginals (correlation extremes)
    assert lo == pytest.approx(-1.0, abs=0.08)
    assert hi == pytest.approx(1.0, abs=0.08)


def test_unequal_sizes_rejected():
    with pytest.raises(ValueError):
        abs_effect_bounds(np.zeros(5), np.zeros(6))