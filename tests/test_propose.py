"""The proposer interface: assumptions validated and priced, never invented."""

import pytest

from worldkernel.propose import evaluate
from worldkernel.witness import frechet_pn_bounds


def test_monotone_pins_pn_to_lower_endpoint():
    nar = evaluate("monotone", 0.5, 0.7)
    assert nar.admissible
    lo, _ = frechet_pn_bounds(0.5, 0.7)
    assert nar.pn_after == pytest.approx((lo, lo))
    assert nar.harmed_after == pytest.approx((0.0, 0.0))
    assert nar.pn_width_bought > 0.4


def test_independent_pins_a_point_inside():
    nar = evaluate("independent", 0.5, 0.7)
    assert nar.admissible
    assert nar.pn_after[0] == pytest.approx(nar.pn_after[1]) == pytest.approx(0.5)
    assert nar.pn_before[0] - 1e-9 <= nar.pn_after[0] <= nar.pn_before[1] + 1e-9


def test_bounded_correlation_partially_narrows():
    nar = evaluate("correlation_at_most", 0.5, 0.7, value=0.3)
    assert nar.admissible
    w_before = nar.pn_before[1] - nar.pn_before[0]
    w_after = nar.pn_after[1] - nar.pn_after[0]
    assert 0.0 < w_after < w_before  # narrows without pinning


def test_inadmissible_coupling_is_refused():
    nar = evaluate("coupling", 0.5, 0.7, value=0.65)  # above min(r0, r1)
    assert not nar.admissible
    assert "INADMISSIBLE" in nar.note
    # the narrowing falls back to the unassumed interval
    assert nar.pn_after == nar.pn_before


def test_assumptions_are_not_findings():
    nar = evaluate("monotone", 0.5, 0.7)
    assert not nar.refutable_from_data
    assert "modelling decision" in nar.note


def test_unknown_assumption_raises():
    with pytest.raises(ValueError, match="vocabulary"):
        evaluate("vibes", 0.5, 0.7)
