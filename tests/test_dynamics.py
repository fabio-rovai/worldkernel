"""Sequential counterfactuals: exact trajectory bounds, MC-validated."""

import numpy as np
import pytest

from worldkernel.dynamics import (
    CorridorWorld,
    cf_slip_prob,
    conditional_truth,
    counterfactual_success_interval,
    independence_point,
    success_prob,
)

P = 0.3
SLIPS = [1, 0, 0, 1, 0, 0, 0, 1, 0]  # an observed 9-step episode, 3 slips
NEEDED = 6


def test_success_prob_edge_cases():
    assert success_prob([0.5] * 4, 0) == 1.0
    assert success_prob([0.5] * 4, 5) == 0.0
    # all-deterministic steps
    assert success_prob([0.0] * 5, 5) == pytest.approx(1.0)
    assert success_prob([1.0] * 5, 1) == pytest.approx(0.0)


def test_cf_slip_prob_canonical_couplings():
    # comonotone (p11 = p): counterfactual mirrors the factual exactly
    assert cf_slip_prob(1, P, P) == pytest.approx(1.0)
    assert cf_slip_prob(0, P, P) == pytest.approx(0.0)
    # independent (p11 = p^2): counterfactual ignores the factual
    assert cf_slip_prob(1, P, P * P) == pytest.approx(P)
    assert cf_slip_prob(0, P, P * P) == pytest.approx(P)


def test_interval_contains_every_admissible_coupling():
    lo, hi = counterfactual_success_interval(SLIPS, P, NEEDED)
    assert 0.0 <= lo < hi <= 1.0  # genuinely unidentified
    box_lo, box_hi = max(0.0, 2 * P - 1), P
    for p11 in np.linspace(box_lo, box_hi, 9):
        truth = conditional_truth(SLIPS, P, p11, NEEDED)
        assert lo - 1e-9 <= truth <= hi + 1e-9


def test_comonotone_truth_is_deterministic_and_inside():
    # under comonotone coupling the counterfactual episode equals the factual
    truth = conditional_truth(SLIPS, P, P, NEEDED)
    factual_success = (len(SLIPS) - sum(SLIPS)) >= NEEDED
    assert truth == pytest.approx(1.0 if factual_success else 0.0)


def test_independence_point_ignores_the_episode():
    a = independence_point([1, 1, 1, 0, 0, 0, 0, 0, 0], P, NEEDED)
    b = independence_point([0, 0, 0, 0, 0, 0, 0, 0, 0], P, NEEDED)
    assert a == pytest.approx(b)  # the predictor cannot condition on evidence
    lo, hi = counterfactual_success_interval(SLIPS, P, NEEDED)
    assert lo - 1e-9 <= a <= hi + 1e-9  # one admissible point among many


def test_monte_carlo_validates_the_dp():
    """Simulate the corridor world under a known coupling and check the DP
    conditional truth against the empirical conditional frequency."""
    rng = np.random.default_rng(11)
    world = CorridorWorld(route_len=4, horizon=6, p_slip=P, p11=0.15)
    # group by factual slip pattern: condition on one concrete episode shape
    target = (1, 0, 0, 1, 0, 0)
    succ, tot = 0, 0
    for _ in range(60000):
        a, b = world.episode(rng)
        if tuple(a) == target:
            tot += 1
            succ += world.cf_success(b)
    assert tot > 200
    mc = succ / tot
    dp = conditional_truth(list(target), P, 0.15, 4)
    assert mc == pytest.approx(dp, abs=0.04)
    lo, hi = counterfactual_success_interval(list(target), P, 4)
    assert lo - 1e-9 <= dp <= hi + 1e-9
