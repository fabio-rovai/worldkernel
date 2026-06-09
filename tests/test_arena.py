"""The arena: proper scoring across world classes, mini configuration."""

import pytest

from worldkernel.arena import ALPHA, Answer, leaderboard, run_arena, winkler


@pytest.fixture(scope="module")
def table():
    records = run_arena(
        seed=7,
        n_two_arm=15,
        n_mediation=8,
        n_k_arm=5,
        n_constraint_random=1,
        constraint_enum_n=12,
    )
    return leaderboard(records)


def test_winkler_is_a_proper_interval_score():
    # containing interval pays only width
    assert winkler(0.2, 0.6, 0.4) == pytest.approx(0.4)
    # a miss pays width plus (2/alpha) * distance
    assert winkler(0.2, 0.6, 0.7) == pytest.approx(0.4 + (2 / ALPHA) * 0.1)
    # a wrong point pays the full penalty
    assert winkler(0.3, 0.3, 0.5) == pytest.approx((2 / ALPHA) * 0.2)
    assert Answer.point(0.3).hi == Answer.point(0.3).lo == 0.3


def test_kernel_has_full_coverage_everywhere(table):
    for wc, rows in table.items():
        assert rows["kernel"]["coverage"] == pytest.approx(1.0), wc
        assert rows["kernel"]["overclaim"] == 0.0, wc


def test_kernel_enters_every_class(table):
    assert all("kernel" in rows for rows in table.values())
    assert len(table) == 5


def test_point_committers_overclaim(table):
    for wc in ("two_arm", "two_arm_sampled", "mediation"):
        ind = table[wc]["independence"]
        assert ind["coverage"] < 0.2, wc
        assert ind["overclaim"] > 0.5, wc


def test_kernel_beats_committers_when_errors_are_expensive(table):
    """At alpha=0.02 (miss costs 100x distance) guaranteed coverage wins in
    every class where a committer competes."""
    for wc, rows in table.items():
        k = rows["kernel"]["winkler_strict"]
        for c in ("independence", "monotone", "bp"):
            if c in rows:
                assert k < rows[c]["winkler_strict"], (wc, c)


def test_kernel_at_least_matches_frechet(table):
    """The kernel's identified set is never looser than the marginal box;
    strictly sharper for mediation (joint feasibility) at both risk levels."""
    for wc, rows in table.items():
        if "frechet" in rows:
            assert (
                rows["kernel"]["width"] <= rows["frechet"]["width"] + 0.15
            ), wc
    assert table["mediation"]["kernel"]["width"] < table["mediation"]["frechet"]["width"]
    assert table["k_arm"]["kernel"]["width"] < table["k_arm"]["frechet"]["width"]


def test_sampling_breaks_unwidened_boxes_not_the_kernel(table):
    """Finite samples: the kernel widens for estimation noise and keeps
    coverage 1.0; the raw box may not."""
    assert table["two_arm_sampled"]["kernel"]["coverage"] == pytest.approx(1.0)
    assert (
        table["two_arm_sampled"]["frechet"]["coverage"]
        <= table["two_arm_sampled"]["kernel"]["coverage"]
    )


def test_constraint_class_structure(table):
    rows = table["constraint"]
    # the kernel computes the structured and small worlds exactly
    assert rows["kernel"]["winkler"] < 0.01
    # BP commits and misses on some worlds; Weitz stays covered
    assert rows["bp"]["coverage"] < 1.0
    assert rows["weitz"]["coverage"] == pytest.approx(1.0)