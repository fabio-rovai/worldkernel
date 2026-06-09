"""Decision under non-identification: rules, dominance, value of information."""

import pytest

from worldkernel.decide import decide, dominated


def test_maximin_prefers_guarantees():
    d = decide({"A": (0.4, 0.9), "B": (0.5, 0.6)}, rule="maximin")
    assert d.action == "B"  # B guarantees 0.5; A only 0.4
    assert not d.determined  # intervals overlap: data alone do not settle it
    assert set(d.contenders) == {"A", "B"}
    assert d.pivotal_widths["A"] == pytest.approx(0.5)


def test_minimax_regret():
    d = decide({"A": (0.4, 0.9), "B": (0.5, 0.6)}, rule="minimax_regret")
    # regret(A) = hi_B - lo_A = 0.2; regret(B) = hi_A - lo_B = 0.4
    assert d.scores["A"] == pytest.approx(0.2)
    assert d.scores["B"] == pytest.approx(0.4)
    assert d.action == "A"


def test_hurwicz_dial():
    iv = {"safe": (0.5, 0.55), "gamble": (0.1, 0.95)}
    assert decide(iv, rule="hurwicz", hurwicz_alpha=0.0).action == "safe"
    assert decide(iv, rule="hurwicz", hurwicz_alpha=1.0).action == "gamble"


def test_interval_dominance_determines_decision():
    d = decide({"A": (0.7, 0.9), "B": (0.2, 0.6)}, rule="maximin")
    assert d.determined  # lo_A > hi_B: every realization prefers A
    assert d.contenders == ["A"]
    assert d.pivotal_widths == {}
    assert "determined" in d.note


def test_dominated_map():
    dom = dominated({"A": (0.7, 0.9), "B": (0.2, 0.6), "C": (0.65, 0.8)})
    assert dom == {"A": False, "B": True, "C": False}


def test_point_committer_is_a_special_case():
    # the predictor's world: every interval is a point; rules coincide
    iv = {"A": (0.6, 0.6), "B": (0.5, 0.5)}
    for rule in ("maximin", "minimax_regret", "hurwicz"):
        assert decide(iv, rule=rule).action == "A"
    assert decide(iv).determined


def test_validation():
    with pytest.raises(ValueError):
        decide({})
    with pytest.raises(ValueError):
        decide({"A": (0.8, 0.2)})
    with pytest.raises(ValueError):
        decide({"A": (0.1, 0.2)}, rule="vibes")
