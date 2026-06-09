"""The WorldModel facade: one object, automatic engine dispatch."""

import pytest

from worldkernel import WorldModel
from worldkernel.tractable import ring_of_cliques
from worldkernel.witness import frechet_pn_bounds


def test_two_world_exact_marginals():
    wm = WorldModel().observe_marginal("control", 0.5).observe_marginal("treated", 0.7)
    v = wm.pn("control", "treated")
    assert v.interval == pytest.approx(frechet_pn_bounds(0.5, 0.7))
    assert v.engine == "closed-form sharp bounds"
    assert v.exact and not v.identified
    a = wm.ace("control", "treated")
    assert a.identified and a.lo == pytest.approx(0.2)


def test_counts_route_to_estimation_engine():
    wm = WorldModel()
    wm.observe_counts("control", 260, 168).observe_counts("treated", 185, 140)
    v = wm.pn("control", "treated")
    assert v.engine == "corner-evaluated confidence box"
    assert not v.exact
    assert v.diagnostics["sampling_inflation"] > 0
    core = v.diagnostics["identified_core"]
    assert v.lo <= core[0] and core[1] <= v.hi


def test_assumption_narrows_and_inadmissible_refused():
    wm = WorldModel().observe_marginal("a", 0.5).observe_marginal("b", 0.7)
    before = wm.pn("a", "b").width
    nar = wm.assume("monotone", "a", "b")
    assert nar.admissible
    after = wm.pn("a", "b")
    assert after.identified
    assert after.width < before
    assert "assumption" in after.diagnostics
    with pytest.raises(ValueError, match="INADMISSIBLE"):
        wm.assume("coupling", "a", "b", value=0.9)


def test_coherence_dispatch():
    wm = WorldModel()
    for i, m in enumerate((0.3, 0.5, 0.7)):
        wm.observe_marginal(f"arm{i}", m)
    v = wm.coherence()
    assert v.engine == "exact response-type LP"
    assert v.exact and v.lo <= v.hi


def test_constraint_graph_dispatch():
    wm = WorldModel().set_constraint_graph(ring_of_cliques(8, 5))
    v = wm.world_marginal(0)
    assert v.engine == "exact variable elimination"
    assert v.identified
    assert v.diagnostics["min_fill_width"] <= 6
    # force the certificate path with a tiny width cap
    v2 = wm.world_marginal(0, width_cap=1)
    assert v2.engine == "Weitz certified interval"
    assert v2.lo - 1e-9 <= v.lo <= v2.hi + 1e-9  # certificate contains truth


def test_trajectory_and_decision_loop():
    wm = WorldModel()
    slips = [1, 0, 0, 1, 0, 0, 0, 1, 0]
    v = wm.trajectory_cf(slips, p=0.3, moves_needed=6)
    assert v.exact and v.width > 0
    ip = v.diagnostics["independence_point"]
    assert v.lo - 1e-9 <= ip <= v.hi + 1e-9
    # decide between retrying route A (point) and switching to B (interval)
    d = wm.decide({"retry": (0.55, 0.55), "switch": v}, rule="maximin")
    assert d.rule == "maximin"
    assert d.action in ("retry", "switch")
    assert isinstance(d.determined, bool)


def test_nde_passthrough_and_explain():
    wm = WorldModel().observe_marginal("x", 0.4)
    p_my = {(x, m): 0.2 for x in (0, 1) for m in (0, 1)}
    p_ydo = {(x, m): 0.4 for x in (0, 1) for m in (0, 1)}
    v = wm.nde((0.5, 0.5), p_my, p_ydo)
    assert v.engine == "response-type polytope LP"
    info = wm.explain()
    assert info["outcomes"] == {"x": 0.4}
    assert info["assumptions"] == {}
