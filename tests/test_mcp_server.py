"""The MCP server: kernel queries as agent tools (requires the mcp extra)."""

import asyncio

import pytest

pytest.importorskip("mcp", reason="MCP server tests require the mcp SDK")

from worldkernel import mcp_server  # noqa: E402
from worldkernel.mediation import random_reference, rung12_summary  # noqa: E402


def test_all_tools_registered():
    tools = asyncio.run(mcp_server.mcp.list_tools())
    names = {t.name for t in tools}
    assert names == {
        "counterfactual_bounds",
        "coupling_query",
        "nde_bounds",
        "coherence_bounds",
        "certified_marginal",
        "exact_marginal_by_width",
        "barrier_diagnostics",
        "mediation_scaling",
        "bounds_from_counts",
        "evaluate_assumption",
        "decide_under_uncertainty",
        "trajectory_counterfactual",
        "verify_entry",
    }


def test_verify_entry_backdoor_roundtrip():
    from worldkernel import backdoor_marginal, find_backdoor, hub_world

    adj = hub_world(10, 3, 1, 6)
    edges = [[u, v] for u in range(len(adj)) for v in adj[u] if u < v]
    B = find_backdoor(adj)
    val = backdoor_marginal(adj, 0, B)
    entry = {
        "query": {"kind": "occupation_marginal",
                  "instance": {"edges": edges}, "params": {"vertex": 0}},
        "answer": {"lo": val, "hi": val},
        "certificate": {"type": "backdoor", "data": {"B": sorted(B)}},
    }
    out = mcp_server.verify_entry(entry)
    assert out["verified"]
    # a false claim with a valid certificate is caught by recomputation
    entry["answer"] = {"lo": val + 0.1, "hi": val + 0.1}
    assert not mcp_server.verify_entry(entry)["verified"]
    # a broken certificate is caught before any computation
    entry["certificate"]["data"]["B"] = []
    out = mcp_server.verify_entry(entry)
    assert not out["verified"] and "certificate" in out["reason"]


def test_verify_entry_malformed():
    assert not mcp_server.verify_entry({"nonsense": 1})["verified"]


def test_agent_loop_tools():
    est = mcp_server.bounds_from_counts(260, 168, 185, 140)
    assert est["pn_sampling_inflation"] > 0
    nar = mcp_server.evaluate_assumption("monotone", 0.5, 0.7)
    assert nar["admissible"] and nar["pn_width_bought"] > 0.4
    bad = mcp_server.evaluate_assumption("coupling", 0.5, 0.7, value=0.9)
    assert not bad["admissible"]
    d = mcp_server.decide_under_uncertainty(
        {"A": [0.4, 0.9], "B": [0.5, 0.6]}, rule="maximin"
    )
    assert d["action"] == "B" and not d["determined_by_data"]
    t = mcp_server.trajectory_counterfactual([1, 0, 0, 1, 0, 0, 0, 1, 0], 0.3, 6)
    lo, hi = t["cf_success_interval"]
    assert lo - 1e-9 <= t["independence_point"] <= hi + 1e-9


def test_counterfactual_bounds_tool():
    out = mcp_server.counterfactual_bounds(0.5, 0.7)
    assert out["ace"] == pytest.approx(0.2)
    assert out["pn_interval"] == pytest.approx([2 / 7, 5 / 7])
    assert out["under_monotonicity"]["harmed"] == pytest.approx(0.0)


def test_coupling_query_tool():
    out = mcp_server.coupling_query(0.5, 0.7, 0.35)
    assert out["admissible"]
    assert out["pn"] == pytest.approx(0.5)
    assert out["helped"] - out["harmed"] == pytest.approx(0.2)


def test_nde_bounds_tool_round_trip():
    """A record generated from a real law must be feasible and reproduce the
    library interval; the seed-0 instance spans zero."""
    s = rung12_summary(random_reference(seed=0))
    p0 = random_reference(seed=0)
    from worldkernel.mediation import ATOMS, m_val, y_val
    import numpy as np

    def val(fn):
        return float(np.array([fn(iM, iY) for (iM, iY) in ATOMS]) @ p0)

    p_my = {
        f"{x},{m}": val(
            lambda iM, iY, x=x, m=m: 1.0
            if (m_val(iM, x) == m and y_val(iY, x, m_val(iM, x)) == 1)
            else 0.0
        )
        for x in (0, 1)
        for m in (0, 1)
    }
    p_ydo = {f"{x},{m}": s[f"P(Y=1|do X={x},do M={m})"] for x in (0, 1) for m in (0, 1)}
    out = mcp_server.nde_bounds(
        s["P(M=1|do X=0)"], s["P(M=1|do X=1)"], p_my, p_ydo
    )
    assert out["feasible"]
    lo, hi = out["nde_interval"]
    assert lo == pytest.approx(-0.381, abs=0.02)
    assert hi == pytest.approx(0.187, abs=0.02)
    assert not out["sign_identified"]


def test_nde_bounds_tool_rejects_inconsistent_record():
    bad = {"0,0": 0.9, "0,1": 0.9, "1,0": 0.9, "1,1": 0.9}  # sums > 1: impossible
    out = mcp_server.nde_bounds(0.5, 0.5, bad, bad)
    assert not out["feasible"]


def test_certified_marginal_tool():
    edges = [[0, 1], [1, 2], [2, 3], [3, 0]]  # 4-cycle
    out = mcp_server.certified_marginal(edges, 0, depth=10)
    lo, hi = out["certified_interval"]
    # C4 hard-core at lam=1: 7 independent sets ({}, 4 singletons, {0,2},
    # {1,3}); vertex 0 occupied in 2 of them -> marginal 2/7
    assert lo - 1e-9 <= 2 / 7 <= hi + 1e-9
    assert out["regime"] == "tractable"


def test_exact_marginal_by_width_tool():
    edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
    out = mcp_server.exact_marginal_by_width(edges, 0)
    assert out["computed"]
    assert out["marginal"] == pytest.approx(2 / 7)
    assert out["min_fill_width"] == 2


def test_barrier_diagnostics_tool():
    out = mcp_server.barrier_diagnostics(8)
    assert out["regime"] == "hard by degree"
    assert out["critical_degree"] == pytest.approx(5.141, abs=2e-3)
