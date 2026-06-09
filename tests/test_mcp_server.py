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
    }


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
