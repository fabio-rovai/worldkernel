"""WorldKernel as an MCP server: the kernel as any agent's world model.

The division of labour this package argues for, made operational: the LLM (or
any orchestrator) is the frontier sensor that reads the world and proposes
structure; the kernel is the calculator that computes rung-3 quantities
exactly, certifies bounds, and surfaces non-identification instead of hiding
it. Every tool below returns intervals where intervals are the truth.

Run:  worldkernel-mcp            (stdio transport; needs the ``mcp`` extra)
Add to an MCP client config:
  {"mcpServers": {"worldkernel": {"command": "worldkernel-mcp"}}}
"""

from __future__ import annotations

from typing import Any

from .barrier import d_critical as _d_critical
from .barrier import order_parameter
from .kernel import exact_interval, frechet_interval
from .mediation import atom_count, nde_interval_from_record
from .tractable import min_fill_order, treewidth_marginal, weitz_interval
from .witness import TwoWorldKernel, frechet_harmed_bounds, frechet_pn_bounds

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "the MCP server requires the mcp SDK: pip install 'worldkernel[mcp]'"
    ) from exc

mcp = FastMCP(
    "worldkernel",
    instructions=(
        "Exact counterfactual computation over potential-outcome kernels. "
        "A randomized experiment identifies only the kernel's diagonal (the "
        "marginals); rung-3 quantities depend on the off-diagonal coupling "
        "and are generally identified only to an interval. These tools "
        "compute those intervals exactly. Treat a returned interval as the "
        "complete answer: do not pick a point inside it without stating the "
        "extra assumption that picks it."
    ),
)


@mcp.tool()
def counterfactual_bounds(r0: float, r1: float) -> dict[str, Any]:
    """Identified intervals for rung-3 quantities of a two-arm experiment.

    Args: r0 = P(Y=1 | do X=0), r1 = P(Y=1 | do X=1), both in [0,1].
    Returns the ACE (point-identified) and the identified intervals for the
    probability of necessity (PN) and the fraction harmed, plus the values
    each quantity takes under the two canonical couplings (monotonicity,
    independence). The intervals are sharp: no rung-1/2 data narrows them."""
    pn_lo, pn_hi = frechet_pn_bounds(r0, r1)
    h_lo, h_hi = frechet_harmed_bounds(r0, r1)
    mono = TwoWorldKernel(r0, r1, p11=min(r0, r1))
    indep = TwoWorldKernel(r0, r1, p11=r0 * r1)
    return {
        "ace": r1 - r0,
        "pn_interval": [pn_lo, pn_hi],
        "fraction_harmed_interval": [h_lo, h_hi],
        "under_monotonicity": {"pn": mono.pn(), "harmed": mono.harmed()},
        "under_independence": {"pn": indep.pn(), "harmed": indep.harmed()},
        "note": "the interval is the answer; a point requires an assumption",
    }


@mcp.tool()
def coupling_query(r0: float, r1: float, p11: float) -> dict[str, Any]:
    """Evaluate every rung-3 quantity given a FULL kernel (an assumed coupling).

    Args: marginals r0, r1 and the cross-world coupling p11 = P(Y0=1, Y1=1).
    Errors if the coupling is inadmissible (outside the Frechet box)."""
    k = TwoWorldKernel(r0, r1, p11)
    return {
        "admissible": k.admissible(),
        "pn": k.pn(),
        "ps": k.ps(),
        "pns": k.pns(),
        "helped": k.helped(),
        "harmed": k.harmed(),
        "cross_world_joint": {f"Y0={i},Y1={j}": v for (i, j), v in k.joint().items()},
    }


@mcp.tool()
def nde_bounds(
    p_m1_do_x0: float,
    p_m1_do_x1: float,
    p_my_do_x: dict[str, float],
    p_y_do_xm: dict[str, float],
) -> dict[str, Any]:
    """Identified interval of the Natural Direct Effect from a measured
    mediation record (X -> M -> Y, all binary).

    Args:
      p_m1_do_x0/x1: P(M=1 | do X=x)
      p_my_do_x: {"x,m": P(M=m, Y=1 | do X=x)} with keys "0,0","0,1","1,0","1,1"
      p_y_do_xm: {"x,m": P(Y=1 | do X=x, do M=m)}, same keys
    Returns the exact identified interval via LP over the 64-atom
    response-type polytope, or infeasibility if the record is inconsistent."""

    def parse(d: dict[str, float]) -> dict[tuple[int, int], float]:
        return {(int(k[0]), int(k[2])): float(v) for k, v in d.items()}

    try:
        lo, hi = nde_interval_from_record(
            (p_m1_do_x0, p_m1_do_x1), parse(p_my_do_x), parse(p_y_do_xm)
        )
    except ValueError as e:
        return {"feasible": False, "error": str(e)}
    return {
        "feasible": True,
        "nde_interval": [lo, hi],
        "width": hi - lo,
        "sign_identified": not (lo < 0.0 < hi),
        "note": "width is off-diagonal freedom; no rung-1/2 data reduces it",
    }


@mcp.tool()
def coherence_bounds(marginals: list[float], method: str = "auto") -> dict[str, Any]:
    """Bounds on the cross-world coherence Q = sum_{i<j} P(Y_i=1, Y_j=1) of a
    k-arm experiment, given only the marginals (the kernel diagonal).

    method: "frechet" (box, any k), "exact" (LP over 2^k response types,
    k <= 20), or "auto" (exact when k <= 14, else frechet)."""
    k = len(marginals)
    fl, fh = frechet_interval(marginals)
    out: dict[str, Any] = {"k": k, "frechet": [fl, fh]}
    if method == "exact" or (method == "auto" and k <= 14):
        if k > 20:
            return {**out, "error": "exact LP infeasible past k=20 (2^k atoms)"}
        el, eh = exact_interval(marginals)
        out["exact"] = [el, eh]
    return out


@mcp.tool()
def certified_marginal(
    edges: list[list[int]], vertex: int, lam: float = 1.0, depth: int = 8
) -> dict[str, Any]:
    """Certified hard-core occupation marginal bounds on a constraint graph
    (Weitz SAW recursion with interval boundaries). Unconditionally valid at
    every depth; contracts geometrically below the Sly-Sun threshold.

    Args: edges as [u, v] pairs over vertices 0..n-1; the query vertex;
    fugacity lam; SAW-tree depth (cost grows ~ (max_degree-1)^depth)."""
    n = max(max(e) for e in edges) + 1
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    lo, hi = weitz_interval(adj, vertex, lam, depth)
    dmax = max(len(a) for a in adj)
    return {
        "certified_interval": [lo, hi],
        "width": hi - lo,
        "max_degree": dmax,
        "order_parameter": order_parameter(dmax, lam),
        "regime": "tractable" if order_parameter(dmax, lam) < 1 else "hard-by-degree",
    }


@mcp.tool()
def exact_marginal_by_width(
    edges: list[list[int]], vertex: int, lam: float = 1.0
) -> dict[str, Any]:
    """EXACT hard-core marginal by variable elimination, with the width
    certificate. Polynomial in n for bounded width, at any degree. Refuses
    (rather than hangs) when the min-fill width exceeds 22."""
    n = max(max(e) for e in edges) + 1
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    order, width = min_fill_order(adj)
    if width > 22:
        return {
            "computed": False,
            "min_fill_width": width,
            "error": "width too large for exact elimination; use certified_marginal",
        }
    p = treewidth_marginal(adj, vertex, lam, order=order)
    return {
        "computed": True,
        "marginal": p,
        "min_fill_width": width,
        "max_degree": max(len(a) for a in adj),
        "note": "exactness is governed by width, not degree",
    }


@mcp.tool()
def barrier_diagnostics(degree: float, lam: float = 1.0) -> dict[str, Any]:
    """Where a constraint structure sits relative to the Sly-Sun counting
    barrier: the order parameter (d-1)*eta, the critical degree, and what is
    and is not computable there."""
    op = order_parameter(degree, lam)
    dc = _d_critical(lam)
    return {
        "order_parameter": op,
        "critical_degree": dc,
        "regime": "correlation decay (tractable)" if op < 1 else "hard by degree",
        "available_anyway": [
            "certified intervals (certified_marginal) at any degree",
            "exact computation if width is bounded (exact_marginal_by_width)",
            "PSD outer bounds on coherence (coherence_bounds)",
        ],
    }


@mcp.tool()
def mediation_scaling(n_mediators: int) -> dict[str, Any]:
    """Size of the response-type space for a mediation chain: where the
    counting barrier lives (64 -> 4096 -> ~4.2M atoms for 1 -> 2 -> 3)."""
    return {
        "n_mediators": n_mediators,
        "response_type_atoms": atom_count(n_mediators),
    }


@mcp.tool()
def bounds_from_counts(
    n_control: int, successes_control: int, n_treated: int,
    successes_treated: int, coverage: float = 0.95,
) -> dict[str, Any]:
    """Rung-3 intervals from raw trial COUNTS with a simultaneous coverage
    guarantee (Wilson + Bonferroni, corner-evaluated). Separates the part of
    the width that shrinks with n (sampling) from the part that never does
    (identification)."""
    from .estimate import harmed_bounds_from_counts, pn_bounds_from_counts

    pn = pn_bounds_from_counts(
        n_control, successes_control, n_treated, successes_treated, coverage
    )
    h = harmed_bounds_from_counts(
        n_control, successes_control, n_treated, successes_treated, coverage
    )
    return {
        "pn_interval": [pn.lo, pn.hi],
        "pn_identified_core": [pn.identified_lo, pn.identified_hi],
        "pn_sampling_inflation": pn.sampling_inflation,
        "harmed_interval": [h.lo, h.hi],
        "coverage": coverage,
        "note": "the identified core never shrinks with n; only an "
        "assumption closes it (see evaluate_assumption)",
    }


@mcp.tool()
def evaluate_assumption(
    name: str, r0: float, r1: float, value: float | None = None
) -> dict[str, Any]:
    """Validate a proposed coupling assumption against the marginals and
    price exactly what it buys. Vocabulary: monotone, independent, no_harm,
    correlation_at_most (needs value), coupling (needs value). Inadmissible
    proposals are refused. Adopting an admissible one is a modelling
    decision, not a finding: rung-1/2 data can never refute it."""
    from .propose import evaluate

    nar = evaluate(name, r0, r1, value)
    return {
        "assumption": nar.assumption,
        "admissible": nar.admissible,
        "coupling_interval": list(nar.coupling_interval),
        "pn_before": list(nar.pn_before),
        "pn_after": list(nar.pn_after),
        "harmed_after": list(nar.harmed_after),
        "pn_width_bought": nar.pn_width_bought,
        "note": nar.note,
    }


@mcp.tool()
def decide_under_uncertainty(
    options: dict[str, list[float]], rule: str = "maximin",
    hurwicz_alpha: float = 0.5,
) -> dict[str, Any]:
    """Choose between actions whose utilities are intervals [lo, hi] (from
    kernel queries). Rules: maximin, minimax_regret, hurwicz. Also reports
    whether the data already determine the choice (interval dominance) and,
    if not, which intervals are pivotal: the value-of-information map."""
    from .decide import decide

    d = decide(
        {k: (float(v[0]), float(v[1])) for k, v in options.items()},
        rule=rule, hurwicz_alpha=hurwicz_alpha,
    )
    return {
        "action": d.action,
        "rule": d.rule,
        "scores": d.scores,
        "determined_by_data": d.determined,
        "undominated": d.contenders,
        "pivotal_widths": d.pivotal_widths,
        "note": d.note,
    }


@mcp.tool()
def trajectory_counterfactual(
    observed_slips: list[int], p_slip: float, moves_needed: int
) -> dict[str, Any]:
    """Episode-level counterfactual: 'I took route A and observed these
    slips; would route B have reached the goal?' Exact identified interval
    over all admissible per-step couplings (Poisson-binomial DP at per-step
    endpoints), plus the independence point a predictive world model commits
    to, which ignores the episode's own evidence."""
    from .dynamics import counterfactual_success_interval, independence_point

    lo, hi = counterfactual_success_interval(observed_slips, p_slip, moves_needed)
    return {
        "cf_success_interval": [lo, hi],
        "independence_point": independence_point(
            observed_slips, p_slip, moves_needed
        ),
        "steps": len(observed_slips),
        "note": "the predictor's point is one admissible coupling among "
        "many and does not condition on the observed slips",
    }


@mcp.tool()
def verify_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Verify a proof-carrying world-model entry (PCWE protocol v0.1, see
    docs/PROOF_CARRYING_PROTOCOL.md). Supports certificate types 'backdoor'
    (linear-time witness check + recomputation) and 'exact-elimination'
    (width check + recomputation). Returns verified true/false and the
    recomputed answer where applicable."""
    from .backdoor import backdoor_marginal, verify_backdoor
    from .tractable import min_fill_order, treewidth_marginal

    try:
        edges = entry["query"]["instance"]["edges"]
        n = max(max(e) for e in edges) + 1
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        cert = entry["certificate"]
        ans = entry["answer"]
        kind = entry["query"]["kind"]
        if kind != "occupation_marginal":
            return {"verified": False, "reason": f"unsupported query kind {kind}"}
        vertex = int(entry["query"]["params"]["vertex"])

        if cert["type"] == "backdoor":
            B = set(cert["data"]["B"])
            if not verify_backdoor(adj, B):
                return {"verified": False, "reason": "backdoor certificate invalid"}
            value = backdoor_marginal(adj, vertex, B)
        elif cert["type"] == "exact-elimination":
            order, width = min_fill_order(adj)
            if width > 22:
                return {"verified": False, "reason": "width too large to recompute"}
            value = treewidth_marginal(adj, vertex, order=order)
        else:
            return {
                "verified": False,
                "reason": f"certificate type {cert['type']} needs the "
                "interactive path (sumcheck) or is not yet supported",
            }
        ok = ans["lo"] - 1e-9 <= value <= ans["hi"] + 1e-9
        return {"verified": bool(ok), "recomputed_value": value,
                "claimed": [ans["lo"], ans["hi"]]}
    except (KeyError, TypeError, ValueError) as e:
        return {"verified": False, "reason": f"malformed entry: {e}"}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
