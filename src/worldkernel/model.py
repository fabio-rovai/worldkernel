"""The WorldModel: one object, every engine, certificates attached.

This is the architecture's front door. Register structure (named potential
outcomes, a constraint graph), feed data (exact marginals or finite counts),
optionally adopt validated assumptions, then ask queries. Every answer is a
``Verdict``: an interval (a point when identified or pinned), the engine
that produced it, whether it is exact or an outer bound, and the
diagnostics that say WHY (width certificate, barrier position, sampling
inflation). Dispatch is automatic:

  two-world rung-3 queries    closed-form sharp bounds; corner-evaluated
                              confidence boxes when data are counts;
                              narrowed by validated assumptions
  k-arm coherence             exact response-type LP (k <= 14), PSD SDP if
                              cvxpy is present, Frechet box otherwise
  mediation NDE               feasibility-checked LP over the 64-atom polytope
  constraint-graph marginals  exact variable elimination when the min-fill
                              width allows, Weitz certified interval when not
  trajectory counterfactuals  exact per-step-endpoint Poisson-binomial bounds

The LLM's place in this architecture is outside this object: it proposes
structure and assumptions; ``assume`` validates them (via worldkernel.propose)
and refuses inadmissible ones. The kernel is the calculator.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .barrier import order_parameter
from .decide import Decision, decide as _decide
from .dynamics import counterfactual_success_interval, independence_point
from .estimate import (
    EstimatedInterval,
    ace_from_counts,
    harmed_bounds_from_counts,
    pn_bounds_from_counts,
)
from .kernel import exact_interval, frechet_interval
from .mediation import nde_interval_from_record
from .propose import Narrowing, evaluate as _evaluate
from .tractable import min_fill_order, treewidth_marginal, weitz_interval
from .witness import frechet_harmed_bounds, frechet_pn_bounds

__all__ = ["Verdict", "WorldModel"]


@dataclass
class Verdict:
    lo: float
    hi: float
    engine: str
    exact: bool  # True when the interval IS the identified set (or a point)
    seconds: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> float:
        return self.hi - self.lo

    @property
    def identified(self) -> bool:
        return self.width < 1e-9

    @property
    def interval(self) -> tuple[float, float]:
        return (self.lo, self.hi)

    def __repr__(self) -> str:  # readable at the REPL and in agent logs
        body = f"{self.lo:.4f}" if self.identified else f"[{self.lo:.4f}, {self.hi:.4f}]"
        return f"Verdict({body}, engine={self.engine}, exact={self.exact})"


class WorldModel:
    """Structure + data + assumptions in; certified verdicts out."""

    def __init__(self) -> None:
        self._marginals: dict[str, float] = {}
        self._counts: dict[str, tuple[int, int]] = {}
        self._coverage = 0.95
        self._assumptions: dict[tuple[str, str], Narrowing] = {}
        self._graph = None
        self._graph_order = None
        self._graph_width = None
        self._lam = 1.0

    # -- structure and data ----------------------------------------------------
    def observe_marginal(self, outcome: str, value: float) -> "WorldModel":
        if not 0.0 <= value <= 1.0:
            raise ValueError("marginal must be in [0, 1]")
        self._marginals[outcome] = float(value)
        self._counts.pop(outcome, None)
        return self

    def observe_counts(
        self, outcome: str, n: int, successes: int, coverage: float = 0.95
    ) -> "WorldModel":
        if not 0 <= successes <= n:
            raise ValueError("need 0 <= successes <= n")
        self._counts[outcome] = (int(n), int(successes))
        self._marginals[outcome] = successes / n
        self._coverage = coverage
        return self

    def set_constraint_graph(self, adj, lam: float = 1.0) -> "WorldModel":
        self._graph = [set(a) for a in adj]
        self._graph_order, self._graph_width = min_fill_order(self._graph)
        self._lam = lam
        return self

    def assume(
        self, name: str, a: str, b: str, value: float | None = None
    ) -> Narrowing:
        """Adopt an assumption about the coupling of outcomes (a, b).
        Validated against the marginals; inadmissible proposals are refused."""
        r0, r1 = self._pair(a, b)
        narrowing = _evaluate(name, r0, r1, value)
        if not narrowing.admissible:
            raise ValueError(narrowing.note)
        self._assumptions[(a, b)] = narrowing
        return narrowing

    # -- two-world rung-3 queries ------------------------------------------------
    def _pair(self, a: str, b: str) -> tuple[float, float]:
        for lbl in (a, b):
            if lbl not in self._marginals:
                raise KeyError(f"no data for outcome {lbl!r}")
        return self._marginals[a], self._marginals[b]

    def _two_world(self, a: str, b: str, kind: str) -> Verdict:
        t0 = time.time()
        r0, r1 = self._pair(a, b)
        sampled = a in self._counts and b in self._counts
        diag: dict[str, Any] = {}

        if (a, b) in self._assumptions:
            nar = self._assumptions[(a, b)]
            lo, hi = nar.pn_after if kind == "pn" else nar.harmed_after
            diag["assumption"] = nar.assumption
            engine = "closed-form + assumption"
            if sampled:
                diag["caveat"] = "assumption narrowing uses point marginals"
        elif sampled:
            (n0, k0), (n1, k1) = self._counts[a], self._counts[b]
            est: EstimatedInterval = (
                pn_bounds_from_counts(n0, k0, n1, k1, self._coverage)
                if kind == "pn"
                else harmed_bounds_from_counts(n0, k0, n1, k1, self._coverage)
            )
            lo, hi = est.lo, est.hi
            diag["sampling_inflation"] = est.sampling_inflation
            diag["identified_core"] = (est.identified_lo, est.identified_hi)
            diag["coverage"] = est.coverage
            engine = "corner-evaluated confidence box"
        else:
            lo, hi = (
                frechet_pn_bounds(r0, r1)
                if kind == "pn"
                else frechet_harmed_bounds(r0, r1)
            )
            engine = "closed-form sharp bounds"
        return Verdict(float(lo), float(hi), engine, exact=not sampled,
                       seconds=time.time() - t0, diagnostics=diag)

    def pn(self, a: str, b: str) -> Verdict:
        """Probability of necessity: P(Y_a=0 | chose b, Y_b=1)."""
        return self._two_world(a, b, "pn")

    def harmed(self, a: str, b: str) -> Verdict:
        """Fraction flipped 1 -> 0 by moving from world a to world b."""
        return self._two_world(a, b, "harmed")

    def ace(self, a: str, b: str) -> Verdict:
        t0 = time.time()
        r0, r1 = self._pair(a, b)
        if a in self._counts and b in self._counts:
            (n0, k0), (n1, k1) = self._counts[a], self._counts[b]
            est = ace_from_counts(n0, k0, n1, k1, self._coverage)
            return Verdict(est.lo, est.hi, "corner-evaluated confidence box",
                           exact=False, seconds=time.time() - t0,
                           diagnostics={"coverage": est.coverage})
        return Verdict(r1 - r0, r1 - r0, "identity", exact=True,
                       seconds=time.time() - t0)

    # -- k-arm coherence -----------------------------------------------------------
    def coherence(self) -> Verdict:
        """Bounds on Q = sum_{i<j} P(Y_i=1, Y_j=1) over all observed outcomes."""
        t0 = time.time()
        labels = sorted(self._marginals)
        d = [self._marginals[lbl] for lbl in labels]
        k = len(d)
        if k < 2:
            raise ValueError("coherence needs at least two outcomes")
        if k <= 14:
            lo, hi = exact_interval(d)
            return Verdict(lo, hi, "exact response-type LP", exact=True,
                           seconds=time.time() - t0, diagnostics={"k": k})
        try:
            from .kernel import psd_interval

            lo, hi = psd_interval(d)
            return Verdict(lo, hi, "PSD kernel SDP (outer bound)", exact=False,
                           seconds=time.time() - t0, diagnostics={"k": k})
        except ImportError:
            lo, hi = frechet_interval(d)
            return Verdict(lo, hi, "Frechet box (outer bound)", exact=False,
                           seconds=time.time() - t0, diagnostics={"k": k})

    # -- mediation --------------------------------------------------------------
    def nde(self, p_m_do_x, p_my_do_x, p_y_do_xm) -> Verdict:
        t0 = time.time()
        lo, hi = nde_interval_from_record(p_m_do_x, p_my_do_x, p_y_do_xm)
        return Verdict(lo, hi, "response-type polytope LP", exact=True,
                       seconds=time.time() - t0,
                       diagnostics={"atoms": 64,
                                    "sign_identified": not (lo < 0.0 < hi)})

    # -- constraint-graph marginals ------------------------------------------------
    def world_marginal(self, vertex: int, width_cap: int = 18,
                       weitz_depth: int = 8) -> Verdict:
        if self._graph is None:
            raise ValueError("set_constraint_graph first")
        t0 = time.time()
        n = len(self._graph)
        dmax = max(len(x) for x in self._graph)
        diag = {
            "min_fill_width": self._graph_width,
            "max_degree": dmax,
            "order_parameter": order_parameter(dmax, self._lam),
        }
        if self._graph_width <= width_cap:
            p = treewidth_marginal(self._graph, vertex, self._lam,
                                   order=self._graph_order)
            return Verdict(p, p, "exact variable elimination", exact=True,
                           seconds=time.time() - t0, diagnostics=diag)
        lo, hi = weitz_interval(self._graph, vertex, self._lam, weitz_depth)
        diag["weitz_depth"] = weitz_depth
        return Verdict(lo, hi, "Weitz certified interval", exact=False,
                       seconds=time.time() - t0, diagnostics=diag)

    # -- trajectory counterfactuals ---------------------------------------------
    def trajectory_cf(
        self, observed_slips: list[int], p: float, moves_needed: int
    ) -> Verdict:
        t0 = time.time()
        lo, hi = counterfactual_success_interval(observed_slips, p, moves_needed)
        return Verdict(lo, hi, "per-step endpoint Poisson-binomial DP",
                       exact=True, seconds=time.time() - t0,
                       diagnostics={
                           "steps": len(observed_slips),
                           "independence_point": independence_point(
                               observed_slips, p, moves_needed),
                       })

    # -- decision ------------------------------------------------------------------
    def decide(
        self, options: dict[str, Verdict | tuple[float, float]],
        rule: str = "maximin", hurwicz_alpha: float = 0.5,
    ) -> Decision:
        intervals = {
            k: (v.interval if isinstance(v, Verdict) else (float(v[0]), float(v[1])))
            for k, v in options.items()
        }
        return _decide(intervals, rule=rule, hurwicz_alpha=hurwicz_alpha)

    # -- introspection ---------------------------------------------------------------
    def explain(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "outcomes": dict(self._marginals),
            "counts": dict(self._counts),
            "assumptions": {
                f"{a}~{b}": nar.assumption
                for (a, b), nar in self._assumptions.items()
            },
        }
        if self._graph is not None:
            out["constraint_graph"] = {
                "n": len(self._graph),
                "min_fill_width": self._graph_width,
            }
        return out
