"""The World Model Arena: many worlds, many contenders, one proper score.

The arena generates worlds whose full law it knows, hands every contender
only what an experiment would reveal (rung-1/2 functionals, or the constraint
graph), asks a rung-3 or off-diagonal question, and scores the answers
against the hidden truth.

The scoring rule is the Winkler interval score at level alpha:

    S(answer=[l, u], truth=y) = (u - l)
                                + (2/alpha) * (l - y)  if y < l
                                + (2/alpha) * (y - u)  if y > u

It is a proper scoring rule for interval forecasts: sharp valid intervals
win, loose intervals pay for width, and a point answer (a zero-width
interval) pays the full penalty whenever it is wrong. This is the arena's
whole point: it prices OVERCLAIMING, the failure mode of every predictor
that answers an unidentified question with one number.

World classes:
  two_arm           random two-world SCMs; queries: PN, fraction harmed
  two_arm_sampled   the same, but contenders see binomial counts, not exact
                    marginals (estimation noise on top of identification)
  mediation         random response-type laws on X -> M -> Y; query: NDE
  k_arm             random laws over k potential outcomes; query: pairwise
                    coherence Q, given only the diagonal
  constraint        hard-core worlds: random regular graphs below and above
                    the Sly-Sun threshold, rings of cliques, disjointness
                    taxonomies; query: an occupation marginal

Contenders:
  kernel            this package: sharp identified intervals (LP / Frechet /
                    width-bounded exact elimination / Weitz certificates),
                    widened for sampling noise where data are finite
  frechet           honest marginal-only boxes (valid, loose)
  independence      the standard predictor: commits to the independent
                    coupling / mediation plug-in formula / BP-style point
  monotone          the optimist: commits to the comonotone coupling
  bp                belief propagation (constraint worlds)
  weitz             the depth-budgeted certificate alone (constraint worlds)

Not every contender enters every class; the leaderboard reports, per class:
mean Winkler score, coverage (truth inside the answer), mean width, and
overclaim rate (zero-width answer that misses the truth by more than tol).
"""

from __future__ import annotations

import random as _random
import time
from dataclasses import dataclass, field

import numpy as np

from .barrier import bp_marginals, exact_marginals, random_regular
from .kernel import exact_interval, frechet_interval
from .mediation import (
    ATOMS,
    m_val,
    nde_interval_from_record,
    nde_vector,
    y_val,
)
from .tractable import (
    disjointness_graph,
    min_fill_order,
    ring_of_cliques,
    transfer_marginals,
    treewidth_marginal,
    weitz_interval,
)

__all__ = ["run_arena", "leaderboard", "winkler", "ALPHA"]

ALPHA = 0.2
OVERCLAIM_TOL = 0.02
WIDTH_CAP = 18  # kernel falls back to certificates above this elimination width


def winkler(lo: float, hi: float, truth: float, alpha: float = ALPHA) -> float:
    s = hi - lo
    if truth < lo:
        s += (2.0 / alpha) * (lo - truth)
    elif truth > hi:
        s += (2.0 / alpha) * (truth - hi)
    return s


@dataclass
class Answer:
    lo: float
    hi: float
    seconds: float = 0.0

    @classmethod
    def point(cls, x: float, seconds: float = 0.0) -> "Answer":
        return cls(x, x, seconds)


@dataclass
class Record:
    world_class: str
    query: str
    truth: float
    answers: dict[str, Answer] = field(default_factory=dict)


# ---- world class: two_arm ----------------------------------------------------

def _two_arm_records(n: int, rng: np.random.Generator, sampled: bool) -> list[Record]:
    out = []
    label = "two_arm_sampled" if sampled else "two_arm"
    for _ in range(n):
        r0, r1 = rng.uniform(0.15, 0.85, size=2)
        lo_box, hi_box = max(0.0, r0 + r1 - 1.0), min(r0, r1)
        p11 = rng.uniform(lo_box, hi_box)  # the hidden coupling
        truth_pn = (r1 - p11) / r1
        truth_h = r0 - p11

        if sampled:
            n_arm = 500
            r0_hat = rng.binomial(n_arm, r0) / n_arm
            r1_hat = rng.binomial(n_arm, r1) / n_arm
            se0 = np.sqrt(r0_hat * (1 - r0_hat) / n_arm)
            se1 = np.sqrt(r1_hat * (1 - r1_hat) / n_arm)
        else:
            r0_hat, r1_hat, se0, se1 = r0, r1, 0.0, 0.0

        def box(a: float, b: float) -> tuple[float, float]:
            return max(0.0, a + b - 1.0), min(a, b)

        for query, truth in (("PN", truth_pn), ("harmed", truth_h)):
            rec = Record(label, query, truth)
            # identified interval from the (estimated) diagonal
            lo_p, hi_p = box(r0_hat, r1_hat)
            if query == "PN":
                k_lo, k_hi = (r1_hat - hi_p) / r1_hat, (r1_hat - lo_p) / r1_hat
                ind = (r1_hat - r0_hat * r1_hat) / r1_hat
                mono = (r1_hat - hi_p) / r1_hat
            else:
                k_lo, k_hi = r0_hat - hi_p, r0_hat - lo_p
                ind = r0_hat - r0_hat * r1_hat
                mono = r0_hat - hi_p
            if sampled:  # widen the identified interval for sampling noise
                pad = 2.0 * (se0 + se1)
                k_lo, k_hi = max(0.0, k_lo - pad), min(1.0, k_hi + pad)
            rec.answers["kernel"] = Answer(k_lo, k_hi)
            rec.answers["frechet"] = Answer(k_lo, k_hi) if not sampled else Answer(
                max(0.0, (r1_hat - hi_p) / r1_hat if query == "PN" else r0_hat - hi_p),
                (r1_hat - lo_p) / r1_hat if query == "PN" else r0_hat - lo_p,
            )
            rec.answers["independence"] = Answer.point(ind)
            rec.answers["monotone"] = Answer.point(mono)
            out.append(rec)
    return out


# ---- world class: mediation ---------------------------------------------------

def _mediation_records(n: int, rng: np.random.Generator) -> list[Record]:
    nde = nde_vector()
    out = []
    for _ in range(n):
        # full Dirichlet over the 64 atoms: the hidden cross-world coupling
        # varies freely (a product law would make the plug-in formula exact
        # by construction and rig the class for the predictor)
        p0 = rng.dirichlet(np.ones(len(ATOMS)) * 0.3)
        truth = float(nde @ p0)

        def val(fn):
            return float(np.array([fn(iM, iY) for (iM, iY) in ATOMS]) @ p0)

        p_m = tuple(val(lambda iM, iY, x=x: m_val(iM, x) == 1) for x in (0, 1))
        p_my = {
            (x, m): val(
                lambda iM, iY, x=x, m=m: m_val(iM, x) == m
                and y_val(iY, x, m_val(iM, x)) == 1
            )
            for x in (0, 1)
            for m in (0, 1)
        }
        p_ydo = {
            (x, m): val(lambda iM, iY, x=x, m=m: y_val(iY, x, m) == 1)
            for x in (0, 1)
            for m in (0, 1)
        }

        rec = Record("mediation", "NDE", truth)
        t0 = time.time()
        lo, hi = nde_interval_from_record(p_m, p_my, p_ydo)
        rec.answers["kernel"] = Answer(lo, hi, time.time() - t0)
        # the textbook plug-in (sequential-ignorability / cross-world
        # independence): the one number every standard pipeline reports
        plug = sum(
            (p_m[0] if m else 1 - p_m[0]) * (p_ydo[(1, m)] - p_ydo[(0, m)])
            for m in (0, 1)
        )
        rec.answers["independence"] = Answer.point(plug)
        # marginal-only box: term-wise Frechet, ignoring joint feasibility
        def term_box(x: int) -> tuple[float, float]:
            lo_t = hi_t = 0.0
            for m in (0, 1):
                pm_ = p_m[0] if m else 1 - p_m[0]
                q = p_ydo[(x, m)]
                lo_t += max(0.0, pm_ + q - 1.0)
                hi_t += min(pm_, q)
            return lo_t, hi_t

        l1, h1 = term_box(1)
        l0, h0 = term_box(0)
        rec.answers["frechet"] = Answer(l1 - h0, h1 - l0)
        out.append(rec)
    return out


# ---- world class: k_arm --------------------------------------------------------

def _k_arm_records(n: int, rng: np.random.Generator, k: int = 8) -> list[Record]:
    out = []
    for _ in range(n):
        # hidden law over 2^k response types
        p = rng.dirichlet(np.ones(2**k) * 0.5)
        types = np.array(
            [[(t >> i) & 1 for i in range(k)] for t in range(2**k)], dtype=float
        )
        d = types.T @ p  # implied diagonal
        pair = 0.0
        for i in range(k):
            for j in range(i + 1, k):
                pair += float(((types[:, i] * types[:, j]) * p).sum())
        npairs = k * (k - 1) / 2  # report per-pair so classes are comparable
        rec = Record("k_arm", "coherence Q / pair", pair / npairs)
        t0 = time.time()
        el, eh = exact_interval(d)
        rec.answers["kernel"] = Answer(el / npairs, eh / npairs, time.time() - t0)
        fl, fh = frechet_interval(d)
        rec.answers["frechet"] = Answer(fl / npairs, fh / npairs)
        ind = sum(d[i] * d[j] for i in range(k) for j in range(i + 1, k))
        rec.answers["independence"] = Answer.point(float(ind) / npairs)
        out.append(rec)
    return out


# ---- world class: constraint ---------------------------------------------------

def _constraint_records(rng_seed: int, n_random: int, enum_n: int) -> list[Record]:
    rng = _random.Random(rng_seed)
    out = []
    worlds: list[tuple[str, list, float]] = []  # (tag, adj, truth at vertex 0)

    for d in (3, 7):
        for _ in range(n_random):
            adj = random_regular(enum_n, d, rng)
            truth = float(exact_marginals(adj, enum_n, 1.0)[0])
            worlds.append((f"random d={d}", adj, truth))
    adj = ring_of_cliques(20, 7)
    worlds.append(("ring s=7", adj, float(transfer_marginals(20, 7)[0])))
    adj = disjointness_graph(6, 3, seed=rng_seed)
    order, _ = min_fill_order(adj)
    worlds.append(
        ("taxonomy b=6", adj, float(treewidth_marginal(adj, 0, order=order)))
    )

    for tag, adj, truth in worlds:
        n = len(adj)
        rec = Record("constraint", f"marginal ({tag})", truth)
        # kernel: width-bounded exact elimination, else Weitz certificate
        t0 = time.time()
        order, width = min_fill_order(adj)
        if width <= WIDTH_CAP:
            p = treewidth_marginal(adj, 0, order=order)
            rec.answers["kernel"] = Answer.point(p, time.time() - t0)
        else:
            lo, hi = weitz_interval(adj, 0, 1.0, depth=8)
            rec.answers["kernel"] = Answer(lo, hi, time.time() - t0)
        t0 = time.time()
        bp = float(bp_marginals(adj, n, 1.0)[0])
        rec.answers["bp"] = Answer.point(bp, time.time() - t0)
        t0 = time.time()
        lo, hi = weitz_interval(adj, 0, 1.0, depth=6)
        rec.answers["weitz"] = Answer(lo, hi, time.time() - t0)
        out.append(rec)
    return out


# ---- the arena ------------------------------------------------------------------

def run_arena(
    seed: int = 11,
    n_two_arm: int = 40,
    n_mediation: int = 25,
    n_k_arm: int = 15,
    n_constraint_random: int = 3,
    constraint_enum_n: int = 14,
) -> list[Record]:
    rng = np.random.default_rng(seed)
    records: list[Record] = []
    records += _two_arm_records(n_two_arm, rng, sampled=False)
    records += _two_arm_records(n_two_arm, rng, sampled=True)
    records += _mediation_records(n_mediation, rng)
    records += _k_arm_records(n_k_arm, rng)
    records += _constraint_records(seed, n_constraint_random, constraint_enum_n)
    return records


def leaderboard(records: list[Record]) -> dict[str, dict[str, dict[str, float]]]:
    """Per world class and contender: Winkler scores at two risk levels,
    coverage, mean width, overclaim rate.

    The two alphas are the arena's risk dial. At alpha=0.2 a miss costs 10x
    its distance: if being wrong is tolerable, a committer with small typical
    error can beat a wide honest interval. At alpha=0.02 a miss costs 100x:
    the safety-critical regime, where guaranteed coverage dominates. The
    interesting result is WHERE each contender wins, not one number."""
    classes = sorted({r.world_class for r in records})
    contenders = sorted({c for r in records for c in r.answers})
    table: dict[str, dict[str, dict[str, float]]] = {}
    for wc in classes:
        rows = [r for r in records if r.world_class == wc]
        table[wc] = {}
        for c in contenders:
            scored = [(r.truth, r.answers[c]) for r in rows if c in r.answers]
            if not scored:
                continue
            ws = [winkler(a.lo, a.hi, y, alpha=ALPHA) for y, a in scored]
            ws_strict = [winkler(a.lo, a.hi, y, alpha=0.02) for y, a in scored]
            cov = [a.lo - 1e-9 <= y <= a.hi + 1e-9 for y, a in scored]
            widths = [a.hi - a.lo for _, a in scored]
            over = [
                (a.hi - a.lo) < 1e-12 and abs(a.lo - y) > OVERCLAIM_TOL
                for y, a in scored
            ]
            table[wc][c] = {
                "winkler": float(np.mean(ws)),
                "winkler_strict": float(np.mean(ws_strict)),
                "coverage": float(np.mean(cov)),
                "width": float(np.mean(widths)),
                "overclaim": float(np.mean(over)),
                "n": len(scored),
            }
    return table
