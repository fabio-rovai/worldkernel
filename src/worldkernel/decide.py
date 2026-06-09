"""Decision under non-identification: act on the interval, not a point.

The kernel returns intervals because intervals are the truth; this module is
what an agent DOES with them. Inputs are per-action utility intervals (from
kernel queries, possibly estimation-widened); outputs are a chosen action
under an explicit decision rule, plus the value-of-information analysis: is
the choice already determined by the data, and if not, which interval widths
are responsible, i.e. which assumption or experiment is worth buying.

Rules:
  maximin          argmax of the lower endpoint (Wald): best guaranteed value
  minimax_regret   argmin of worst-case regret R(a) = max_{b != a} hi_b - lo_a,
                   valid when actions' utilities can co-vary freely across
                   the identified set (the conservative reading)
  hurwicz(alpha)   argmax of alpha * hi + (1 - alpha) * lo

A point-committing agent is the special case where every interval is a
point; the arena shows what that costs when the point is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Decision", "decide", "dominated"]


@dataclass
class Decision:
    action: str
    rule: str
    scores: dict[str, float]
    determined: bool  # True iff the best action is the same for EVERY
    # realization in the intervals (interval dominance)
    contenders: list[str] = field(default_factory=list)  # undominated actions
    pivotal_widths: dict[str, float] = field(default_factory=dict)
    note: str = ""


def dominated(intervals: dict[str, tuple[float, float]]) -> dict[str, bool]:
    """a is dominated iff some b guarantees more: lo_b > hi_a."""
    out = {}
    for a, (_, hi_a) in intervals.items():
        out[a] = any(lo_b > hi_a for b, (lo_b, _) in intervals.items() if b != a)
    return out


def decide(
    intervals: dict[str, tuple[float, float]],
    rule: str = "maximin",
    hurwicz_alpha: float = 0.5,
) -> Decision:
    if not intervals:
        raise ValueError("no actions to decide between")
    for a, (lo, hi) in intervals.items():
        if lo > hi + 1e-12:
            raise ValueError(f"action {a!r} has an inverted interval")

    if rule == "maximin":
        scores = {a: lo for a, (lo, _) in intervals.items()}
        best = max(scores, key=scores.get)
    elif rule == "minimax_regret":
        scores = {}
        for a, (lo_a, _) in intervals.items():
            others = [hi_b for b, (_, hi_b) in intervals.items() if b != a]
            scores[a] = (max(others) - lo_a) if others else 0.0
        best = min(scores, key=scores.get)
    elif rule == "hurwicz":
        scores = {
            a: hurwicz_alpha * hi + (1 - hurwicz_alpha) * lo
            for a, (lo, hi) in intervals.items()
        }
        best = max(scores, key=scores.get)
    else:
        raise ValueError(f"unknown rule {rule!r}")

    dom = dominated(intervals)
    contenders = [a for a, d in dom.items() if not d]
    determined = len(contenders) == 1

    # value of information: among undominated actions, the overlap that keeps
    # the decision ambiguous; collapsing these widths settles the choice
    pivotal = {}
    if not determined:
        for a in contenders:
            pivotal[a] = intervals[a][1] - intervals[a][0]
    note = (
        "decision determined by the data: one action interval-dominates"
        if determined
        else f"{len(contenders)} undominated actions; an off-diagonal "
        "assumption or more data on the pivotal intervals settles it"
    )
    return Decision(
        action=best,
        rule=rule,
        scores={k: float(v) for k, v in scores.items()},
        determined=determined,
        contenders=sorted(contenders),
        pivotal_widths={k: float(v) for k, v in pivotal.items()},
        note=note,
    )
