"""The proposer interface: assumptions in, validated narrowings out.

The architecture's loop is: an LLM (or analyst, or search procedure)
PROPOSES structure; the kernel VALIDATES it and computes exactly what it
buys. This module is the validation half, the kernel-side primitive the
proposer talks to. It never invents assumptions; it prices them.

An assumption over a two-world kernel is a constraint on the coupling
p11 = P(Y_0=1, Y_1=1). The vocabulary:

  monotone                 treatment never hurts: p11 = min(r0, r1)
  independent              independent potential outcomes: p11 = r0 * r1
  no_harm                  alias of monotone (harmed = 0)
  correlation_at_most(r)   |corr(Y_0, Y_1)| <= r: an interval of couplings
  coupling(x)              an explicit value

``evaluate`` returns whether the assumption is admissible (consistent with
the marginals), the implied interval for each rung-3 query before and after,
and the honest epistemic note: rung-1/2 data can never refute an admissible
coupling, so adopting one is a modelling decision, priced here, not a
finding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .witness import TwoWorldKernel, frechet_harmed_bounds, frechet_pn_bounds

__all__ = ["Narrowing", "evaluate", "VOCABULARY"]

VOCABULARY = ("monotone", "independent", "no_harm", "correlation_at_most", "coupling")


@dataclass
class Narrowing:
    assumption: str
    admissible: bool
    coupling_interval: tuple[float, float]
    pn_before: tuple[float, float]
    pn_after: tuple[float, float]
    harmed_before: tuple[float, float]
    harmed_after: tuple[float, float]
    refutable_from_data: bool
    note: str

    @property
    def pn_width_bought(self) -> float:
        return (self.pn_before[1] - self.pn_before[0]) - (
            self.pn_after[1] - self.pn_after[0]
        )


def _coupling_set(
    name: str, r0: float, r1: float, value: float | None
) -> tuple[float, float]:
    lo_box, hi_box = max(0.0, r0 + r1 - 1.0), min(r0, r1)
    if name in ("monotone", "no_harm"):
        return hi_box, hi_box
    if name == "independent":
        p = r0 * r1
        return p, p
    if name == "correlation_at_most":
        if value is None or not 0.0 <= value <= 1.0:
            raise ValueError("correlation_at_most needs a value in [0, 1]")
        sd = np.sqrt(r0 * (1 - r0) * r1 * (1 - r1))
        centre = r0 * r1
        return (
            max(lo_box, centre - value * sd),
            min(hi_box, centre + value * sd),
        )
    if name == "coupling":
        if value is None:
            raise ValueError("coupling needs an explicit value")
        return value, value
    raise ValueError(f"unknown assumption {name!r}; vocabulary: {VOCABULARY}")


def evaluate(
    name: str, r0: float, r1: float, value: float | None = None
) -> Narrowing:
    """Validate an assumption against the marginals and price what it buys."""
    lo_box, hi_box = max(0.0, r0 + r1 - 1.0), min(r0, r1)
    c_lo, c_hi = _coupling_set(name, r0, r1, value)
    admissible = c_lo <= c_hi and c_lo >= lo_box - 1e-12 and c_hi <= hi_box + 1e-12

    pn_b = frechet_pn_bounds(r0, r1)
    h_b = frechet_harmed_bounds(r0, r1)
    if admissible:
        ks = [TwoWorldKernel(r0, r1, p) for p in (c_lo, c_hi)]
        pn_a = (min(k.pn() for k in ks), max(k.pn() for k in ks))
        h_a = (min(k.harmed() for k in ks), max(k.harmed() for k in ks))
        note = (
            "admissible; rung-1/2 data can never refute an admissible "
            "coupling, so this is a priced modelling decision, not a finding"
        )
    else:
        pn_a, h_a = pn_b, h_b
        note = (
            f"INADMISSIBLE: the proposed coupling set [{c_lo:.4f}, {c_hi:.4f}] "
            f"leaves the Frechet box [{lo_box:.4f}, {hi_box:.4f}]; "
            "reject the proposal"
        )
    return Narrowing(
        assumption=name if value is None else f"{name}({value})",
        admissible=bool(admissible),
        coupling_interval=(float(c_lo), float(c_hi)),
        pn_before=pn_b,
        pn_after=(float(pn_a[0]), float(pn_a[1])),
        harmed_before=h_b,
        harmed_after=(float(h_a[0]), float(h_a[1])),
        refutable_from_data=False,
        note=note,
    )
