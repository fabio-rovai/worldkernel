"""The off-diagonal witness: two worlds, one treatment.

The smallest kernel that already separates a world model from a predictor.
T0 is the potential world "no treatment" (outcome Y_0), T1 is "treatment"
(outcome Y_1). The kernel is the 2 x 2 second-moment matrix of (Y_0, Y_1):

  diagonal      P(Y_0 = 1), P(Y_1 = 1)      -- what rung-1/2 data identify
  off-diagonal  p11 = P(Y_0 = 1, Y_1 = 1)   -- the cross-world coupling

Two kernels with the SAME diagonal but DIFFERENT off-diagonal agree on every
observational table and every interventional table, yet disagree on the
probability of necessity

  PN = P(Y_0 = 0 | X = 1, Y = 1) = (P(Y_1=1) - p11) / P(Y_1=1):

of the treated who recovered, the fraction who would not have recovered
untreated. A reasoner holding only rung-1/2 data must return one number and
therefore collapses the two worlds; a reasoner holding the off-diagonal
separates them exactly. The off-diagonal is the load-bearing sufficient
statistic, not decoration.

``witness_pair()`` builds the canonical verified pair: identical rungs 1-2
(ACE = 0.20), PN = 0.286 vs PN = 0.500.
"""

from __future__ import annotations

import numpy as np

from .kernel import CouplingKernel

__all__ = [
    "TwoWorldKernel",
    "witness_pair",
    "frechet_pn_bounds",
    "frechet_harmed_bounds",
]


class TwoWorldKernel(CouplingKernel):
    """Coupling kernel over the pair of potential worlds (Y_0, Y_1)."""

    def __init__(self, r0: float, r1: float, p11: float) -> None:
        lo, hi = max(0.0, r0 + r1 - 1.0), min(r0, r1)
        if not (lo - 1e-12 <= p11 <= hi + 1e-12):
            raise ValueError(
                f"coupling p11={p11} outside Frechet box [{lo:.4f}, {hi:.4f}]"
            )
        super().__init__([[r0, p11], [p11, r1]])
        self.r0, self.r1, self.p11 = float(r0), float(r1), float(p11)

    # -- the full cross-world joint -------------------------------------------
    def joint(self) -> dict[tuple[int, int], float]:
        """P(Y_0 = i, Y_1 = j), the off-diagonal unpacked to a distribution."""
        p11 = self.p11
        return {
            (1, 1): p11,
            (1, 0): self.r0 - p11,
            (0, 1): self.r1 - p11,
            (0, 0): 1.0 - self.r0 - self.r1 + p11,
        }

    # -- rungs 1 and 2: the diagonal data --------------------------------------
    def observational(self, p_treat: float = 0.5) -> dict[tuple[str, str], float]:
        """P(X, Y) under randomized X. Depends only on the diagonal."""
        return {
            ("X=1", "Y=1"): p_treat * self.r1,
            ("X=1", "Y=0"): p_treat * (1.0 - self.r1),
            ("X=0", "Y=1"): (1.0 - p_treat) * self.r0,
            ("X=0", "Y=0"): (1.0 - p_treat) * (1.0 - self.r0),
        }

    @property
    def ace(self) -> float:
        """Average causal effect P(Y=1|do X=1) - P(Y=1|do X=0): rung 2."""
        return self.r1 - self.r0

    # -- rung 3: reads the off-diagonal ----------------------------------------
    def pn(self) -> float:
        """Probability of necessity P(Y_0=0, Y_1=1) / P(Y_1=1)."""
        return (self.r1 - self.p11) / self.r1

    def ps(self) -> float:
        """Probability of sufficiency P(Y_0=0, Y_1=1) / P(Y_0=0)."""
        return (self.r1 - self.p11) / (1.0 - self.r0)

    def pns(self) -> float:
        """Probability of necessity and sufficiency P(Y_0=0, Y_1=1)."""
        return self.r1 - self.p11

    def helped(self) -> float:
        """Fraction of units the treatment flips 0 -> 1: P(Y_0=0, Y_1=1)."""
        return self.r1 - self.p11

    def harmed(self) -> float:
        """Fraction of units the treatment flips 1 -> 0: P(Y_0=1, Y_1=0).

        Reads the off-diagonal: two worlds with the same ACE can have
        harmed = 0 (monotonic) or harmed > 0 (heterogeneous response)."""
        return self.r0 - self.p11


def witness_pair(
    r0: float = 0.5, r1: float = 0.7
) -> tuple[TwoWorldKernel, TwoWorldKernel]:
    """The canonical witness: same diagonal, maximally different couplings.

    Returns (A, B) where A is the monotonic world (treatment never hurts,
    coupling at the Frechet maximum) and B is the independent-potential-outcomes
    world (p11 = r0 * r1). Both reproduce identical rung-1 and rung-2 tables;
    their PN values differ. With the defaults: PN_A = 0.286, PN_B = 0.500."""
    a = TwoWorldKernel(r0, r1, p11=min(r0, r1))
    b = TwoWorldKernel(r0, r1, p11=r0 * r1)
    assert np.allclose(a.diagonal, b.diagonal)
    return a, b


def frechet_pn_bounds(r0: float, r1: float) -> tuple[float, float]:
    """The identified PN interval from rung-1/2 data alone: everything the
    diagonal pins down. The off-diagonal selects the point inside it."""
    lo_p11, hi_p11 = max(0.0, r0 + r1 - 1.0), min(r0, r1)
    return (r1 - hi_p11) / r1, (r1 - lo_p11) / r1


def frechet_harmed_bounds(r0: float, r1: float) -> tuple[float, float]:
    """The identified interval for the fraction harmed P(Y_0=1, Y_1=0) from
    rung-1/2 data alone. The lower bound is max(0, r0 - r1): a positive ACE
    forces nothing about harm, which is exactly the off-diagonal point."""
    lo_p11, hi_p11 = max(0.0, r0 + r1 - 1.0), min(r0, r1)
    return r0 - hi_p11, r0 - lo_p11
