"""Time: sequential potential outcomes and trajectory counterfactuals.

The witness, extended to episodes. A sequential world has, at every step, a
potential outcome per action (here: does the step slip under route A, under
route B) whose per-action marginals are identified by randomization while
the WITHIN-STEP cross-action coupling is not. Latents are independent across
steps (the explicit dynamics assumption); each step carries its own coupling
p11_t, free inside its Frechet box.

The trajectory query an agent actually asks after a failed episode: "I took
route A and observed these slips; would route B have reached the goal in
time?" The counterfactual per-step slip probability given the factual slip is

  q_t = p11_t / p          if the factual step slipped,
  q_t = (p - p11_t)/(1-p)  if it did not,

and counterfactual success is a Poisson-binomial tail in the q_t, computed
exactly by dynamic programming. Success is monotone decreasing in every q_t
and each q_t is monotone in its own p11_t, so the identified interval over
the whole product of coupling boxes is attained at per-step endpoints:
exact trajectory-level bounds, no relaxation.

A predictive world model commits to one coupling (independence: q_t = p at
every step regardless of what was observed) and collapses worlds that the
episode's own evidence distinguishes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "cf_slip_prob",
    "success_prob",
    "counterfactual_success_interval",
    "independence_point",
    "CorridorWorld",
]


def cf_slip_prob(factual_slip: int, p: float, p11: float) -> float:
    """P(counterfactual action slips | factual action slipped or not)."""
    if factual_slip:
        return p11 / p
    return (p - p11) / (1.0 - p)


def success_prob(slip_probs: list[float], moves_needed: int) -> float:
    """P(at least moves_needed non-slip steps) for independent steps with the
    given slip probabilities: exact Poisson-binomial tail by DP."""
    T = len(slip_probs)
    if moves_needed <= 0:
        return 1.0
    if moves_needed > T:
        return 0.0
    dp = np.zeros(T + 1)
    dp[0] = 1.0
    for q in slip_probs:
        nxt = np.zeros(T + 1)
        nxt[1:] += dp[:-1] * (1.0 - q)  # step succeeded
        nxt += dp * q  # step slipped
        dp = nxt
    return float(dp[moves_needed:].sum())


def _coupling_box(p: float) -> tuple[float, float]:
    return max(0.0, 2.0 * p - 1.0), p


def counterfactual_success_interval(
    observed_slips: list[int], p: float, moves_needed: int
) -> tuple[float, float]:
    """Identified interval for P(counterfactual route succeeds | episode).

    Exact over the product of per-step coupling boxes: success is monotone
    decreasing in each q_t, q_t(slip=1) increases with p11_t and q_t(slip=0)
    decreases with it, so the interval endpoints use, per step, the coupling
    endpoint that pushes q_t up (lower bound) or down (upper bound)."""
    lo_p11, hi_p11 = _coupling_box(p)
    q_max = [
        cf_slip_prob(s, p, hi_p11 if s else lo_p11) for s in observed_slips
    ]
    q_min = [
        cf_slip_prob(s, p, lo_p11 if s else hi_p11) for s in observed_slips
    ]
    return (
        success_prob(q_max, moves_needed),
        success_prob(q_min, moves_needed),
    )


def independence_point(
    observed_slips: list[int], p: float, moves_needed: int
) -> float:
    """The predictive answer: cross-action independence, q_t = p always.
    Note it does not depend on the observed slips at all: the predictor
    cannot condition on the episode's own counterfactual evidence."""
    return success_prob([p] * len(observed_slips), moves_needed)


def conditional_truth(
    observed_slips: list[int], p: float, p11: float, moves_needed: int
) -> float:
    """Ground truth under a known per-step coupling (for validation)."""
    qs = [cf_slip_prob(s, p, p11) for s in observed_slips]
    return success_prob(qs, moves_needed)


@dataclass
class CorridorWorld:
    """Two routes to a goal; per-step slip with hidden cross-route coupling.

    Route A (taken) and route B (counterfactual) each need ``route_len``
    successful moves within ``horizon`` steps. ``p11`` is the true coupling,
    known only to the simulator: the kernel sees the episode and p."""

    route_len: int = 6
    horizon: int = 9
    p_slip: float = 0.3
    p11: float = 0.3  # default: comonotone (= p_slip)

    def episode(self, rng: np.random.Generator) -> tuple[list[int], list[int]]:
        """Simulate one episode: returns (factual slips A, latent slips B)."""
        p, p11 = self.p_slip, self.p11
        a, b = [], []
        for _ in range(self.horizon):
            sa = rng.random() < p
            if sa:
                sb = rng.random() < p11 / p
            else:
                sb = rng.random() < (p - p11) / (1.0 - p)
            a.append(int(sa))
            b.append(int(sb))
        return a, b

    def factual_success(self, slips_a: list[int]) -> bool:
        return (len(slips_a) - sum(slips_a)) >= self.route_len

    def cf_success(self, slips_b: list[int]) -> bool:
        return (len(slips_b) - sum(slips_b)) >= self.route_len
