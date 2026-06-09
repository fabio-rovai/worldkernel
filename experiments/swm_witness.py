"""Experiment 8: the witness inside a real world-model platform.

Target: stable-worldmodel (github.com/galilai-group/stable-worldmodel), the
"platform for reproducible world model research and evaluation". We embed the
off-diagonal witness in its TwoRoom environment and audit, on the platform's
own Gym API, exactly what any world model trained there can and cannot learn.

Construction. The agent picks between two canonical actions (left, right).
Each step a slip can occur (the action is nulled). Two worlds:

  World A (common cause):  ONE latent U ~ Bernoulli(p) per step; if U = 1 the
                           step slips whichever action is taken. Potential
                           slips are comonotone: P(slip_L, slip_R) = p.
  World B (independent):   each action has its own coin: coupling p^2.

Both give P(slip | do action) = p for every action, and the environment is
otherwise deterministic, so the transition law P(s' | s, a), i.e. THE ENTIRE
DISTRIBUTION any world model is trained on, is identical in A and B. This is
not an approximation claim; it holds by construction, and the script audits
it empirically on rollouts collected through the platform's own step() API.

The rung-3 query a planner actually cares about: "this step slipped; would it
also have slipped under the OTHER action?" (replan or wait?). Ground truth:
1.0 in world A, p in world B. The worlds' potential outcomes are logged by
the wrapper (never shown to the learner) so the truth is measured, not
asserted. The kernel computes the identified interval from rung-1/2 data and
the exact answer once the coupling is supplied; a world model fitted to the
rollouts, tabular or neural, answers with one number and collapses A and B.

Run from the stable-worldmodel environment:
  PYTHONPATH=/path/to/worldkernel/src .venv/bin/python experiments/swm_witness.py
Requires: pip install stable-worldmodel (or a clone) + gymnasium.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gymnasium  # noqa: E402
import stable_worldmodel  # noqa: E402,F401  (registers swm/ environments)

from worldkernel import TwoWorldKernel, frechet_pn_bounds  # noqa: E402

P_SLIP = 0.30
ACTIONS = {0: np.array([-1.0, 0.0]), 1: np.array([1.0, 0.0])}  # left, right
EPISODES = 250
STEPS = 40
SEED = 11


class CoupledSlip(gymnasium.Wrapper):
    """Slip noise with controlled CROSS-ACTION coupling.

    The marginal slip probability is p for every action in both modes, so the
    transition kernel (rungs 1-2) is mode-invariant. Only the joint law of
    the potential slips {slip_a} differs: 'common' couples them through one
    latent; 'independent' gives each action its own coin. The wrapper logs
    the full potential-outcome vector as ground truth for rung-3 scoring."""

    def __init__(self, env, p: float, coupling: str, seed: int):
        super().__init__(env)
        self.p, self.coupling = p, coupling
        self.rng = np.random.default_rng(seed)
        self.potential_log: list[dict[int, bool]] = []

    def step_with_choice(self, action_id: int):
        if self.coupling == "common":
            u = self.rng.random() < self.p
            slips = {a: u for a in ACTIONS}
        else:
            slips = {a: bool(self.rng.random() < self.p) for a in ACTIONS}
        self.potential_log.append(slips)
        act = np.zeros(2, dtype=np.float32) if slips[action_id] else ACTIONS[action_id]
        obs, r, term, trunc, info = self.env.step(act.astype(np.float32))
        return obs, slips[action_id], term or trunc


def rollout(coupling: str):
    env = CoupledSlip(
        gymnasium.make("swm/TwoRoom-v1"), P_SLIP, coupling, seed=SEED
    )
    pol = np.random.default_rng(SEED + 1)  # same policy stream in both worlds
    data = []  # (action_id, slipped)
    for ep in range(EPISODES):
        env.reset(seed=SEED + 1000 + ep)
        for _ in range(STEPS):
            a = int(pol.integers(0, 2))
            _, slipped, done = env.step_with_choice(a)
            data.append((a, slipped))
            if done:
                break
    return np.array(data, dtype=int), env.potential_log


def main() -> None:
    print(f"stable-worldmodel TwoRoom-v1, slip p = {P_SLIP}, "
          f"{EPISODES} episodes x {STEPS} steps per world\n")

    rates, truths, ns = {}, {}, {}
    for mode in ("common", "independent"):
        data, log = rollout(mode)
        ns[mode] = len(data)
        rates[mode] = [data[data[:, 0] == a, 1].mean() for a in (0, 1)]
        both = np.array([[s[0], s[1]] for s in log], dtype=int)
        truths[mode] = both[both[:, 0] == 1, 1].mean()  # P(slip_R | slip_L)

    print("RUNG 1-2 AUDIT (everything a world model trains on):")
    print(f"{'world':12s} | {'P(slip|do L)':>12} | {'P(slip|do R)':>12} | n steps")
    for mode in rates:
        print(f"{mode:12s} | {rates[mode][0]:>12.4f} | {rates[mode][1]:>12.4f} | {ns[mode]}")
    n_per = min(ns.values()) / 2  # steps per (world, action) cell
    # two-proportion comparison: each rate carries its own sampling noise
    se_diff = np.sqrt(2 * P_SLIP * (1 - P_SLIP) / n_per)
    gap = max(abs(rates["common"][i] - rates["independent"][i]) for i in (0, 1))
    print(f"max cross-world rate gap {gap:.4f} vs 2-sigma band for a difference "
          f"of proportions {2 * se_diff:.4f}: "
          f"{'INDISTINGUISHABLE' if gap < 2 * se_diff else 'distinguishable'}")
    print("(the env is otherwise deterministic: the slip rate IS the transition law,")
    print(" so identical rates = identical training distribution, by construction)\n")

    print("RUNG 3 (measured from the logged potential outcomes):")
    print("query: this step slipped under LEFT; would it also slip under RIGHT?")
    for mode in truths:
        print(f"  world {mode:12s}: P(slip_R | slip_L) = {truths[mode]:.3f}")

    print("\nTHE KERNEL:")
    lo, hi = frechet_pn_bounds(P_SLIP, P_SLIP)
    # P(slip_R=1 | slip_L=1) = p11 / p; Frechet box on p11 = [max(0,2p-1), p]
    c_lo, c_hi = max(0.0, 2 * P_SLIP - 1) / P_SLIP, 1.0
    print(f"  identified from rungs 1-2 alone: P(slip_R|slip_L) in [{c_lo:.3f}, {c_hi:.3f}]")
    a = TwoWorldKernel(P_SLIP, P_SLIP, p11=P_SLIP)          # comonotone
    b = TwoWorldKernel(P_SLIP, P_SLIP, p11=P_SLIP ** 2)     # independent
    print(f"  kernel, coupling=common:      {a.p11 / P_SLIP:.3f}   (truth {truths['common']:.3f})")
    print(f"  kernel, coupling=independent: {b.p11 / P_SLIP:.3f}   (truth {truths['independent']:.3f})")
    print(f"  (PN interval of the same pair, for the record: [{lo:.3f}, {hi:.3f}])")

    print("\nVERDICT: the two worlds give the platform identical training data, so")
    print("any world model trained on it, tabular or DINO-WM, answers the rung-3")
    print("query with ONE number and is wrong in at least one world. The kernel")
    print("exposes the unidentified interval and, handed the coupling, is exact in")
    print("both. The off-diagonal is invisible to the platform and load-bearing")
    print("for the planner.")


if __name__ == "__main__":
    main()
