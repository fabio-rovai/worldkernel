"""Kobs: Online prior estimator — infer door_y from observed episodes.

This is the Kobs knowledge source: given a stream of observed trajectories
(10-dim observations from TwoRoom), estimate the latent door-position prior
θ by gradient ascent on the smooth trajectory-replay loss.

After estimation, a KobsSubstrateCostModel is returned — a planning substrate
parameterised by the inferred θ rather than a hand-specified constant.

Usage
-----
    from stable_worldmodel.wm.substrate.obs_adapter import ObsAdapter

    adapter = ObsAdapter(init_theta=112.0)   # wrong prior as starting point
    adapter.observe(traj)                    # add a (T, 10) observation
    ...
    model = adapter.compile(n_steps=300)     # returns KobsSubstrateCostModel
    model.theta_hat                          # estimated door_y

Standalone demo
---------------
    python -m stable_worldmodel.wm.substrate.obs_adapter
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from stable_worldmodel.wm.substrate.smooth_substrate import estimate_theta
from stable_worldmodel.wm.substrate.code_compiler import (
    CodeCompiledSubstrateCostModel,
    compile_from_source,
)


# ── Online observation buffer ──────────────────────────────────────────────────

class ObsAdapter:
    """Accumulate observed trajectories and estimate door_y via gradient descent.

    Parameters
    ----------
    init_theta : float
        Starting prior for θ (door_y). Default 112 = wall centre (wrong).
    n_steps    : int
        Gradient steps for each estimation call.
    lr         : float
        Adam learning rate for θ.
    tau        : float
        Smoothness temperature passed to smooth_substrate.
    """

    def __init__(
        self,
        init_theta: float = 112.0,
        n_steps: int = 300,
        lr: float = 1.0,
        tau: float = 1.5,
    ):
        self.init_theta = init_theta
        self.n_steps    = n_steps
        self.lr         = lr
        self.tau        = tau
        self._trajectories: list[np.ndarray] = []

    def observe(self, trajectory: np.ndarray) -> None:
        """Add a (T, 10) observation trajectory.

        Parameters
        ----------
        trajectory : (T, 10) ndarray
            Sequence of 10-dim TwoRoom observations from a single episode.
            Must have T >= 2 to be informative.
        """
        traj = np.asarray(trajectory, dtype=np.float32)
        if traj.ndim != 2 or traj.shape[1] != 10:
            raise ValueError(f"Expected (T, 10) trajectory, got {traj.shape}")
        if len(traj) >= 2:
            self._trajectories.append(traj)

    def n_observed(self) -> int:
        return len(self._trajectories)

    def compile(
        self,
        n_steps: Optional[int] = None,
        verbose: bool = False,
    ) -> "KobsSubstrateCostModel":
        """Estimate θ from all collected trajectories and return a cost model.

        Parameters
        ----------
        n_steps : int, optional
            Override gradient steps (default: self.n_steps).
        verbose : bool
            Print θ and loss progress.

        Returns
        -------
        KobsSubstrateCostModel
            Drop-in Costable with door_y set to the estimated θ.
        """
        if not self._trajectories:
            warnings.warn("[Kobs] No trajectories observed — returning default θ.")
            return KobsSubstrateCostModel(theta_hat=self.init_theta)

        steps = n_steps if n_steps is not None else self.n_steps
        theta_hat, losses = estimate_theta(
            self._trajectories,
            init_theta=self.init_theta,
            n_steps=steps,
            lr=self.lr,
            tau=self.tau,
            verbose=verbose,
        )

        if verbose:
            print(f"[Kobs] Estimated θ = {theta_hat:.2f}  (init={self.init_theta})")

        return KobsSubstrateCostModel(theta_hat=theta_hat)


# ── Substrate parameterised by estimated θ ────────────────────────────────────

class KobsSubstrateCostModel(CodeCompiledSubstrateCostModel):
    """CodeCompiled substrate whose door_y comes from observed trajectories (Kobs).

    Inherits all physics from CodeCompiledSubstrateCostModel; the only
    difference is door_y is an estimate, not a static prior.
    """

    def __init__(self, theta_hat: float, **kwargs):
        super().__init__(door_y=theta_hat, **kwargs)
        self.theta_hat = theta_hat

    def __repr__(self) -> str:
        return f"KobsSubstrateCostModel(theta_hat={self.theta_hat:.2f})"


# ── Standalone demo ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import time
    import gymnasium as gym

    sys.path.insert(0, ".")
    warnings.filterwarnings("ignore")

    N_EPISODES   = 20        # episodes to collect before estimating
    INIT_THETA   = 112.0     # wrong prior (wall centre)
    TRUE_THETA   = 49.0      # ground-truth door_y
    SEED_START   = 42
    ENV_INIT     = {"wall.axis": 0}    # horizontal wall → door on x-axis y=49

    print("=" * 60)
    print(f"Kobs demo: collect {N_EPISODES} episodes, estimate θ from trajectories")
    print(f"  init θ = {INIT_THETA}  (wrong)   true θ = {TRUE_THETA}")
    print("=" * 60)

    adapter = ObsAdapter(init_theta=INIT_THETA, n_steps=400, lr=1.0)

    for ep in range(N_EPISODES):
        env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array",
                       max_episode_steps=100, init_value=ENV_INIT)
        obs, _ = env.reset(seed=SEED_START + ep)
        traj = [np.asarray(obs, dtype=np.float32)]

        done = False
        rng = np.random.default_rng(SEED_START + ep)
        while not done:
            action = rng.uniform(-1, 1, size=2)
            obs, _, terminated, truncated, _ = env.step(action)
            traj.append(np.asarray(obs, dtype=np.float32))
            done = terminated or truncated
        env.close()

        adapter.observe(np.stack(traj, axis=0))

    print(f"\nCollected {adapter.n_observed()} trajectories — running estimation...")
    t0 = time.time()
    model = adapter.compile(verbose=True)
    print(f"  Elapsed: {time.time()-t0:.1f}s")

    print(f"\n[Kobs] θ_hat = {model.theta_hat:.2f}   (true = {TRUE_THETA})")
    err = abs(model.theta_hat - TRUE_THETA)
    ok = err < 15.0
    print(f"  Error |θ_hat − θ_true| = {err:.2f}   {'PASS (< 15px)' if ok else 'FAIL'}")

    # Quick functional test: step an agent toward the wall and check the door zone
    # agent at (106,49), action right → should pass through door at y≈49
    s = np.array([[106, 49, 160, 49, 0, 49, 0, 0, 0, 0]], dtype=np.float32)
    a = np.array([[1.0, 0.0]], dtype=np.float32)
    ns = model._batch_step(s, a)
    ax_after = float(ns[0, 0])
    print(f"\n  Functional test (through door at θ_hat={model.theta_hat:.1f}): ax {106}→{ax_after:.1f}  "
          f"{'PASS' if ax_after > 107 else 'FAIL'}")
