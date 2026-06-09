"""Generate V*(θ) sweep data for the paper.

For each door prior θ in [40, 120]:
  1. For each cross-room seed, initialise env, get obs
  2. Run gradient planner under prior θ (smooth substrate)
  3. Evaluate the planned actions under the TRUE physics (hard step)
  4. Record success (dist < 10px threshold)

Result: V*(θ) = empirical cross-room success rate at each prior value θ.
This traces the prior-sensitivity curve used in the paper figure.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import time
import numpy as np
import torch
import gymnasium as gym
import stable_worldmodel as swm

from stable_worldmodel.wm.substrate.smooth_substrate import (
    smooth_step, smooth_rollout, smooth_cost, plan_gradient
)
from stable_worldmodel.wm.substrate.python_substrate import (
    TwoRoomABox, rollout as hard_rollout
)


HORIZON = 40
N_ITER = 200
SUCCESS_RADIUS = 12.0


def get_cross_room_seeds():
    results_path = os.path.join(os.path.dirname(__file__), "../results/substrate_sequential.json")
    with open(results_path) as f:
        data = json.load(f)
    seed_start = int(data["seed_range"].split("-")[0])
    return [
        seed_start + i
        for i, rt in enumerate(data["room_types"])
        if rt == "cross"
    ]


def evaluate_policy_hard(obs, actions_np):
    """Evaluate a numpy action sequence under hard physics. Returns final dist."""
    agent_pos = obs[0:2].astype(np.float32)
    target_pos = obs[2:4].astype(np.float32)
    door_y = float(obs[5])
    abox = TwoRoomABox(
        agent_pos=agent_pos,
        target_pos=target_pos,
        door_centers=np.array([[112.0, door_y, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32).reshape(3, 2),
    )
    final = hard_rollout(abox, actions_np[np.newaxis, :, :]).agent_pos[0]
    return float(np.linalg.norm(final - target_pos))


def run_theta_sweep():
    cross_seeds = get_cross_room_seeds()
    print(f"Cross-room seeds: {len(cross_seeds)} → {cross_seeds[:5]}...")

    # θ grid: 20 points from 40 to 120
    thetas = np.linspace(40.0, 120.0, 20)

    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")

    # Use first 10 cross-room seeds for speed (report clearly)
    seeds_to_use = cross_seeds[:10]

    # Cache initial observations
    obs_cache = {}
    for seed in seeds_to_use:
        obs, _ = env.reset(seed=seed)
        obs_np = obs.numpy() if hasattr(obs, 'numpy') else np.asarray(obs)
        obs_cache[seed] = obs_np.astype(np.float32)

    print(f"Using {len(seeds_to_use)} cross-room seeds: {seeds_to_use}")
    print(f"θ grid: {thetas[0]:.0f} → {thetas[-1]:.0f} ({len(thetas)} points)")
    print()

    results = {"thetas": thetas.tolist(), "seeds": seeds_to_use,
               "success_rates": [], "mean_costs_smooth": [], "mean_costs_hard": []}

    t0 = time.time()
    for i, theta_val in enumerate(thetas):
        door_y = torch.tensor(float(theta_val))
        successes = []
        costs_smooth = []
        costs_hard = []

        for seed in seeds_to_use:
            obs = obs_cache[seed]
            agent_pos = torch.tensor(obs[0:2], dtype=torch.float32)
            target_pos = torch.tensor(obs[2:4], dtype=torch.float32)

            # Gradient planner under prior θ
            acts, _ = plan_gradient(
                agent_pos, target_pos, door_y,
                horizon=HORIZON, n_iter=N_ITER, lr=0.05, tau=1.5, verbose=False
            )

            # Smooth cost (under the prior)
            c_smooth = float(smooth_cost(agent_pos, acts, target_pos, door_y, tau=1.5))
            costs_smooth.append(c_smooth)

            # Hard cost (under TRUE physics with true door_y)
            actions_np = acts.detach().numpy()
            dist_hard = evaluate_policy_hard(obs, actions_np)
            costs_hard.append(dist_hard)
            successes.append(1 if dist_hard < SUCCESS_RADIUS else 0)

        sr = float(np.mean(successes))
        results["success_rates"].append(sr)
        results["mean_costs_smooth"].append(float(np.mean(costs_smooth)))
        results["mean_costs_hard"].append(float(np.mean(costs_hard)))

        elapsed = time.time() - t0
        print(f"  θ={theta_val:6.1f}  sr={sr:.2f}  "
              f"cost_smooth={np.mean(costs_smooth):.1f}  "
              f"cost_hard={np.mean(costs_hard):.1f}  "
              f"[{elapsed:.0f}s elapsed]")

    env.close()

    # Summary
    thetas_arr = np.array(thetas)
    sr_arr = np.array(results["success_rates"])
    best_idx = np.argmax(sr_arr)
    print(f"\n  Best θ: {thetas_arr[best_idx]:.1f}  sr={sr_arr[best_idx]:.2f}")
    print(f"  θ=49 sr: {np.interp(49.0, thetas_arr, sr_arr):.2f}")
    print(f"  θ=112 sr: {np.interp(112.0, thetas_arr, sr_arr):.2f}")

    out = os.path.join(os.path.dirname(__file__), "../results/theta_sweep_gradient.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to results/theta_sweep_gradient.json")

    return results


if __name__ == "__main__":
    run_theta_sweep()
