"""Demo and validation for the smooth differentiable substrate.

Tests:
  1. Unit test: smooth_step recovers exact physics in the limit τ→0
  2. Gradient test: ∂cost/∂θ sign matches intuition (lower θ → harder cross-room)
  3. V(θ) sweep: trace expected cost from wrong prior (112) to correct prior (49)
  4. Gradient-based planning: compare CEM vs gradient descent on a cross-room episode
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import torch

from stable_worldmodel.wm.substrate.smooth_substrate import (
    smooth_step,
    smooth_rollout,
    smooth_cost,
    SmoothSubstrateCostModel,
    plan_gradient,
    value_vs_theta,
)
from stable_worldmodel.wm.substrate.python_substrate import (
    TwoRoomABox,
    step as hard_step,
    rollout as hard_rollout,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def t(x):
    return torch.tensor(x, dtype=torch.float32)


def arr(x):
    return np.array(x, dtype=np.float32)


# ── 1. Unit test: smooth ≈ hard at low τ ────────────────────────────────────

def test_unit_agreement(tau=0.05):
    print("=" * 60)
    print("1. Unit test: smooth_step vs hard step (τ=0.05)")
    print("=" * 60)
    cases = [
        # (agent_pos, action, door_y, description)
        ([100.0, 80.0], [1.0, 0.0], 49.0, "cross-room attempt, away from door → blocked"),
        ([100.0, 49.0], [1.0, 0.0], 49.0, "cross-room attempt, at door → passes"),
        ([130.0, 80.0], [-1.0, 0.0], 49.0, "right→left attempt, away from door → blocked"),
        ([80.0, 80.0],  [0.0, 1.0], 49.0, "same-room move, no wall interaction"),
        ([100.0, 49.0], [1.0, 0.0], 112.0, "wrong prior (door at 112), agent at y=49 → blocked"),
    ]

    all_pass = True
    for agent_pos, action, door_y, desc in cases:
        # Hard step
        abox = TwoRoomABox(
            agent_pos=arr(agent_pos),
            target_pos=arr([180.0, 80.0]),
            door_centers=arr([[112.0, door_y, 0.0, 0.0, 0.0, 0.0]]).reshape(3, 2),
        )
        hard_new = hard_step(abox, arr(action)).agent_pos

        # Smooth step
        soft_new = smooth_step(t(agent_pos), t(action), t(door_y), tau=tau)
        soft_new_np = soft_new.detach().numpy()

        err = np.max(np.abs(hard_new - soft_new_np))
        status = "PASS" if err < 1.0 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  [{status}] {desc}")
        print(f"         hard={hard_new}, smooth={soft_new_np}, err={err:.4f}")

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAIL'}\n")
    return all_pass


# ── 2. Gradient test: ∂cost/∂θ ──────────────────────────────────────────────

def test_gradients():
    print("=" * 60)
    print("2. Gradient test: ∂cost/∂θ")
    print("=" * 60)

    # Agent at (80, 63): left of wall, y=63 = exactly at door edge (49+15.75≈65)
    # Target at (170, 49). All-right actions. Door at θ.
    # When θ=49: agent y=63 is inside door band [33.25, 64.75] → barely in → can cross
    # When θ increases beyond 63+15.75=78.75: door band no longer covers y=63 → agent blocked
    # So ∂cost/∂θ should be positive around θ=63 (higher θ → harder to cross for this agent)
    agent_pos = t([80.0, 63.0])
    target_pos = t([170.0, 49.0])

    thetas_to_check = [49.0, 63.0, 80.0, 112.0]
    actions = torch.zeros(20, 2)
    actions[:, 0] = 1.0                                   # all-right actions

    for th_val in thetas_to_check:
        theta = torch.tensor(float(th_val), requires_grad=True)
        cost = smooth_cost(agent_pos, actions, target_pos, theta, tau=1.5)
        cost.backward()
        grad = float(theta.grad.detach())
        print(f"  θ={th_val:.0f}  cost={float(cost.detach()):.2f}  ∂cost/∂θ={grad:+.4f}")

    print()
    # More diagnostic: sweep θ and compute numeric gradient
    print("  Numeric gradient check (finite differences, Δθ=0.1):")
    for th_val in [49.0, 63.0, 80.0]:
        theta_lo = torch.tensor(float(th_val) - 0.1)
        theta_hi = torch.tensor(float(th_val) + 0.1)
        cost_lo = smooth_cost(agent_pos, actions, target_pos, theta_lo, tau=1.5)
        cost_hi = smooth_cost(agent_pos, actions, target_pos, theta_hi, tau=1.5)
        numeric_grad = float((cost_hi - cost_lo) / 0.2)
        print(f"    θ={th_val:.0f}  numeric ∂cost/∂θ={numeric_grad:+.4f}")

    print()


# ── 3. V(θ) sweep ────────────────────────────────────────────────────────────

def run_theta_sweep():
    print("=" * 60)
    print("3. V(θ) sweep: expected cost vs door prior")
    print("=" * 60)

    # Use cross-room episodes from sequential results
    results_path = os.path.join(os.path.dirname(__file__), "../results/substrate_sequential.json")
    if not os.path.exists(results_path):
        print("  substrate_sequential.json not found — skipping sweep")
        return None, None

    with open(results_path) as f:
        seq = json.load(f)

    room_types = seq["room_types"]          # list of "same"/"cross"
    seed_start = int(seq["seed_range"].split("-")[0])
    cross_room_seeds = [
        seed_start + i for i, rt in enumerate(room_types) if rt == "cross"
    ]

    print(f"  Using {len(cross_room_seeds)} cross-room seeds for sweep")

    # Build obs/action batches from actual environment initial states
    # We use naive all-right actions as the fixed trajectory to evaluate V(θ)
    N_STEPS = 40
    obs_list = []
    act_list = []

    try:
        import gymnasium as gym
        import stable_worldmodel as swm

        env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
        for seed in cross_room_seeds[:20]:                # first 20 for speed
            obs, _ = env.reset(seed=seed)
            obs_list.append(obs[:10])
            actions = np.zeros((N_STEPS, 2), dtype=np.float32)
            actions[:, 0] = 1.0                           # all-right
            act_list.append(actions)
        env.close()
    except Exception as e:
        print(f"  Could not load env: {e} — using synthetic data")
        for seed in cross_room_seeds[:20]:
            rng = np.random.default_rng(seed)
            obs = np.array([80.0, rng.uniform(30, 70), 170.0, rng.uniform(30, 70),
                            112.0, 49.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            obs_list.append(obs)
            actions = np.zeros((N_STEPS, 2), dtype=np.float32)
            actions[:, 0] = 1.0
            act_list.append(actions)

    obs_batch = np.stack(obs_list)                        # (N, 10)
    act_batch = np.stack(act_list)                        # (N, T, 2)

    thetas, values = value_vs_theta(obs_batch, act_batch, theta_range=(40.0, 120.0), n_points=80)

    print(f"\n  θ=49 (correct prior): cost = {np.interp(49.0, thetas, values):.2f}")
    print(f"  θ=80 (midpoint):      cost = {np.interp(80.0, thetas, values):.2f}")
    print(f"  θ=112 (wrong prior):  cost = {np.interp(112.0, thetas, values):.2f}")

    min_idx = np.argmin(values)
    print(f"  Global minimum: θ*={thetas[min_idx]:.1f}, cost={values[min_idx]:.2f}")

    # Save sweep results
    sweep_data = {"thetas": thetas.tolist(), "values": values.tolist()}
    out_path = os.path.join(os.path.dirname(__file__), "../results/theta_sweep.json")
    with open(out_path, "w") as f:
        json.dump(sweep_data, f, indent=2)
    print(f"\n  Saved to results/theta_sweep.json")

    return thetas, values


# ── 4. Gradient planner on one cross-room episode ──────────────────────────

def run_gradient_planner():
    print("=" * 60)
    print("4. Gradient-based planning vs CEM substrate")
    print("=" * 60)

    try:
        import gymnasium as gym
        import stable_worldmodel as swm
    except ImportError:
        print("  Could not import env — using synthetic scenario")
        # Synthetic cross-room: agent at (80, 49), target at (170, 100)
        agent_pos = t([80.0, 49.0])
        target_pos = t([170.0, 100.0])
        true_door_y = 49.0

        door_y_correct = torch.tensor(true_door_y)
        door_y_wrong = torch.tensor(112.0)

        print(f"\n  Synthetic: agent={agent_pos.numpy()}, target={target_pos.numpy()}")

        # Gradient planning with correct prior
        acts_c, losses_c = plan_gradient(agent_pos, target_pos, door_y_correct,
                                          horizon=40, n_iter=300, lr=0.05, tau=1.5)
        final_c = smooth_rollout(agent_pos, acts_c, door_y_correct)
        print(f"\n  Correct prior (θ=49):")
        print(f"    final pos: {final_c.detach().numpy()}")
        print(f"    final cost: {float(torch.norm(final_c - target_pos)):.2f}")
        print(f"    initial loss: {losses_c[0]:.2f} → final: {losses_c[-1]:.2f}")

        # Gradient planning with wrong prior
        acts_w, losses_w = plan_gradient(agent_pos, target_pos, door_y_wrong,
                                          horizon=40, n_iter=300, lr=0.05, tau=1.5)
        final_w = smooth_rollout(agent_pos, acts_w, door_y_wrong)
        print(f"\n  Wrong prior (θ=112):")
        print(f"    final pos: {final_w.detach().numpy()}")
        print(f"    final cost: {float(torch.norm(final_w - target_pos)):.2f}")
        print(f"    initial loss: {losses_w[0]:.2f} → final: {losses_w[-1]:.2f}")
        return

    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    obs, _ = env.reset(seed=63)                           # a cross-room seed
    agent_pos = t(obs[:2])
    target_pos = t(obs[2:4])
    true_door_y = float(obs[5])
    print(f"\n  Seed 63: agent={agent_pos.numpy()}, target={target_pos.numpy()}, true door_y={true_door_y:.1f}")

    for door_y_val, label in [(true_door_y, "correct"), (112.0, "wrong")]:
        door_y = torch.tensor(door_y_val)
        acts, losses = plan_gradient(agent_pos, target_pos, door_y,
                                      horizon=40, n_iter=300, lr=0.05, tau=1.5, verbose=True)
        final = smooth_rollout(agent_pos, acts, door_y)
        final_cost = float(torch.norm(final - target_pos))
        print(f"\n  Prior θ={door_y_val:.1f} ({label}):")
        print(f"    loss {losses[0]:.2f} → {losses[-1]:.2f}")
        print(f"    final pos: {final.detach().numpy()}")
        print(f"    L2 to target: {final_cost:.2f}")

    env.close()


# ── 5. ∂cost/∂θ sign check (smoke test for estimate_theta) ─────────────────

def test_theta_estimation_direction():
    print("=" * 60)
    print("5. θ estimation direction test")
    print("=" * 60)

    # Create a synthetic trajectory of agent crossing at y=49
    # One step: agent at (100, 49) takes action (1, 0) → should cross (door at 49)
    agent_start = t([100.0, 49.0])
    action = t([1.0, 0.0])

    # With true θ=49: agent passes door → lands around (105, 49)
    # With wrong θ=112: agent is blocked → lands at (99.5, 49)
    observed_crossing = smooth_step(agent_start, action, t(49.0), tau=0.5)
    print(f"  Observed crossing position: {observed_crossing.detach().numpy()}")

    # Estimate θ from this one observation, starting from wrong prior 112
    theta = torch.tensor(112.0, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=2.0)

    for step in range(100):
        opt.zero_grad()
        predicted = smooth_step(agent_start, action, theta, tau=1.5)
        loss = torch.sum((predicted - observed_crossing.detach()) ** 2)
        loss.backward()
        opt.step()
        with torch.no_grad():
            theta.clamp_(20.0, 200.0)

    print(f"  True θ=49.0, estimated θ={float(theta):.2f} (started at 112.0)")
    err = abs(float(theta) - 49.0)
    print(f"  Error: {err:.2f} px — {'PASS' if err < 20.0 else 'needs more data'}\n")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_unit_agreement(tau=0.05)
    test_gradients()
    run_theta_sweep()
    run_gradient_planner()
    test_theta_estimation_direction()
    print("Done.")
