"""PushT substrate: K_spec ⊕ K_demo demonstration.

Illustrates a *structurally incomplete* Kspec — the OWL TBox formalises
agent kinematics but has no rule for block contact physics (pymunk rigid-body
dynamics are not axiomatisable in OWL DL).

Architecture
------------
K_spec alone   : f̂_spec(s,a) — agent PD kinematics, block stays fixed
K_spec ⊕ K_demo: f̂_spec(s,a) + g_θ(s,a) — MLP residual learns block dynamics

State vector (7D):
    [agent_x, agent_y, block_x, block_y, block_angle, agent_vx, agent_vy]

Action (2D): continuous relative velocity command ∈ [-1, 1]²
    effective target = agent_pos + action * action_scale (=100)

Cost:
    block_pos_err = ||block_pos_final - goal_block_pos||
    angle_err     = |block_angle_final - goal_angle| (wrapped)
    cost          = block_pos_err + W_ANGLE * angle_err
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ── PushT physics constants (from envs/pusht/env.py) ──────────────────────────
WINDOW_SIZE  = 512.0
ACTION_SCALE = 100.0
K_P          = 100.0   # PD proportional gain
K_V          = 20.0    # PD velocity gain
DT           = 0.01    # physics dt
N_PHYS       = 10      # physics steps per control step (=int(1/(DT*control_hz)))
W_ANGLE      = 30.0    # angle error weight (pixels equivalent, ~1 body width)

# Success thresholds (from PushT.eval_state)
POS_SUCCESS_THRESH   = 20.0
ANGLE_SUCCESS_THRESH = np.pi / 9


# ── Agent dynamics ─────────────────────────────────────────────────────────────

def _step_agent_batch(
    pos: np.ndarray,   # (..., 2)
    vel: np.ndarray,   # (..., 2)
    action: np.ndarray,  # (..., 2) in [-1,1]
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate one control step (N_PHYS physics steps) of agent PD controller.

    Returns new (pos, vel).  Block contact is ignored (Kspec = incomplete).
    """
    target = pos + action * ACTION_SCALE
    p, v = pos.copy(), vel.copy()
    for _ in range(N_PHYS):
        accel = K_P * (target - p) + K_V * (0.0 - v)
        v = v + accel * DT
        p = p + v * DT
    p = np.clip(p, 0.0, WINDOW_SIZE)
    return p, v


# ── Kspec substrate (agent kinematics only, block static) ─────────────────────

class PushTCompiledSubstrate:
    """Planning substrate compiled from the PushT agent-kinematics TBox.

    Kspec models agent PD dynamics exactly; block state is held fixed
    (OWL TBox has no axiom for contact physics).

    Costable interface: get_cost(info_dict, action_candidates) → torch.Tensor
    """

    def get_cost(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        """Costable.get_cost implementation.

        info_dict["state"]      : (B, S, T, 7)  — state history
        info_dict["goal_state"] : (B, S, T, 7) or (B, S, 7) — goal state
        action_candidates       : (B, S, H, 2)   — action sequences
        Returns cost            : (B, S)
        """
        state_arr = info_dict["state"]
        goal_arr  = info_dict["goal_state"]

        if hasattr(state_arr, "numpy"):
            state_arr = state_arr.numpy()
        if hasattr(goal_arr, "numpy"):
            goal_arr = goal_arr.numpy()
        if hasattr(action_candidates, "numpy"):
            action_candidates = action_candidates.numpy()

        state = np.asarray(state_arr, dtype=np.float64)
        goal  = np.asarray(goal_arr,  dtype=np.float64)
        acts  = np.asarray(action_candidates, dtype=np.float64)

        # Last timestep in history; goal_state is constant but may also have T dim
        s = state[..., -1, :]   # (B, S, 7)
        g = goal[..., -1, :] if goal.ndim == state.ndim else goal  # (B, S, 7)
        B, Ss, _ = s.shape

        # acts: (B, S, H, action_block * 2) — flatten to (B, S, T, 2)
        _B, _S, H, ad = acts.shape
        micro = acts.reshape(B, Ss, H * (ad // 2), 2)  # (B, S, T, 2)
        T = micro.shape[-2]

        agent_pos = s[..., 0:2].copy()  # (B, S, 2)
        agent_vel = s[..., 5:7].copy()  # (B, S, 2)
        block_pos = s[..., 2:4].copy()  # (B, S, 2) — static in Kspec
        block_ang = s[..., 4  ].copy()  # (B, S)    — static in Kspec

        for t in range(T):
            act = micro[..., t, :]  # (B, S, 2)
            agent_pos, agent_vel = _step_agent_batch(agent_pos, agent_vel, act)

        goal_block_pos = g[..., 2:4]  # (B, S, 2)
        goal_block_ang = g[..., 4]    # (B, S)

        pos_err   = np.linalg.norm(block_pos - goal_block_pos, axis=-1)
        raw_diff  = block_ang - goal_block_ang
        ang_err   = np.abs(((raw_diff + np.pi) % (2 * np.pi)) - np.pi)
        cost      = pos_err + W_ANGLE * ang_err

        return torch.as_tensor(cost, dtype=torch.float32)

    def parameters(self):
        return iter([torch.zeros(1, dtype=torch.float32)])


# ── Residual MLP for block dynamics ───────────────────────────────────────────

class PushTResidualMLP(nn.Module):
    """Predict block state change given (state, action).

    Input  : (7 + 2) = 9D
    Output : (3D) = [Δblock_x, Δblock_y, Δblock_angle]
    """

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


# ── Hybrid substrate ───────────────────────────────────────────────────────────

class PushTHybridSubstrate:
    """K_spec ⊕ K_demo hybrid: agent kinematics (compiled) + block MLP.

    The MLP is trained on demonstration data to predict block dynamics.
    This is the 'structurally incomplete spec' showcase: Kspec is formally
    correct for the agent but silent on block-contact physics; Kdemo fills it.

    Costable interface: get_cost(info_dict, action_candidates) → torch.Tensor
    """

    def __init__(self, weights_path: str | Path, hidden: int = 128):
        self.mlp = PushTResidualMLP(hidden=hidden)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.mlp.load_state_dict(state)
        self.mlp.eval()

    def _block_residual(
        self, state_np: np.ndarray, action_np: np.ndarray
    ) -> np.ndarray:
        """Predict [Δblock_x, Δblock_y, Δblock_angle]. Shape (..., 3)."""
        shape = state_np.shape[:-1]
        s_flat = state_np.reshape(-1, 7).astype(np.float32)
        a_flat = action_np.reshape(-1, 2).astype(np.float32)
        with torch.no_grad():
            delta = self.mlp(
                torch.from_numpy(s_flat), torch.from_numpy(a_flat)
            ).numpy()
        return delta.reshape(*shape, 3)

    def get_cost(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        """Costable.get_cost implementation (hybrid rollout).

        info_dict["state"]      : (B, S, T, 7)
        info_dict["goal_state"] : (B, S, T, 7) or (B, S, 7) — goal state
        action_candidates       : (B, S, H, 2)
        Returns cost            : (B, S)
        """
        state_arr = info_dict["state"]
        goal_arr  = info_dict["goal_state"]

        if hasattr(state_arr, "numpy"):
            state_arr = state_arr.numpy()
        if hasattr(goal_arr, "numpy"):
            goal_arr = goal_arr.numpy()
        if hasattr(action_candidates, "numpy"):
            action_candidates = action_candidates.numpy()

        state = np.asarray(state_arr, dtype=np.float64)
        goal  = np.asarray(goal_arr,  dtype=np.float64)
        acts  = np.asarray(action_candidates, dtype=np.float64)

        s = state[..., -1, :]   # (B, S, 7)
        g = goal[..., -1, :] if goal.ndim == state.ndim else goal  # (B, S, 7)
        B, Ss, _ = s.shape

        # acts: (B, S, H, action_block * 2) — flatten to (B, S, T, 2)
        _B, _S, H, ad = acts.shape
        micro = acts.reshape(B, Ss, H * (ad // 2), 2)  # (B, S, T, 2)
        T = micro.shape[-2]

        agent_pos = s[..., 0:2].copy().astype(np.float64)
        agent_vel = s[..., 5:7].copy().astype(np.float64)
        block_pos = s[..., 2:4].copy().astype(np.float64)
        block_ang = s[..., 4  ].copy().astype(np.float64)

        cur_state = s.copy()

        for t in range(T):
            act = micro[..., t, :]   # (B, S, 2)
            # Agent kinematics (Kspec)
            agent_pos, agent_vel = _step_agent_batch(
                agent_pos, agent_vel, act
            )
            # Block dynamics residual (Kdemo)
            delta = self._block_residual(cur_state, act.astype(np.float32))  # (..., 3)
            block_pos = block_pos + delta[..., 0:2]
            block_ang = block_ang + delta[..., 2]
            # Clamp block to window
            block_pos = np.clip(block_pos, 0.0, WINDOW_SIZE)
            # Update cur_state for next residual call
            cur_state = cur_state.copy()
            cur_state[..., 0:2] = agent_pos
            cur_state[..., 2:4] = block_pos
            cur_state[..., 4  ] = block_ang
            cur_state[..., 5:7] = agent_vel

        goal_block_pos = g[..., 2:4]
        goal_block_ang = g[..., 4]

        pos_err  = np.linalg.norm(block_pos - goal_block_pos, axis=-1)
        raw_diff = block_ang - goal_block_ang
        ang_err  = np.abs(((raw_diff + np.pi) % (2 * np.pi)) - np.pi)
        cost     = pos_err + W_ANGLE * ang_err

        return torch.as_tensor(cost, dtype=torch.float32)

    def parameters(self):
        return iter([torch.zeros(1, dtype=torch.float32)])


# ── Training ──────────────────────────────────────────────────────────────────

def train_pusht_mlp(
    demo_data: dict,
    hidden: int = 128,
    epochs: int = 300,
    lr: float = 3e-4,
    batch_size: int = 512,
    verbose: bool = True,
) -> PushTResidualMLP:
    """Train PushTResidualMLP on (state, action, next_state) tuples.

    demo_data keys:
        "state"      : (N, 7)  — current state
        "action"     : (N, 2)  — applied action
        "next_state" : (N, 7)  — next state

    Trains to predict next_state[2:5] - state[2:5] (block pose delta).
    Agent dynamics are handled by Kspec; only block residual is learned.
    """
    import torch.optim as optim

    obs   = torch.tensor(demo_data["state"],      dtype=torch.float32)
    act   = torch.tensor(demo_data["action"],     dtype=torch.float32)
    ns    = torch.tensor(demo_data["next_state"], dtype=torch.float32)

    # Block pose residual target
    target = ns[:, 2:5] - obs[:, 2:5]  # (N, 3): [Δbx, Δby, Δba]

    mlp = PushTResidualMLP(hidden=hidden)
    opt = optim.Adam(mlp.parameters(), lr=lr)
    dataset = torch.utils.data.TensorDataset(obs, act, target)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    N = len(obs)
    for epoch in range(epochs):
        total = 0.0
        for ob, ac, tgt in loader:
            pred = mlp(ob, ac)
            loss = nn.functional.mse_loss(pred, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(ob)
        if verbose and (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch+1}/{epochs}  loss={total/N:.6f}")

    return mlp
