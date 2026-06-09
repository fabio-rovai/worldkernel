"""Hybrid symbolic+neural substrate — K_spec ⊕ K_demo knowledge source.

Architecture: f̂_hybrid(s,a) = f̂_spec(s,a) + g_θ(s,a)

  f̂_spec  = compiled substrate from OWL TBox (deterministic, interpretable)
  g_θ     = residual MLP trained on demonstration data (B, S, 12) → (B, S, 2)

The residual MLP learns to correct systematic errors in f̂_spec that arise
from structural misspecification — e.g., wrong wall axis (vertical vs.
horizontal).  This demonstrates K_spec + K_demo > K_spec alone.

Training signal:
  δ(s,a) = s'[0:2] − f̂_spec(s,a).agent_pos   (position residual)

The door-encoding in the observation provides implicit wall-orientation
signal: vertical wall → obs[4]≈112, obs[5]=door_y; horizontal wall →
obs[4]=door_x, obs[5]≈112.  The MLP can read this without explicit wall_axis.

Usage:
    model = HybridSubstrateCostModel(weights_path="results/residual_mlp.pt")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from stable_worldmodel.wm.substrate.compiled_substrate import (
    TwoRoomABox,
    rollout as compiled_rollout,
    WALL_CENTER,
)


# ── Residual MLP ───────────────────────────────────────────────────────────────

class ResidualMLP(nn.Module):
    """Tiny MLP: (obs 10D + action 2D) → (Δagent_pos 2D).

    Learns to correct systematic errors in the compiled substrate.
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


# ── Hybrid step ────────────────────────────────────────────────────────────────

def hybrid_step(
    abox: TwoRoomABox,
    action: np.ndarray,
    residual_fn,          # callable(obs_np, action_np) → delta_pos_np
    obs_np: np.ndarray,   # current full 10-dim obs (..., 10)
) -> TwoRoomABox:
    """One hybrid step: compiled step + residual MLP correction."""
    import importlib
    compiled_mod = importlib.import_module(
        "stable_worldmodel.wm.substrate.compiled_substrate"
    )
    # Symbolic base step
    from stable_worldmodel.wm.substrate.compiled_substrate import step as compiled_step
    new_abox = compiled_step(abox, action)

    # Residual correction on agent_pos only
    delta = residual_fn(obs_np, action)
    new_abox.agent_pos = new_abox.agent_pos + delta

    # Reclamp to valid range (21, 203)
    new_abox.agent_pos = np.clip(new_abox.agent_pos, 21.0, 203.0)
    return new_abox


# ── Costable model ─────────────────────────────────────────────────────────────

GATE_MODES = frozenset(["none", "spatial", "crossing_intent", "axis_y_only"])
# none            : apply MLP everywhere (naive)
# spatial         : apply when |agent_y - 112| < gate_radius
# crossing_intent : spatial AND action_y moving toward y=112
# axis_y_only     : zero x-component of delta; apply y-component everywhere
#                   (physical rationale: x-dynamics identical across wall orientations;
#                    only y-dynamics are misspecified for horizontal wall)


class HybridSubstrateCostModel:
    """Hybrid K_spec + K_demo substrate conforming to the Costable protocol.

    Parameters
    ----------
    weights_path  : path to .pt file with ResidualMLP state_dict
    door_y        : door prior
    hidden        : MLP hidden size (must match trained model)
    gate_radius   : proximity threshold in pixels (used by spatial / crossing_intent)
    gate_mode     : one of GATE_MODES — controls how the MLP correction is masked
    """

    def __init__(
        self,
        weights_path: str | Path,
        door_y: float = 49.0,
        hidden: int = 64,
        gate_radius: float = 30.0,
        gate_mode: str = "spatial",
    ):
        if gate_mode not in GATE_MODES:
            raise ValueError(f"gate_mode must be one of {GATE_MODES}, got {gate_mode!r}")
        self.door_y = door_y
        self.gate_radius = gate_radius
        self.gate_mode = gate_mode
        self.mlp = ResidualMLP(hidden=hidden)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.mlp.load_state_dict(state)
        self.mlp.eval()

    def _residual(self, obs_np: np.ndarray, action_np: np.ndarray) -> np.ndarray:
        """Apply gated MLP residual. obs_np (..., 10), action_np (..., 2) → (..., 2)."""
        shape = obs_np.shape[:-1]
        obs_flat = obs_np.reshape(-1, 10)
        act_flat = action_np.reshape(-1, 2)
        with torch.no_grad():
            delta = self.mlp(
                torch.from_numpy(obs_flat),
                torch.from_numpy(act_flat),
            ).numpy()
        delta = delta.reshape(*shape, 2)

        mode = self.gate_mode
        agent_y  = obs_np[..., 1]       # (...,)
        action_y = action_np[..., 1]     # (...,)

        if mode == "none":
            pass  # apply everywhere

        elif mode == "spatial":
            near = np.abs(agent_y - WALL_CENTER) < self.gate_radius
            delta = delta * near[..., np.newaxis]

        elif mode == "crossing_intent":
            # Spatial gate AND action_y is moving toward y=112
            near     = np.abs(agent_y - WALL_CENTER) < self.gate_radius
            toward   = (action_y * (WALL_CENTER - agent_y)) > 0  # same sign
            mask     = near & toward
            delta    = delta * mask[..., np.newaxis]

        elif mode == "axis_y_only":
            # Zero x-component everywhere; keep y-component always.
            # Rationale: compiled substrate x-physics is correct for horiz wall
            # (no wall at x=112); only y-axis dynamics are misspecified.
            delta[..., 0] = 0.0

        return delta

    def parameters(self):
        return iter([torch.zeros(1, dtype=torch.float32)])

    def get_cost(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        """Costable.get_cost implementation.

        info_dict["observation"] : (B, S, T, 10)
        action_candidates        : (B, S, H, action_block * 2)
        Returns cost             : (B, S)
        """
        obs = info_dict["observation"]
        if hasattr(obs, "numpy"):
            obs = obs.numpy()
        if hasattr(action_candidates, "numpy"):
            action_candidates = action_candidates.numpy()

        obs  = np.asarray(obs,  dtype=np.float32)
        acts = np.asarray(action_candidates, dtype=np.float32)

        state = obs[..., -1, :]     # (B, S, 10)
        B, S, _ = state.shape

        # Override door prior
        state[..., 5] = self.door_y

        abox = TwoRoomABox.from_state_vec(state)

        _B, _S, H, ad = acts.shape
        micro = acts.reshape(B, S, H * (ad // 2), 2)  # (B, S, T, 2)
        T = micro.shape[-2]

        # Roll out T steps with hybrid step
        cur_abox = abox
        cur_obs  = state.copy()  # (B, S, 10)
        for t in range(T):
            act_t = micro[..., t, :]                   # (B, S, 2)
            delta = self._residual(cur_obs, act_t)     # (B, S, 2)
            cur_abox = _compiled_rollout_one(cur_abox, act_t)
            cur_abox.agent_pos = np.clip(
                cur_abox.agent_pos + delta, 21.0, 203.0
            )
            # Update obs for next residual call
            cur_obs = cur_obs.copy()
            cur_obs[..., 0:2] = cur_abox.agent_pos

        diff = cur_abox.agent_pos - cur_abox.target_pos
        cost = np.linalg.norm(diff, axis=-1).astype(np.float32)
        return torch.as_tensor(cost)


def _compiled_rollout_one(abox: TwoRoomABox, action: np.ndarray) -> TwoRoomABox:
    """One compiled step (imports locally to avoid circular dependency)."""
    from stable_worldmodel.wm.substrate.compiled_substrate import step
    return step(abox, action)


# ── Training utilities ─────────────────────────────────────────────────────────

def train_residual_mlp(
    demo_data: dict,
    hidden: int = 64,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 256,
    verbose: bool = True,
) -> ResidualMLP:
    """Train ResidualMLP on demonstration (s, a, s') tuples.

    demo_data keys: "obs" (N, 10), "action" (N, 2), "next_obs" (N, 10)

    Trains to predict: next_obs[0:2] - compiled_step(obs, action).agent_pos
    """
    import torch.optim as optim

    obs     = torch.tensor(demo_data["obs"],      dtype=torch.float32)
    action  = torch.tensor(demo_data["action"],   dtype=torch.float32)
    next_pos = torch.tensor(demo_data["next_obs"][:, 0:2], dtype=torch.float32)

    # Compute compiled substrate prediction (vectorised over batch)
    obs_np = demo_data["obs"].astype(np.float32)
    act_np = demo_data["action"].astype(np.float32)

    from stable_worldmodel.wm.substrate.compiled_substrate import (
        TwoRoomABox as CompABox, step as comp_step
    )

    N = len(obs_np)
    compiled_pos = np.zeros((N, 2), dtype=np.float32)
    for i in range(N):
        s = obs_np[i : i + 1]                                # (1, 10)
        abox = CompABox.from_state_vec(s)
        out  = comp_step(abox, act_np[i : i + 1])
        compiled_pos[i] = out.agent_pos[0]

    residual_target = next_pos - torch.tensor(compiled_pos, dtype=torch.float32)

    mlp = ResidualMLP(hidden=hidden)
    opt = optim.Adam(mlp.parameters(), lr=lr)
    dataset = torch.utils.data.TensorDataset(obs, action, residual_target)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0
        for ob, ac, tgt in loader:
            pred = mlp(ob, ac)
            loss = nn.functional.mse_loss(pred, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(ob)
        if verbose and (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch+1}/{epochs}  loss={total_loss/N:.6f}")

    return mlp
