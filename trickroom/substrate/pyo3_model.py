"""Pyo3SubstrateCostModel: native Python extension bridge to the Rust TwoRoom physics.

v2 of the substrate bridge.  Replaces the JSON-RPC subprocess (rust_model.py)
with a direct call into the compiled PyO3 extension module `tworoom_substrate`.
No serialisation overhead — Rust physics runs in-process via rayon batch
parallelism.  Should recover the ~91 ms/ep wall-clock of the numpy v0 while
preserving the "separate language, from ontological spec" uncontestability.

Build the extension once before using:
    cd $SUBSTRATE_RUST_DIR/tworoom_py
    VIRTUAL_ENV=<this_project>/.venv maturin develop --release
"""

from __future__ import annotations

import numpy as np
import torch

try:
    import tworoom_substrate as _lib
except ImportError as exc:
    raise ImportError(
        "tworoom_substrate native module not found. "
        "Build it with:\n"
        "  cd $SUBSTRATE_RUST_DIR/tworoom_py\n"
        "  VIRTUAL_ENV=<project>/.venv maturin develop --release"
    ) from exc


class Pyo3SubstrateCostModel:
    """Costable backed by the native Rust physics extension (tworoom_substrate).

    Conforms structurally to stable_worldmodel.protocols.Costable.
    Same get_cost(info_dict, action_candidates) -> Tensor[B, S] contract as
    SubstrateCostModel and RustSubstrateCostModel.

    No subprocess, no JSON serialisation — Rust physics runs in-process.
    """

    def __init__(
        self,
        *,
        agent_radius: float = 7.0,
        agent_speed: float = 5.0,
        wall_axis: int = 1,
        wall_thickness: int = 10,
        door_positions: list | None = None,
        door_half_extents: list | None = None,
        num_doors: int = 1,
        img_size: int = 224,
        border_size: int = 14,
        action_block: int = 5,
    ) -> None:
        self.action_block = action_block
        self._dynamics: dict = {
            "agent_radius":     float(agent_radius),
            "agent_speed":      float(agent_speed),
            "wall_axis":        int(wall_axis),
            "wall_thickness":   int(wall_thickness),
            "door_positions":   list(door_positions) if door_positions is not None else [112.0],
            "door_half_extents": list(door_half_extents) if door_half_extents is not None else [14.0],
            "num_doors":        int(num_doors),
            "img_size":         int(img_size),
            "border_size":      int(border_size),
        }

    def parameters(self):
        return iter([torch.zeros(1, dtype=torch.float32)])

    def get_cost(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        """Cost = final L2 distance from agent to target after rollout.

        Expects:
          info_dict['observation']: (B, S, T, 10)
          info_dict['goal_state']:  (B, S, T, 2)  (optional)
          action_candidates:        (B, S, H, action_block * 2)
        Returns: cost Tensor (B, S).
        """
        obs = info_dict["observation"]
        obs_np = obs.detach().cpu().numpy() if torch.is_tensor(obs) else np.asarray(obs)
        obs_last = obs_np[..., -1, :]  # (B, S, 10)
        if obs_last.shape[-1] != 10:
            raise ValueError(f"expected 10-dim TwoRoom obs, got {obs_last.shape}")

        B, S = obs_last.shape[:2]

        target_pos = obs_last[..., 2:4].astype(np.float32)  # (B, S, 2)
        goal = info_dict.get("goal_state")
        if goal is not None:
            goal_np = goal.detach().cpu().numpy() if torch.is_tensor(goal) else np.asarray(goal)
            target_pos = goal_np[..., -1, :].astype(np.float32)

        obs_flat = obs_last.reshape(B * S, 10).astype(np.float32)

        acts = action_candidates.detach().cpu().numpy()  # (B, S, H, ab*2)
        H = acts.shape[2]
        micro = acts.reshape(B, S, H * self.action_block, 2)
        micro_flat = micro.reshape(B * S, H * self.action_block, 2).astype(np.float32)

        # Pass numpy arrays directly — zero-copy buffer protocol via pyo3-numpy
        final_pos_raw = _lib.rollout_batch(obs_flat, micro_flat, **self._dynamics)
        final_pos_np = np.asarray(final_pos_raw, dtype=np.float32).reshape(B, S, 2)

        info_dict["_substrate_final_pos"] = final_pos_np
        info_dict["_substrate_target_pos"] = target_pos

        diff = final_pos_np - target_pos
        cost_np = np.linalg.norm(diff, axis=-1).astype(np.float32)
        return torch.from_numpy(cost_np)
