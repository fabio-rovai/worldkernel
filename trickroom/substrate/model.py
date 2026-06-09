"""SubstrateCostModel: typed-RDF substrate plugged into upstream Costable.

CEM-MPC calls get_cost(info_dict, action_candidates) -> Tensor[B, S].
We run the substrate forward H * action_block micro-steps per sample and
return final-distance-to-target as the cost.
"""

from __future__ import annotations

import numpy as np
import torch

from stable_worldmodel.wm.substrate.python_substrate import TwoRoomABox, rollout


class SubstrateCostModel:
    """Conforms structurally to stable_worldmodel.protocols.Costable.

    v0: pure-Python substrate engine via python_substrate.py.
    v1: subprocess JSON-RPC bridge to the Rust substrate (TODO).
    """

    def __init__(
        self,
        *,
        tbox_path: str | None = None,
        action_block: int = 5,
        agent_radius: float = 7.0,
        agent_speed: float = 5.0,
        wall_axis: int = 1,
        wall_thickness: int = 10,
        door_half_extents: tuple = (14, 14, 14),
        num_doors: int = 1,
    ) -> None:
        self.tbox_path = tbox_path                      # informational for v0
        self.action_block = action_block
        self._dyn_kwargs = dict(
            agent_radius=agent_radius,
            agent_speed=agent_speed,
            wall_axis=wall_axis,
            wall_thickness=wall_thickness,
            door_half_extents=door_half_extents,
            num_doors=num_doors,
        )

    def parameters(self):                               # for CEMSolver dtype probe
        return iter([torch.zeros(1, dtype=torch.float32)])

    def criterion(self, info_dict: dict) -> torch.Tensor:
        """Pure cost from already-rolled-out final positions in info_dict."""
        final = info_dict['_substrate_final_pos']       # (B, S, 2) np
        target = info_dict['_substrate_target_pos']     # (B, S, 2) np
        return torch.from_numpy(
            np.linalg.norm(final - target, axis=-1).astype(np.float32)
        )

    def get_cost(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        """Cost = final euclidean distance from agent to target after rollout.

        Upstream MegaWrapper provides:
          info_dict['observation']: (B, S, T, 10) — full Two-Room obs vector
          info_dict['goal_state']:  (B, S, T, 2)  — target xy (used for cost goal)
          action_candidates:        (B, S, H, action_block * 2)
        Returns: cost (B, S).
        """
        obs = info_dict['observation']
        if torch.is_tensor(obs):
            obs_np = obs.detach().cpu().numpy()
        else:
            obs_np = np.asarray(obs)
        obs_last = obs_np[..., -1, :]                                  # squeeze history T
        if obs_last.shape[-1] != 10:
            raise ValueError(f"expected 10-dim TwoRoom obs, got {obs_last.shape}")

        B, S = obs_last.shape[:2]
        abox = TwoRoomABox.from_state_vec(obs_last, **self._dyn_kwargs)

        # Override target_pos with the canonical goal_state from the wrapper
        # (obs[..., 2:4] is the agent's current view of the target; goal_state
        # is what the env was reset to as the goal).
        goal = info_dict.get('goal_state')
        if goal is not None:
            goal_np = goal.detach().cpu().numpy() if torch.is_tensor(goal) else np.asarray(goal)
            abox.target_pos = goal_np[..., -1, :].astype(np.float32, copy=True)

        acts = action_candidates.detach().cpu().numpy()                # (B,S,H, ab*2)
        H = acts.shape[2]
        micro = acts.reshape(B, S, H * self.action_block, 2)           # (B,S, H*ab, 2)

        final_abox = rollout(abox, micro)

        info_dict['_substrate_final_pos'] = final_abox.agent_pos       # (B, S, 2)
        info_dict['_substrate_target_pos'] = final_abox.target_pos     # (B, S, 2)
        return self.criterion(info_dict)
