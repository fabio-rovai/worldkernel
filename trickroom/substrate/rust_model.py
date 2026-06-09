"""RustSubstrateCostModel: JSON-RPC subprocess bridge to the Rust TwoRoom physics engine.

Spawns `tworoom_substrate_rpc` as a persistent child process and communicates
via newline-delimited JSON-RPC on its stdin/stdout.  Implements the same
get_cost interface as SubstrateCostModel.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import torch


class RustSubstrateCostModel:
    """Costable backed by the Rust substrate subprocess.

    Conforms structurally to stable_worldmodel.protocols.Costable.
    Same get_cost(info_dict, action_candidates) -> Tensor[B, S] contract.
    """

    _BINARY_NAME = "tworoom_substrate_rpc"

    @classmethod
    def _find_binary(cls) -> Path:
        # 1. Explicit location of the Rust substrate engine (private repo)
        import os

        rust_dir = os.environ.get("SUBSTRATE_RUST_DIR")
        if rust_dir:
            local = Path(rust_dir) / "target/release" / cls._BINARY_NAME
            if local.exists():
                return local
        # 2. Workspace-level shared-target (cargo config CARGO_TARGET_DIR)
        shared = Path.home() / ".cargo/shared-target/release" / cls._BINARY_NAME
        if shared.exists():
            return shared
        raise RuntimeError(
            "Rust substrate binary not found. Set SUBSTRATE_RUST_DIR to the "
            "Rust substrate engine checkout and run: "
            "cargo build --release --bin tworoom_substrate_rpc "
            "(the engine is not part of this repository; "
            "python_substrate.py is the self-contained fallback)"
        )

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
        self._dynamics = {
            "agent_radius": agent_radius,
            "agent_speed": agent_speed,
            "wall_axis": wall_axis,
            "wall_thickness": wall_thickness,
            "door_positions": door_positions if door_positions is not None else [112.0],
            "door_half_extents": door_half_extents if door_half_extents is not None else [14.0],
            "num_doors": num_doors,
            "img_size": img_size,
            "border_size": border_size,
        }

        binary_path = self._find_binary()
        self._binary_path = binary_path
        self._proc = subprocess.Popen(
            [str(binary_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
        )

        # Verify the process is alive with a ping
        result = self._send_rpc_method("ping", None)
        if result != "pong":
            raise RuntimeError(
                f"Rust substrate process did not respond to ping (got {result!r})"
            )

    def _check_alive(self) -> None:
        if self._proc.poll() is not None:
            raise RuntimeError(
                f"Rust substrate process died: check binary exists at {self._binary_path}"
            )

    def _send_rpc(self, params: dict) -> dict:
        """Send a rollout RPC and return the parsed result dict."""
        self._check_alive()
        req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "rollout",
            "params": params,
        }
        line = json.dumps(req) + "\n"
        assert self._proc.stdin is not None
        assert self._proc.stdout is not None
        self._proc.stdin.write(line.encode())
        self._proc.stdin.flush()
        response_line = self._proc.stdout.readline()
        resp = json.loads(response_line)
        if "error" in resp:
            raise RuntimeError(f"Rust RPC error: {resp['error']}")
        return resp["result"]

    def _send_rpc_method(self, method: str, params) -> object:
        """Send an arbitrary method RPC and return the parsed result."""
        self._check_alive()
        req = {"jsonrpc": "2.0", "id": 1, "method": method}
        if params is not None:
            req["params"] = params
        line = json.dumps(req) + "\n"
        assert self._proc.stdin is not None
        assert self._proc.stdout is not None
        self._proc.stdin.write(line.encode())
        self._proc.stdin.flush()
        response_line = self._proc.stdout.readline()
        resp = json.loads(response_line)
        if "error" in resp:
            raise RuntimeError(f"Rust RPC error: {resp['error']}")
        return resp["result"]

    def parameters(self):  # for CEMSolver dtype probe
        return iter([torch.zeros(1, dtype=torch.float32)])

    def get_cost(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        """Cost = final euclidean distance from agent to target after rollout.

        Expects:
          info_dict['observation']: (B, S, T, 10)
          info_dict['goal_state']:  (B, S, T, 2)  (optional)
          action_candidates:        (B, S, H, action_block * 2)
        Returns: cost Tensor (B, S).
        """
        obs = info_dict["observation"]
        if torch.is_tensor(obs):
            obs_np = obs.detach().cpu().numpy()
        else:
            obs_np = np.asarray(obs)

        # Squeeze history dimension T to get last observation
        obs_last = obs_np[..., -1, :]  # (B, S, 10)
        if obs_last.shape[-1] != 10:
            raise ValueError(f"expected 10-dim TwoRoom obs, got {obs_last.shape}")

        B, S = obs_last.shape[:2]

        # Extract agent pos from state[:, :, 0:2]
        agent_pos_init = obs_last[..., 0:2]  # (B, S, 2)

        # Extract target pos from state[:, :, 2:4] (overridden below if goal_state present)
        target_pos = obs_last[..., 2:4].astype(np.float32)  # (B, S, 2)

        # Override with canonical goal_state if provided
        goal = info_dict.get("goal_state")
        if goal is not None:
            goal_np = (
                goal.detach().cpu().numpy() if torch.is_tensor(goal) else np.asarray(goal)
            )
            target_pos = goal_np[..., -1, :].astype(np.float32)  # (B, S, 2)

        # Build flat state vectors: (B*S, 10)
        obs_flat = obs_last.reshape(B * S, 10).astype(np.float32)

        # Build action sequences: (B*S, H*ab, 2)
        acts = action_candidates.detach().cpu().numpy()  # (B, S, H, ab*2)
        H = acts.shape[2]
        micro = acts.reshape(B, S, H * self.action_block, 2)  # (B, S, H*ab, 2)
        micro_flat = micro.reshape(B * S, H * self.action_block, 2).astype(np.float32)

        # Serialize for RPC
        states_list: list[list[float]] = obs_flat.tolist()
        actions_list: list[list[list[float]]] = micro_flat.tolist()

        result = self._send_rpc(
            {
                "states": states_list,
                "actions": actions_list,
                "dynamics": self._dynamics,
            }
        )

        # Parse final_pos: list of [x, y], length B*S
        final_pos_np = np.array(result["final_pos"], dtype=np.float32)  # (B*S, 2)
        final_pos = final_pos_np.reshape(B, S, 2)

        # Store for downstream criterion use
        info_dict["_substrate_final_pos"] = final_pos
        info_dict["_substrate_target_pos"] = target_pos

        # L2 distance to target
        diff = final_pos - target_pos  # (B, S, 2)
        cost_np = np.linalg.norm(diff, axis=-1).astype(np.float32)  # (B, S)
        return torch.from_numpy(cost_np)

    def close(self) -> None:
        """Terminate the Rust subprocess."""
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
