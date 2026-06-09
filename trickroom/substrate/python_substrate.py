"""v0 substrate engine: literal port of Two-Room dynamics into a typed ABox.

Mirrors stable_worldmodel/envs/two_room/env.py step() + _apply_collisions().
Vectorised over (batch, samples) with NumPy. No learned weights.

When the Rust substrate at the private Rust substrate engine (set SUBSTRATE_RUST_DIR)
is plumbed via JSON-RPC, this module becomes the spec the bridge must match.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


IMG_SIZE = 224
BORDER_SIZE = 14
WALL_CENTER = 112
MAX_DOOR = 3


@dataclass
class TwoRoomABox:
    """Initial conditions for one or many parallel Two-Room scenes.

    Arrays are shaped (..., d). Leading dims propagate through rollout.
    """

    agent_pos: np.ndarray              # (..., 2)
    target_pos: np.ndarray             # (..., 2)
    door_centers: np.ndarray           # (..., MAX_DOOR, 2)  zeros for unused
    agent_radius: float = 7.0
    agent_speed: float = 5.0
    wall_axis: int = 1                 # 0 horizontal, 1 vertical
    wall_thickness: int = 10
    door_half_extents: tuple = (14, 14, 14)
    num_doors: int = 1

    @classmethod
    def from_state_vec(
        cls,
        state: np.ndarray,
        *,
        agent_radius: float = 7.0,
        agent_speed: float = 5.0,
        wall_axis: int = 1,
        wall_thickness: int = 10,
        door_half_extents: tuple = (14, 14, 14),
        num_doors: int = 1,
    ) -> 'TwoRoomABox':
        """Build an ABox batch from upstream's 10-dim observation vector.

        state[..., 0:2]   agent position
        state[..., 2:4]   target position
        state[..., 4:10]  three (x, y) door centers
        """
        assert state.shape[-1] == 10, state.shape
        return cls(
            agent_pos=state[..., 0:2].astype(np.float32, copy=True),
            target_pos=state[..., 2:4].astype(np.float32, copy=True),
            door_centers=state[..., 4:10].reshape(*state.shape[:-1], MAX_DOOR, 2).astype(np.float32, copy=True),
            agent_radius=agent_radius,
            agent_speed=agent_speed,
            wall_axis=wall_axis,
            wall_thickness=wall_thickness,
            door_half_extents=door_half_extents,
            num_doors=num_doors,
        )


def _in_any_door_1d(coord: np.ndarray, door_centers_1d: np.ndarray, half_extents: np.ndarray, num_doors: int, margin: float) -> np.ndarray:
    """Vectorised door-membership test. coord shape (...). Returns bool (...)."""
    out = np.zeros_like(coord, dtype=bool)
    for i in range(num_doors):
        c = door_centers_1d[..., i]
        s = half_extents[i]
        out |= (coord >= c - s - margin) & (coord <= c + s + margin)
    return out


def step(abox: TwoRoomABox, action: np.ndarray) -> TwoRoomABox:
    """Apply one env step: MoveProposed + CollideBorder + CollideWall.

    Vectorised: action shape (..., 2), agent_pos shape (..., 2). Other ABox
    fields (radius, speed, wall geom, door geom) are scalars for now; the
    factors-of-variation OOD eval will lift them to per-sample arrays.
    """
    action = np.clip(action, -1.0, 1.0)
    proposed = abox.agent_pos + action * abox.agent_speed

    # CollideBorder
    bs = BORDER_SIZE
    r = abox.agent_radius
    proposed = np.clip(proposed, bs + r, IMG_SIZE - bs - r)

    # CollideWall
    half = abox.wall_thickness // 2
    c = WALL_CENTER
    effective_low = c - half - r
    effective_high = c + half + r
    door_margin = 1.75
    half_extents = np.asarray(abox.door_half_extents, dtype=np.float32)

    if abox.wall_axis == 1:                                # vertical wall
        x_start = abox.agent_pos[..., 0]
        x_prop = proposed[..., 0]
        y_prop = proposed[..., 1]
        door_1d = abox.door_centers[..., :, 1]             # door y coords
        in_door = _in_any_door_1d(y_prop, door_1d, half_extents, abox.num_doors, door_margin)
        started_left = x_start < c
        push_left = started_left & (x_prop > effective_low) & ~in_door
        push_right = (~started_left) & (x_prop < effective_high) & ~in_door
        new_x = np.where(push_left, effective_low - 0.5,
                np.where(push_right, effective_high + 0.5, x_prop))
        new_y = y_prop
    else:                                                  # horizontal wall
        y_start = abox.agent_pos[..., 1]
        y_prop = proposed[..., 1]
        x_prop = proposed[..., 0]
        door_1d = abox.door_centers[..., :, 0]
        in_door = _in_any_door_1d(x_prop, door_1d, half_extents, abox.num_doors, door_margin)
        started_top = y_start < c
        push_top = started_top & (y_prop > effective_low) & ~in_door
        push_bottom = (~started_top) & (y_prop < effective_high) & ~in_door
        new_y = np.where(push_top, effective_low - 0.5,
                np.where(push_bottom, effective_high + 0.5, y_prop))
        new_x = x_prop

    new_pos = np.stack([new_x, new_y], axis=-1).astype(np.float32)

    return TwoRoomABox(
        agent_pos=new_pos,
        target_pos=abox.target_pos,
        door_centers=abox.door_centers,
        agent_radius=abox.agent_radius,
        agent_speed=abox.agent_speed,
        wall_axis=abox.wall_axis,
        wall_thickness=abox.wall_thickness,
        door_half_extents=abox.door_half_extents,
        num_doors=abox.num_doors,
    )


def rollout(abox: TwoRoomABox, actions: np.ndarray) -> TwoRoomABox:
    """Run a sequence of micro-actions. actions shape (..., T, 2)."""
    T = actions.shape[-2]
    cur = abox
    for t in range(T):
        cur = step(cur, actions[..., t, :])
    return cur
