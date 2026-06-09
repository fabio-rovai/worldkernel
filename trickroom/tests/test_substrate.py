"""Sanity tests for v0 SubstrateCostModel.

Loads substrate modules via importlib to avoid triggering the upstream
package init (which pulls heavy training deps we don't need here).

Verifies:
1. get_cost runs end-to-end with shapes matching upstream CEM expectations.
2. Zero action keeps distance unchanged.
3. Door at agent's perpendicular coord lets agent cross the wall.
4. Closed wall (door elsewhere) clamps agent x at the effective wall edge.
5. Border clamp respects agent radius.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


def _load(name: str, relpath: str):
    p = Path(__file__).resolve().parents[2] / relpath
    spec = importlib.util.spec_from_file_location(name, p)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module                            # register before exec
    spec.loader.exec_module(module)
    return module


_ps = _load(
    'stable_worldmodel.wm.substrate.python_substrate',
    'stable_worldmodel/wm/substrate/python_substrate.py',
)
_model_mod = _load(
    'stable_worldmodel.wm.substrate.model',
    'stable_worldmodel/wm/substrate/model.py',
)

SubstrateCostModel = _model_mod.SubstrateCostModel
TwoRoomABox = _ps.TwoRoomABox
rollout = _ps.rollout
BORDER_SIZE = _ps.BORDER_SIZE
WALL_CENTER = _ps.WALL_CENTER


def _info_dict(agent_xy, target_xy, door_y=49):
    """Build an info_dict matching upstream MegaWrapper shapes.

    observation shape: (B=1, S=1, T=1, 10)
    goal_state shape:  (B=1, S=1, T=1, 2)
    """
    obs = np.zeros((1, 1, 1, 10), dtype=np.float32)
    obs[0, 0, 0, 0:2] = agent_xy
    obs[0, 0, 0, 2:4] = target_xy
    obs[0, 0, 0, 4:6] = [WALL_CENTER, door_y]
    goal = np.zeros((1, 1, 1, 2), dtype=np.float32)
    goal[0, 0, 0] = target_xy
    return {'observation': obs, 'goal_state': goal}


def test_get_cost_shape_and_dtype():
    model = SubstrateCostModel()
    info = _info_dict([60, 112], [164, 112])
    actions = torch.zeros(1, 1, 5, 10)
    cost = model.get_cost(info, actions)
    assert cost.shape == (1, 1)
    assert cost.dtype == torch.float32


def test_zero_action_keeps_distance_unchanged():
    model = SubstrateCostModel()
    info = _info_dict([60, 112], [164, 112])
    cost = model.get_cost(info, torch.zeros(1, 1, 5, 10))
    expected = np.linalg.norm(np.array([164 - 60, 0]))
    assert abs(float(cost[0, 0]) - expected) < 1e-3


def test_open_door_lets_agent_pass():
    info = _info_dict([60, 112], [164, 112], door_y=112)
    model = SubstrateCostModel()
    actions = torch.zeros(1, 1, 5, 10)
    actions[..., 0::2] = 1.0
    cost = model.get_cost(info, actions)
    expected_dist = abs((60 + 25 * 5) - 164)
    assert abs(float(cost[0, 0]) - expected_dist) < 1e-2, (
        f"Expected door pass to land 21px from target, got {float(cost[0, 0])}"
    )


def test_closed_wall_blocks_agent():
    info = _info_dict([60, 112], [164, 112], door_y=49)
    model = SubstrateCostModel(wall_thickness=10)
    actions = torch.zeros(1, 1, 5, 10)
    actions[..., 0::2] = 1.0
    cost = model.get_cost(info, actions)
    assert float(cost[0, 0]) > 60.0, (
        f"Expected agent blocked at wall, cost={float(cost[0, 0])}"
    )


def test_border_clamps_with_radius():
    abox = TwoRoomABox(
        agent_pos=np.array([[20.0, 112.0]], dtype=np.float32),
        target_pos=np.array([[164.0, 112.0]], dtype=np.float32),
        door_centers=np.array(
            [[[WALL_CENTER, 49], [0, 0], [0, 0]]], dtype=np.float32
        ),
    )
    actions = -np.ones((1, 1, 2), dtype=np.float32)
    out = rollout(abox, actions)
    assert abs(out.agent_pos[0, 0] - (BORDER_SIZE + 7.0)) < 1e-3
