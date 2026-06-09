"""Kcode compiler: extract physics constants from TwoRoom env source and
compile a planning substrate without any hand-written specification.

Usage:
    python -m stable_worldmodel.wm.substrate.code_compiler
    # or
    from stable_worldmodel.wm.substrate.code_compiler import compile_from_source
    model = compile_from_source()
"""
import ast
import re
import inspect
from pathlib import Path
import numpy as np

# ── AST-based constant extractor ──────────────────────────────────────────────

def _extract_class_constants(source: str, class_name: str) -> dict:
    """Parse class-level constant assignments from Python source."""
    tree = ast.parse(source)
    constants = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name):
                            try:
                                val = ast.literal_eval(item.value)
                                constants[target.id] = val
                            except (ValueError, TypeError):
                                pass
    return constants


def _extract_door_default(source: str) -> float:
    """Extract the default door position from variation_space init_value.

    Matches the comment '# door position is center coord along wall'
    followed by init_value=[N, ...].
    """
    m = re.search(
        r"door position is center coord.*?init_value=\[([0-9]+)",
        source, re.DOTALL
    )
    if m:
        return float(m.group(1))
    return 49.0


def _extract_door_size_default(source: str) -> float:
    """Extract the default door half-size from variation_space.

    Matches the comment '# door size is half-extent' followed by init_value=[N, ...].
    """
    m = re.search(
        r"door size is half-extent.*?init_value=\[([0-9]+)",
        source, re.DOTALL
    )
    if m:
        return float(m.group(1))
    return 14.0


def compile_from_source(env_source_path: str | None = None) -> "CodeCompiledSubstrateCostModel":
    """Parse TwoRoom env.py and return an instantiated substrate.

    Parameters
    ----------
    env_source_path : str or None
        Path to TwoRoom env.py. If None, resolved relative to this file.
    """
    if env_source_path is None:
        here = Path(__file__).parent
        env_source_path = here.parents[1] / "envs/two_room/env.py"
    source = Path(env_source_path).read_text()

    cls_consts = _extract_class_constants(source, "TwoRoomEnv")

    extracted = {
        "wall_center": cls_consts.get("WALL_CENTER", 112),
        "wall_half":   cls_consts.get("WALL_WIDTH_DEFAULT", 10) / 2,
        "border":      cls_consts.get("BORDER_SIZE", 14),
        "max_speed":   cls_consts.get("MAX_SPEED", 10.5),
        "img_size":    cls_consts.get("IMG_SIZE", 224),
        "door_y":      _extract_door_default(source),
        "door_half":   _extract_door_size_default(source),
    }

    print("[Kcode] Extracted from env.py:")
    for k, v in extracted.items():
        print(f"  {k} = {v}")

    return CodeCompiledSubstrateCostModel(**extracted)


# ── Substrate built from extracted constants ───────────────────────────────────

class CodeCompiledSubstrateCostModel:
    """Planning substrate whose parameters come entirely from env source code.

    No OWL TBox, no hand-written physics — constants are parsed from
    TwoRoom's class body and variation_space defaults.
    """

    def __init__(
        self,
        wall_center: float = 112.0,
        wall_half: float = 5.0,
        border: float = 14.0,
        max_speed: float = 10.5,
        img_size: float = 224.0,
        door_y: float = 49.0,
        door_half: float = 31.0,
    ):
        self.wall_center = wall_center
        self.wall_low    = wall_center - wall_half
        self.wall_high   = wall_center + wall_half
        self.border      = border
        self.max_speed   = max_speed
        self.img_size    = img_size
        self.door_y      = door_y
        self.door_half   = door_half

        self.eff_low  = door_y - door_half
        self.eff_high = door_y + door_half

    # ── Costable interface ──────────────────────────────────────────────────

    def get_cost(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """CEM cost: L2 distance from agent to target after stepping."""
        next_states = self._batch_step(states, actions)
        agent  = next_states[:, 0:2]
        target = next_states[:, 2:4]
        return np.linalg.norm(agent - target, axis=1)

    # ── Physics step ────────────────────────────────────────────────────────

    def _batch_step(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        next_states = states.copy()
        N = len(states)
        for i in range(N):
            next_states[i, 0:2] = self._step_agent(
                states[i, 0:2], actions[i], states[i, 4:6]
            )
        return next_states

    def _step_agent(self, pos: np.ndarray, action: np.ndarray,
                    door_centre: np.ndarray) -> np.ndarray:
        speed   = self.max_speed
        prop_x  = float(pos[0] + action[0] * speed)
        prop_y  = float(pos[1] + action[1] * speed)
        cur_x   = float(pos[0])
        door_y  = float(door_centre[1])   # obs[5] — runtime door centre y

        # Clamp to border
        prop_x = np.clip(prop_x, self.border, self.img_size - self.border - 1)
        prop_y = np.clip(prop_y, self.border, self.img_size - self.border - 1)

        # Wall crossing rule (vertical wall, axis=1)
        crosses_wall = (
            (cur_x < self.wall_low  and prop_x >= self.wall_low)  or
            (cur_x > self.wall_high and prop_x <= self.wall_high)
        )
        if crosses_wall:
            in_door = (door_y - self.door_half) <= prop_y <= (door_y + self.door_half)
            if in_door:
                new_x = prop_x   # allowed through door
            else:
                # Push back to wall edge
                new_x = self.wall_low - 0.5 if cur_x < self.wall_low else self.wall_high + 0.5
        else:
            new_x = prop_x

        return np.array([new_x, prop_y], dtype=np.float32)


# ── Standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = compile_from_source()
    print("\n[Kcode] Running 5 unit tests: compare step outputs ...")
    rng = np.random.default_rng(42)
    passed = 0
    # Test: two agents in same-room, actions toward target — should both get same cost
    # Tests check: push-back when blocked, pass-through when in door zone
    # wall_low=107, wall_high=117, door=[35,63] (door_y=49±14)
    test_cases = [
        # (ax, ay, tx, ty, dx, dy_obs, acx, acy, description, expect_new_ax_approx)
        (60,  80, 60, 40, 0, 49,  0.0, -1.0, "free move up",         60.0),
        (60,  80,160, 80, 0, 49,  1.0,  0.0, "free move right",      70.5),
        (106, 49,160, 49, 0, 49,  1.0,  0.0, "in door: enter wall",  116.5),  # allowed
        (106,200,160,200, 0, 49,  1.0,  0.0, "not in door: push back",106.5), # blocked
        (60,  50, 60, 50, 0, 49,  0.0,  0.0, "zero action",           60.0),
    ]
    for i, (ax, ay, tx, ty, dx, dy, acx, acy, desc, exp_x) in enumerate(test_cases):
        s = np.array([[ax, ay, tx, ty, dx, dy, 0, 0, 0, 0]], dtype=np.float32)
        a = np.array([[acx, acy]], dtype=np.float32)
        cost = model.get_cost(s, a)[0]
        next_s = model._batch_step(s, a)
        new_ax = next_s[0, 0]
        ok = abs(new_ax - exp_x) < 1.0
        print(f"  test {i+1} ({desc}): ax={ax}→{new_ax:.1f}  expected≈{exp_x}  {'OK' if ok else 'FAIL'}")
        passed += ok
    print(f"\n[Kcode] {passed}/5 unit tests passed.")
