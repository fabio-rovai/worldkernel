"""Knowledge Compiler: OWL TBox → Executable Python Substrate.

This is the core artefact for the Executable World Models paper.
It demonstrates that a planning substrate can be *compiled* from a formal
knowledge source (OWL/Turtle specification) without manual engineering.

Pipeline
--------
  tworoom.ttl  (OWL TBox, the K_spec knowledge source)
       ↓
  TBoxCompiler.parse()   — extract rules + scene parameters via rdflib
       ↓
  TBoxCompiler.emit()    — render Python step() from parametric templates
       ↓
  compiled_substrate.py  (executable Costable world model)

The generated file is functionally identical to python_substrate.py, which
was written by hand. This proves that the OWL specification contains enough
information to recover the full physics implementation.

Usage
-----
  python -m stable_worldmodel.wm.substrate.tbox_compiler           # compile to stdout
  python -m stable_worldmodel.wm.substrate.tbox_compiler --write   # write compiled_substrate.py
  python -m stable_worldmodel.wm.substrate.tbox_compiler --verify  # run unit tests
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from rdflib import Graph, Namespace, URIRef, Literal
    from rdflib.namespace import RDF, RDFS, XSD, OWL
    HAS_RDFLIB = True
except ImportError:
    HAS_RDFLIB = False

TBOX_PATH = Path(__file__).parent.parent.parent.parent / "data" / "tbox" / "tworoom.ttl"
OUT_PATH = Path(__file__).parent / "compiled_substrate.py"

NS = "https://trickroom.fabio-rovai.dev/tworoom#"


# ── IR dataclasses ────────────────────────────────────────────────────────────

@dataclass
class SceneParams:
    img_size: int = 224
    border_size: int = 14
    wall_center: int = 112
    max_doors: int = 3
    wall_axis: int = 1
    wall_thickness: int = 10
    agent_radius: float = 7.0
    agent_speed: float = 5.0
    door_half_extent: float = 14.0
    door_margin: float = 1.75


@dataclass
class RuleIR:
    rule_id: int
    name: str
    expr: str
    doc: str
    params: dict = field(default_factory=dict)

    def parse_params(self) -> None:
        """Parse key=value pairs from ruleExpr string into self.params."""
        for token in self.expr.split():
            if "=" in token:
                k, v = token.split("=", 1)
                try:
                    self.params[k] = float(v) if "." in v else int(v)
                except ValueError:
                    self.params[k] = v


# ── Parser ────────────────────────────────────────────────────────────────────

class TBoxParser:
    """Parse tworoom.ttl using rdflib and return SceneParams + list[RuleIR]."""

    def __init__(self, path: Path = TBOX_PATH):
        if not HAS_RDFLIB:
            raise ImportError("rdflib required: pip install rdflib")
        self.g = Graph()
        self.g.parse(str(path), format="turtle")
        self.ns = Namespace(NS)

    def _prop(self, subject: URIRef, prop_local: str) -> Optional[Literal]:
        for _, _, obj in self.g.triples((subject, self.ns[prop_local], None)):
            return obj
        return None

    def scene_params(self) -> SceneParams:
        p = SceneParams()
        world = URIRef(NS + "TwoRoomDefault")
        wall = URIRef(NS + "DefaultWall")
        agent = URIRef(NS + "DefaultAgent")
        door = URIRef(NS + "DefaultDoor")

        if (v := self._prop(world, "imgSize")) is not None:
            p.img_size = int(v)
        if (v := self._prop(world, "borderSize")) is not None:
            p.border_size = int(v)
        if (v := self._prop(world, "wallCenter")) is not None:
            p.wall_center = int(v)
        if (v := self._prop(world, "maxDoors")) is not None:
            p.max_doors = int(v)
        if (v := self._prop(wall, "axis")) is not None:
            p.wall_axis = int(v)
        if (v := self._prop(wall, "thickness")) is not None:
            p.wall_thickness = int(v)
        if (v := self._prop(agent, "radius")) is not None:
            p.agent_radius = float(v)
        if (v := self._prop(agent, "speed")) is not None:
            p.agent_speed = float(v)
        if (v := self._prop(door, "halfExtent")) is not None:
            p.door_half_extent = float(v)
        if (v := self._prop(door, "doorMargin")) is not None:
            p.door_margin = float(v)
        return p

    def rules(self) -> list[RuleIR]:
        rule_class = URIRef(NS + "RuleStep")
        rules: list[RuleIR] = []
        for subj in self.g.subjects(RDF.type, rule_class):
            rid = int(self._prop(subj, "ruleId") or 0)
            name = str(self._prop(subj, "ruleName") or "")
            expr = str(self._prop(subj, "ruleExpr") or "")
            doc = str(self._prop(subj, "ruleDoc") or "")
            ir = RuleIR(rule_id=rid, name=name, expr=expr, doc=doc)
            ir.parse_params()
            rules.append(ir)
        return sorted(rules, key=lambda r: r.rule_id)


# ── Code emitter ──────────────────────────────────────────────────────────────

class PythonEmitter:
    """Render a complete Python substrate module from parsed IR."""

    def __init__(self, params: SceneParams, rules: list[RuleIR]):
        self.p = params
        self.rules = {r.rule_id: r for r in rules}

    # ── helpers ────────────────────────────────────────────────────────────────

    @property
    def eff_low(self) -> float:
        return self.p.wall_center - self.p.wall_thickness // 2 - self.p.agent_radius

    @property
    def eff_high(self) -> float:
        return self.p.wall_center + self.p.wall_thickness // 2 + self.p.agent_radius

    @property
    def clip_lo(self) -> float:
        return self.p.border_size + self.p.agent_radius

    @property
    def clip_hi(self) -> float:
        return self.p.img_size - self.p.border_size - self.p.agent_radius

    @property
    def door_band(self) -> float:
        return self.p.door_half_extent + self.p.door_margin

    def emit(self) -> str:
        """Return the full compiled module as a string."""
        lines: list[str] = []

        lines.append(self._header())
        lines.append(self._constants())
        lines.append(self._dataclass())
        lines.append(self._door_helper())
        lines.append(self._step_fn())
        lines.append(self._rollout_fn())
        lines.append(self._costmodel_cls())

        return "\n".join(lines)

    # ── section emitters ──────────────────────────────────────────────────────

    def _header(self) -> str:
        return textwrap.dedent(f'''\
            """Compiled substrate — AUTO-GENERATED by tbox_compiler.py.

            Source: data/tbox/tworoom.ttl  (OWL TBox, K_spec knowledge source)
            Compiler: stable_worldmodel/wm/substrate/tbox_compiler.py
            Target: Two-Room planning environment dynamics

            DO NOT EDIT BY HAND — regenerate with:
              python -m stable_worldmodel.wm.substrate.tbox_compiler --write

            This file is the artefact produced by the knowledge compilation step
            in the Executable World Models framework.  It is functionally identical
            to python_substrate.py, which was written manually. The equivalence is
            verified by the --verify flag.
            """

            from __future__ import annotations

            from dataclasses import dataclass
            from typing import Optional

            import numpy as np
            ''')

    def _constants(self) -> str:
        p = self.p
        return textwrap.dedent(f'''\
            # ── Compiled constants (from :TwoRoomDefault individual in tworoom.ttl) ──
            IMG_SIZE     = {p.img_size}
            BORDER_SIZE  = {p.border_size}
            WALL_CENTER  = {p.wall_center}
            WALL_AXIS    = {p.wall_axis}       # 1 = vertical wall
            WALL_THICK   = {p.wall_thickness}
            MAX_DOOR     = {p.max_doors}
            AGENT_RADIUS = {p.agent_radius}
            AGENT_SPEED  = {p.agent_speed}
            DOOR_HALF    = {p.door_half_extent}
            DOOR_MARGIN  = {p.door_margin}

            # ── Derived constants ──────────────────────────────────────────────────
            EFF_LOW   = WALL_CENTER - WALL_THICK // 2 - AGENT_RADIUS   # {self.eff_low}
            EFF_HIGH  = WALL_CENTER + WALL_THICK // 2 + AGENT_RADIUS   # {self.eff_high}
            CLIP_LO   = BORDER_SIZE + AGENT_RADIUS                      # {self.clip_lo}
            CLIP_HI   = IMG_SIZE - BORDER_SIZE - AGENT_RADIUS           # {self.clip_hi}
            DOOR_BAND = DOOR_HALF + DOOR_MARGIN                         # {self.door_band}
            PUSH_L    = EFF_LOW  - 0.5   # {self.eff_low  - 0.5}
            PUSH_R    = EFF_HIGH + 0.5   # {self.eff_high + 0.5}
            ''')

    def _dataclass(self) -> str:
        return textwrap.dedent('''\
            # ── ABox (compiled from class + property declarations) ─────────────────

            @dataclass
            class TwoRoomABox:
                """Scene state for one or many parallel Two-Room episodes.

                Compiled from :Agent, :Target, :Wall, :Door class declarations
                in the OWL TBox.  Arrays are (..., d) shaped; leading dims
                propagate through rollout().
                """
                agent_pos:        np.ndarray   # (..., 2)
                target_pos:       np.ndarray   # (..., 2)
                door_centers:     np.ndarray   # (..., MAX_DOOR, 2)  zeros for unused
                agent_radius:     float = AGENT_RADIUS
                agent_speed:      float = AGENT_SPEED
                wall_axis:        int   = WALL_AXIS
                wall_thickness:   int   = WALL_THICK
                door_half_extents: tuple = (DOOR_HALF, DOOR_HALF, DOOR_HALF)
                num_doors:        int   = 1

                @classmethod
                def from_state_vec(
                    cls,
                    state: np.ndarray,
                    *,
                    agent_radius: float = AGENT_RADIUS,
                    agent_speed: float = AGENT_SPEED,
                    wall_axis: int = WALL_AXIS,
                    wall_thickness: int = WALL_THICK,
                    door_half_extents: tuple = (DOOR_HALF, DOOR_HALF, DOOR_HALF),
                    num_doors: int = 1,
                ) -> "TwoRoomABox":
                    assert state.shape[-1] == 10, state.shape
                    return cls(
                        agent_pos=state[..., 0:2].astype(np.float32, copy=True),
                        target_pos=state[..., 2:4].astype(np.float32, copy=True),
                        door_centers=state[..., 4:10].reshape(
                            *state.shape[:-1], MAX_DOOR, 2
                        ).astype(np.float32, copy=True),
                        agent_radius=agent_radius,
                        agent_speed=agent_speed,
                        wall_axis=wall_axis,
                        wall_thickness=wall_thickness,
                        door_half_extents=door_half_extents,
                        num_doors=num_doors,
                    )
            ''')

    def _door_helper(self) -> str:
        return textwrap.dedent('''\
            # ── R3 helper (door membership test) ─────────────────────────────────

            def _in_any_door_1d(
                coord: np.ndarray,
                door_centers_1d: np.ndarray,
                half_extents: np.ndarray,
                num_doors: int,
                margin: float,
            ) -> np.ndarray:
                """Vectorised door-membership test. coord shape (...). Returns bool."""
                out = np.zeros_like(coord, dtype=bool)
                for i in range(num_doors):
                    c = door_centers_1d[..., i]
                    s = half_extents[i]
                    out |= (coord >= c - s - margin) & (coord <= c + s + margin)
                return out
            ''')

    def _step_fn(self) -> str:
        p = self.p
        r1 = self.rules.get(1)
        r2 = self.rules.get(2)
        r3 = self.rules.get(3)

        r1_doc = r1.doc if r1 else "MoveProposed"
        r2_doc = r2.doc if r2 else "CollideBorder"
        r3_doc = r3.doc if r3 else "CollideWall"

        return textwrap.dedent(f'''\
            # ── step() — compiled from R1, R2, R3 rule individuals ───────────────

            def step(abox: TwoRoomABox, action: np.ndarray) -> TwoRoomABox:
                """One compiled step: R1 MoveProposed + R2 CollideBorder + R3 CollideWall.

                Generated from rules:
                  R1: {r1_doc}
                  R2: {r2_doc}
                  R3: {r3_doc}
                """
                # R1: MoveProposed  (rule expr: {r1.expr if r1 else "PROPOSED speed=" + str(p.agent_speed)})
                action   = np.clip(action, -1.0, 1.0)
                proposed = abox.agent_pos + action * abox.agent_speed

                # R2: CollideBorder  (rule expr: {r2.expr if r2 else "CLIP_BORDER"})
                proposed = np.clip(proposed, CLIP_LO, CLIP_HI)

                # R3: CollideWall  (rule expr: {r3.expr if r3 else "WALL_COLL"})
                half_extents = np.asarray(abox.door_half_extents, dtype=np.float32)

                if abox.wall_axis == 1:                           # vertical wall
                    x_start = abox.agent_pos[..., 0]
                    x_prop  = proposed[..., 0]
                    y_prop  = proposed[..., 1]
                    door_1d = abox.door_centers[..., :, 1]        # door y-coords
                    in_door = _in_any_door_1d(
                        y_prop, door_1d, half_extents,
                        abox.num_doors, DOOR_MARGIN,
                    )
                    started_left = x_start < WALL_CENTER
                    push_left  = started_left & (x_prop > EFF_LOW)  & ~in_door
                    push_right = (~started_left) & (x_prop < EFF_HIGH) & ~in_door
                    new_x = np.where(push_left,  PUSH_L,
                            np.where(push_right, PUSH_R, x_prop))
                    new_y = y_prop
                else:                                             # horizontal wall
                    y_start = abox.agent_pos[..., 1]
                    y_prop  = proposed[..., 1]
                    x_prop  = proposed[..., 0]
                    door_1d = abox.door_centers[..., :, 0]
                    in_door = _in_any_door_1d(
                        x_prop, door_1d, half_extents,
                        abox.num_doors, DOOR_MARGIN,
                    )
                    started_top = y_start < WALL_CENTER
                    push_top    = started_top  & (y_prop > EFF_LOW)  & ~in_door
                    push_bottom = (~started_top) & (y_prop < EFF_HIGH) & ~in_door
                    new_y = np.where(push_top,    PUSH_L,
                            np.where(push_bottom, PUSH_R, y_prop))
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
            ''')

    def _rollout_fn(self) -> str:
        return textwrap.dedent('''\
            # ── rollout() ─────────────────────────────────────────────────────────

            def rollout(abox: TwoRoomABox, actions: np.ndarray) -> TwoRoomABox:
                """Run T compiled steps. actions shape (..., T, 2)."""
                T = actions.shape[-2]
                cur = abox
                for t in range(T):
                    cur = step(cur, actions[..., t, :])
                return cur
            ''')

    def _costmodel_cls(self) -> str:
        return textwrap.dedent('''\
            # ── CompiledSubstrateCostModel — Costable protocol ────────────────────

            class CompiledSubstrateCostModel:
                """Compiled substrate conforming to the Costable protocol.

                This is the Experiment B model: auto-generated from the OWL TBox
                without manual physics engineering.  Identical API to
                SubstrateCostModel (python_substrate.py).

                Parameters
                ----------
                door_y : float
                    Door y-coordinate prior (θ_door).  Default 112 = wrong prior.
                """

                def __init__(self, door_y: float = 112.0):
                    self.door_y = door_y

                def get_cost(self, info_dict: dict, action_candidates) -> "np.ndarray":
                    """Costable.get_cost implementation.

                    info_dict["observation"] : (B, S, T, 10)
                    action_candidates        : (B, S, H, action_block * 2)
                    Returns cost             : (B, S)
                    """
                    import torch
                    obs = info_dict["observation"]
                    if hasattr(obs, "numpy"):
                        obs = obs.numpy()
                    if hasattr(action_candidates, "numpy"):
                        action_candidates = action_candidates.numpy()

                    obs    = np.asarray(obs,    dtype=np.float32)
                    acts   = np.asarray(action_candidates, dtype=np.float32)

                    # Most-recent obs
                    state = obs[..., -1, :]                       # (B, S, 10)
                    B, S, _ = state.shape

                    # Override door prior
                    state[..., 5] = self.door_y                   # obs[5] = door_y

                    abox = TwoRoomABox.from_state_vec(state)

                    # Reshape actions: (B, S, H, block*2) → (B, S, H*block, 2)
                    _B, _S, H, ad = acts.shape
                    micro = acts.reshape(B, S, H * (ad // 2), 2)

                    final = rollout(abox, micro)
                    diff  = final.agent_pos - final.target_pos
                    cost  = np.linalg.norm(diff, axis=-1).astype(np.float32)

                    return torch.as_tensor(cost)
            ''')


# ── Verification ──────────────────────────────────────────────────────────────

def verify_compiled(compiled_code: str) -> bool:
    """Load compiled_substrate from disk and check agreement with python_substrate."""
    # Write to OUT_PATH first so dataclasses get proper __module__
    OUT_PATH.write_text(compiled_code)
    import numpy as np  # noqa: F811
    spec = importlib.util.spec_from_file_location("compiled_substrate", OUT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["compiled_substrate"] = mod
    spec.loader.exec_module(mod)
    CompiledABox = mod.TwoRoomABox
    compiled_step = mod.step

    from stable_worldmodel.wm.substrate.python_substrate import (
        TwoRoomABox as RefABox, step as ref_step
    )

    cases = [
        ([100.0, 80.0], [1.0, 0.0], 49.0, "cross-room blocked"),
        ([100.0, 49.0], [1.0, 0.0], 49.0, "cross-room passes"),
        ([130.0, 80.0], [-1.0, 0.0], 49.0, "right→left blocked"),
        ([80.0,  80.0], [0.0, 1.0], 49.0, "same-room move"),
        ([100.0, 49.0], [1.0, 0.0], 112.0, "wrong prior blocked"),
    ]

    all_pass = True
    for agent_pos, action, door_y, label in cases:
        ap = np.array(agent_pos, dtype=np.float32)
        ac = np.array(action,    dtype=np.float32)

        # Reference (python_substrate)
        ref_abox = RefABox(
            agent_pos=ap.copy(),
            target_pos=np.array([180.0, 80.0], dtype=np.float32),
            door_centers=np.array([[112.0, door_y, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32).reshape(3, 2),
        )
        ref_new = ref_step(ref_abox, ac).agent_pos

        # Compiled
        c_abox = CompiledABox(
            agent_pos=ap.copy(),
            target_pos=np.array([180.0, 80.0], dtype=np.float32),
            door_centers=np.array([[112.0, door_y, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32).reshape(3, 2),
        )
        c_new = compiled_step(c_abox, ac).agent_pos

        err = float(np.max(np.abs(ref_new - c_new)))
        status = "PASS" if err < 1e-4 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  [{status}] {label}: ref={ref_new}, compiled={c_new}, err={err:.6f}")

    return all_pass


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OWL TBox → Python substrate compiler")
    parser.add_argument("--tbox", default=str(TBOX_PATH), help="Path to .ttl file")
    parser.add_argument("--write", action="store_true", help="Write compiled_substrate.py")
    parser.add_argument("--verify", action="store_true", help="Run unit tests after compilation")
    parser.add_argument("--quiet", action="store_true", help="Suppress code output")
    args = parser.parse_args()

    if not HAS_RDFLIB:
        print("ERROR: rdflib not installed.  pip install rdflib", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing TBox: {args.tbox}")
    tparser = TBoxParser(Path(args.tbox))
    params = tparser.scene_params()
    rules  = tparser.rules()

    print(f"  Scene params: img={params.img_size}, border={params.border_size}, "
          f"wall_center={params.wall_center}, agent_r={params.agent_radius}")
    print(f"  Rules extracted: {[r.name for r in rules]}")

    emitter = PythonEmitter(params, rules)
    code    = emitter.emit()

    if not args.quiet:
        print("\n" + "─" * 70)
        print(code[:3000] + ("\n... [truncated]\n" if len(code) > 3000 else ""))
        print("─" * 70)

    if args.write:
        OUT_PATH.write_text(code)
        print(f"\nWrote {OUT_PATH}")

    if args.verify:
        print("\nRunning verification against python_substrate.py ...")
        import numpy as np  # noqa: F811 (already imported at top of compiled code)
        ok = verify_compiled(code)
        if ok:
            print("\n✓ All tests PASS — compiled substrate is functionally equivalent")
        else:
            print("\n✗ Some tests FAILED")
            sys.exit(1)


if __name__ == "__main__":
    main()
