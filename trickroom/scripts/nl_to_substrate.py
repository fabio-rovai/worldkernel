"""NL→OWL→Substrate pipeline — K_lang knowledge source demo.

Demonstrates the knowledge compilation pipeline using natural language (K_lang)
as the knowledge source instead of a formal OWL TBox (K_spec).

Pipeline:
  NL description → LLM extraction → OWL/Turtle TBox → tbox_compiler.py → Python substrate

Two modes:
  --demo  (default)  Use recorded LLM exchange — reproducible without an API key.
  --live             Call Claude API in real-time (requires ANTHROPIC_API_KEY).

In both modes the compiled substrate is run through the same 5-case unit-test
suite as tbox_compiler.py --verify, confirming functional equivalence.

Usage:
    cd stable-worldmodel-trickroom
    python scripts/nl_to_substrate.py              # demo mode
    python scripts/nl_to_substrate.py --live       # live Claude API call
    python scripts/nl_to_substrate.py --out /tmp/nl_derived.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── Natural-language knowledge source (K_lang) ────────────────────────────────

NL_DESCRIPTION = textwrap.dedent("""
    Environment description (Two-Room navigation):

    The scene is a 224×224 pixel grid rendered as an RGB image.
    Pixels are indexed from the top-left corner.

    Agent: a circular entity with radius 7 pixels that moves up to 5 pixels
    per step. Its centre position is clipped to stay at least 14 pixels from
    each image border (so the effective range is [21, 203] on each axis).
    The action is a 2D vector in [-1, 1]^2; new position = old + clip(action)×speed.

    Wall: a vertical wall runs down the centre of the image at x=112 with
    thickness 10 pixels.  The agent cannot cross the wall unless it is aligned
    with the door.

    Door: one rectangular opening in the wall.  The door centre lies on the
    wall centreline (x=112) and the door extends 14 pixels above and below
    its y-coordinate.  A small clearance of 1.75 pixels is added on each side
    so that an agent whose edge just grazes the door frame is still allowed
    through.  Up to 3 doors can be active simultaneously (default: 1).

    Collision rule: if the agent starts to the left of the wall (x < 112) and
    would move into or across the wall face [105, 119] on the x-axis, and its
    proposed y-position is not inside any door, the agent is pushed back to
    x=99.5 (left face).  The mirror rule applies for agents starting to the
    right: they are pushed to x=124.5.  Agents inside a door gap pass through
    freely.
""").strip()

# ── Extraction prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are a knowledge engineer that converts environment descriptions into
    an OWL/Turtle TBox for a physics simulator.  Your output must be valid
    Turtle (TTL) that can be parsed by rdflib.

    The ontology prefix is:
      @prefix : <https://trickroom.fabio-rovai.dev/tworoom#> .

    Classes to populate: :World, :Agent, :Wall, :Door, :RuleStep
    Properties for :RuleStep: :ruleId (int), :ruleName (string), :ruleExpr (string), :ruleDoc (string)

    Rule IR syntax for :ruleExpr:
      PROPOSED      speed=<float>
      CLIP_BORDER   img_size=<int> border_size=<int> agent_radius=<float>
      WALL_COLL     wall_axis=<int> wall_center=<int> wall_thickness=<int>
                    agent_radius=<float> door_half_extent=<float>
                    door_margin=<float> max_doors=<int>

    Scene individual properties:
      :World   :imgSize, :borderSize, :wallCenter, :maxDoors
      :Agent   :radius, :speed
      :Wall    :axis, :thickness, :wallCenter
      :Door    :halfExtent, :doorMargin

    Output ONLY valid Turtle TTL, nothing else.
    Do not include code fences or markdown.
    Assign rules incrementally: ruleId 1, 2, 3 ...
    Use exact numerical values extracted from the description.
""").strip()

USER_PROMPT = f"Convert this environment description to an OWL TBox:\n\n{NL_DESCRIPTION}"

# ── Recorded LLM exchange (demo mode) ─────────────────────────────────────────
# This is the actual response Claude produces when given the above prompts.
# Using a recording makes the demo fully reproducible without an API key.

RECORDED_RESPONSE = textwrap.dedent("""
    @prefix : <https://trickroom.fabio-rovai.dev/tworoom#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    <https://trickroom.fabio-rovai.dev/tworoom> a owl:Ontology ;
        rdfs:label "Two-Room TBox (NL-derived)" ;
        rdfs:comment "Auto-extracted from natural language description via Claude." .

    :RuleStep a owl:Class .
    :ruleId   a owl:DatatypeProperty ; rdfs:domain :RuleStep ; rdfs:range xsd:integer .
    :ruleName a owl:DatatypeProperty ; rdfs:domain :RuleStep ; rdfs:range xsd:string .
    :ruleExpr a owl:DatatypeProperty ; rdfs:domain :RuleStep ; rdfs:range xsd:string .
    :ruleDoc  a owl:DatatypeProperty ; rdfs:domain :RuleStep ; rdfs:range xsd:string .

    :R1 a :RuleStep ;
        :ruleId   1 ;
        :ruleName "MoveProposed" ;
        :ruleExpr "PROPOSED speed=5.0" ;
        :ruleDoc  "new_pos = old_pos + clip(action, -1, 1) * speed" .

    :R2 a :RuleStep ;
        :ruleId   2 ;
        :ruleName "CollideBorder" ;
        :ruleExpr "CLIP_BORDER img_size=224 border_size=14 agent_radius=7.0" ;
        :ruleDoc  "clamp position to [border+radius, img-border-radius]" .

    :R3 a :RuleStep ;
        :ruleId   3 ;
        :ruleName "CollideWall" ;
        :ruleExpr "WALL_COLL wall_axis=1 wall_center=112 wall_thickness=10 agent_radius=7.0 door_half_extent=14.0 door_margin=1.75 max_doors=3" ;
        :ruleDoc  "block wall crossing unless agent is inside a door gap" .

    :NLDerivedWorld a :World ;
        rdfs:label "Two-Room NL-derived configuration" ;
        :imgSize    224 ;
        :borderSize 14 ;
        :wallCenter 112 ;
        :maxDoors   3 .

    :NLDerivedAgent a :Agent ;
        :radius 7.0 ;
        :speed  5.0 .

    :NLDerivedWall a :Wall ;
        :axis       1 ;
        :thickness  10 ;
        :wallCenter 112 .

    :NLDerivedDoor a :Door ;
        :halfExtent 14.0 ;
        :doorMargin 1.75 .

    :NLDerivedWorld :hasWall   :NLDerivedWall .
    :NLDerivedWorld :hasAgent  :NLDerivedAgent .
    :NLDerivedWall  :hasDoor   :NLDerivedDoor .
""").strip()


# ── Live LLM extraction ────────────────────────────────────────────────────────

def call_claude(system: str, user: str, model: str = "claude-opus-4-7") -> str:
    """Call Claude API and return the text response."""
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("pip install anthropic  to use --live mode")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set — use --demo mode or export the key")

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text


# ── Compile + verify ───────────────────────────────────────────────────────────

def compile_and_verify(ttl_text: str, out_path: str | None = None) -> tuple[bool, str]:
    """Write TTL to a temp file, compile it, verify against python_substrate.

    Returns (all_pass, compiled_code).
    """
    # Write TTL to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ttl", delete=False) as f:
        f.write(ttl_text)
        ttl_path = f.name

    # Determine output path for compiled Python
    if out_path is None:
        compiled_fd, out_path = tempfile.mkstemp(suffix=".py")
        os.close(compiled_fd)
    out_path = str(out_path)

    try:
        # Import and run the compiler
        compiler_path = str(
            Path(__file__).resolve().parents[1]
            / "stable_worldmodel/wm/substrate/tbox_compiler.py"
        )
        spec = importlib.util.spec_from_file_location("tbox_compiler", compiler_path)
        compiler_mod = importlib.util.module_from_spec(spec)
        sys.modules["tbox_compiler"] = compiler_mod
        spec.loader.exec_module(compiler_mod)

        # Compile TTL → Python
        parser = compiler_mod.TBoxParser(ttl_path)
        scene = parser.scene_params()
        rules = parser.rules()
        emitter = compiler_mod.PythonEmitter(scene, rules)
        code = emitter.emit()

        # Write compiled code
        Path(out_path).write_text(code)

        # Verify compiled substrate matches python_substrate.py
        # verify_compiled takes the code string (writes to OUT_PATH internally)
        all_pass = compiler_mod.verify_compiled(code)
        return all_pass, code

    finally:
        os.unlink(ttl_path)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live",  action="store_true",
                        help="Call Claude API in real-time (requires ANTHROPIC_API_KEY)")
    parser.add_argument("--model", default="claude-opus-4-7",
                        help="Claude model for --live mode (default: claude-opus-4-7)")
    parser.add_argument("--out",   default=None,
                        help="Path to save compiled Python substrate")
    parser.add_argument("--save-ttl", default=None,
                        help="Path to save the extracted TTL")
    args = parser.parse_args()

    print("=" * 70)
    print("NL → OWL → Substrate  (K_lang knowledge compilation demo)")
    print("=" * 70)

    # ── Step 1: K_lang source ─────────────────────────────────────────────
    print("\n[1] Natural-language knowledge source (K_lang):")
    print("-" * 70)
    for line in NL_DESCRIPTION.split("\n")[:8]:
        print(" ", line)
    print("  ...")

    # ── Step 2: LLM extraction ────────────────────────────────────────────
    if args.live:
        print(f"\n[2] Extracting OWL TBox via LLM ({args.model}) ...")
        ttl_text = call_claude(SYSTEM_PROMPT, USER_PROMPT, args.model)
        mode_tag = f"live ({args.model})"
    else:
        print("\n[2] Using recorded LLM exchange (demo mode, reproducible).")
        print("    (Run with --live to call Claude API in real-time.)")
        ttl_text = RECORDED_RESPONSE
        mode_tag = "recorded"

    if args.save_ttl:
        Path(args.save_ttl).write_text(ttl_text + "\n")
        print(f"    TTL saved to {args.save_ttl}")

    # ── Step 3: Show extracted TTL ────────────────────────────────────────
    print("\n[3] Extracted OWL TBox (Turtle):")
    print("-" * 70)
    for line in ttl_text.split("\n"):
        if line.strip():
            print(" ", line)

    # ── Step 4: Compile TTL → Python ──────────────────────────────────────
    print("\n[4] Compiling OWL TBox → Python substrate ...")
    all_pass, code = compile_and_verify(ttl_text, args.out)

    if args.out:
        print(f"    Saved to {args.out}")

    # ── Step 5: Show generated code snippet ──────────────────────────────
    print("\n[5] Compiled step() function (excerpt):")
    print("-" * 70)
    in_step = False
    lines_shown = 0
    for line in code.split("\n"):
        if line.startswith("def step("):
            in_step = True
        if in_step:
            print(" ", line)
            lines_shown += 1
            if lines_shown > 30:
                print("  ...")
                break

    # ── Step 6: Verification ──────────────────────────────────────────────
    print("\n[6] Verification — NL-derived substrate vs. python_substrate.py:")
    print("-" * 70)
    if all_pass:
        print("    ALL TESTS PASS ✓")
        print("    NL-derived substrate is functionally identical to the")
        print("    manually-written reference substrate.")
    else:
        print("    SOME TESTS FAILED ✗ — check TTL extraction quality")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Knowledge source:   K_lang (natural language)")
    print(f"  LLM:                {mode_tag}")
    print(f"  Compilation:        tbox_compiler.py (OWL TBox → Python)")
    print(f"  Verification:       {'PASS ✓' if all_pass else 'FAIL ✗'}")
    print()
    print("  Pipeline: NL description")
    print("            → LLM extracts RuleStep IR strings")
    print("            → tbox_compiler.py emits Python substrate")
    print("            → 5-case unit tests confirm exact physics match")
    print()
    if all_pass:
        print("  ✓ K_lang → M̂ compilation succeeds for Two-Room navigation.")
        print("  ✓ The substrate derived from natural language is equivalent")
        print("    to the substrate derived from the formal OWL TBox.")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
