"""DiagonalRoom TBox Compiler — OWL-RL multi-hop reasoning demonstration.

This module demonstrates three OWL-RL inference mechanisms that distinguish
OWL-based compilation from a Python dict or class-constant lookup:

  1. MULTI-LEVEL SUBCLASS INFERENCE (TWO PARAMETERS, ZERO ASSERTIONS):
     :diagonalWall is typed only as :ThickDiagonalWall.  The OWL-RL closure infers:
       • wallTangentAngleDeg = 45.0  (from DiagonalWall, 2 hops up the subClassOf chain)
       • wallHalfThickness   = 0.05  (from ThickDiagonalWall, 1 hop up)
     Both properties are absent from the asserted ABox.  The compiler queries
     the *expanded* graph for both values before emitting any gate code.

  2. CROSS-INDIVIDUAL PROPERTY CHAIN (DOOR TOLERANCE FROM WALL):
     The property chain  :doorTolerance owl:propertyChainAxiom (:servesWall :wallHalfThickness)
     causes OWL-RL to infer :door1 :doorTolerance 0.05  by traversing:
       door1 → (servesWall) → diagonalWall → (wallHalfThickness) → 0.05
     The door's effective tolerance is never written on the door individual —
     it emerges from the combination of the :servesWall arc and the wall's
     (itself inferred) :wallHalfThickness.

  3. COMPILE-TIME INCONSISTENCY DETECTION:
     Both wallTangentAngleDeg and wallHalfThickness are owl:FunctionalProperty.
     A bad TBox that types a wall as :ThickDiagonalWall but asserts a conflicting
     value for either property acquires two values for a FunctionalProperty after
     reasoning, triggering TBoxInconsistencyError before any code is emitted.

In Python, mechanism (1) can be approximated with multiple inheritance + MRO,
but (2) cross-individual property chain inference and (3) compile-time
inconsistency checking across class hierarchies have no native Python equivalent.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

try:
    import owlrl
    from rdflib import Graph, Namespace, URIRef, Literal, BNode
    from rdflib.namespace import RDF, RDFS, OWL, XSD
    HAS_OWLRL = True
except ImportError:
    HAS_OWLRL = False

NS_STR = "http://trickroom.org/diagonalroom#"
NS = Namespace(NS_STR)

TBOX_PATH = Path(__file__).parent.parent.parent.parent / "docs" / "diagonalroom.ttl"


class TBoxInconsistencyError(ValueError):
    """Raised when OWL-RL reasoning reveals a FunctionalProperty violation."""


# ── OWL-RL reasoning layer ────────────────────────────────────────────────────

def load_and_reason(ttl_path: Path) -> Graph:
    if not HAS_OWLRL:
        raise ImportError("owlrl required: pip install owlrl")
    g = Graph()
    g.parse(str(ttl_path), format="turtle")
    owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(g)
    return g


def check_functional_property_consistency(g: Graph) -> None:
    """Raise TBoxInconsistencyError if any owl:FunctionalProperty has >1 value
    on any individual after OWL-RL expansion."""
    func_props = set(g.subjects(RDF.type, OWL.FunctionalProperty))
    for prop in func_props:
        seen: dict[URIRef, list] = {}
        for subj, _, obj in g.triples((None, prop, None)):
            if isinstance(subj, URIRef):
                seen.setdefault(subj, []).append(obj)
        for subj, vals in seen.items():
            unique = list({str(v): v for v in vals}.values())
            if len(unique) > 1:
                short_prop = str(prop).split("#")[-1]
                raise TBoxInconsistencyError(
                    f"FunctionalProperty violation: "
                    f"{str(subj).split('#')[-1]} has {len(unique)} values for "
                    f":{short_prop} ({[str(v) for v in unique]}).  "
                    f"Typed as a subclass whose owl:hasValue conflicts with explicit assertion."
                )


# ── Parameter extraction ──────────────────────────────────────────────────────

def _inferred(g: Graph, uri: URIRef, prop: URIRef, label: str) -> float:
    """Return the (inferred) value of a FunctionalProperty for a URI.

    Raises ValueError if not found — which means the TBox class hierarchy
    does not entail this property for the given individual.
    """
    vals = list(g.objects(uri, prop))
    if not vals:
        raise ValueError(
            f"No {str(prop).split('#')[-1]} found for {str(uri).split('#')[-1]} "
            f"after OWL-RL expansion.  ({label})"
        )
    return float(vals[0])


def compile_diagonal_room(ttl_path: Path = TBOX_PATH, verbose: bool = True) -> dict:
    """Full compilation pipeline with three OWL-RL inference mechanisms."""
    if verbose:
        print(f"Loading TBox: {ttl_path}")

    g = load_and_reason(ttl_path)
    if verbose:
        print("  OWL-RL deductive closure applied.")

    check_functional_property_consistency(g)
    if verbose:
        print("  FunctionalProperty consistency: OK")

    wall_uri  = URIRef(NS_STR + "diagonalWall")
    door_uri  = URIRef(NS_STR + "door1")

    # ── MECHANISM 1: multi-level subClassOf inference ─────────────────────────
    raw = Graph(); raw.parse(str(ttl_path), format="turtle")
    asserted_angle = list(raw.objects(wall_uri, NS.wallTangentAngleDeg))
    asserted_thick = list(raw.objects(wall_uri, NS.wallHalfThickness))

    angle_deg     = _inferred(g, wall_uri, NS.wallTangentAngleDeg,
                               "expected from DiagonalWall class hierarchy")
    half_thickness = _inferred(g, wall_uri, NS.wallHalfThickness,
                               "expected from ThickDiagonalWall class hierarchy")

    if verbose:
        src_a = "asserted" if asserted_angle else "*** INFERRED via DiagonalWall class (2-hop subClassOf)"
        src_t = "asserted" if asserted_thick else "*** INFERRED via ThickDiagonalWall class (1-hop)"
        print(f"  :diagonalWall :wallTangentAngleDeg = {angle_deg}°   [{src_a}]")
        print(f"  :diagonalWall :wallHalfThickness   = {half_thickness}m  [{src_t}]")

    # ── MECHANISM 2: cross-individual property chains ──────────────────────────
    room_a_uri = URIRef(NS_STR + "roomA")
    door_tolerance_vals  = list(g.objects(door_uri,   NS.doorTolerance))
    door_wall_vals       = list(g.objects(door_uri,   NS.doorWall))      # class-inferred
    room_clearance_a     = list(g.objects(room_a_uri, NS.crossingClearance))
    room_clearance_b     = list(g.objects(room_a_uri, NS.roomClearance))  # strongest chain
    if verbose:
        if door_wall_vals:
            dw = str(door_wall_vals[0]).split('#')[-1]
            print(f"  :door1 :doorWall = {dw}   "
                  f"[*** CLASS-INFERRED from DiagonalDoor (arc-2 in depth-3 chain B)]")
        if door_tolerance_vals:
            print(f"  :door1 :doorTolerance = {door_tolerance_vals[0]}   "
                  f"[INFERRED depth-2: servesWall∘wallHalfThickness]")
        if room_clearance_a:
            print(f"  :roomA :crossingClearance = {room_clearance_a[0]}   "
                  f"[INFERRED depth-3A: containsDoor∘servesWall∘wallHalfThickness]")
        if room_clearance_b:
            print(f"  :roomA :roomClearance = {room_clearance_b[0]}   "
                  f"[*** INFERRED depth-3B: containsDoor∘doorWall∘wallHalfThickness "
                  f"— BOTH arc-2 (doorWall) AND arc-3 (wallHalfThickness) are class-inferred]")

    # ── Gate derivation from inferred parameters ───────────────────────────────
    theta = math.radians(angle_deg)
    tx = math.cos(theta)
    ty = math.sin(theta)

    door_cx   = _inferred(g, door_uri, NS.doorCentreX, "door centre X")
    door_cy   = _inferred(g, door_uri, NS.doorCentreY, "door centre Y")
    door_half = _inferred(g, door_uri, NS.doorHalfWidth, "door half-width")
    door_proj = tx * door_cx + ty * door_cy

    push_diag = half_thickness * math.sqrt(2)  # derived from inferred thickness

    gate = {
        "gate_type":       "dot_product",
        "angle_deg":       angle_deg,
        "tangent":         (tx, ty),
        "door_proj":       door_proj,
        "door_half":       door_half,
        "half_thickness":  half_thickness,
        "push_diag":       push_diag,
        "tolerance_cert":  door_half,
        "inference_hops":  2,   # 2 class levels traversed for angle; 1 for thickness
    }

    if verbose:
        print(f"\n  Gate formula:  "
              f"|{tx:.4f}*x + {ty:.4f}*y - {door_proj:.4f}| < {door_half}")
        print(f"  Push distance: push_diag = halfThickness * sqrt(2) "
              f"= {half_thickness} * 1.414 = {push_diag:.5f}")
        print(f"  Tolerance cert: ε* = {door_half} (reads from TBox, no calibration)")
        print(f"  OWL inference hops: {gate['inference_hops']} (angle from class depth-2, "
              f"thickness from class depth-1, door tolerance via property chain)")

    return gate


# ── Inconsistency demo ────────────────────────────────────────────────────────

def demo_inconsistency() -> None:
    """Show two types of compile-time inconsistency detection."""
    print("═══ Inconsistency demo ════════════════════════════════════════════")

    def try_conflict(desc: str, prop: URIRef, bad_val: float):
        g = load_and_reason(TBOX_PATH)
        bad = URIRef(NS_STR + "badWall")
        g.add((bad, RDF.type, NS.ThickDiagonalWall))
        g.add((bad, prop, Literal(bad_val, datatype=XSD.float)))
        owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(g)
        inferred_vals = list(g.objects(bad, prop))
        print(f"\n  {desc}")
        short = str(prop).split('#')[-1]
        print(f"  Python dict: {{{short!r}: {bad_val}}}  (silently wrong)")
        print(f"  OWL values after reasoning: {[str(v) for v in inferred_vals]}")
        try:
            check_functional_property_consistency(g)
            print("  No inconsistency detected (unexpected).")
        except TBoxInconsistencyError as e:
            print(f"  TBoxInconsistencyError: {e}")

    try_conflict(
        ":ThickDiagonalWall + wallTangentAngleDeg='90.0' (wrong angle type)",
        NS.wallTangentAngleDeg, 90.0
    )
    try_conflict(
        ":ThickDiagonalWall + wallHalfThickness='0.1' (wrong thickness)",
        NS.wallHalfThickness, 0.1
    )
    print()


# ── Verification tests ────────────────────────────────────────────────────────

def verify() -> bool:
    all_pass = True

    def check(name: str, condition: bool):
        nonlocal all_pass
        status = "PASS" if condition else "FAIL"
        if not condition:
            all_pass = False
        print(f"  [{status}] {name}")

    print("Running DiagonalRoom compiler verification ...")
    print()

    raw = Graph(); raw.parse(str(TBOX_PATH), format="turtle")
    g   = load_and_reason(TBOX_PATH)
    wall_uri = URIRef(NS_STR + "diagonalWall")
    door_uri = URIRef(NS_STR + "door1")

    # ── Mechanism 1: multi-level subClassOf inference ──────────────────────────
    print("  [Mechanism 1: multi-level subClassOf — two parameters inferred]")
    a_asserted = list(raw.objects(wall_uri, NS.wallTangentAngleDeg))
    t_asserted = list(raw.objects(wall_uri, NS.wallHalfThickness))
    a_inferred = list(g.objects(wall_uri, NS.wallTangentAngleDeg))
    t_inferred = list(g.objects(wall_uri, NS.wallHalfThickness))
    check("angle NOT asserted on :diagonalWall (raw graph)",    len(a_asserted) == 0)
    check("thickness NOT asserted on :diagonalWall (raw graph)", len(t_asserted) == 0)
    check("angle=45.0 INFERRED (from DiagonalWall, 2-hop chain)",
          len(a_inferred) == 1 and abs(float(a_inferred[0]) - 45.0) < 1e-3)
    check("halfThickness=0.05 INFERRED (from ThickDiagonalWall, 1-hop)",
          len(t_inferred) == 1 and abs(float(t_inferred[0]) - 0.05) < 1e-5)

    # Check full class chain
    is_diagonal = (wall_uri, RDF.type, NS.DiagonalWall) in g
    is_wall     = (wall_uri, RDF.type, NS.Wall) in g
    check(":ThickDiagonalWall instance inferred rdf:type :DiagonalWall", is_diagonal)
    check(":ThickDiagonalWall instance inferred rdf:type :Wall",          is_wall)

    # ── Mechanism 2: property chains ─────────────────────────────────────────
    print()
    print("  [Mechanism 2: property chains — depth-2 + depth-3A + depth-3B (strongest)]")
    room_a_uri   = URIRef(NS_STR + "roomA")
    dt_asserted  = list(raw.objects(door_uri,   NS.doorTolerance))
    dw_asserted  = list(raw.objects(door_uri,   NS.doorWall))
    rca_asserted = list(raw.objects(room_a_uri, NS.crossingClearance))
    rcb_asserted = list(raw.objects(room_a_uri, NS.roomClearance))
    dt_inferred  = list(g.objects(door_uri,     NS.doorTolerance))
    dw_inferred  = list(g.objects(door_uri,     NS.doorWall))
    rca_inferred = list(g.objects(room_a_uri,   NS.crossingClearance))
    rcb_inferred = list(g.objects(room_a_uri,   NS.roomClearance))
    check("doorTolerance NOT asserted on :door1 (raw)",      len(dt_asserted) == 0)
    check("doorWall NOT asserted on :door1 (raw)",            len(dw_asserted) == 0)
    check("crossingClearance NOT asserted on :roomA (raw)",   len(rca_asserted) == 0)
    check("roomClearance NOT asserted on :roomA (raw)",       len(rcb_asserted) == 0)
    check("doorTolerance=0.05 INFERRED (depth-2: servesWall∘wallHalfThickness)",
          len(dt_inferred) == 1 and abs(float(dt_inferred[0]) - 0.05) < 1e-5)
    check("doorWall=:diagonalWall INFERRED (arc-2 CLASS-INFERRED from DiagonalDoor)",
          len(dw_inferred) == 1 and str(dw_inferred[0]).split('#')[-1] == 'diagonalWall')
    check("crossingClearance=0.05 INFERRED (depth-3A: containsDoor∘servesWall∘wallHalfThickness)",
          len(rca_inferred) == 1 and abs(float(rca_inferred[0]) - 0.05) < 1e-5)
    check("roomClearance=0.05 INFERRED (depth-3B STRONGEST: containsDoor∘doorWall∘wallHalfThickness, "
          "arc-2 AND arc-3 both class-inferred)",
          len(rcb_inferred) == 1 and abs(float(rcb_inferred[0]) - 0.05) < 1e-5)

    # ── Gate parameters ────────────────────────────────────────────────────────
    print()
    print("  [Gate derivation from inferred parameters]")
    gate = compile_diagonal_room(verbose=False)
    tx, ty = gate["tangent"]
    expected = 1.0 / math.sqrt(2)
    check("gate_type == 'dot_product'",              gate["gate_type"] == "dot_product")
    check("tangent = (1/√2, 1/√2)",
          abs(tx - expected) < 1e-5 and abs(ty - expected) < 1e-5)
    check("door_proj = (3.5+3.5)/√2",
          abs(gate["door_proj"] - 7.0 / math.sqrt(2)) < 1e-4)
    check("push_diag = halfThick * √2 = 0.05 * 1.414 = 0.07071",
          abs(gate["push_diag"] - 0.05 * math.sqrt(2)) < 1e-5)

    # ── Mechanism 3: inconsistency detection ───────────────────────────────────
    print()
    print("  [Mechanism 3: compile-time inconsistency detection]")
    check_functional_property_consistency(g)   # valid TBox — should not raise
    check("valid TBox passes consistency check", True)

    def bad_raises(prop: URIRef, bad_val: float, label: str) -> bool:
        gb = load_and_reason(TBOX_PATH)
        bw = URIRef(NS_STR + "badWall")
        gb.add((bw, RDF.type, NS.ThickDiagonalWall))
        gb.add((bw, prop, Literal(bad_val, datatype=XSD.float)))
        owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(gb)
        try:
            check_functional_property_consistency(gb)
            return False
        except TBoxInconsistencyError:
            return True

    check("bad angle (ThickDiagonalWall + wallTangentAngleDeg=90) raises error",
          bad_raises(NS.wallTangentAngleDeg, 90.0, "angle conflict"))
    check("bad thickness (ThickDiagonalWall + wallHalfThickness=0.1) raises error",
          bad_raises(NS.wallHalfThickness, 0.1, "thickness conflict"))

    print()
    return all_pass


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DiagonalRoom OWL TBox → gate compiler (v2)")
    parser.add_argument("--tbox", default=str(TBOX_PATH))
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--demo-inconsistency", action="store_true")
    args = parser.parse_args()

    if not HAS_OWLRL:
        print("ERROR: owlrl not installed.  pip install owlrl"); return

    if args.verify:
        ok = verify()
        print("All tests PASS." if ok else "Some tests FAILED."); return

    if args.demo_inconsistency:
        demo_inconsistency(); return

    compile_diagonal_room(Path(args.tbox), verbose=True)


if __name__ == "__main__":
    main()
