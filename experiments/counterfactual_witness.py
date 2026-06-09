"""Experiment 1: the off-diagonal witness.

Two worlds with identical rung-1 (observational) and rung-2 (interventional)
data but different cross-world coupling, hence different probability of
necessity (PN 0.286 vs 0.500). Any reasoner holding only rung-1/2 data returns
one number and collapses the pair; the kernel separates them exactly.

Optional LLM baseline (requires the `claude` CLI): run with --llm to ask
claude for the PN with and without the off-diagonal. Verified behaviour:
blind, it returns a single excess-risk-style number for both worlds; handed
the coupling, it separates them.

Usage: python experiments/counterfactual_witness.py [--llm]
"""

import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from worldkernel import frechet_pn_bounds, witness_pair

A, B = witness_pair(r0=0.5, r1=0.7)
NAMES = {
    "A": ("monotonic: treatment never hurts", A),
    "B": ("independent potential outcomes", B),
}


def rung_report() -> None:
    print("RUNG 1 (observational P(X,Y)), identical for both worlds:")
    for k, v in A.observational().items():
        print(f"   P({k[0]},{k[1]}) = {v:.3f}")
    print("\nRUNG 2 (interventional), identical for both worlds:")
    print(f"   P(Y=1|do X=1) = {A.r1:.3f}   P(Y=1|do X=0) = {A.r0:.3f}   ACE = {A.ace:.3f}")
    print("\nRUNG 3 (counterfactual PN), DIFFERS: this is the off-diagonal:")
    for tag, (desc, k) in NAMES.items():
        print(f"   {tag} ({desc:35s})  P(Y0=1,Y1=1) = {k.p11:.2f}  ->  PN = {k.pn():.4f}")
    lo, hi = frechet_pn_bounds(A.r0, A.r1)
    print(f"\nIdentified PN interval from rung-1/2 alone: [{lo:.3f}, {hi:.3f}]")
    print("The off-diagonal coupling selects the point inside it.")


def ask(prompt: str):
    out = subprocess.run(
        ["claude", "-p", prompt], capture_output=True, text=True, timeout=90
    ).stdout.strip()
    m = re.findall(r"[01]?\.\d+|\b[01]\b|\d+%", out)
    if not m:
        return None
    tok = m[-1]
    return float(tok[:-1]) / 100 if tok.endswith("%") else float(tok)


def llm_baseline() -> None:
    obs = "; ".join(f"P({k[0]},{k[1]})={v:.2f}" for k, v in A.observational().items())
    do = f"P(Y=1|do(X=1))={A.r1:.2f}; P(Y=1|do(X=0))={A.r0:.2f}"
    q = (
        "Among patients who were treated (X=1) and recovered (Y=1), what is the "
        "probability they would NOT have recovered if untreated (probability of "
        "necessity)? Answer with ONE number in [0,1], nothing else."
    )
    base = f"A treatment X (1/0) affects recovery Y (1/0). Randomized trial.\nObservational: {obs}\nInterventional: {do}\n"
    tasks = [("blind (rung 1+2 only)", base + "\n" + q)] * 3
    for tag, (_, k) in NAMES.items():
        coup = "; ".join(f"P(Y0={i},Y1={j})={v:.2f}" for (i, j), v in k.joint().items())
        tasks += [(f"+ off-diagonal of {tag}", base + f"Cross-world joint (Y0=untreated, Y1=treated): {coup}\n\n{q}")] * 3
    with ThreadPoolExecutor(max_workers=6) as ex:
        answers = list(ex.map(lambda t: (t[0], ask(t[1])), tasks))
    agg: dict[str, list] = {}
    for label, val in answers:
        agg.setdefault(label, []).append(val)
    print(f"\n{'condition':28s} | {'claude mean':>11} | raw")
    for label, vals in agg.items():
        clean = [v for v in vals if v is not None]
        mean = sum(clean) / len(clean) if clean else float("nan")
        print(f"{label:28s} | {mean:>11.3f} | {vals}")


def main() -> None:
    rung_report()
    if "--llm" in sys.argv:
        llm_baseline()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = ["PN_A\n(kernel)", "PN_B\n(kernel)"]
        vals = [A.pn(), B.pn()]
        plt.figure(figsize=(5.5, 4))
        plt.bar(labels, vals, color="#1f77b4")
        plt.ylabel("probability of necessity (PN)")
        plt.title("Same diagonal, different off-diagonal:\nonly the kernel separates the worlds")
        for i, v in enumerate(vals):
            plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
        plt.tight_layout()
        out = Path(__file__).parent / "counterfactual_witness_repro.png"
        plt.savefig(out, dpi=130)
        print(f"\nChart saved: {out}")
    except ImportError:
        print("\n(matplotlib not installed: chart skipped)")


if __name__ == "__main__":
    main()
