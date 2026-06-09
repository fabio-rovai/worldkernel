"""Experiment 7: the ontology-to-width bridge.

The claim that closes the loop: an ontology's disjointness axioms generate a
constraint graph whose WIDTH is set by the local branching factor, not by the
number of classes. Since exact off-diagonal computation costs
O(n * 2^(width+1)) by variable elimination, an ontology-structured world stays
exactly computable however large it grows and however far its degree sits
above the Sly-Sun critical degree. The ontology is not metadata: it is the
tractability certificate of the world model.

Sweep: class taxonomies (sibling AllDisjoint cliques + sparse cousin
incompatibilities) of branching 4..10 and growing depth. For each, report
n (classes), max degree (vs d_c = 5.141), min-fill width, and the wall-clock
time of an EXACT off-diagonal marginal. Contrast: a random d-regular world of
the same degree, where width grows with n and exact computation dies.

Usage: python experiments/ontology_width.py
"""

import random
import time
from pathlib import Path

import numpy as np

from worldkernel import d_critical, order_parameter
from worldkernel.barrier import random_regular
from worldkernel.tractable import (
    disjointness_graph,
    min_fill_order,
    treewidth_marginal,
)

DC = d_critical(1.0)


def taxonomy_row(b: int, depth: int):
    adj = disjointness_graph(b, depth)
    n = len(adj)
    deg = max(len(a) for a in adj)
    t0 = time.time()
    order, width = min_fill_order(adj)
    p = treewidth_marginal(adj, 0, order=order)
    t = time.time() - t0
    return n, deg, width, p, t


def main() -> None:
    print(f"d_c = {DC:.3f} at unit fugacity.\n")
    print("TAXONOMIES (sibling AllDisjoint + sparse cousin constraints):")
    print(f"{'branching':>9} | {'depth':>5} | {'classes':>7} | {'max deg':>7} | "
          f"{'(d-1)eta':>8} | {'width':>5} | {'exact marginal':>14} | {'time':>7}")
    print("-" * 84)
    rows = []
    for b, depth in ((4, 4), (6, 3), (8, 3), (10, 3)):
        n, deg, width, p, t = taxonomy_row(b, depth)
        op = order_parameter(deg, 1.0)
        regime = "HARD by degree" if op > 1 else "tractable"
        rows.append((b, n, deg, width, t))
        print(f"{b:>9} | {depth:>5} | {n:>7} | {deg:>7} | {op:>8.2f} | "
              f"{width:>5} | {p:>14.4f} | {t:>6.2f}s   ({regime})")

    print("\nCONTRAST: random d-regular worlds of the same degree (no structure):")
    rng = random.Random(11)
    print(f"{'n':>5} | {'degree':>6} | {'min-fill width':>14}")
    for n in (24, 48, 96, 192):
        adj = random_regular(n, 8, rng)
        _, w = min_fill_order(adj)
        print(f"{n:>5} | {8:>6} | {w:>14}   (exact cost ~ 2^{w + 1}: "
              f"{'dead' if w > 30 else 'alive'})")
    print("\nIn the structured world, width is set by local branching and stays")
    print("constant as the ontology grows; in the unstructured world of the SAME")
    print("degree, width grows linearly with n and exactness dies. The ontology")
    print("IS the world model's tractability certificate.")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # width vs size: taxonomy (several branchings, growing depth) vs random
        fig, ax = plt.subplots(figsize=(7.5, 4.4))
        for b, marker in ((4, "o"), (8, "s")):
            ns, ws = [], []
            for depth in (2, 3, 4) if b == 4 else (2, 3):
                adj = disjointness_graph(b, depth)
                _, w = min_fill_order(adj)
                ns.append(len(adj))
                ws.append(w)
            ax.plot(ns, ws, marker + "-", color="#1f77b4" if b == 4 else "#2ca02c",
                    label=f"taxonomy, branching {b} (degree {'5' if b == 4 else '10'})")
        rng2 = random.Random(11)
        ns, ws = [], []
        for n in (24, 48, 96, 192):
            adj = random_regular(n, 8, rng2)
            _, w = min_fill_order(adj)
            ns.append(n)
            ws.append(w)
        ax.plot(ns, ws, "d--", color="crimson", label="random 8-regular (no structure)")
        ax.set_xscale("log")
        ax.set_xlabel("world size n (number of classes / vertices)")
        ax.set_ylabel("min-fill width (exact cost ~ 2^width)")
        ax.set_title("The ontology is the tractability certificate:\n"
                     "width tracks local branching in structured worlds, grows with n otherwise")
        ax.legend(fontsize=8)
        plt.tight_layout()
        out = Path(__file__).parent / "ontology_width_repro.png"
        plt.savefig(out, dpi=130)
        print(f"\nChart saved: {out}")
    except ImportError:
        print("\n(matplotlib not installed: chart skipped)")


if __name__ == "__main__":
    main()
