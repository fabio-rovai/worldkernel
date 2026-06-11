"""Experiment 20: the tractable counterfactual query class, widened.

The query-scar result handled a single marginal. This shows the kernel's
actual queries, the pairwise off-diagonal P(Y_i, Y_j) and higher k-local
counterfactuals, are computed exactly in poly(width) time on worlds ABOVE
the Sly-Sun degree threshold, where the partition function |C| is
exponential and counting is hard. The coupling rank (<= 2^k) is the query's
scar dimension; width is the per-pattern cost. Validated against full
enumeration at small scale.

Usage: python experiments/query_class_demo.py
"""

import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worldkernel.barrier import order_parameter  # noqa: E402
from worldkernel.query_class import (  # noqa: E402
    coupling_rank,
    necessity_from_couplings,
    occupation_pattern_prob,
    pairwise_offdiagonal,
)
from worldkernel.tractable import disjointness_graph, ring_of_cliques  # noqa: E402


def enum_pattern_prob(adj, pattern, lam=1.0):
    """Brute-force P(pattern) by enumerating admissible worlds (validation)."""
    n = len(adj)
    mask = [0] * n
    for i in range(n):
        for j in adj[i]:
            mask[i] |= 1 << j
    z = 0.0
    num = 0.0
    for s in range(1 << n):
        ok = True
        t = s
        while t:
            i = (t & -t).bit_length() - 1
            if mask[i] & s:
                ok = False
                break
            t &= t - 1
        if not ok:
            continue
        w = lam ** bin(s).count("1")
        z += w
        if all(((s >> v) & 1) == b for v, b in pattern.items()):
            num += w
    return num / z, z


def main() -> None:
    print("k-local counterfactual queries ABOVE the threshold; |C| exponential,")
    print("query computed in poly(width). Validated vs enumeration.\n")
    print(f"  {'graph':>13} | {'deg':>3} | {'(d-1)eta':>8} | {'|C|':>8} | "
          f"{'query':>22} | {'rank':>4} | {'width':>5} | {'value':>9} | {'ok':>4}")

    worlds = [
        ("ring 3x6", ring_of_cliques(3, 6)),    # degree 6, above d_c
        ("ring 3x7", ring_of_cliques(3, 7)),    # degree 7
        ("taxonomy", disjointness_graph(4, 2, seed=11)),
    ]
    for name, adj in worlds:
        deg = max(len(a) for a in adj)
        op = order_parameter(deg, 1.0)
        # k=1: marginal; k=2: pairwise off-diagonal; k=3: triple
        queries = [
            ("P(Y_0=1)  [k=1]", {0: 1}),
            ("P(Y_0=1,Y_h=1) [k=2]", {0: 1, len(adj) - 1: 1}),
            ("triple [k=3]", {0: 1, 1: 0, len(adj) - 1: 1}),
        ]
        for label, pat in queries:
            v = occupation_pattern_prob(adj, pat)
            truth, Z = enum_pattern_prob(adj, pat)
            ok = abs(v.value - truth) < 1e-9
            print(f"  {name:>13} | {deg:>3} | {op:>8.2f} | {Z:>8.0f} | "
                  f"{label:>22} | {v.coupling_rank:>4} | {v.width:>5} | "
                  f"{v.value:>9.5f} | {str(ok):>4}")

    print("\nThe off-diagonal P(Y_i, Y_j) is the kernel's central object; it is")
    print("a rank-4 query computed in poly(width) on a degree-6 world where")
    print("counting is Sly-Sun hard. A downstream necessity functional stays")
    print("k=2 hence tractable:")
    adj = ring_of_cliques(3, 6)
    pn = necessity_from_couplings(adj, treat=0, ctrl=len(adj) - 1)
    print(f"  PN-analogue on ring 3x6: {pn.value:.5f}  (rank {pn.coupling_rank}, "
          f"width {pn.width})")

    print("\nCoupling-rank hierarchy (scar dimension by query locality k):")
    for k in (1, 2, 3, 4):
        print(f"  k={k}: coupling rank <= {coupling_rank(k)}  "
              f"(independent of n and |C|)")

    print("\nTheorem (the tractable query class): every k-local counterfactual")
    print("query on a treewidth-w world is exact in O(2^k * n * 2^{w+1}) time,")
    print("independent of |C|. Bounded k and w => polynomial, even above the")
    print("Sly-Sun threshold where counting is hard. This is the kernel's")
    print("working regime: pairwise off-diagonals, nested three-world")
    print("counterfactuals, and their functionals all live here.")


if __name__ == "__main__":
    main()
