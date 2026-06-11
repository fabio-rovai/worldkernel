"""Experiment 22: long-range off-diagonals through a low-rank channel.

The final widening: a genuinely LONG-RANGE counterfactual coupling
P(Y_i, Y_j) between vertices in far-apart cliques, computed exactly through
the rank-(s+1) transfer channel in poly(interaction rank), independent of
|C| AND exponentially below the 2^treewidth width engine. The interaction
rank chi = s+1 is linear in the clique-separator size s, while the width
engine pays 2^s and counting pays |C| (exponential in the number of
cliques). The K(m,m) bottleneck is the extreme: chi = 2 versus treewidth m.

Usage: python experiments/interaction_rank_demo.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from worldkernel.interaction import (  # noqa: E402
    _enum_pair,
    interaction_rank,
    ring_clique_pair,
    ring_clique_transfer,
    treewidth_cost,
)
from worldkernel.phases import kmm_exact_marginal  # noqa: E402
from worldkernel.query_scar import kmm_marginal_via_scar  # noqa: E402


def main() -> None:
    print("1. Long-range off-diagonal exact via the rank-(s+1) channel")
    print("   (validated vs enumeration on small rings)\n")
    print(f"  {'m':>3} | {'s':>2} | {'pair (far cliques)':>20} | "
          f"{'transfer':>10} | {'enum':>10} | {'ok':>4}")
    for m, s, ga, va, gb, vb in [(3, 4, 0, 2, 2, 3), (4, 3, 0, 0, 2, 1),
                                 (4, 4, 0, 3, 2, 2), (5, 3, 0, 0, 3, 2)]:
        mine = ring_clique_pair(m, s, ga, va, gb, vb)
        truth = _enum_pair(m, s, ga, va, gb, vb)
        label = f"({ga},{va})-({gb},{vb})"
        print(f"  {m:>3} | {s:>2} | {label:>20} | {mine:>10.6f} | "
              f"{truth:>10.6f} | {str(abs(mine - truth) < 1e-9):>4}")

    print("\n2. The separation: interaction rank vs width-engine cost vs |C|")
    print("   long-range pair on rings far too large to enumerate or to run")
    print("   the 2^treewidth width engine on\n")
    print(f"  {'m (cliques)':>11} | {'s':>3} | {'chi = s+1':>9} | "
          f"{'2^treewidth':>12} | {'|C| (mag)':>12} | {'pair value':>11} | "
          f"{'time':>8}")
    for m, s in [(100, 10), (1000, 20), (5000, 30)]:
        chi = interaction_rank(s)
        wcost = treewidth_cost(s)
        T = ring_clique_transfer(s)
        top = float(np.linalg.eigvals(T).real.max())
        logZ = m * np.log10(top)        # |C| ~ top^m: report its magnitude
        t0 = time.time()
        val = ring_clique_pair(m, s, 0, 2, m // 2, 3)   # cliques m/2 apart
        dt = time.time() - t0
        print(f"  {m:>11} | {s:>3} | {chi:>9} | {wcost:>12.3g} | "
              f"10^{logZ:>9.0f} | {val:>11.4g} | {dt * 1000:>6.1f}ms")
    print("  chi stays linear (11, 21, 31); the width engine would pay 2^s")
    print("  (10^3, 10^6, 10^9); |C| is astronomically large; the long-range")
    print("  coupling is computed in milliseconds through the rank-chi channel.")

    print("\n3. The extreme: K(m,m), interaction rank 2 vs treewidth m")
    print(f"  {'m':>4} | {'chi':>3} | {'treewidth':>9} | {'|C|':>10} | "
          f"{'marginal':>10}")
    for m in (20, 100, 200):
        sq = kmm_marginal_via_scar(m)
        assert abs(sq.value - kmm_exact_marginal(m)) < 1e-12
        print(f"  {m:>4} | {2:>3} | {m:>9} | {sq.n_admissible:>10.3g} | "
              f"{sq.value:>10.6f}")
    print("  the central K(m,m) cut has bond dimension 2^m but transfer RANK 2")
    print("  (the right side only distinguishes 'left empty' vs 'not'); the")
    print("  query is read from that rank-2 phase channel in O(1).")

    print("\nTheorem (interaction-rank tractability): every bounded-rank")
    print("counterfactual query on a world of interaction rank chi is exact in")
    print("poly(n, chi) time, independent of |C|. chi can be exponentially")
    print("below the treewidth (clique separators: chi = s+1 vs 2^s; K(m,m):")
    print("chi = 2 vs treewidth m), so this strictly widens the width-based")
    print("class. Ontology worlds, whose disjointness axioms are dense cliques,")
    print("have small interaction rank by construction.")


if __name__ == "__main__":
    main()
