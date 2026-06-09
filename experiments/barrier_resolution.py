"""Experiment 6: resolving the barrier, constructively.

The Sly-Sun theorem forbids exactly one thing: a general efficient algorithm
for the off-diagonal aggregate above the critical degree (that would give
NP = RP). This experiment demonstrates the two routes it does NOT forbid,
both implemented in worldkernel.tractable:

ROUTE 1 (certify): Weitz's self-avoiding-walk recursion with interval
boundaries returns rigorous upper and lower bounds on every marginal, at
every depth, on every graph. Below d_c the interval contracts geometrically;
approaching and crossing d_c the contraction rate collapses while the cost
per depth level grows as (d-1)^depth. The barrier becomes a measurable
property of the certificate instead of a silent failure. (On a finite graph
every walk eventually terminates, so this is a rate statement, not a cliff:
the same honesty as the keystone figure.)

ROUTE 2 (structure): Sly-Sun is a worst-case statement about DEGREE; exact
computation is governed by WIDTH. A ring of m cliques of size 9 has internal
degree 8, far above d_c = 5.141, yet treewidth 9: its hard-core marginals are
computed EXACTLY by a transfer matrix in O(m s^3), at n = 360 where
enumeration has 2^360 states and where belief propagation is measurably
wrong. Worlds whose constraints come from structured ontologies live in this
class by design: restriction is the design constraint that keeps the
off-diagonal computable.

Usage: python experiments/barrier_resolution.py
"""

import random
import time
from pathlib import Path

import numpy as np

from worldkernel import (
    d_critical,
    order_parameter,
    ring_of_cliques,
    transfer_marginals,
    weitz_interval,
)
from worldkernel.barrier import bp_marginals, exact_marginals, random_regular

N = 60
SWEEP = {3: range(2, 15), 4: range(2, 13), 5: range(2, 11), 7: range(2, 9)}


def route1():
    print("ROUTE 1: certified marginal intervals (Weitz SAW tree, n=60)\n")
    rng = random.Random(11)
    curves = {}
    for d, depths in SWEEP.items():
        adj = random_regular(N, d, rng)
        op = order_parameter(d, 1.0)
        widths = []
        t0 = time.time()
        for L in depths:
            lo, hi = weitz_interval(adj, 0, 1.0, L)
            widths.append(hi - lo)
        curves[d] = (list(depths), widths)
        print(f"  d={d} ((d-1)eta={op:.2f}): width {widths[0]:.4f} -> {widths[-1]:.6f} "
              f"over depths {list(depths)[0]}..{list(depths)[-1]}  ({time.time()-t0:.1f}s)")
    print(f"\n  certified width at comparable compute: d=3 {curves[3][1][-1]:.6f} vs "
          f"d=7 {curves[7][1][-1]:.4f}; the contraction rate tracks (d-1)eta.")
    return curves


def route2():
    print("\nROUTE 2: structure beats degree (ring of cliques, s=9, degree 8-9)\n")
    # validation against brute force where it exists
    for m, s in ((3, 4), (3, 5)):
        err = np.max(np.abs(transfer_marginals(m, s) - exact_marginals(ring_of_cliques(m, s), m * s)))
        print(f"  validation m={m} s={s} (n={m*s}): |transfer - enumeration| = {err:.2e}")

    m, s = 40, 9
    t0 = time.time()
    tm = transfer_marginals(m, s)
    t_tm = time.time() - t0
    adj = ring_of_cliques(m, s)
    t0 = time.time()
    bp = bp_marginals(adj, m * s)
    t_bp = time.time() - t0
    err = np.abs(bp - tm)
    dc = d_critical(1.0)
    print(f"\n  n = {m*s}, internal degree {s-1} > d_c = {dc:.3f} "
          f"((d-1)eta = {order_parameter(s-1, 1.0):.2f}: 'hard' regime by degree)")
    print(f"  transfer matrix (exact):  {t_tm*1000:.2f} ms   (enumeration: 2^{m*s} states)")
    print(f"  belief propagation:       {t_bp:.1f} s, WRONG: mean error {err.mean():.4f}, "
          f"max {err.max():.4f}")
    print("  The tractability parameter is width, not degree: treewidth ~ s makes the")
    print("  off-diagonal exactly computable in the regime degree calls hard.")
    return tm, bp, err


def main():
    curves = route1()
    tm, bp, err = route2()

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        colors = {3: "#2ca02c", 4: "#1f77b4", 5: "#ff7f0e", 7: "crimson"}
        for d, (depths, widths) in curves.items():
            op = order_parameter(d, 1.0)
            ax1.semilogy(depths, np.maximum(widths, 1e-7), "o-", color=colors[d],
                         label=f"d={d}, (d-1)η={op:.2f}")
        ax1.set_xlabel("SAW-tree depth")
        ax1.set_ylabel("certified interval width (log)")
        ax1.set_title("Route 1: the certificate contracts geometrically\n"
                      "below the threshold; the rate collapses above")
        ax1.legend(fontsize=8)

        idx = np.arange(27)  # three cliques' worth of vertices
        ax2.plot(idx, tm[:27], "o-", color="#1f77b4", label="transfer matrix (exact)")
        ax2.plot(idx, bp[:27], "s--", color="crimson", alpha=0.7,
                 label=f"belief propagation (mean err {err.mean():.3f})")
        ax2.set_xlabel("vertex (first 3 of 40 cliques, n=360)")
        ax2.set_ylabel("occupation marginal")
        ax2.set_title("Route 2: degree 8 > d_c, yet exact in milliseconds\n"
                      "(treewidth 9): structure beats degree")
        ax2.legend(fontsize=8)
        plt.tight_layout()
        out = Path(__file__).parent / "barrier_resolution_repro.png"
        plt.savefig(out, dpi=130)
        print(f"\nChart saved: {out}")
    except ImportError:
        print("\n(matplotlib not installed: chart skipped)")


if __name__ == "__main__":
    main()
