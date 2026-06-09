"""Experiment 4: the keystone. Barrier and measurement on one axis.

The aggregate counterfactual over units with mutual-exclusion constraints is a
hard-core model over independent sets of a constraint graph. Sweep the graph
degree d: belief-propagation marginal error (vs exact enumeration at n=16)
rises with the Sly-Sun order parameter (d-1)*eta, both turning at d_c = 5.141
(unit fugacity). Below d_c the off-diagonal aggregate is tractably computable;
above it is not. Finite n gives a monotonic correspondence, not a cliff: the
sharp transition is the asymptotic theorem.

Usage: python experiments/barrier_sweep.py
"""

import random
from pathlib import Path

import numpy as np

from worldkernel import d_critical, order_parameter
from worldkernel.barrier import bp_marginals, exact_marginals, random_regular

LAM = 1.0
N = 16
GRAPHS_PER_D = 8
DEGREES = [2, 3, 4, 5, 6, 7, 8]


def main() -> None:
    rng = random.Random(11)
    dc = d_critical(LAM)
    print(f"Sly-Sun threshold at lambda={LAM}: d_c = {dc:.3f}\n")
    print(f"{'d':>3} | {'(d-1)eta':>9} | {'BP marginal err':>16} | regime")
    print("-" * 52)

    ds, ops, errs = [], [], []
    for d in DEGREES:
        op = order_parameter(d, LAM)
        per_graph = []
        for _ in range(GRAPHS_PER_D):
            adj = random_regular(N, d, rng)
            ex = exact_marginals(adj, N, LAM)
            bp = bp_marginals(adj, N, LAM)
            per_graph.append(float(np.mean(np.abs(ex - bp))))
        err = float(np.mean(per_graph))
        ds.append(d)
        ops.append(op)
        errs.append(err)
        regime = "tractable (Weitz)" if op < 1 else "HARD (Sly-Sun)"
        print(f"{d:>3} | {op:>9.3f} | {err:>16.4f} | {regime}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(7.8, 4.6))
        ax1.plot(ds, errs, "o-", color="crimson", lw=2.2, label="BP marginal error")
        ax1.set_xlabel("constraint-graph degree d")
        ax1.set_ylabel("BP marginal error vs exact", color="crimson")
        ax2 = ax1.twinx()
        ax2.plot(ds, ops, "s--", color="#1f77b4", lw=2.0, label="(d-1)η")
        ax2.axhline(1.0, color="#1f77b4", ls=":", alpha=0.6)
        ax2.set_ylabel("(d-1)η", color="#1f77b4")
        ax1.axvline(dc, color="k", ls="--", alpha=0.5)
        ax1.axvspan(dc, max(DEGREES) + 0.3, color="grey", alpha=0.12)
        plt.title("Computing the counterfactual aggregate tracks the Sly-Sun threshold")
        plt.tight_layout()
        out = Path(__file__).parent / "barrier_sweep_repro.png"
        plt.savefig(out, dpi=130)
        print(f"\nChart saved: {out}")
    except ImportError:
        print("\n(matplotlib not installed: chart skipped)")


if __name__ == "__main__":
    main()
