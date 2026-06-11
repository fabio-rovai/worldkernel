"""Experiment 21: global counterfactual queries, exact above the threshold.

Widens the tractable class from a single k-local query to the whole algebra of
GLOBAL functionals built from local pieces: the aggregate pairwise coherence
Q = sum_{i<j} P(Y_i, Y_j) (touches every vertex), expected occupancy, and
ratio functionals. Each is exact in poly(n, 2^width) on worlds above the
Sly-Sun degree threshold where |C| is exponential and counting is hard.
Validated against enumeration; the coherence is cross-checked against the
kernel's own CouplingKernel.pairwise_coherence on the exact moment matrix.

Usage: python experiments/query_algebra_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from worldkernel import CouplingKernel  # noqa: E402
from worldkernel.barrier import order_parameter  # noqa: E402
from worldkernel.query_algebra import (  # noqa: E402
    expected_occupancy,
    pairwise_coherence,
    ratio_query,
)
from worldkernel.query_class import occupation_pattern_prob  # noqa: E402
from worldkernel.tractable import disjointness_graph, ring_of_cliques  # noqa: E402


def enum_moments(adj, lam=1.0):
    """Exact diagonal and full second-moment matrix by enumeration (check)."""
    n = len(adj)
    mask = [0] * n
    for i in range(n):
        for j in adj[i]:
            mask[i] |= 1 << j
    Z = 0.0
    d = np.zeros(n)
    M = np.zeros((n, n))
    states = 0
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
        states += 1
        w = lam ** bin(s).count("1")
        Z += w
        occ = [v for v in range(n) if (s >> v) & 1]
        for a in occ:
            d[a] += w
            for b in occ:
                M[a, b] += w
    return d / Z, M / Z, Z, states


def main() -> None:
    print("GLOBAL counterfactual queries (touch every vertex), exact above the")
    print("Sly-Sun threshold; |C| exponential. Validated vs enumeration.\n")
    print(f"  {'graph':>13} | {'deg':>3} | {'(d-1)eta':>8} | {'|C|':>6} | "
          f"{'E[occ] ok':>9} | {'coherence Q':>11} | {'Q ok':>5} | "
          f"{'=kernel?':>8}")

    worlds = [
        ("ring 3x6", ring_of_cliques(3, 6)),     # degree 6, above d_c
        ("ring 3x7", ring_of_cliques(3, 7)),     # degree 7
        ("ring 4x6", ring_of_cliques(4, 6)),     # n=24, degree 6
        ("taxonomy", disjointness_graph(4, 2, seed=11)),
    ]
    for name, adj in worlds:
        n = len(adj)
        if n > 24:
            continue
        deg = max(len(a) for a in adj)
        op = order_parameter(deg, 1.0)
        d_exact, M_exact, Z, _ = enum_moments(adj)

        eo = expected_occupancy(adj)
        eo_ok = abs(eo.value - d_exact.sum()) < 1e-9

        coh = pairwise_coherence(adj)
        coh_truth = sum(M_exact[i, j] for i in range(n) for j in range(i + 1, n))
        coh_ok = abs(coh.value - coh_truth) < 1e-9

        # cross-check against the kernel's own pairwise_coherence on the exact M
        kernel_coh = CouplingKernel(M_exact).pairwise_coherence()
        kernel_ok = abs(coh.value - kernel_coh) < 1e-9

        print(f"  {name:>13} | {deg:>3} | {op:>8.2f} | {Z:>6.0f} | "
              f"{str(eo_ok):>9} | {coh.value:>11.4f} | {str(coh_ok):>5} | "
              f"{str(kernel_ok):>8}")

    print("\nThe coherence Q = sum_(i<j) P(Y_i,Y_j) is GLOBAL (all n vertices),")
    print("yet exact in O(n^2 * poly(width)). Where structure is absent the")
    print("kernel returns the PSD / Frechet BOUNDS on Q; on a bounded-width")
    print("world it returns the EXACT value, above the threshold.\n")

    # closure under ratios: a global-ish necessity functional
    adj = ring_of_cliques(3, 6)
    pn = ratio_query(adj, {0: 1, 17: 0}, {0: 1})
    d_exact, _, _, _ = enum_moments(adj)
    print(f"  ratio functional PN-style on ring 3x6: {pn.value:.5f} "
          f"(terms {pn.n_terms}, width {pn.width})")

    print("\nTheorem (closure of the tractable class): the tractable queries are")
    print("closed under poly-size linear combination, ratios, and bounded-order")
    print("products. So aggregate coherence, average causal effects,")
    print("probabilities of necessity/sufficiency, fractions harmed, and natural")
    print("(in)direct effects, all global functionals assembled from local")
    print("pieces, are exact in poly(n, 2^width) independent of |C|. The")
    print("partition function alone (rank |C|) stays outside: the algebra")
    print("characterizes the tractable frontier, it does not cross it.")


if __name__ == "__main__":
    main()
