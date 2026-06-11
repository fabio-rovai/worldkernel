"""Experiment 18: the FULL Farago pipeline, end to end, above the threshold.

The single-round test (farago_falsifier.py) showed the H-PM selection engine
is exact. This runs the ENTIRE Algorithm 1: edge-by-edge graph construction
G_1 subset ... subset G_m = G, the real BIDC Markov chain building each
matrix's rows, the value-vector outputs of each round FED FORWARD as the next
round's chain seeds, with EXACT uniform H-perfect-matchings (permanent-ratio
sampling, no JSV). The target graph is K_7: every vertex has degree 6, so it
is squarely above the Sly-Sun degree-6 threshold where the BIDC chain mixes
exponentially slowly and approximate counting is NP-hard unless NP=RP.

The question: does the full composition produce a uniform sample of I(G)?

What this CAN show: whether the algorithm's logic and multi-round composition
are internally correct, isolated from the JSV approximation (which the eq-(20)
analysis already found to be a patchable error). With exact matchings and
stationary inputs, the ideal analysis predicts an exactly uniform output; a
deviation would expose a composition bug the single-round test missed.

What this CANNOT show: whether the asymptotic NP=RP claim holds. The Sly-Sun
hardness is a statement about N -> infinity; at N=7 there is no real threshold
and the ground truth is enumerable. A uniform output here confirms internal
consistency, NOT that the construction beats the barrier asymptotically (that
hinges on whether JSV stays accurate enough as N grows, which simulation at
enumerable N cannot decide).

Usage: python experiments/farago_full_pipeline.py
"""

import itertools
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from farago_falsifier import permanent, sample_uniform_pm, tv  # noqa: E402

N = 7                  # wheel W_6: hub of degree 6 (above the threshold)
MATRIX_N = 10          # matrix size; paper needs 2N^2 so H-PM failures are
# exponentially rare (its Theorem 4). At feasible exact-matching sizes we
# cannot reach 2N^2=98, so we pick a graph whose per-edge independent-set
# density stays high (failures rare) to avoid conditioning bias on completion.
CHAIN_STEPS = 10       # BIDC steps per row
RUNS = 1500
SEED = 11


def wheel_edges(n):
    """W_(n-1): a hub (vertex n-1) joined to an (n-1)-cycle. The hub has
    degree n-1 = 6 for n=7 (above the Sly-Sun threshold); the rim density
    keeps H-PMs plentiful so the failure rate stays low at small matrices."""
    hub = n - 1
    rim = list(range(hub))
    edges = [(rim[i], rim[(i + 1) % len(rim)]) for i in range(len(rim))]
    edges += [(hub, r) for r in rim]
    return edges


def independent_sets(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    out = []
    for r in range(n + 1):
        for c in itertools.combinations(range(n), r):
            s = set(c)
            if all(not (adj[v] & s) for v in c):
                out.append(frozenset(c))
    return out, adj


def bidc_step(state: frozenset, adj, n, rng) -> frozenset:
    """One Basic Insert/Delete Chain transition on I(G)."""
    u = rng.randrange(n)
    if u in state:
        return state - {u}
    if not (adj[u] & state):  # u not adjacent to any member: can insert
        return state | {u}
    return state


def run_round(A, adj_k, n, plus_set, rng):
    """One round: build the matrix by running the BIDC chain on G_k from the
    seeds A (rows independent), then extract the value vector of an EXACT
    uniform H-PM with H = I(G_{k+1}). Returns the n new samples, or None on
    failure (no H-PM)."""
    # Step 3+4: row i is the chain trajectory of length MATRIX_N from A[i]
    X = []
    for i in range(MATRIX_N):
        st = A[i]
        row = []
        for _ in range(MATRIX_N):
            st = bidc_step(st, adj_k, n, rng)
            row.append(st)
        X.append(row)
    # Step 5/6: H-skeleton + exact uniform H-PM
    B = [[1 if X[i][j] in plus_set else 0 for j in range(MATRIX_N)]
         for i in range(MATRIX_N)]
    choice = sample_uniform_pm(B, rng)
    if choice is None:
        return None
    return [X[i][choice[i]] for i in range(MATRIX_N)]  # value vector


def main() -> None:
    rng = random.Random(SEED)

    # K_7 and its edge order; the successive subgraphs G_1 .. G_m
    edges = list(itertools.combinations(range(N), 2))  # 21 edges, all degree 6
    rng.shuffle(edges)
    m = len(edges)
    IS_full, _ = independent_sets(N, edges)
    print(f"target G = K_7 (every vertex degree 6, above the threshold); "
          f"{m} edges, {m - 1} rounds")
    print(f"|I(G)| = {len(IS_full)} independent sets; matrix {MATRIX_N}x{MATRIX_N}, "
          f"EXACT uniform H-PMs, real BIDC chain\n")

    # precompute the independent-set families and adjacency for each G_k
    fams, adjs = [], []
    for k in range(1, m + 1):
        f, a = independent_sets(N, edges[:k])
        fams.append(set(f))
        adjs.append(a)

    out_counts: dict = {}
    failures = 0
    completed = 0
    for run in range(RUNS):
        # base case: start from a uniform sample of I(G_1) per row
        G1 = list(fams[0])
        A = [G1[rng.randrange(len(G1))] for _ in range(MATRIX_N)]
        ok = True
        for k in range(1, m):                       # round k: G_k -> G_{k+1}
            res = run_round(A, adjs[k - 1], N, fams[k], rng)
            if res is None:
                failures += 1
                ok = False
                break
            A = res
        if ok:
            completed += 1
            for v in A:                              # final samples in I(G)
                out_counts[v] = out_counts.get(v, 0) + 1

    n_samples = sum(out_counts.values())
    tv_out = tv(out_counts, IS_full)
    # control: direct uniform sampling at the same sample size
    ctrl: dict = {}
    for _ in range(n_samples):
        v = IS_full[rng.randrange(len(IS_full))]
        ctrl[v] = ctrl.get(v, 0) + 1
    tv_ctrl = tv(ctrl, IS_full)

    print(f"completed runs: {completed}/{RUNS} (failures: {failures})")
    print(f"output samples of I(G): {n_samples}\n")
    print(f"TV( full-pipeline output , uniform on I(G) ) = {tv_out:.4f}")
    print(f"TV( direct uniform , uniform ) [control]     = {tv_ctrl:.4f}  "
          f"(sampling-noise floor)\n")

    total = n_samples
    u = 1.0 / len(IS_full)
    print("per-set empirical frequency vs uniform:")
    for s in IS_full:
        f = out_counts.get(s, 0) / total
        print(f"   {sorted(s) if s else '(empty)'}: {f:.4f}  (dev {f - u:+.4f})")

    print("\nVerdict:")
    if tv_out > tv_ctrl * 4 and tv_out > 0.02:
        print(f"  DEVIATES. The full pipeline output is {tv_out:.4f} from uniform,")
        print(f"  {tv_out / max(tv_ctrl,1e-9):.0f}x the noise floor {tv_ctrl:.4f}, with")
        print(f"  EXACT matchings: a multi-round composition bug the single-round")
        print(f"  test missed. Concrete refutation of the construction's logic.")
    else:
        print(f"  UNIFORM within sampling noise ({tv_out:.4f} vs floor {tv_ctrl:.4f}).")
        print(f"  The full algorithm's LOGIC and multi-round composition are")
        print(f"  internally correct end-to-end on an above-threshold graph, with")
        print(f"  exact matchings. This does NOT confirm the asymptotic NP=RP")
        print(f"  claim: at N=7 there is no real threshold, and the open question")
        print(f"  is whether JSV stays accurate as N grows (eq-20 analysis: the")
        print(f"  separation-distance error is patchable). The fatal flaw, if any,")
        print(f"  is asymptotic, in the Appendix-B matching-existence / chain-")
        print(f"  mixing interaction, and is not visible at enumerable scale.")


if __name__ == "__main__":
    main()
