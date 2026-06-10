"""Experiment 12: the Valiant-Vazirani parity probe (Route A).

The proposal: do not recover the isolated witness; compute the PARITY of the
hashed satisfying set. Isolation makes the bucket size 0 or 1; parity
distinguishes them; SAT lands in RP if the sliced parity is polynomial-time.

Theory placement, stated before measuring: this pipeline is the first half
of the Valiant-Vazirani/Toda chain (NP is in RP applied to ParityP), and the
sliced parity is itself ParityP-complete: the affine constraints fold back
into parity-3SAT. The hammer lemma ('parity is easy after hashing') is
therefore equivalent to collapsing ParityP on VV-hashed distributions. What
no theorem settles is the proposal's EMPIRICAL signal:

    does the parity REPRESENTATION WIDTH drop sharply at isolation?

This probe measures exactly that. For random 3-CNF instances near the
clause density 4.26n, with random XOR hashes of sweeping rank:

  parity ground truth   by vectorized evaluation over the slice (exact);
  representation width  by multiplying the clause indicator polynomials in
                        ANF over F_2 on the sliced variables, tracking the
                        FINAL sparsity and the MAX INTERMEDIATE sparsity
                        (the proxy for any structure-exploiting method's
                        working set);

then groups widths by bucket cardinality (0 / 1 / 2 / >= 3). The route's
decisive signal would be singleton buckets showing systematically smaller
widths. If width is statistically blind to the bucket size, the wall is the
proposal's own fallback: parity pseudorandomness under isolating slices.

Usage: python experiments/vv_parity_probe.py
"""

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

N = 18  # original variables
DENSITY = 4.0
INSTANCES = 6
HASHES_PER_INSTANCE = 10
MIN_SOLUTIONS, MAX_SOLUTIONS = 8, 2000  # keep instances satisfiable and
# the VV-matched hash rank r ~ log2(#solutions) in the bucket-mixing regime


def random_3cnf(n: int, m: int, rng: random.Random):
    return [
        [(v, rng.randint(0, 1)) for v in rng.sample(range(n), 3)] for _ in range(m)
    ]


def random_slice(n: int, k: int, rng: random.Random):
    """A random affine slice x = a + B y with B of full column rank k."""
    while True:
        B = np.array([[rng.randint(0, 1) for _ in range(k)] for _ in range(n)],
                     dtype=np.uint8)
        # rank over GF(2)
        M = [int("".join(map(str, row)), 2) for row in B.T]
        basis = {}
        ok = True
        for r in M:
            while r:
                p = r.bit_length() - 1
                if p in basis:
                    r ^= basis[p]
                else:
                    basis[p] = r
                    break
            else:
                ok = False
        if ok:
            break
    a = np.array([rng.randint(0, 1) for _ in range(n)], dtype=np.uint8)
    return a, B


def slice_eval(clauses, a, B):
    """Evaluate the sliced formula on all 2^k points: returns the boolean
    satisfaction vector (ground truth for bucket size and parity)."""
    k = B.shape[1]
    ys = ((np.arange(1 << k)[:, None] >> np.arange(k)) & 1).astype(np.uint8)
    xs = (a[None, :] + ys @ B.T) % 2  # (2^k, n)
    sat = np.ones(1 << k, dtype=bool)
    for cl in clauses:
        cv = np.zeros(1 << k, dtype=bool)
        for v, s in cl:
            cv |= xs[:, v] == s
        sat &= cv
    return sat


def clause_anf(cl, a, B, k):
    """ANF (as a boolean vector over 2^k monomials, index = variable subset)
    of the sliced clause indicator 1 + prod(lit_false)."""
    # each literal-false factor is an affine form in y
    factors = []
    for v, s in cl:
        # literal is [x_v = s]; its falsity indicator over F_2 is x_v + s,
        # which after x_v = a_v + sum_j B[v,j] y_j has constant a_v + s
        vec = np.zeros(1 << k, dtype=np.uint8)
        vec[0] = (int(a[v]) + s) % 2
        for j in range(k):
            if B[v, j]:
                vec[1 << j] ^= 1
        factors.append(vec)
    prod = factors[0]
    for f in factors[1:]:
        prod = anf_mul(prod, f, k)
    prod = prod.copy()
    prod[0] ^= 1  # 1 + product
    return prod


def anf_mul(p: np.ndarray, q: np.ndarray, k: int) -> np.ndarray:
    """Multiply two multilinear ANF polynomials over F_2 (idempotent vars:
    monomial product = bitwise OR of subset indices)."""
    out = np.zeros(1 << k, dtype=np.uint8)
    idx = np.arange(1 << k)
    q_monos = np.nonzero(q)[0]
    p_monos = np.nonzero(p)[0]
    if len(p_monos) > len(q_monos):
        p_monos, q_monos = q_monos, p_monos
    for m in p_monos:
        targets = q_monos | m
        np.bitwise_xor.at(out, targets, 1)
    return out


def anf_product_width(clauses, a, B, k):
    """Multiply all sliced clause ANFs; return (parity, final sparsity,
    max intermediate sparsity)."""
    prod = np.zeros(1 << k, dtype=np.uint8)
    prod[0] = 1
    max_width = 1
    for cl in clauses:
        prod = anf_mul(prod, clause_anf(cl, a, B, k), k)
        max_width = max(max_width, int(prod.sum()))
        if not prod.any():
            break
    parity = int(prod[(1 << k) - 1])  # coefficient of the full monomial
    return parity, int(prod.sum()), max_width


def count_solutions(clauses, n: int) -> int:
    xs = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(np.uint8)
    sat = np.ones(1 << n, dtype=bool)
    for cl in clauses:
        cv = np.zeros(1 << n, dtype=bool)
        for v, s in cl:
            cv |= xs[:, v] == s
        sat &= cv
    return int(sat.sum())


def main() -> None:
    rng = random.Random(11)
    m = int(DENSITY * N)
    buckets: dict[str, list[tuple[int, int]]] = {}
    checked = 0
    print(f"3-CNF n={N}, m={m}; hash rank matched to log2(#solutions) per "
          f"instance (the VV regime);\n{INSTANCES} instances x "
          f"{HASHES_PER_INSTANCE} hashes\n")
    made = 0
    while made < INSTANCES:
        clauses = random_3cnf(N, m, rng)
        n_sol = count_solutions(clauses, N)
        if not MIN_SOLUTIONS <= n_sol <= MAX_SOLUTIONS:
            continue
        made += 1
        r = max(1, round(np.log2(n_sol)))  # VV-matched hash rank
        k = N - r
        print(f"  instance {made}: {n_sol} solutions, hash rank {r}, "
              f"slice dim k={k}")
        for _ in range(HASHES_PER_INSTANCE):
            a, B = random_slice(N, k, rng)
            sat = slice_eval(clauses, a, B)
            count = int(sat.sum())
            parity_true = count % 2
            parity_anf, final_w, max_w = anf_product_width(clauses, a, B, k)
            assert parity_anf == parity_true, "ANF parity disagrees with truth"
            checked += 1
            label = str(count) if count < 3 else ">=3"
            buckets.setdefault(label, []).append((final_w, max_w))

    print(f"{checked} slices computed; ANF parity == enumerated parity on all.\n")
    print(f"{'bucket size':>11} | {'n slices':>8} | {'final ANF width':>15} | "
          f"{'max intermediate width':>22}")
    for label in sorted(buckets, key=lambda s: (len(s), s)):
        rows = buckets[label]
        fw = np.mean([r[0] for r in rows])
        mw = np.mean([r[1] for r in rows])
        print(f"{label:>11} | {len(rows):>8} | {fw:>15.1f} | {mw:>22.1f}")

    print("\nThe route's decisive signal would be singleton buckets (size 1)")
    print("showing systematically SMALLER widths than size-0 and size->=3")
    print("buckets. Width tracking the formula rather than the bucket size is")
    print("the proposal's own fallback wall: parity pseudorandomness under")
    print("isolating affine slices.")


if __name__ == "__main__":
    main()
