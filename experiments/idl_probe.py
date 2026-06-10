"""Experiment 11: the Isolate-Densify-Linearize probe.

Empirical test of the 'random affine degree-collapse' route at SAT: encode a
uniquely satisfiable 3-CNF as cubic equations over F_2, optionally scramble
with a random invertible affine map x = Ay + c, build the degree-D Boolean
Macaulay matrix (all multilinearized products m * h_C with deg <= D, reduced
mod y_i^2 = y_i), row-reduce over F_2, and record the first D at which every
coordinate equation y_i + y_i* lies in the row span (witness extracted).

The decisive signal the proposal names: D_scrambled staying O(1) while
D_unsliced grows would be a serious attack on NP; D growing kills the affine
route cleanly. The theory verdict is already known and stated in the
proposal's own Proposition (affine maps preserve degree, so polynomial-
calculus degree lower bounds for hidden unsat cores transfer verbatim:
random affine scrambling CANNOT give constant-degree collapse). This probe
measures what actually happens at small n, where the asymptotic lower
bounds have not yet taken hold.

Usage: python experiments/idl_probe.py
"""

import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# ---- GF(2) multilinear polynomials: a poly is a set of monomials, ----------
# ---- a monomial a frozenset of variable indices ----------------------------


def poly_mul(p: set, q: set) -> set:
    out: set = set()
    for a in p:
        for b in q:
            m = a | b  # idempotent: y^2 = y
            if m in out:
                out.discard(m)
            else:
                out.add(m)
    return out


def affine_form(row_bits: list[int], const: int) -> set:
    """The linear form sum_j row_bits[j] * y_j + const as a polynomial."""
    p = {frozenset([j]) for j, b in enumerate(row_bits) if b}
    if const:
        p ^= {frozenset()}
    return p


# ---- 3-SAT instances ---------------------------------------------------------

def random_unique_sat(n: int, rng: random.Random):
    """A uniquely satisfiable 3-CNF: plant x*, add random clauses satisfied
    by x* until enumeration confirms uniqueness."""
    xstar = [rng.randint(0, 1) for _ in range(n)]
    clauses: list[list[tuple[int, int]]] = []  # [(var, sign)] sign=1 positive

    def n_models() -> int:
        count = 0
        for bits in itertools.product((0, 1), repeat=n):
            ok = all(
                any(bits[v] == s for v, s in cl) for cl in clauses
            )
            count += ok
            if count > 1:
                return 2
        return count

    while True:
        vs = rng.sample(range(n), 3)
        cl = [(v, rng.randint(0, 1)) for v in vs]
        if not any(xstar[v] == s for v, s in cl):
            cl[0] = (cl[0][0], xstar[cl[0][0]])  # force satisfaction by x*
        clauses.append(cl)
        if len(clauses) >= 3 * n and n_models() == 1:
            return clauses, xstar


def clause_poly(cl, subst: list[set]) -> set:
    """g_C(x) substituted with x_v = subst[v]: product over literals of
    (x_v + s) (the clause is violated iff every literal is false)."""
    p = {frozenset()}  # 1
    for v, s in cl:
        factor = set(subst[v])
        if s == 1:  # positive literal false means x_v = 0: factor (x_v + 1)
            factor ^= {frozenset()}
        p = poly_mul(p, factor)
    return p


# ---- Macaulay matrix over GF(2) ------------------------------------------------

def macaulay_derives_witness(polys: list[set], n: int, D: int,
                             ystar: list[int]) -> bool:
    monoms = []
    for d in range(D + 1):
        monoms += [frozenset(c) for c in itertools.combinations(range(n), d)]
    col = {m: i for i, m in enumerate(monoms)}

    rows = []
    mults = [frozenset(c) for d in range(D - 3 + 1)
             for c in itertools.combinations(range(n), d)]
    for h in polys:
        for m in mults:
            r = 0
            ok = True
            prod = poly_mul({m}, h)
            for mono in prod:
                if len(mono) > D:
                    ok = False
                    break
                r ^= 1 << col[mono]
            if ok and r:
                rows.append(r)

    # row reduce (bitmask Gaussian elimination)
    basis: dict[int, int] = {}
    for r in rows:
        while r:
            piv = r.bit_length() - 1
            if piv in basis:
                r ^= basis[piv]
            else:
                basis[piv] = r
                break

    def in_span(vec: int) -> bool:
        while vec:
            piv = vec.bit_length() - 1
            if piv not in basis:
                return False
            vec ^= basis[piv]
        return True

    for i in range(n):
        target = (1 << col[frozenset([i])])
        if ystar[i]:
            target ^= 1 << col[frozenset()]
        if not in_span(target):
            return False
    return True


def first_degree(clauses, n, xstar, scramble: bool, rng, dmax: int = 5):
    if scramble:
        while True:  # random invertible A over GF(2)
            A = [[rng.randint(0, 1) for _ in range(n)] for _ in range(n)]
            c = [rng.randint(0, 1) for _ in range(n)]
            if _gf2_invertible(A, n):
                break
        subst = [affine_form(A[i], c[i]) for i in range(n)]
        # y* solves A y* + c = x*: y* = A^{-1}(x* + c)
        ystar = _gf2_solve(A, [xstar[i] ^ c[i] for i in range(n)], n)
    else:
        subst = [{frozenset([i])} for i in range(n)]
        ystar = list(xstar)
    polys = [clause_poly(cl, subst) for cl in clauses]
    for D in range(3, dmax + 1):
        if macaulay_derives_witness(polys, n, D, ystar):
            return D
    return None


def _gf2_invertible(A, n) -> bool:
    M = [int("".join(map(str, row)), 2) for row in A]
    basis = {}
    for r in M:
        while r:
            p = r.bit_length() - 1
            if p in basis:
                r ^= basis[p]
            else:
                basis[p] = r
                break
        else:
            return False
    return len(basis) == n


def _gf2_solve(A, b, n):
    rows = [(int("".join(map(str, A[i])), 2), b[i]) for i in range(n)]
    for i in range(n):
        piv_bit = n - 1 - i
        pr = next(j for j in range(i, n) if rows[j][0] >> piv_bit & 1)
        rows[i], rows[pr] = rows[pr], rows[i]
        for j in range(n):
            if j != i and rows[j][0] >> piv_bit & 1:
                rows[j] = (rows[j][0] ^ rows[i][0], rows[j][1] ^ rows[i][1])
    y = [0] * n
    for i in range(n):
        piv_bit = rows[i][0].bit_length() - 1
        y[n - 1 - piv_bit] = rows[i][1]
    return y


def main() -> None:
    rng = random.Random(11)
    print("IDL probe: first Macaulay degree deriving the unique witness")
    print(f"{'n':>3} | {'plain (no scramble)':>20} | {'random affine scramble':>22}")
    for n in (8, 10, 12):
        plain, scram = [], []
        for _ in range(4):
            clauses, xstar = random_unique_sat(n, rng)
            plain.append(first_degree(clauses, n, xstar, False, rng))
            scram.append(first_degree(clauses, n, xstar, True, rng))
        fmt = lambda xs: ",".join("?" if x is None else str(x) for x in xs)
        print(f"{n:>3} | {fmt(plain):>20} | {fmt(scram):>22}")
    print("\nReading: '?' = not derived by degree 5. The proposal's own")
    print("Proposition already proves the affine route cannot give constant-")
    print("degree collapse in general (degree preservation transfers PC lower")
    print("bounds); this probe shows the small-n behaviour those asymptotics")
    print("eventually dominate.")


if __name__ == "__main__":
    main()
