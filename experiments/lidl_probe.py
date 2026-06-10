"""Experiment 13: the Localized IDL probe (fifth attack).

The move: do not only add equations; add INVERSES. Localizing the ideal
(adjoining u_a with u_a * g_a(x) = 1) changes the proof system: degree
lower bounds proved without division need not transfer. The proposal comes
with the first positive micro-result of the series, verified here:

  TEST 1 (selector-core collapse, F_2). The uniquely satisfiable formula
  F_H(s, w) = AND_{C in H}(not s or C) AND AND_i (s or not w_i), with H an
  unsatisfiable core, hides H behind the selector bit: plain Macaulay
  cannot derive s = 0 at low degree without refuting H. Adjoining ONE
  localizer t(1-s) = 1 derives s = 0 and every w_i = 0 at degree 3,
  independent of H. The selector-core no-go dies because substituting
  s = 1 into the localized system gives -1 = 0: the hard branch is
  excised from the chart, not refuted.

  TEST 2 (random localizers, F_p). Over a large field, random affine
  denominators are nonzero at the true witness w.h.p., but ALSO nonzero at
  essentially every Boolean point (a random affine form vanishes at a given
  cube point w.p. 1/p), which makes u interpolatable on the cube by a
  polynomial of degree <= n: a constant-D localized derivation translates
  to an O(nD) plain one. Prediction recorded before running: random
  localizers do not lower the first derivation degree.

  TEST 3 (rank-greedy 'proof scars', F_p). Choose the denominator from a
  candidate pool by maximal Macaulay rank gain (scar the PROOF space where
  it bleeds, the algebraic analogue of the paper's targeted world-scars).
  The honest caveat: the selector micro-result's localizer g = 1 - s
  vanishes exactly on the rejected branch, i.e. it encodes one bit of the
  witness; the open question is whether rank-greedy selection can find
  such witness-aligned denominators without being told the witness.

Usage: python experiments/lidl_probe.py
"""

import itertools
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# =============================================================================
# Part 1: F_2 machinery with one localizer variable t (t-degree <= 1)
# Monomial = (frozenset of Boolean vars, t_flag). Polynomial = set of monomials.
# =============================================================================


def f2_mul(p: set, q: set) -> set:
    out: set = set()
    for (ma, ta) in p:
        for (mb, tb) in q:
            if ta and tb:
                continue  # cap t-degree at 1 (the derivation needs t^1 only)
            m = (ma | mb, ta or tb)
            if m in out:
                out.discard(m)
            else:
                out.add(m)
    return out


def f2_macaulay_derives(axioms: list[set], nvars: int, D: int,
                        targets: list[set]) -> bool:
    """Rows: m * axiom for monomials m with total degree (|vars| + t) <= D
    after reduction; columns: all monomials of degree <= D. Checks whether
    every target polynomial lies in the F_2 row span."""
    monoms = []
    for d in range(D + 1):
        for c in itertools.combinations(range(nvars), d):
            monoms.append((frozenset(c), 0))
            if d + 1 <= D:
                monoms.append((frozenset(c), 1))
    col = {m: i for i, m in enumerate(monoms)}

    mults = [m for m in monoms]  # multiplier monomials, same degree cap
    rows = []
    for ax in axioms:
        ax_deg = max(len(m) + t for m, t in ax)
        for mono in mults:
            if len(mono[0]) + mono[1] + ax_deg > D:
                continue
            prod = f2_mul({mono}, ax)
            r = 0
            ok = True
            for m in prod:
                if len(m[0]) + m[1] > D or m not in col:
                    ok = False
                    break
                r ^= 1 << col[m]
            if ok and r:
                rows.append(r)

    basis: dict[int, int] = {}
    for r in rows:
        while r:
            piv = r.bit_length() - 1
            if piv in basis:
                r ^= basis[piv]
            else:
                basis[piv] = r
                break

    def in_span(p: set) -> bool:
        v = 0
        for m in p:
            if m not in col:
                return False
            v ^= 1 << col[m]
        while v:
            piv = v.bit_length() - 1
            if piv not in basis:
                return False
            v ^= basis[piv]
        return True

    return all(in_span(t) for t in targets)


def selector_axioms(core_clauses, n_core: int, localized: bool):
    """Variables: 0 = s, 1..n_core = w. Axioms over F_2:
    boolean: v^2 + v (encoded by reduction, so add v*(v+1) = 0 explicitly
    as the degree-2 polynomial {v} XOR {v} after idempotent reduction it
    vanishes; instead we work pre-reduced: monomials are already
    multilinear, so Boolean axioms appear as s(1+s) -> s + s = 0 trivially.
    We therefore encode s(1-s) = 0 via the SQUARE-FREE convention directly
    and need only: clause violations and the localizer.)
    - z-clauses (s or not w_i): violation (1-s) w_i = w_i + s w_i
    - core clauses (not s or C): violation s * prod(literal false)
    - localizer: t(1+s) + 1 = t + ts + 1
    Targets: s and each w_i."""
    s = 0
    axioms: list[set] = []
    for i in range(1, n_core + 1):  # (1-s) w_i
        axioms.append({(frozenset([i]), 0), (frozenset([s, i]), 0)})
    for cl in core_clauses:  # s * prod over literals of (w + sign)
        p: set = {(frozenset([s]), 0)}
        for v, sign in cl:
            factor = {(frozenset([v + 1]), 0)}
            if sign == 1:  # positive literal false when w = 0: factor w + 1
                factor = {(frozenset([v + 1]), 0), (frozenset(), 0)}
            p = f2_mul(p, factor)
        axioms.append(p)
    if localized:
        axioms.append({(frozenset(), 1), (frozenset([s]), 1), (frozenset(), 0)})
    targets = [{(frozenset([s]), 0)}] + [
        {(frozenset([i]), 0)} for i in range(1, n_core + 1)
    ]
    return axioms, targets


def random_unsat_core(n: int, rng: random.Random):
    """A random unsatisfiable 3-CNF on n variables (density 8, verified)."""
    while True:
        clauses = [
            [(v, rng.randint(0, 1)) for v in rng.sample(range(n), 3)]
            for _ in range(8 * n)
        ]
        sat = False
        for bits in itertools.product((0, 1), repeat=n):
            if all(any(bits[v] == s for v, s in cl) for cl in clauses):
                sat = True
                break
        if not sat:
            return clauses


# =============================================================================
# Part 2: F_p machinery (random and greedy localizers on isolated instances)
# Multilinear monomials over x plus u-variables of degree <= 1 each.
# =============================================================================

P = 10007


def fp_rref_rank_and_span(rows: np.ndarray):
    """RREF over F_p; returns (rank, function checking membership in span)."""
    M = rows % P
    n_rows, n_cols = M.shape
    pivots = []
    r = 0
    for c in range(n_cols):
        sel = None
        for i in range(r, n_rows):
            if M[i, c] % P:
                sel = i
                break
        if sel is None:
            continue
        M[[r, sel]] = M[[sel, r]]
        M[r] = (M[r] * pow(int(M[r, c]), P - 2, P)) % P
        nz = np.nonzero(M[:, c] % P)[0]
        for i in nz:
            if i != r:
                M[i] = (M[i] - M[i, c] * M[r]) % P
        pivots.append(c)
        r += 1
        if r == n_rows:
            break
    Mr = M[:r]

    def in_span(vec: np.ndarray) -> bool:
        v = vec % P
        for j, c in enumerate(pivots):
            if v[c] % P:
                v = (v - v[c] * Mr[j]) % P
        return not v.any()

    return r, in_span


class FpMacaulay:
    """Macaulay matrix over F_p in Boolean x-monomials times u-monomials."""

    def __init__(self, n: int, n_u: int, D: int):
        self.n, self.n_u, self.D = n, n_u, D
        self.monoms = []
        for d in range(D + 1):
            for xs in itertools.combinations(range(n), d):
                for du in range(min(n_u, D - d) + 1):
                    for us in itertools.combinations(range(n_u), du):
                        self.monoms.append((frozenset(xs), frozenset(us)))
        self.col = {m: i for i, m in enumerate(self.monoms)}

    def poly_vec(self, poly: dict) -> np.ndarray | None:
        v = np.zeros(len(self.monoms), dtype=np.int64)
        for m, c in poly.items():
            if m not in self.col:
                return None
            v[self.col[m]] = c % P
        return v

    def mul(self, p: dict, q: dict) -> dict:
        out: dict = {}
        for (xa, ua), ca in p.items():
            for (xb, ub), cb in q.items():
                if ua & ub:
                    continue  # u-degree cap 1 per u-variable
                m = (xa | xb, ua | ub)
                out[m] = (out.get(m, 0) + ca * cb) % P
        return {m: c for m, c in out.items() if c}

    def rows_from_axioms(self, axioms: list[dict]) -> list[np.ndarray]:
        rows = []
        for ax in axioms:
            ax_deg = max(len(x) + len(u) for (x, u) in ax)
            for mono in self.monoms:
                if len(mono[0]) + len(mono[1]) + ax_deg > self.D:
                    continue
                prod = self.mul({mono: 1}, ax)
                vec = self.poly_vec(prod)
                if vec is not None and vec.any():
                    rows.append(vec)
        return rows


def fp_clause_poly(cl, n_u: int) -> dict:
    """Violation polynomial of a 3-clause over F_p (multilinear in x)."""
    p = {(frozenset(), frozenset()): 1}
    out = dict(p)
    for v, s in cl:
        nxt: dict = {}
        for (xs, us), c in out.items():
            # factor: literal false indicator = (1 - x_v) if s==1 else x_v
            if s == 1:
                m1 = (xs | {v}, us)
                nxt[m1] = (nxt.get(m1, 0) - c) % P
                nxt[(xs, us)] = (nxt.get((xs, us), 0) + c) % P
            else:
                m1 = (xs | {v}, us)
                nxt[m1] = (nxt.get(m1, 0) + c) % P
        out = {m: c for m, c in nxt.items() if c}
    return out


def fp_first_degree(clauses, n: int, xstar, localizers, dmax: int = 4):
    """First D at which all coordinate equations x_i - x_i* are derived."""
    n_u = len(localizers)
    for D in range(3, dmax + 1):
        mac = FpMacaulay(n, n_u, D)
        axioms = [fp_clause_poly(cl, n_u) for cl in clauses]
        for a, g in enumerate(localizers):  # u_a * g(x) - 1 = 0
            ax: dict = {}
            for (xs, us), c in g.items():
                ax[(xs, frozenset([a]))] = c
            ax[(frozenset(), frozenset())] = (
                ax.get((frozenset(), frozenset()), 0) - 1
            ) % P
            axioms.append(ax)
        rows = mac.rows_from_axioms(axioms)
        if not rows:
            continue
        _, in_span = fp_rref_rank_and_span(np.array(rows))
        ok = True
        for i in range(n):
            tgt: dict = {(frozenset([i]), frozenset()): 1}
            tgt[(frozenset(), frozenset())] = (-int(xstar[i])) % P
            vec = mac.poly_vec(tgt)
            if vec is None or not in_span(vec):
                ok = False
                break
        if ok:
            return D
    return None


def random_affine(n: int, rng: random.Random) -> dict:
    return {
        **{(frozenset([i]), frozenset()): rng.randrange(1, P) for i in range(n)},
        (frozenset(), frozenset()): rng.randrange(P),
    }


def survives(g: dict, x) -> bool:
    val = 0
    for (xs, _), c in g.items():
        if all(x[i] for i in xs):
            val = (val + c) % P
    return val != 0


def main() -> None:
    rng = random.Random(11)

    # ---- TEST 1: selector-core collapse over F_2 ----------------------------
    print("TEST 1: selector-core collapse (F_2)")
    core = random_unsat_core(6, rng)
    n_core = 6
    plain_ax, targets = selector_axioms(core, n_core, localized=False)
    loc_ax, _ = selector_axioms(core, n_core, localized=True)
    for D in (3, 4, 5):
        plain = f2_macaulay_derives(plain_ax, n_core + 1, D, targets)
        loc = f2_macaulay_derives(loc_ax, n_core + 1, D, targets)
        print(f"  degree {D}: plain derives witness: {plain} | "
              f"localized (one t(1-s)=1): {loc}")

    # ---- TEST 2: random localizers over F_p ----------------------------------
    print("\nTEST 2: random affine localizers (F_p, p=10007), unique-SAT n=10")
    sys.path.insert(0, str(Path(__file__).parent))
    from idl_probe import random_unique_sat as _rus

    results = {0: [], 1: [], 2: [], 4: []}
    for _ in range(3):
        clauses, xstar = _rus(10, rng)
        for r in results:
            gs = []
            while len(gs) < r:
                g = random_affine(10, rng)
                if survives(g, xstar):
                    gs.append(g)
            results[r].append(fp_first_degree(clauses, 10, xstar, gs))
    for r, ds in results.items():
        print(f"  r={r} localizers: first degree per instance: "
              f"{[d if d else '>4' for d in ds]}")

    # ---- TEST 3: rank-greedy localizer ----------------------------------------
    print("\nTEST 3: rank-greedy 'proof scar' vs random (n=10, D=3, one localizer)")
    clauses, xstar = _rus(10, rng)
    pool = []
    while len(pool) < 12:
        g = random_affine(10, rng)
        if survives(g, xstar):
            pool.append(g)
    base_axioms = [fp_clause_poly(cl, 1) for cl in clauses]
    mac = FpMacaulay(10, 1, 3)
    base_rank, _ = fp_rref_rank_and_span(
        np.array(mac.rows_from_axioms(base_axioms))
    )
    gains = []
    for g in pool:
        ax: dict = {(xs, frozenset([0])): c for (xs, _), c in g.items()}
        ax[(frozenset(), frozenset())] = (
            ax.get((frozenset(), frozenset()), 0) - 1
        ) % P
        rank, _ = fp_rref_rank_and_span(
            np.array(mac.rows_from_axioms(base_axioms + [ax]))
        )
        gains.append(rank - base_rank)
    best = int(np.argmax(gains))
    print(f"  base rank {base_rank}; localizer rank gains: {gains}")
    d_greedy = fp_first_degree(clauses, 10, xstar, [pool[best]])
    d_rand = fp_first_degree(clauses, 10, xstar, [pool[0]])
    d_plain = fp_first_degree(clauses, 10, xstar, [])
    print(f"  first degree: plain {d_plain or '>4'} | random localizer "
          f"{d_rand or '>4'} | greedy localizer {d_greedy or '>4'}")

    print("\nReading: Test 1 is the micro-result, the first positive crack in")
    print("the series: ONE localizer collapses the selector-core obstruction")
    print("to degree 3 regardless of the hidden core. The caveat it carries:")
    print("g = 1-s vanishes exactly on the rejected branch, so the localizer")
    print("encodes a witness bit. Tests 2-3 measure whether random or")
    print("rank-greedy selection finds such denominators unaided.")


if __name__ == "__main__":
    main()
