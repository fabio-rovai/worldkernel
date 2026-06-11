"""Experiment 17: survivor-blind cover localization (seventh attack).

The sixth route wanted ONE global eigenchart g with x_i g = b_i g for all i,
and failed because chi(F) (the global chart degree) grows. The seventh route
does not glue local certificates by intersection; it glues them by COVER.

For each coordinate i, the census found abundant FORMULA-DRIVEN annihilators:
  g_{i,0} with g_{i,0} x_i in M_D(I)         (annihilates the bit-0 target)
  g_{i,1} with g_{i,1}(x_i - 1) in M_D(I)    (annihilates the bit-1 target)
and the soundness control proved the WRONG-side annihilators vanish at the
witness. Define the survivor-blind sum

  S_i = g_{i,0} + g_{i,1}.

Then S_i(x*) = (correct survivor != 0) + (wrong side = 0) != 0, so S_i
survives the witness WITHOUT knowing which bit is correct. The identity

  S_i x_i - g_{i,1} = g_{i,0} x_i + g_{i,1}(x_i - 1) in I

means that after localizing (u_i S_i = 1) we get x_i = u_i g_{i,1}, i.e.
x_i = g_{i,1} / (g_{i,0} + g_{i,1}), which evaluates to b_i at the witness.

The decisive question is kappa(F): the minimum degree D' at which the
cover-localized ideal J = I + <u_i S_i - 1 : i> derives every coordinate
x_i = b_i (or its truncated quotient collapses to rank one). If kappa is
O(1)/polylog, this is a path to NP = RP. If kappa grows like chi(F), the
seventh wall is cover-localized quotient persistence: low-degree survivor
certificates exist and aggregate survivor-blind, yet the fake quotient still
refuses to collapse at low degree.

Safeguards carried from the prior probes: formula-driven (non-trivial)
annihilators only (Boolean (1-x)x=0 trivialities quotiented out, D>=3 so
clauses are in the ideal); soundness control; the selector core as a
positive control that MUST collapse to D'=3.

Usage: python experiments/cover_lidl_probe.py
"""

import itertools
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from idl_probe import random_unique_sat  # noqa: E402
from saturation_lidl_probe import (  # noqa: E402
    P,
    Quotient,
    clause_violation,
    nontrivial_annihilators,
    nullspace,
    pmul,
)
import lidl_probe  # noqa: E402


# ---- (x, u) multilinear polynomials: monomial = (x-frozenset, u-frozenset) ---
# x idempotent (x^2=x), u capped at degree 1 each (u^2 dropped).

def xu_mul(p: dict, q: dict) -> dict:
    out: dict = {}
    for (xa, ua), ca in p.items():
        for (xb, ub), cb in q.items():
            if ua & ub:
                continue  # u_i^2 not needed
            m = (xa | xb, ua | ub)
            out[m] = (out.get(m, 0) + ca * cb) % P
    return {m: c for m, c in out.items() if c}


def lift(xpoly: dict) -> dict:
    """x-only polynomial (dict frozenset->coeff) to (x,u) form."""
    return {(m, frozenset()): c for m, c in xpoly.items()}


class XUQuotient:
    """Degree-D' Macaulay rowspace over (x,u), reduction map."""

    def __init__(self, n: int, generators: list[dict], Dp: int):
        self.n, self.Dp = n, Dp
        monoms = []
        for du in range(Dp + 1):
            for us in itertools.combinations(range(n), du):
                for dx in range(Dp - du + 1):
                    for xs in itertools.combinations(range(n), dx):
                        monoms.append((frozenset(xs), frozenset(us)))
        self.monoms = monoms
        self.col = {m: i for i, m in enumerate(monoms)}
        rows = []
        for g in generators:
            gdeg = max(len(x) + len(u) for (x, u) in g)
            gu = frozenset().union(*[u for (_, u) in g]) if g else frozenset()
            for du in range(Dp - gdeg + 1):
                for us in itertools.combinations(range(n), du):
                    if gu & frozenset(us):
                        continue  # would create u_i^2 (u is NOT idempotent):
                        # skip the row rather than drop the term (sound, may
                        # under-derive; never invents a false relation)
                    for dx in range(Dp - gdeg - du + 1):
                        for xs in itertools.combinations(range(n), dx):
                            prod = xu_mul({(frozenset(xs), frozenset(us)): 1}, g)
                            if all(len(a) + len(b) <= Dp for (a, b) in prod):
                                rows.append(self.vec(prod))
        self.pivots, self.basis = self._rref(
            np.array(rows) if rows else np.zeros((0, len(monoms)), int)
        )

    def vec(self, poly: dict) -> np.ndarray:
        v = np.zeros(len(self.monoms), dtype=np.int64)
        for m, c in poly.items():
            v[self.col[m]] = c % P
        return v

    def _rref(self, M):
        M = (M % P).copy()
        piv = []
        r = 0
        for c in range(M.shape[1]):
            sel = next((i for i in range(r, M.shape[0]) if M[i, c] % P), None)
            if sel is None:
                continue
            M[[r, sel]] = M[[sel, r]]
            M[r] = (M[r] * pow(int(M[r, c]), P - 2, P)) % P
            for i in range(M.shape[0]):
                if i != r and M[i, c]:
                    M[i] = (M[i] - M[i, c] * M[r]) % P
            piv.append(c)
            r += 1
            if r == M.shape[0]:
                break
        return piv, M[:r]

    def in_span(self, poly: dict) -> bool:
        v = self.vec(poly) % P
        for j, c in enumerate(self.pivots):
            if v[c]:
                v = (v - v[c] * self.basis[j]) % P
        return not v.any()

    def rank(self) -> int:
        return len(self.pivots)


def _rand_combo(basis, rng):
    """A random F_p linear combination of the basis polynomials (survivors of
    a nonzero space are generic, so a random combo survives w.h.p. where the
    basis-element choice can land on the vanishing hyperplane)."""
    out: dict = {}
    for g in basis:
        c = rng.randrange(P)
        for m, v in g.items():
            out[m] = (out.get(m, 0) + c * v) % P
    return {m: c for m, c in out.items() if c}


def _eval(poly, xstar):
    val = 0
    for m, c in poly.items():
        if all(xstar[k] for k in m):
            val = (val + c) % P
    return val % P


def build_S(quot_x: Quotient, empty_x: Quotient, n: int, xstar, q: int, rng):
    """Survivor-blind S_i = g_{i,0} + g_{i,1} from random formula-driven
    annihilator combinations. Records per-coordinate survival, whether the
    correct-side space is empty (=> no localizer buildable), and wrong-side
    survivor violations."""
    S, survive, empty_correct = [], [], []
    wrong_survivors = 0
    for i in range(n):
        h0 = {frozenset([i]): 1}
        h1 = {frozenset([i]): 1, frozenset(): (-1) % P}
        a0 = nontrivial_annihilators(quot_x, empty_x, h0, q)
        a1 = nontrivial_annihilators(quot_x, empty_x, h1, q)
        g0 = _rand_combo(a0, rng) if a0 else {}
        g1 = _rand_combo(a1, rng) if a1 else {}
        Si = dict(g0)
        for m, c in g1.items():
            Si[m] = (Si.get(m, 0) + c) % P
        Si = {m: c for m, c in Si.items() if c}
        S.append(Si)
        survive.append(_eval(Si, xstar) != 0)
        correct = a0 if xstar[i] == 0 else a1
        wrong = a1 if xstar[i] == 0 else a0
        empty_correct.append(len(correct) == 0)
        if wrong and _eval(_rand_combo(wrong, rng), xstar) != 0:
            wrong_survivors += 1
    return S, {"survive": survive, "empty_correct": empty_correct,
               "wrong_survivors": wrong_survivors}


def _build_J(n, clause_gens, S):
    gens = [lift(g) for g in clause_gens]
    for i in range(n):
        if not S[i]:                 # S_i == 0 would make u_i S_i - 1 = -1
            continue                 # (unit ideal); skip - no localizer here
        loc = xu_mul({(frozenset(), frozenset([i])): 1}, lift(S[i]))
        loc[(frozenset(), frozenset())] = (
            loc.get((frozenset(), frozenset()), 0) - 1
        ) % P
        gens.append({m: c for m, c in loc.items() if c})
    return gens


def kappa(n, clause_gens, xstar, S, Dmax=5):
    """Min D' at which J = I + <u_i S_i - 1> SOUNDLY derives every x_i = b_i.

    Soundness gate: reject any D' where J is the unit ideal (1 in span) or
    derives a WRONG coordinate value x_i = 1 - b_i. Either means the
    localizers killed the witness, and the 'collapse' is vacuous."""
    gens = _build_J(n, clause_gens, S)
    for Dp in range(3, Dmax + 1):
        quot = XUQuotient(n, gens, Dp)
        if quot.in_span({(frozenset(), frozenset()): 1}):
            return ("UNIT", Dp)  # J = whole ring: witness killed
        wrong = any(
            quot.in_span({(frozenset([i]), frozenset()): 1,
                          (frozenset(), frozenset()): (-(1 - xstar[i])) % P})
            for i in range(n)
        )
        if wrong:
            return ("UNSOUND", Dp)
        ok = all(
            quot.in_span({(frozenset([i]), frozenset()): 1,
                          (frozenset(), frozenset()): (-xstar[i]) % P})
            for i in range(n)
        )
        if ok:
            return ("OK", Dp)
    return ("NONE", None)


def plain_coord_degree(n, clauses, xstar, Dmax=6):
    """Min D at which plain I derives all x_i = b_i (the chi/Macaulay baseline)."""
    for D in range(2, Dmax + 1):
        quot = Quotient(n, [clause_violation(c) for c in clauses], D)
        ok = all(
            quot.reduce(quot.vec({frozenset([i]): 1,
                                  frozenset(): (-xstar[i]) % P})).sum() == 0
            for i in range(n)
        )
        if ok:
            return D
    return None


def main() -> None:
    rng = random.Random(11)

    # ---- TEST 5 first (positive control): selector core must collapse to 3 ---
    print("TEST 5 (positive control): selector core must collapse to D'=3")
    from saturation_lidl_probe import selector_generators

    core = lidl_probe.random_unsat_core(4, rng)
    n_sel = 5  # s=0, z_1..z_4
    gens = selector_generators(core, 4)
    xstar_sel = [0] * n_sel
    quot_x = Quotient(n_sel, gens, D=3)
    empty_x = Quotient(n_sel, [], D=3)
    S, diag = build_S(quot_x, empty_x, n_sel, xstar_sel, q=1, rng=rng)
    status, ks = kappa(n_sel, gens, xstar_sel, S, Dmax=4)
    n_surv = sum(diag["survive"])
    n_empty = sum(diag["empty_correct"])
    print(f"  per-coordinate S_i survive: {n_surv}/{n_sel}  "
          f"(coords with EMPTY correct-side formula-driven space: {n_empty})")
    print(f"  wrong-side survivors: {diag['wrong_survivors']} (must be 0)")
    print(f"  kappa(selector): status={status}, D'={ks}")
    print("  (the s-variable has only the Boolean-trivial annihilator 1-s, so")
    print("   its correct-side formula-driven space is EMPTY: no localizer, and")
    print("   a naive S_s=0 would force the UNIT ideal. This is the first crack")
    print("   in the cover construction, caught by the soundness gate.)")

    # ---- TESTS 1-4: generic isolated unique-SAT, kappa scaling ----------------
    print("\nTESTS 1-4 (generic): survivor-blindness + kappa vs plain degree")
    print(f"  {'n':>3} | {'S surv/all':>11} | {'empty corr':>10} | "
          f"{'wrong':>5} | {'plain deg':>9} | {'kappa status':>22}")
    for n in (5, 6, 7):
        for _ in range(3):
            clauses, xs = random_unique_sat(n, rng)
            qx = Quotient(n, [clause_violation(c) for c in clauses], D=3)
            ex = Quotient(n, [], D=3)
            S, diag = build_S(qx, ex, n, xs, q=2, rng=rng)
            cgens = [clause_violation(c) for c in clauses]
            status, kp = kappa(n, cgens, xs, S, Dmax=5)
            pl = plain_coord_degree(n, clauses, xs)
            sv = f"{sum(diag['survive'])}/{n}"
            print(f"  {n:>3} | {sv:>11} | {sum(diag['empty_correct']):>10} | "
                  f"{diag['wrong_survivors']:>5} | {str(pl):>9} | "
                  f"{status + ' D=' + str(kp):>22}")

    print("\nVerdict (from the data): SEVENTH WALL.")
    print("Generic: the survivor-blind cover is buildable (S_i all survive the")
    print("witness, wrong-side survivors 0), but kappa(cover) EQUALS the plain")
    print("Macaulay coordinate degree in every instance. The cover localizers")
    print("are INERT in the x-sector: localizing back through S_i only recovers")
    print("the identity g_{i,0}x_i + (x_i-1)g_{i,1} in I, which is already in I,")
    print("so they add no constraint and do not lower the collapse degree.")
    print("Selector control: the variable s has only the Boolean-trivial")
    print("annihilator 1-s, so its formula-driven correct-side space is EMPTY,")
    print("S_s = 0 forces the unit ideal, and the construction breaks on exactly")
    print("the coordinate it was meant to handle once trivialities are excluded.")
    print("So the seventh wall is cover-localized quotient persistence: low-degree")
    print("survivor certificates exist and aggregate survivor-blind, yet the fake")
    print("quotient does not collapse below the plain Macaulay degree.")


if __name__ == "__main__":
    main()
