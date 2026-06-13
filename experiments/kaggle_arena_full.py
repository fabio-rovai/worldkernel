"""
World Model Arena (full) -- three LP-verified classes of non-identified
counterfactual worlds. Run with scipy available (uv run --with numpy --with scipy).

Every generated task is verified: the TRUE counterfactual value provably lies
inside the computed identified interval. If any task fails that check the build
aborts. Output: benchmark_data_full.json (uniform records the kbench task reads).

Classes:
  A two_arm_pn      binary X->Y, Probability of Necessity
  B mediation_nde   X->M->Y, Natural Direct Effect (interval can span zero)
  C karm_coherence  k=3 arms, cross-arm joint potential-outcome query
"""
import json, random, itertools
import numpy as np
from scipy.optimize import linprog


def lp_interval(c, A_eq, b_eq):
    """min and max of c.x over {x>=0, A_eq x = b_eq}. Returns (lo, hi) or None."""
    c = np.asarray(c, float)
    lo = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    hi = linprog(-c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if not (lo.success and hi.success):
        return None
    return float(c @ lo.x), float(-(-c @ hi.x))  # (min, max)


# ===================== Class A: two-arm PN =====================
# atoms (type,x): type in (Y0,Y1) {0,1}^2 (4), x {0,1} -> 8; index = t*2 + x
_TYPESA = [(0, 0), (0, 1), (1, 0), (1, 1)]

def _A_eq_A():
    R = []
    R.append([1] * 8)                                                   # norm
    R.append([1 if (x == 0 and _TYPESA[t][0] == 1) else 0 for t in range(4) for x in range(2)])  # P(X0,Y1)
    R.append([1 if (x == 1 and _TYPESA[t][1] == 1) else 0 for t in range(4) for x in range(2)])  # P(X1,Y1)
    R.append([1 if (x == 1 and _TYPESA[t][1] == 0) else 0 for t in range(4) for x in range(2)])  # P(X1,Y0)
    R.append([1 if _TYPESA[t][1] == 1 else 0 for t in range(4) for x in range(2)])               # P(Y1=1)
    R.append([1 if _TYPESA[t][0] == 1 else 0 for t in range(4) for x in range(2)])               # P(Y0=1)
    return np.array(R, float)

_AeqA = _A_eq_A()

def _gen_A(rng):
    q = np.array([rng.randint(0, 12) for _ in range(8)], float)
    q /= q.sum()
    b = _AeqA @ q
    denom = b[2]                                   # P(X=1,Y=1)
    if denom < 1e-6:
        return None
    c = np.zeros(8); c[1 * 2 + 1] = 1.0            # atom type (0,1) at x=1
    iv = lp_interval(c, _AeqA, b)
    if iv is None:
        return None
    L, H = iv[0] / denom, iv[1] / denom
    true = q[1 * 2 + 1] / denom
    if not (0.15 <= H - L <= 0.65):
        return None
    ev = (f"Observational joint P(X,Y):\n"
          f"  P(X=0,Y=0)={1-b[1]-b[2]-b[3]:.3f}  P(X=0,Y=1)={b[1]:.3f}\n"
          f"  P(X=1,Y=0)={b[3]:.3f}  P(X=1,Y=1)={b[2]:.3f}\n"
          f"Interventional: P(Y=1|do X=0)={b[5]:.3f}  P(Y=1|do X=1)={b[4]:.3f}")
    qn = ("Probability of Necessity (PN): among units with X=1 and Y=1, the "
          "fraction for whom Y would have been 0 had X been 0.")
    return _rec("two_arm_pn", ev, qn, true, L, H)


# ===================== Class B: mediation NDE =====================
# stratum = (a, b): a=(M(0),M(1)) in {0,1}^2 ; b = Y(x,m) bits for (x,m) in
# [(0,0),(0,1),(1,0),(1,1)] -> 16 ; 64 strata.
_AM = list(itertools.product([0, 1], repeat=2))           # 4
_XMM = [(0, 0), (0, 1), (1, 0), (1, 1)]
_STRATA = [(a, yb) for a in _AM for yb in range(16)]      # 64

def _Yfun(yb, x, m):
    return (yb >> _XMM.index((x, m))) & 1

def _gen_B(rng):
    n = len(_STRATA)
    # observed: P(M=m, Y=y | do X=x) for x,(m,y) in {0,1}^2  -> 8 rows + norm
    rows, = ([],)
    rows.append([1.0] * n)                                            # norm
    for x in (0, 1):                                                  # P(M,Y | do X) X-randomized
        for m in (0, 1):
            for y in (0, 1):
                rows.append([1.0 if (a[x] == m and _Yfun(yb, x, m) == y) else 0.0
                             for (a, yb) in _STRATA])
    for x in (0, 1):                                                  # controlled P(Y=1 | do(X,M))
        for m in (0, 1):
            rows.append([1.0 if _Yfun(yb, x, m) == 1 else 0.0 for (a, yb) in _STRATA])
    A_eq = np.array(rows, float)
    q = np.array([rng.random() for _ in range(n)], float); q /= q.sum()
    b = A_eq @ q
    # NDE = E[Y(1,M(0))] - E[Y(0,M(0))]
    c = np.array([_Yfun(yb, 1, a[0]) - _Yfun(yb, 0, a[0]) for (a, yb) in _STRATA], float)
    iv = lp_interval(c, A_eq, b)
    if iv is None:
        return None
    L, H = iv
    true = float(c @ q)
    if not (0.25 <= H - L <= 0.85):
        return None
    pm = {(x, m): sum(q[i] for i, (a, yb) in enumerate(_STRATA) if a[x] == m) for x in (0, 1) for m in (0, 1)}
    ev = ("Randomized mediation experiment, X, M, Y all binary. X randomized; M and Y "
          "observed; M also separately manipulable (controlled direct effects known).\n"
          f"  P(M=1|do X=0)={pm[(0,1)]:.3f}   P(M=1|do X=1)={pm[(1,1)]:.3f}\n")
    for x in (0, 1):
        for m in (0, 1):
            py = sum(q[i] for i, (a, yb) in enumerate(_STRATA)
                     if a[x] == m and _Yfun(yb, x, m) == 1)
            ev += f"  P(Y=1, M={m} | do X={x})={py:.3f}\n"
    for x in (0, 1):
        for m in (0, 1):
            pc = sum(q[i] for i, (a, yb) in enumerate(_STRATA) if _Yfun(yb, x, m) == 1)
            ev += f"  P(Y=1 | do(X={x}, M={m}))={pc:.3f}\n"
    qn = ("Natural Direct Effect (NDE) = E[Y(1, M(0))] - E[Y(0, M(0))]: the effect "
          "of X on Y holding the mediator at the value it would take under X=0. "
          "This may be only partially identified and the interval can span zero.")
    return _rec("mediation_nde", ev, qn.strip(), true, L, H)


# ===================== Class C: k-arm coherence =====================
# k=3 arms, potential outcomes (Y1,Y2,Y3) in {0,1}^3 -> 8 response types.
_RT3 = list(itertools.product([0, 1], repeat=3))

def _gen_C(rng):
    n = 8
    rows = [[1.0] * n]                                  # norm
    for j in range(3):                                  # P(Y_j = 1)
        rows.append([1.0 if _RT3[i][j] == 1 else 0.0 for i in range(n)])
    A_eq = np.array(rows, float)
    q = np.array([rng.randint(0, 12) for _ in range(n)], float); q /= q.sum()
    b = A_eq @ q
    # query: P(Y1=1 and Y2=0) -- arm1 helps, arm2 does not
    c = np.array([1.0 if (_RT3[i][0] == 1 and _RT3[i][1] == 0) else 0.0 for i in range(n)], float)
    iv = lp_interval(c, A_eq, b)
    if iv is None:
        return None
    L, H = iv
    true = float(c @ q)
    if not (0.15 <= H - L <= 0.75):
        return None
    ev = ("Three treatments (arms 1, 2, 3) each randomized; binary outcome Y. "
          "Per-arm success rates (marginals of the potential outcomes):\n"
          f"  P(Y=1 | arm 1)={b[1]:.3f}  P(Y=1 | arm 2)={b[2]:.3f}  P(Y=1 | arm 3)={b[3]:.3f}")
    qn = ("Joint counterfactual: the fraction of units for whom arm 1 would succeed "
          "AND arm 2 would fail (Y(arm1)=1 and Y(arm2)=0). The marginals do not "
          "identify this joint; it is bounded to an interval.")
    return _rec("karm_coherence", ev, qn, true, L, H)


# ===================== assembly =====================
def _rec(cls, ev, qn, true, L, H):
    return {"class": cls, "evidence_text": ev, "question_text": qn,
            "true_value": round(true, 6),
            "gt_lo": round(L, 6), "gt_hi": round(H, 6)}

def build(per_class=25, seed=11):
    rng = random.Random(seed)
    out = []
    for gen, cls in ((_gen_A, "two_arm_pn"), (_gen_B, "mediation_nde"), (_gen_C, "karm_coherence")):
        got = 0
        while got < per_class:
            r = gen(rng)
            if r is None:
                continue
            # VERIFY: true value must lie inside the identified interval
            assert r["gt_lo"] - 1e-6 <= r["true_value"] <= r["gt_hi"] + 1e-6, (cls, r)
            r["id"] = f"{cls}_{got:03d}"
            out.append(r); got += 1
    return out

if __name__ == "__main__":
    tasks = build(per_class=25, seed=11)
    json.dump(tasks, open("benchmark_data_full.json", "w"), indent=1)
    print(f"built + verified {len(tasks)} tasks")
    for cls in ("two_arm_pn", "mediation_nde", "karm_coherence"):
        ts = [t for t in tasks if t["class"] == cls]
        w = np.mean([t["gt_hi"] - t["gt_lo"] for t in ts])
        spans0 = np.mean([1 if (t["gt_lo"] < 0 < t["gt_hi"]) else 0 for t in ts])
        print(f"  {cls:16s} n={len(ts)}  mean width={w:.3f}  spans_zero={spans0:.0%}")
