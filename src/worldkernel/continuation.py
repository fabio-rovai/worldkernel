"""The elastic wall: Sly-Sun as the shadow of complex zero geometry.

Reframing (the fourth escape hatch, and the sharpest limitation statement).
The partition function Z_G(lam) is a polynomial with positive coefficients;
log Z_G is analytic wherever Z_G is zero-free, and the Barvinok/Patel-Regts
mechanism turns zero-freeness into deterministic approximation: truncate the
Taylor series of log Z at m = O(log(n/eps)) terms inside a zero-free region.
The real-axis Sly-Sun wall is therefore the projection of a complex object:
the ZERO VARIETY of Z_G. A polynomial-clearance analytic path from 0 to the
target fugacity would collapse the wall (and NP to RP with it); so, unless
NP = RP, hard instances must hide an exponential zero moat, an exponential
condition number, or #P-hard Taylor data. Hardness is zero-moat/precision
hardness, not degree hardness.

What this module makes concrete and testable:

  THE DISK ALWAYS FAILS. The product of the root moduli of the independence
  polynomial is i_0 / i_alpha = 1 / i_alpha << 1, so SOME zero always lies
  inside |t| < 1: the naive Barvinok disk centred at 0 never reaches lam = 1
  on any nontrivial instance. The elastic-path framing is forced, not
  optional.

  TRUNCATION OBEYS THE MOAT. log-Taylor truncation converges geometrically
  for |t| below the closest zero and breaks beyond it, on real instances.

  THE SEGMENT'S CLEARANCE IS THE SHEARER MOAT. At enumerable sizes the
  distance from the zero set to the real segment [0, lam] is governed by the
  negative-axis zeros near -lam*(d) = -(d-1)^{d-1}/d^d, decaying smoothly in
  d. Honest negative finding, recorded by test: the positive-axis pinching
  near lam_c that the asymptotic theory predicts for d >= 6 is NOT visible
  at n <= 20 (no zeros with positive real part at all); the wrinkle that
  carries the hardness lives beyond enumerable sizes.

Coefficients here are computed exactly by enumeration (test scale); for
bounded degree, Patel-Regts compute them in polynomial time, which is what
the theorem-level statements assume.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "indep_poly",
    "zero_moat",
    "shearer_radius",
    "log_taylor_estimate",
]


def indep_poly(adj) -> np.ndarray:
    """Coefficients [i_0, i_1, ..., i_alpha] of Z_G(t) = sum_k i_k t^k:
    the number of independent sets of each size. Exact, by enumeration."""
    n = len(adj)
    mask = [0] * n
    for i in range(n):
        for j in adj[i]:
            mask[i] |= 1 << j
    counts: dict[int, int] = {}
    for s in range(1 << n):
        ok = True
        t = s
        while t:
            i = (t & -t).bit_length() - 1
            if mask[i] & s:
                ok = False
                break
            t &= t - 1
        if ok:
            k = bin(s).count("1")
            counts[k] = counts.get(k, 0) + 1
    deg = max(counts)
    return np.array([counts.get(k, 0) for k in range(deg + 1)], dtype=float)


def zero_moat(coeffs: np.ndarray, lam: float = 1.0) -> dict[str, float]:
    """The zero geometry of one instance: minimum root modulus (the radius
    of the largest zero-free disk at 0) and the distance from the zero set
    to the real segment [0, lam] (the clearance a real-axis path enjoys)."""
    roots = np.roots(coeffs[::-1])
    moduli = np.abs(roots)
    seg = []
    for r in roots:
        x = min(max(r.real, 0.0), lam)
        seg.append(abs(r - x))
    return {
        "min_root_modulus": float(moduli.min()),
        "segment_clearance": float(min(seg)),
        "n_roots": len(roots),
    }


def shearer_radius(d: int) -> float:
    """The worst-case zero-free radius lam*(d) = (d-1)^(d-1) / d^d for
    degree-d graphs: the floor under the negative-axis moat."""
    return (d - 1) ** (d - 1) / d**d


def log_taylor_estimate(coeffs: np.ndarray, t: float, m: int) -> float:
    """Barvinok estimate of Z(t): truncate the Taylor series of log Z at
    order m and exponentiate. Standard log-of-power-series recursion
    (a_0 = 1 for the independence polynomial: the empty set):

        b_k = a_k - (1/k) * sum_{j=1}^{k-1} j * b_j * a_{k-j}.

    Converges geometrically iff |t| is inside the zero moat; the failure
    beyond it is the point, not a bug."""
    a = coeffs / coeffs[0]
    deg = len(a) - 1
    b = np.zeros(m + 1)
    for k in range(1, m + 1):
        ak = a[k] if k <= deg else 0.0
        s = sum(j * b[j] * (a[k - j] if k - j <= deg else 0.0) for j in range(1, k))
        b[k] = ak - s / k
    log_z = float(sum(b[k] * t**k for k in range(1, m + 1)))
    return float(coeffs[0] * np.exp(log_z))
