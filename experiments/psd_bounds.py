"""Experiment 3: the kernel's PSD structure is real partial-identifying information.

k-arm trial identifies only the diagonal of the kernel M = E[v v^T]. Bound the
cross-world coherence aggregate Q = sum_{i<j} P(Y_i=1, Y_j=1) three ways:

  Frechet box (marginals only, poly time)
  PSD kernel SDP (adds M >= 0, poly time)         <- the kernel's contribution
  exact response-type LP (tight, 2^k variables)   <- dead past modest k

Verified: PSD is a valid outer bound on exact in every instance, strictly
tighter than Frechet (gap grows with k), and computable at k=40 where the
exact LP has 2^40 variables.

Usage: python experiments/psd_bounds.py   (requires cvxpy: pip install 'worldkernel[sdp]')
"""

import numpy as np

from worldkernel import exact_interval, frechet_interval, psd_interval

RNG = np.random.default_rng(11)


def run(k: int, trials: int) -> None:
    gaps, valid = [], 0
    wf_all, wp_all, we_all = [], [], []
    for _ in range(trials):
        d = RNG.uniform(0.15, 0.85, size=k)
        fl, fh = frechet_interval(d)
        pl, ph = psd_interval(d)
        wf, wp = fh - fl, ph - pl
        gaps.append((wf - wp) / max(wf, 1e-9))
        wf_all.append(wf)
        wp_all.append(wp)
        if k <= 14:
            el, eh = exact_interval(d)
            we_all.append(eh - el)
            if pl <= el + 1e-5 and ph >= eh - 1e-5:
                valid += 1
    line = (
        f"k={k:>2}: Frechet width {np.mean(wf_all):6.3f} | PSD width {np.mean(wp_all):6.3f} "
        f"| PSD tightens Frechet by {100 * np.mean(gaps):4.1f}%"
    )
    if k <= 14:
        line += f" | exact width {np.mean(we_all):6.3f} | PSD outer-valid {valid}/{trials}"
    print(line)


def main() -> None:
    print("Cross-world query Q = sum_{i<j} P(Y_i=1, Y_j=1), bounds from the diagonal only.\n")
    for k in (3, 4, 6, 8, 10, 12, 14):
        run(k, trials=20)
    print("\nWhere the exact LP is dead (2^k response types):")
    for k in (20, 30, 40):
        run(k, trials=5)
    print(f"\n  exact LP variables at k=40: 2^40 = {2**40:,} (intractable)")
    print(f"  PSD kernel SDP at k=40:     {40 * 41 // 2} free entries (trivial)")


if __name__ == "__main__":
    main()
