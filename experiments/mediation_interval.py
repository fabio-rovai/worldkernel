"""Experiment 2: the off-diagonal at scale (mediation NDE interval).

Fix everything a randomized mediation experiment measures (rungs 1 and 2) and
compute the identified interval of the Natural Direct Effect by LP over the
64-atom response-type polytope. Verified result with seed 0: the interval is
approximately [-0.381, +0.187], width 0.568, spanning zero. The same
experimental record is consistent with the direct effect being harmful or
helpful; the cross-world coupling decides the sign.

Usage: python experiments/mediation_interval.py
"""

from pathlib import Path

import numpy as np

from worldkernel import atom_count, nde_interval, random_reference, rung12_summary
from worldkernel.mediation import nde_vector, rung12_constraints

p0 = random_reference(seed=0)
A, b = rung12_constraints(p0)
lo, hi, p_lo, p_hi = nde_interval(p0)

print("X -> M -> Y mediation, 64 response-type atoms.\n")
print("RUNG 1/2 (fixed for BOTH endpoint models):")
for k, v in rung12_summary(p0).items():
    print(f"   {k} = {v:.3f}")

print(f"\nRUNG 3 (NDE) IDENTIFIED INTERVAL: [{lo:.3f}, {hi:.3f}]   width = {hi - lo:.3f}")
print("   Non-zero width = the off-diagonal freedom; rung 3 is not identified.")
assert np.allclose(A @ p_lo, b, atol=1e-6) and np.allclose(A @ p_hi, b, atol=1e-6)
print("   Endpoint models verified to reproduce the same rung-1/2 record.")

print("\nSCALING (the counting barrier, concretely):")
for k in (1, 2, 3):
    print(f"   {k} mediator(s): {atom_count(k):,} response-type atoms")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nde = nde_vector()
    plt.figure(figsize=(7, 3.6))
    plt.axhspan(lo, hi, color="#1f77b4", alpha=0.18, label="rung-3 unidentified interval")
    plt.scatter([0, 0], [float(nde @ p_lo), float(nde @ p_hi)], color="#1f77b4", s=70, zorder=5)
    plt.axhline(0, color="k", lw=0.8)
    plt.xticks([])
    plt.ylabel("Natural Direct Effect")
    plt.title("Rungs 1-2 fix everything except the NDE.\nThe off-diagonal coupling sets it.")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    out = Path(__file__).parent / "mediation_interval_repro.png"
    plt.savefig(out, dpi=130)
    print(f"\nChart saved: {out}")
except ImportError:
    print("\n(matplotlib not installed: chart skipped)")
