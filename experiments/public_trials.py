"""Experiment 5: the kernel on real public randomized trials.

A randomized trial identifies the kernel's DIAGONAL: the potential-outcome
marginals r0 = P(Y_0=1), r1 = P(Y_1=1). It identifies nothing about the
OFF-DIAGONAL p11 = P(Y_0=1, Y_1=1), so rung-3 quantities (probability of
necessity, fraction harmed) are identified only to intervals. This script
computes those intervals on three classic public datasets and separates the
two kinds of uncertainty:

  sampling uncertainty       shrinks with n (bootstrap CIs on the endpoints)
  off-diagonal freedom       does NOT shrink with n; it is what the kernel
                             holds and a predictor cannot

Datasets (all public):
  NSW   Lalonde / Dehejia-Wahba experimental sample (n=445), job training,
        Y = employed in 1978 (re78 > 0).
        https://raw.githubusercontent.com/scunning1975/mixtape/master/nsw_mixtape.dta
  IST   International Stroke Trial (n=19,435), aspirin arm vs no aspirin,
        Y = alive at 14 days (ID14 == 0). Sandercock et al., ODC-BY.
        https://datashare.ed.ac.uk/bitstream/handle/10283/124/IST_corrected.csv
  STAR  Tennessee STAR kindergarten (n=5,789 with reading scores), three
        class-type arms: a genuine k=3 kernel. Y = reading score at or above
        the pooled median.
        https://vincentarelbundock.github.io/Rdatasets/csv/AER/STAR.csv

Headline (IST): with 19,435 patients the sampling error on the diagonal is
under a percentage point, while the identified PN interval stays ~10 points
wide. More data polishes the diagonal; only an off-diagonal assumption
(e.g. monotonicity: aspirin never kills a patient who would have survived)
closes the rest.

Usage: python experiments/public_trials.py   (downloads ~7 MB to ./data on first run)
"""

import subprocess
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from worldkernel import (
    CouplingKernel,
    TwoWorldKernel,
    exact_interval,
    frechet_harmed_bounds,
    frechet_interval,
    frechet_pn_bounds,
)

DATA = Path(__file__).parent / "data"
SOURCES = {
    "nsw.dta": "https://raw.githubusercontent.com/scunning1975/mixtape/master/nsw_mixtape.dta",
    "ist.csv": "https://datashare.ed.ac.uk/bitstream/handle/10283/124/IST_corrected.csv",
    "star.csv": "https://vincentarelbundock.github.io/Rdatasets/csv/AER/STAR.csv",
    "bank.zip": "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip",
}


def fetch(name: str) -> Path:
    DATA.mkdir(exist_ok=True)
    path = DATA / name
    if not path.exists():
        print(f"downloading {name} ...")
        try:
            urlretrieve(SOURCES[name], path)
        except URLError:  # e.g. python.org macOS builds without certificates
            subprocess.run(
                ["curl", "-sSL", "-o", str(path), SOURCES[name]], check=True
            )
    return path


# ---- two-arm analysis --------------------------------------------------------

def two_arm_report(label, n0, k0, n1, k1, bootstrap=2000, seed=11):
    """Counts (n, successes) per arm -> diagonal, rung-3 intervals, bootstrap CIs."""
    r0, r1 = k0 / n0, k1 / n1
    pn_lo, pn_hi = frechet_pn_bounds(r0, r1)
    h_lo, h_hi = frechet_harmed_bounds(r0, r1)
    mono = TwoWorldKernel(r0, r1, p11=min(r0, r1))
    indep = TwoWorldKernel(r0, r1, p11=r0 * r1)

    print(f"\n=== {label} ===")
    print(f"diagonal: r0 = {k0}/{n0} = {r0:.4f}   r1 = {k1}/{n1} = {r1:.4f}   ACE = {r1 - r0:+.4f}")
    print(f"PN identified interval:        [{pn_lo:.3f}, {pn_hi:.3f}]  width {pn_hi - pn_lo:.3f}")
    print(f"  point under monotonicity     {mono.pn():.3f}")
    print(f"  point under independence     {indep.pn():.3f}")
    print(f"fraction-harmed interval:      [{h_lo:.3f}, {h_hi:.3f}]")
    print(f"  (monotonicity sets it to {mono.harmed():.3f}; the data alone cannot)")

    rng = np.random.default_rng(seed)
    b0 = rng.binomial(n0, r0, size=bootstrap) / n0
    b1 = rng.binomial(n1, r1, size=bootstrap) / n1
    lo_bs = np.array([frechet_pn_bounds(a, b)[0] for a, b in zip(b0, b1)])
    hi_bs = np.array([frechet_pn_bounds(a, b)[1] for a, b in zip(b0, b1)])
    lo_ci = np.percentile(lo_bs, [2.5, 97.5])
    hi_ci = np.percentile(hi_bs, [2.5, 97.5])
    samp = max(lo_ci[1] - lo_ci[0], hi_ci[1] - hi_ci[0])
    print(f"bootstrap 95% CI on endpoints: lower in [{lo_ci[0]:.3f}, {lo_ci[1]:.3f}], "
          f"upper in [{hi_ci[0]:.3f}, {hi_ci[1]:.3f}]")
    print(f"sampling spread {samp:.3f} vs off-diagonal width {pn_hi - pn_lo:.3f} "
          f"(ratio {(pn_hi - pn_lo) / max(samp, 1e-9):.1f}x)")
    return dict(label=label, pn=(pn_lo, pn_hi), lo_ci=lo_ci, hi_ci=hi_ci,
                mono=mono.pn(), indep=indep.pn())


def nsw_counts():
    df = pd.read_stata(fetch("nsw.dta"))
    out = {}
    for t in (0, 1):
        sub = df[df.treat == t]
        out[t] = (len(sub), int((sub.re78 > 0).sum()))
    return out


def ist_counts():
    df = pd.read_csv(fetch("ist.csv"), low_memory=False, encoding="latin-1")
    out = {}
    for arm, t in (("N", 0), ("Y", 1)):
        sub = df[df.RXASP == arm]
        out[t] = (len(sub), int((sub.ID14 == 0).sum()))  # alive at 14 days
    return out


def bank_marketing_counts():
    """UCI Bank Marketing (Moro et al. 2014). Treatment = prior-campaign outcome
    (success vs failure), Y = subscribed a term deposit. A large identified ACE
    whose probability of necessity is still only bounded to an interval."""
    import zipfile

    with zipfile.ZipFile(fetch("bank.zip")) as z, z.open("bank.csv") as f:
        df = pd.read_csv(f, sep=";")
    sub = df[df.poutcome.isin(["success", "failure"])]
    out = {}
    for arm, t in (("failure", 0), ("success", 1)):
        s = sub[sub.poutcome == arm]
        out[t] = (len(s), int((s.y == "yes").sum()))
    return out


# ---- three-arm analysis (STAR): a real k=3 kernel ---------------------------

def star_analysis():
    df = pd.read_csv(fetch("star.csv"))
    k = df[df.stark.notna() & df.readk.notna()]
    med = k.readk.median()
    arms = ["regular", "small", "regular+aide"]
    d, ns = [], []
    print(f"\n=== STAR kindergarten (k=3 arms, Y = reading >= pooled median {med:.0f}) ===")
    for arm in arms:
        sub = k[k.stark == arm]
        rate = float((sub.readk >= med).mean())
        d.append(rate)
        ns.append(len(sub))
        print(f"  {arm:13s} n={len(sub)}  P(Y=1) = {rate:.4f}")

    fl, fh = frechet_interval(d)
    el, eh = exact_interval(d)
    print(f"cross-world coherence Q = sum P(Y_i=1, Y_j=1) over the 3 arm pairs:")
    print(f"  Frechet box  [{fl:.3f}, {fh:.3f}]   exact identified set [{el:.3f}, {eh:.3f}]")
    assert fl - 1e-9 <= el and eh <= fh + 1e-9

    # the kernel at the independence point is admissible (a sanity exhibit)
    M = np.outer(d, d)
    np.fill_diagonal(M, d)
    assert CouplingKernel(M).admissible()

    # pairwise cross-arm counterfactual: of the kids above median in a small
    # class, what fraction would have been below median in a regular class?
    r_reg, r_small = d[0], d[1]
    lo, hi = frechet_pn_bounds(r_reg, r_small)
    print(f"PN(small class vs regular): identified interval [{lo:.3f}, {hi:.3f}]")
    print("  rungs 1-2 of a three-arm trial leave every pairwise coupling free;")
    print("  only the joint feasibility of one law over all three arms tightens Q.")
    return d, (fl, fh), (el, eh)


def main() -> None:
    n = nsw_counts()
    nsw = two_arm_report("NSW job training (Lalonde/Dehejia-Wahba, Y = employed 1978)",
                         n[0][0], n[0][1], n[1][0], n[1][1])
    i = ist_counts()
    ist = two_arm_report("IST aspirin (Y = alive at 14 days)",
                         i[0][0], i[0][1], i[1][0], i[1][1])
    b = bank_marketing_counts()
    bank = two_arm_report(
        "UCI Bank Marketing (prior-campaign success, Y = subscribed term deposit)",
        b[0][0], b[0][1], b[1][0], b[1][1])
    star_analysis()

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.4))
        for y, r in ((2, bank), (1, nsw), (0, ist)):
            lo, hi = r["pn"]
            ax.plot([lo, hi], [y, y], lw=8, color="#1f77b4", alpha=0.35,
                    solid_capstyle="butt",
                    label="identified PN interval (off-diagonal freedom)" if y == 2 else None)
            ax.plot([r["lo_ci"][0], r["lo_ci"][1]], [y, y], lw=2.5, color="k",
                    label="bootstrap 95% CI on endpoints" if y == 2 else None)
            ax.plot([r["hi_ci"][0], r["hi_ci"][1]], [y, y], lw=2.5, color="k")
            ax.scatter([r["mono"]], [y], marker="v", color="#2ca02c", zorder=5,
                       label="monotonicity point" if y == 2 else None)
            ax.scatter([r["indep"]], [y], marker="o", color="crimson", zorder=5,
                       label="independence point" if y == 2 else None)
        ax.set_yticks([2, 1, 0])
        ax.set_yticklabels(["Bank Mktg\n(n=619)", "NSW\n(n=445)", "IST\n(n=19,435)"])
        ax.set_ylim(-0.55, 2.55)
        ax.set_xlabel("probability of necessity (PN)")
        ax.set_title("Real trials identify the diagonal; rung 3 stays an interval.\n"
                     "Sampling error (black) shrinks with n; off-diagonal freedom (blue) does not.")
        ax.legend(fontsize=7.5, loc="center right")
        plt.tight_layout()
        out = Path(__file__).parent / "public_trials_repro.png"
        plt.savefig(out, dpi=130)
        print(f"\nChart saved: {out}")
    except ImportError:
        print("\n(matplotlib not installed: chart skipped)")


if __name__ == "__main__":
    main()
