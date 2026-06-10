"""Experiment 10: WorldKernel audit of a frontier model (Claude).

The audit question for any world model: WHEN THE TRUTH IS AN INTERVAL, WHAT
DOES IT REPORT? We put a frontier LLM (via the local `claude -p` CLI, zero
marginal cost) through arena questions under two harness conditions:

  point-forced       "Answer with ONE number" (how world models are
                     normally consumed: a rollout, a value head, a scalar)
  interval-permitted "If not identified, answer with an interval [lo, hi]"

and score both against the kernel's identified interval with the Winkler
rule, coverage, and overclaim rate. The delta between conditions is the
HARNESS TAX: how much honesty the standard point-consuming interface
destroys even when the underlying model can do better.

Usage: python experiments/frontier_audit.py   (requires the `claude` CLI)
"""

import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worldkernel import frechet_pn_bounds  # noqa: E402
from worldkernel.arena import winkler  # noqa: E402
from worldkernel.mediation import (  # noqa: E402
    ATOMS,
    m_val,
    nde_interval_from_record,
    nde_vector,
    y_val,
)

N_TWO_ARM = 8
N_MEDIATION = 6
POINT_SUFFIX = " Answer with ONE number, nothing else."
INTERVAL_SUFFIX = (
    " If the quantity is not identified by the data given, answer with an "
    "interval in the exact form [lo, hi]; otherwise one number. No other text."
)


def ask(prompt: str):
    try:
        out = subprocess.run(
            ["claude", "-p", prompt], capture_output=True, text=True, timeout=120
        ).stdout.strip()
    except subprocess.TimeoutExpired:
        return None, ""
    m = re.search(r"\[\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*\]", out)
    if m:
        lo, hi = sorted((float(m.group(1)), float(m.group(2))))
        return (lo, hi), out
    nums = re.findall(r"-?\d*\.?\d+", out)
    if nums:
        x = float(nums[-1])
        return (x, x), out
    return None, out


def build_questions(seed: int = 11):
    rng = np.random.default_rng(seed)
    qs = []  # (id, text, truth, kernel_lo, kernel_hi)
    for i in range(N_TWO_ARM):
        r0, r1 = rng.uniform(0.15, 0.85, size=2)
        p11 = rng.uniform(max(0.0, r0 + r1 - 1.0), min(r0, r1))
        truth = (r1 - p11) / r1
        k_lo, k_hi = frechet_pn_bounds(r0, r1)
        text = (
            f"A randomized trial of binary treatment X on outcome Y reports "
            f"P(Y=1|do(X=0)) = {r0:.3f} and P(Y=1|do(X=1)) = {r1:.3f}. Among "
            f"treated units with Y=1, what is the probability they would have "
            f"had Y=0 without treatment (probability of necessity)?"
        )
        qs.append((f"pn_{i}", text, truth, k_lo, k_hi))
    nde = nde_vector()
    for i in range(N_MEDIATION):
        p0 = rng.dirichlet(np.ones(len(ATOMS)) * 0.3)
        truth = float(nde @ p0)

        def val(fn):
            return float(np.array([fn(iM, iY) for (iM, iY) in ATOMS]) @ p0)

        p_m = tuple(val(lambda iM, iY, x=x: m_val(iM, x) == 1) for x in (0, 1))
        p_my = {
            (x, m): val(
                lambda iM, iY, x=x, m=m: m_val(iM, x) == m
                and y_val(iY, x, m_val(iM, x)) == 1
            )
            for x in (0, 1)
            for m in (0, 1)
        }
        p_ydo = {
            (x, m): val(lambda iM, iY, x=x, m=m: y_val(iY, x, m) == 1)
            for x in (0, 1)
            for m in (0, 1)
        }
        k_lo, k_hi = nde_interval_from_record(p_m, p_my, p_ydo)
        text = (
            "Randomized mediation experiment, all binary, X -> M -> Y.\n"
            f"P(M=1|do X=0)={p_m[0]:.3f}, P(M=1|do X=1)={p_m[1]:.3f}\n"
            + ", ".join(f"P(M={m},Y=1|do X={x})={v:.3f}" for (x, m), v in p_my.items())
            + "\n"
            + ", ".join(
                f"P(Y=1|do X={x},do M={m})={v:.3f}" for (x, m), v in p_ydo.items()
            )
            + "\nWhat is the Natural Direct Effect "
            "NDE = P(Y_{X=1,M=M_0}=1) - P(Y_{X=0,M=M_0}=1)?"
        )
        qs.append((f"nde_{i}", text, truth, k_lo, k_hi))
    return qs


def main() -> None:
    qs = build_questions()
    tasks = []
    for qid, text, truth, k_lo, k_hi in qs:
        tasks.append((qid, "point", text + POINT_SUFFIX, truth, k_lo, k_hi))
        tasks.append((qid, "interval", text + INTERVAL_SUFFIX, truth, k_lo, k_hi))

    print(f"Auditing claude -p on {len(qs)} questions x 2 harness conditions...")
    with ThreadPoolExecutor(max_workers=6) as ex:
        answers = list(ex.map(lambda t: ask(t[2]), tasks))

    rows = {"point": [], "interval": []}
    for (qid, cond, _, truth, k_lo, k_hi), (ans, raw) in zip(tasks, answers):
        kernel_w = winkler(k_lo, k_hi, truth)
        if ans is None:
            rows[cond].append(
                dict(qid=qid, w=kernel_w + 10, cov=0.0, over=1.0, reg=10.0, width=0.0)
            )
            continue
        lo, hi = ans
        w = winkler(lo, hi, truth)
        rows[cond].append(
            dict(
                qid=qid,
                w=w,
                cov=float(lo - 1e-9 <= truth <= hi + 1e-9),
                over=float(hi - lo < 1e-12 and abs(lo - truth) > 0.02),
                reg=w - kernel_w,
                width=hi - lo,
            )
        )

    print(f"\n{'condition':20s} | {'winkler':>8} | {'coverage':>8} | "
          f"{'overclaim':>9} | {'mean width':>10} | {'regret vs kernel':>16}")
    for cond, rs in rows.items():
        print(
            f"{cond:20s} | {np.mean([r['w'] for r in rs]):>8.3f} | "
            f"{np.mean([r['cov'] for r in rs]):>8.0%} | "
            f"{np.mean([r['over'] for r in rs]):>9.0%} | "
            f"{np.mean([r['width'] for r in rs]):>10.3f} | "
            f"{np.mean([r['reg'] for r in rs]):>16.3f}"
        )
    # split by question class
    for cls in ("pn", "nde"):
        print(f"\n  class {cls}:")
        for cond, rs in rows.items():
            sel = [r for r in rs if r["qid"].startswith(cls)]
            print(
                f"    {cond:18s} winkler {np.mean([r['w'] for r in sel]):.3f}, "
                f"coverage {np.mean([r['cov'] for r in sel]):.0%}, "
                f"overclaim {np.mean([r['over'] for r in sel]):.0%}"
            )
    print("\nReading: 'point' is how world models are consumed (rollouts, value")
    print("heads); 'interval' is what an honest interface permits. The gap is the")
    print("harness tax. Kernel regret > 0 everywhere the model commits or returns")
    print("non-sharp intervals; the kernel's own regret is 0 by construction.")


if __name__ == "__main__":
    main()
