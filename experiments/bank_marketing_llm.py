"""LLM overclaiming on the UCI Bank Marketing necessity question.

Diagonal (prior-campaign success vs failure, Y = subscribed a term deposit):
  P(Y=1 | prior failure) = 63/490 = 0.1286,  P(Y=1 | prior success) = 83/129 = 0.6434.
The probability of necessity of the prior success, for a success-and-subscribed
customer, is identified ONLY to the Frechet interval [0.800, 1.000]. We pose the
question to claude -p under two harness conditions and show it reports a POINT
inside the band -- hiding exactly the quantity the data cannot determine. The ERR
point estimate (1 - r0/r1) sits at the interval's floor; the LLM clusters there.

Usage: python experiments/bank_marketing_llm.py   (requires the `claude` CLI)
"""
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from worldkernel import frechet_pn_bounds  # noqa: E402

R0, R1 = 63 / 490, 83 / 129
K_LO, K_HI = frechet_pn_bounds(R0, R1)
ERR = 1 - R0 / R1

Q = (
    "A bank's prior marketing campaign on a customer either succeeded or failed; "
    "the customer was then contacted again and either subscribed a term deposit or "
    "not. The data give P(subscribe | prior failure) = "
    f"{R0:.3f} and P(subscribe | prior success) = {R1:.3f}. Among customers whose "
    "prior campaign succeeded and who then subscribed, what is the probability they "
    "would NOT have subscribed had the prior campaign failed (probability of necessity)?"
)
POINT = " Answer with ONE number, nothing else."
INTERVAL = (
    " If the quantity is not identified by the data given, answer with an interval in "
    "the exact form [lo, hi]; otherwise one number. No other text."
)


def ask(prompt):
    try:
        out = subprocess.run(
            ["claude", "-p", prompt], capture_output=True, text=True, timeout=120
        ).stdout.strip()
    except subprocess.TimeoutExpired:
        return None, ""
    m = re.search(r"\[\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*\]", out)
    if m:
        return tuple(sorted((float(m.group(1)), float(m.group(2))))), out
    nums = re.findall(r"-?\d*\.?\d+", out)
    if nums:
        x = float(nums[-1])
        return (x, x), out
    return None, out


def main(n=5):
    print(f"Identified PN interval the data permits: [{K_LO:.3f}, {K_HI:.3f}]  "
          f"(width {K_HI - K_LO:.3f})")
    print(f"ERR point estimate 1 - r0/r1 = {ERR:.3f}  <- the floor of that interval\n")
    conds = [("POINT-FORCED", POINT), ("INTERVAL-PERMITTED", INTERVAL)]
    tasks = [(lbl, Q + suf) for lbl, suf in conds for _ in range(n)]
    with ThreadPoolExecutor(max_workers=6) as ex:
        out = list(ex.map(lambda t: ask(t[1]), tasks))

    for lbl, _ in conds:
        print(f"== {lbl} ==")
        points = covers = 0
        for (l, _), (ans, _raw) in zip(tasks, out):
            if l != lbl:
                continue
            if ans is None:
                print("  (no parse)")
                continue
            lo, hi = ans
            w = hi - lo
            cov = lo <= K_LO + 1e-9 and hi >= K_HI - 1e-9
            points += w == 0
            covers += cov
            tag = "POINT" if w == 0 else f"interval width {w:.3f}"
            print(f"  [{lo:.3f}, {hi:.3f}]  {tag:18s} "
                  f"{'covers band' if cov else 'MISSES band (overclaims)'}")
        print(f"  -> {points}/{n} collapsed to a point; {covers}/{n} recovered "
              f"the identified band\n")


if __name__ == "__main__":
    main()
