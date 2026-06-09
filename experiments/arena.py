"""Experiment 9: the World Model Arena, full run.

Generates hundreds of worlds across five classes, lets six contender
strategies answer the off-diagonal questions, and scores everything with the
Winkler interval score (proper: sharp valid intervals win, points pay in
full when wrong). Produces the leaderboard and the arena heat map.

Usage: python experiments/arena.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worldkernel.arena import ALPHA, leaderboard, run_arena  # noqa: E402

CONTENDER_ORDER = ["kernel", "frechet", "independence", "monotone", "bp", "weitz"]
LABELS = {
    "kernel": "WorldKernel",
    "frechet": "Frechet box",
    "independence": "Independence (predictor)",
    "monotone": "Monotone (optimist)",
    "bp": "Belief propagation",
    "weitz": "Weitz certificate",
}
CLASS_ORDER = ["two_arm", "two_arm_sampled", "mediation", "k_arm", "constraint"]


def main() -> None:
    print("Running the arena (seed 11)...")
    records = run_arena(seed=11)
    table = leaderboard(records)
    n_total = len(records)
    print(f"{n_total} scored queries across {len(table)} world classes; "
          f"Winkler interval score at alpha={ALPHA} (lower is better)\n")

    for wc in CLASS_ORDER:
        if wc not in table:
            continue
        rows = table[wc]
        n = next(iter(rows.values()))["n"]
        print(f"=== {wc} ({n} queries) ===")
        print(f"{'contender':26s} | {'W(a=.2)':>8} | {'W(a=.02)':>9} | "
              f"{'coverage':>8} | {'width':>6} | {'overclaim':>9}")
        for c in CONTENDER_ORDER:
            if c not in rows:
                continue
            m = rows[c]
            print(f"{LABELS[c]:26s} | {m['winkler']:>8.3f} | "
                  f"{m['winkler_strict']:>9.3f} | {m['coverage']:>8.0%} | "
                  f"{m['width']:>6.3f} | {m['overclaim']:>9.0%}")
        print()

    print("=== mean rank per class entered (1 = best) ===")
    for metric, label in (("winkler", "errors tolerable (a=0.2)"),
                          ("winkler_strict", "errors expensive (a=0.02)")):
        ranks: dict[str, list[int]] = {}
        for wc in table:
            ordered = sorted(table[wc], key=lambda c: table[wc][c][metric])
            for pos, c in enumerate(ordered, 1):
                ranks.setdefault(c, []).append(pos)
        print(f"  {label}:")
        for c in CONTENDER_ORDER:
            if c in ranks:
                print(f"    {LABELS[c]:26s} {np.mean(ranks[c]):.2f} "
                      f"({len(ranks[c])}/{len(table)} classes)")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        classes = [wc for wc in CLASS_ORDER if wc in table]
        conts = [c for c in CONTENDER_ORDER if any(c in table[wc] for wc in classes)]
        M = np.full((len(conts), len(classes)), np.nan)
        for j, wc in enumerate(classes):
            for i, c in enumerate(conts):
                if c in table[wc]:
                    M[i, j] = table[wc][c]["winkler"]

        fig, ax = plt.subplots(figsize=(9.5, 4.6))
        masked = np.ma.masked_invalid(np.log10(M + 1e-3))
        im = ax.imshow(masked, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, fontsize=9)
        ax.set_yticks(range(len(conts)))
        ax.set_yticklabels([LABELS[c] for c in conts], fontsize=9)
        for i in range(len(conts)):
            for j in range(len(classes)):
                if not np.isnan(M[i, j]):
                    cell = table[classes[j]][conts[i]]
                    ax.text(j, i, f"{cell['winkler']:.2f}\ncov {cell['coverage']:.0%}",
                            ha="center", va="center", fontsize=8)
                else:
                    ax.text(j, i, "n/a", ha="center", va="center",
                            fontsize=8, color="grey")
        fig.colorbar(im, ax=ax, label="log10 Winkler score (lower = better)")
        ax.set_title("World Model Arena: proper interval scoring across five world classes\n"
                     "(cell: mean Winkler score + coverage; green = better)")
        plt.tight_layout()
        out = Path(__file__).parent / "arena_repro.png"
        plt.savefig(out, dpi=130)
        print(f"\nChart saved: {out}")
    except ImportError:
        print("\n(matplotlib not installed: chart skipped)")


if __name__ == "__main__":
    main()
