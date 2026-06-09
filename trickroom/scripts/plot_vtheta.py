"""Generate V*(θ) prior-sensitivity figure for the paper.

Reads results/theta_sweep_gradient.json (from theta_sweep_experiment.py)
and produces docs/paper/figures/vtheta_curve.pdf.
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter1d

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "text.usetex": False,
})

DATA_PATH = os.path.join(os.path.dirname(__file__), "../results/theta_sweep_gradient.json")
OUT_DIR = os.path.join(os.path.dirname(__file__), "../docs/paper/figures")
OUT_PATH = os.path.join(OUT_DIR, "vtheta_curve.pdf")


def load_data():
    with open(DATA_PATH) as f:
        d = json.load(f)
    thetas = np.array(d["thetas"])
    sr = np.array(d["success_rates"])
    cost_hard = np.array(d["mean_costs_hard"])
    return thetas, sr, cost_hard


def plot(thetas, sr, cost_hard):
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))

    # ── left: success rate ────────────────────────────────────────────
    ax = axes[0]
    sr_smooth = gaussian_filter1d(sr, sigma=1.0)
    ax.plot(thetas, sr, "o", color="#999999", markersize=4, alpha=0.7, label="observed")
    ax.plot(thetas, sr_smooth, "-", color="#2166ac", linewidth=2, label="smoothed")
    ax.axvline(49.0, color="#d73027", linestyle="--", linewidth=1.2, label=r"$\theta^*=49$ (true)")
    ax.axvline(112.0, color="#f4a582", linestyle="--", linewidth=1.2, label=r"$\theta=112$ (wrong)")
    ax.fill_between(thetas, sr_smooth - 0.05, sr_smooth + 0.05,
                    color="#2166ac", alpha=0.12)

    # Mark the two experimental data points from discrete evaluation
    # (n=129 cross-room episodes, seeds 42-291)
    ax.scatter([49.0, 112.0], [0.519, 0.240],
               s=70, zorder=5, color=["#d73027", "#f4a582"],
               edgecolors="black", linewidth=0.8,
               label="Substrate (Kspec, n=129)")

    # ── Neural baseline reference lines (PEG visualisation) ──────────
    LEWM_CROSS   = 0.155   # LeWM cross-room SR (n=129)
    MLP_CROSS    = 0.504   # MLP-WM cross-room SR (n=129)
    CORRECT_PRIOR = 0.519  # substrate correct-prior (n=129)
    WRONG_PRIOR  = 0.240   # substrate wrong-prior baseline (n=129)

    ax.axhline(MLP_CROSS,  color="#33a02c", linestyle="-.",
               linewidth=1.6, zorder=3, label=f"MLP-WM state ({MLP_CROSS:.1%})")
    ax.axhline(LEWM_CROSS, color="#e31a1c", linestyle=":",
               linewidth=1.6, zorder=3, label=f"LeWM visual ({LEWM_CROSS:.1%})")

    # PEG bracket on the right margin — two annotations separated horizontally
    x_peg  = thetas[-1] + 3   # PEG (MLP vs LeWM): leftmost bracket
    x_plan = thetas[-1] + 8   # planning value (wrong→correct prior): rightmost bracket
    # PEG: MLP vs LeWM
    ax.annotate("", xy=(x_peg, MLP_CROSS), xytext=(x_peg, LEWM_CROSS),
                arrowprops=dict(arrowstyle="<->", color="#e31a1c", lw=1.2))
    ax.text(x_peg + 1, (MLP_CROSS + LEWM_CROSS) / 2,
            "PEG\n34.9pp", fontsize=6.5, va="center", ha="left", color="#e31a1c")
    # Full planning value (wrong → correct prior)
    ax.annotate("", xy=(x_plan, CORRECT_PRIOR), xytext=(x_plan, WRONG_PRIOR),
                arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.2))
    ax.text(x_plan + 1, (CORRECT_PRIOR + WRONG_PRIOR) / 2, "27.9pp\navailable",
            fontsize=6.5, va="center", ha="left", color="#555555")

    ax.set_xlabel(r"Door-position prior $\theta$")
    ax.set_ylabel(r"Cross-room success rate $V^*(\theta)$")
    ax.set_xlim(thetas[0] - 2, thetas[-1] + 14)
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9, ncol=1)
    ax.set_title(r"$V^*(\theta)$ with neural baseline reference lines")

    # ── right: planning cost ──────────────────────────────────────────
    ax = axes[1]
    cost_smooth = gaussian_filter1d(cost_hard, sigma=1.0)
    ax.plot(thetas, cost_hard, "o", color="#999999", markersize=4, alpha=0.7)
    ax.plot(thetas, cost_smooth, "-", color="#1a9641", linewidth=2)
    ax.axvline(49.0, color="#d73027", linestyle="--", linewidth=1.2, label=r"$\theta^*=49$ (true)")
    ax.axvline(112.0, color="#f4a582", linestyle="--", linewidth=1.2, label=r"$\theta=112$ (wrong)")

    ax.set_xlabel(r"Door-position prior $\theta$")
    ax.set_ylabel(r"Mean L2 dist to target (hard physics)")
    ax.set_xlim(thetas[0] - 2, thetas[-1] + 2)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    ax.set_title("Planning cost under true physics")

    fig.tight_layout()
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")

    # Also save PNG for quick inspection
    png_path = OUT_PATH.replace(".pdf", ".png")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    print(f"Saved {png_path}")


if __name__ == "__main__":
    thetas, sr, cost_hard = load_data()
    plot(thetas, sr, cost_hard)
    print(f"θ=49 sr={np.interp(49.0, thetas, sr):.3f}  "
          f"θ=112 sr={np.interp(112.0, thetas, sr):.3f}")
