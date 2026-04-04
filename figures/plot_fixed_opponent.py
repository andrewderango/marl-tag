"""
figures/plot_fixed_opponent.py — Fixed-opponent learning curves.

Reads the CSVs produced by eval/evaluate_fixed_opponent.py and plots
two solo learning curves on one axis:

  Blue  (solid):  historical runners vs fixed latest tagger
                  → duration should INCREASE (runner improving)
  Red   (dashed): historical taggers vs fixed latest runner
                  → duration should DECREASE (tagger improving, catches faster)

Because the opponent is fixed, these curves reflect each agent's solo
improvement — much less noisy than the co-evolution plot.

Usage:
    python figures/plot_fixed_opponent.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

matplotlib.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "legend.fontsize":   10,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

RUNNER_COLOR = "#2166ac"
TAGGER_COLOR = "#d6604d"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--output_dir",  type=str, default="figures/output")
    return p.parse_args()


def main():
    args = parse_args()

    path_C = os.path.join(args.results_dir, "historical_runners_vs_fixed_tagger.csv")
    path_D = os.path.join(args.results_dir, "historical_taggers_vs_fixed_runner.csv")

    if not os.path.exists(path_C) or not os.path.exists(path_D):
        print("CSVs not found. Run eval/evaluate_fixed_opponent.py first.")
        return

    df_C = pd.read_csv(path_C)
    df_D = pd.read_csv(path_D)

    from env.tag_env import MAX_STEPS

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Plot 1: Runner solo learning curve ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        df_C["snapshot_cycle"], df_C["mean_duration"],
        color=RUNNER_COLOR, linestyle="-", linewidth=2, marker="o", markersize=4,
    )
    ax.fill_between(
        df_C["snapshot_cycle"],
        df_C["mean_duration"] - df_C["std_duration"],
        df_C["mean_duration"] + df_C["std_duration"],
        color=RUNNER_COLOR, alpha=0.15,
    )
    ax.axhline(MAX_STEPS, color="grey", linestyle=":", linewidth=1,
               label=f"Max steps ({MAX_STEPS})")
    ax.set_xlabel("Runner training cycle (tagger frozen at cycle 500)")
    ax.set_ylabel("Mean episode duration (steps)")
    ax.set_title("Runner solo improvement vs fixed best tagger")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(args.output_dir, f"runner_solo_learning.{ext}")
        fig.savefig(p); print(f"  Saved: {p}")
    plt.close(fig)

    # --- Plot 2: Tagger solo learning curve ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        df_D["snapshot_cycle"], df_D["mean_duration"],
        color=TAGGER_COLOR, linestyle="--", linewidth=2, marker="s", markersize=4,
    )
    ax.fill_between(
        df_D["snapshot_cycle"],
        df_D["mean_duration"] - df_D["std_duration"],
        df_D["mean_duration"] + df_D["std_duration"],
        color=TAGGER_COLOR, alpha=0.15,
    )
    ax.axhline(MAX_STEPS, color="grey", linestyle=":", linewidth=1,
               label=f"Max steps ({MAX_STEPS})")
    ax.set_xlabel("Tagger training cycle (runner frozen at cycle 500)")
    ax.set_ylabel("Mean episode duration (steps)")
    ax.set_title("Tagger solo improvement vs fixed best runner")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(args.output_dir, f"tagger_solo_learning.{ext}")
        fig.savefig(p); print(f"  Saved: {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
