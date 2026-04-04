"""
figures/plot_heatmaps.py — Emergent behavior visualization via Log-Scale Heatmaps.

Usage:
    python figures/plot_heatmaps.py [--snapshots_dir DIR] [--output_dir DIR]
                                   [--initial N] [--middle N] [--final N]
                                   [--n_episodes N] [--deterministic]

This script generates a 2x3 grid of occupancy heatmaps comparing Seeker (Tagger)
and Hider (Runner) behaviors at three distinct stages of training.

Key Features:
  - Logarithmic Scaling: Uses LogNorm to ensure low-density exploration paths 
    are visible alongside high-density strategic hotspots.
  - Robust Selection: Automatically finds the nearest available .zip snapshots 
    to the requested cycles, handling asynchronous save intervals.
  - Flexible Timing:
      - Default: Picks 10%, 50%, and 100% of the latest available snapshot.
      - Anchored: Providing --final 1000 scales others to 100 and 500 automatically.
      - Manual: Explicitly set --initial, --middle, or --final to compare specific 
        points of interest (e.g., before/after a strategy shift).

The resulting figure illustrates the transition from high-entropy random walks 
to structured, emergent coordination and environmental exploitation.
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from stable_baselines3 import PPO

from env.tag_env import GRID_SIZE, RunnerEnv, TaggerEnv

matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "legend.fontsize":    10,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
})


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate positional heatmaps for seeker and hider policies."
    )
    p.add_argument(
        "--snapshots_dir", type=str, default="snapshots",
        help="Directory containing runner_NNNN.zip and tagger_NNNN.zip snapshots.",
    )
    p.add_argument(
        "--output_dir", type=str, default="figures/output",
        help="Directory to save heatmap figures.",
    )
    p.add_argument(
        "--n_episodes", type=int, default=200,
        help="Episodes to run per snapshot stage (default: 200).",
    )
    p.add_argument(
        "--deterministic", action="store_true", default=False,
        help="Use deterministic policy actions for heatmap rollouts.",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="Base random seed for episode sampling.",
    )
    # New arguments for specific cycle control
    p.add_argument(
        "--initial", type=int, 
        help="Specific cycle for the first heatmap.",
    )
    p.add_argument(
        "--middle", type=int, 
        help="Specific cycle for the second heatmap.",
    )
    p.add_argument(
        "--final", type=int, 
        help="Specific cycle for the third heatmap.",
    )
    return p.parse_args()


def find_snapshots(snapshots_dir: str, prefix: str) -> dict:
    pattern = os.path.join(snapshots_dir, f"{prefix}_*.zip")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No {prefix} snapshots found in '{snapshots_dir}'. "
            f"Run agents/train.py first."
        )

    snapshots = {}
    for f in files:
        basename = os.path.basename(f)
        cycle_str = basename.replace(f"{prefix}_", "").replace(".zip", "")
        try:
            cycle = int(cycle_str)
        except ValueError:
            continue
        snapshots[cycle] = f

    if not snapshots:
        raise FileNotFoundError(
            f"No valid {prefix} snapshots found in '{snapshots_dir}'."
        )

    return dict(sorted(snapshots.items()))


def choose_snapshot_at_fraction(snapshots: dict, fraction: float) -> tuple:
    cycles = sorted(snapshots.keys())
    if not cycles:
        raise ValueError("Snapshot dictionary is empty.")
    target = fraction * cycles[-1]
    chosen_cycle = min(cycles, key=lambda c: abs(c - target))
    return snapshots[chosen_cycle], chosen_cycle

def find_closest_cycle(snapshots: dict, target: int, prefix: str = "") -> tuple:
    """Finds the existing snapshot key closest to the target integer."""
    if not snapshots:
        raise FileNotFoundError(f"No {prefix} snapshots found.")
    
    cycles = sorted(snapshots.keys())
    chosen_cycle = min(cycles, key=lambda c: abs(c - target))
    
    # Calculate the 'drift' from the user's requested target
    offset = abs(chosen_cycle - target)
    if offset > 0:
        print(f"  [INFO] Requested {prefix} {target}, using closest available: {chosen_cycle} (offset: {offset})")
        
    return snapshots[chosen_cycle], chosen_cycle

def load_model(path: str, env):
    return PPO.load(path, env=env)


def collect_position_counts(
    active_model: PPO,
    opponent_model: PPO,
    env_class,
    set_opponent_method_name: str,
    position_attr: str,
    n_episodes: int,
    deterministic: bool,
    seed: int,
) -> np.ndarray:
    counts = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
    env = env_class(seed=seed)
    getattr(env, set_opponent_method_name)(opponent_model)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        pos = getattr(env.grid_state, position_attr)
        counts[int(pos[0]), int(pos[1])] += 1

        done = False
        while not done:
            action, _ = active_model.predict(obs, deterministic=deterministic)
            obs, _, terminated, truncated, _ = env.step(action)
            pos = getattr(env.grid_state, position_attr)
            counts[int(pos[0]), int(pos[1])] += 1
            done = terminated or truncated

    return counts


def normalize_counts(counts: np.ndarray) -> np.ndarray:
    total = np.sum(counts)
    if total == 0:
        return counts.astype(np.float32)
    return (counts.astype(np.float32) / total)


def save_fig(fig: plt.Figure, output_dir: str, name: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"{name}.{ext}")
        fig.savefig(path)
        print(f"  Saved: {path}")

from matplotlib.colors import LogNorm

def plot_heatmaps(
    snapshot_fractions,
    snapshot_labels,
    seeker_maps,
    hider_maps,
    wall_mask,
    output_dir,
):
    # Use constrained_layout=True to prevent legend overlap automatically
    fig, axes = plt.subplots(2, len(snapshot_fractions), figsize=(15, 8), constrained_layout=True)
    fig.suptitle("Seeker and Hider Occupancy Heatmaps (Log Scale)", fontsize=16)
    
    cmap = plt.get_cmap("viridis") # Viridis is often easier on the eyes for heatmaps
    cmap.set_bad("lightgrey")

    global_max = max(
        np.max(map_values) if np.any(map_values) else 1e-4
        for map_values in seeker_maps + hider_maps
    )
    
    # Define a Logarithmic Normalization
    # vmin=1e-4 ensures we can see values as low as 0.01% occupancy
    norm = LogNorm(vmin=1e-4, vmax=global_max)

    for row, (maps, role_label) in enumerate(zip([seeker_maps, hider_maps], ["Seeker", "Hider"])):
        for col, label in enumerate(snapshot_labels):
            ax = axes[row, col]
            
            # Mask the walls
            heat = np.ma.masked_where(wall_mask, maps[col])
            
            # Apply the LogNorm here
            im = ax.imshow(heat, origin="upper", cmap=cmap, norm=norm)
            
            if row == 0:
                ax.set_title(f"{label}")
            if col == 0:
                ax.set_ylabel(role_label, fontsize=12, fontweight='bold')
                
            ax.set_xticks([])
            ax.set_yticks([])

    # Create the colorbar using the last 'im' generated
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("Occupancy Probability (Log Scale)", rotation=270, labelpad=15)

    save_fig(fig, output_dir, "position_heatmaps_log")
    plt.close(fig)

def main():
    args = parse_args()

    tagger_snapshots = find_snapshots(args.snapshots_dir, "tagger")
    runner_snapshots = find_snapshots(args.snapshots_dir, "runner")

    # Determine reference points
    max_available = max(tagger_snapshots.keys())
    final_target = args.final if args.final is not None else max_available

    # Determine target cycles for all three columns
    # If --final is set but others aren't, they scale proportionally (10%, 50%)
    target_cycles = [
            args.initial if args.initial is not None else int(final_target * 0.1),
            args.middle if args.middle is not None else int(final_target * 0.5),
            final_target
        ]
    
    # 3. Generate dynamic labels
    # If user provided any specific cycle args, use "Cycle N" labels. 
    # Otherwise, stick to the percentage labels.
    seeker_maps = []
    hider_maps = []
    actual_labels = [] # We will build labels based on what we actually fin

    wall_mask = TaggerEnv(seed=args.seed).grid_state.walls.astype(bool)

    for i, target in enumerate(target_cycles):
        # Find the actual files closest to our targets
        t_path, t_actual = find_closest_cycle(tagger_snapshots, target, "Tagger")
        r_path, r_actual = find_closest_cycle(runner_snapshots, target, "Runner")
        
        # Robustness Check: Are the agents significantly out of sync?
        if abs(t_actual - r_actual) > 0:
            print(f"  [WARNING] Agent Mismatch at Stage {i+1}: Seeker is {t_actual}, Hider is {r_actual}")

        print(f"Target {target} -> Using: Tagger {t_actual}, Runner {r_actual}")

        # Update the label to show the ACTUAL cycle used
        actual_labels.append(f"Cycle {t_actual}")

        tagger_model = load_model(t_path, TaggerEnv(seed=args.seed))
        runner_model = load_model(r_path, RunnerEnv(seed=args.seed + 1))

        seeker_counts = collect_position_counts(
            active_model=tagger_model,
            opponent_model=runner_model,
            env_class=TaggerEnv,
            set_opponent_method_name="set_opponent",
            position_attr="tagger_pos",
            n_episodes=args.n_episodes,
            deterministic=args.deterministic,
            seed=args.seed + target,
        )
        hider_counts = collect_position_counts(
            active_model=runner_model,
            opponent_model=tagger_model,
            env_class=RunnerEnv,
            set_opponent_method_name="set_opponent",
            position_attr="runner_pos",
            n_episodes=args.n_episodes,
            deterministic=args.deterministic,
            seed=args.seed + target + 100,
        )

        seeker_maps.append(normalize_counts(seeker_counts))
        hider_maps.append(normalize_counts(hider_counts))

    print("Plotting heatmaps...")
    # Passing target_cycles as fractions just to satisfy the plot_heatmaps signature, 
    # though only labels are used for the UI.
    plot_heatmaps(target_cycles, actual_labels, seeker_maps, hider_maps, wall_mask, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
