"""
eval/evaluate.py — Historical evaluation scheme (core research contribution).

Usage:
    python eval/evaluate.py [--snapshots_dir DIR] [--results_dir DIR]
                            [--n_episodes N] [--deterministic]

This script implements the two-sided historical evaluation that distinguishes
genuine co-evolution from policy cycling:

  Experiment A — latest runner vs historical taggers:
    If the runner is genuinely improving, it should survive longer against
    older (weaker) taggers.  Duration should INCREASE with snapshot age.

  Experiment B — latest tagger vs historical runners:
    If the tagger is genuinely improving, it should catch older (weaker)
    runners faster.  Duration should DECREASE with snapshot age.

If BOTH trends hold simultaneously, both agents are co-evolving — neither is
merely exploiting a fixed opponent.  If only one trend holds, one agent has
dominated and the other stagnated.  Flat trends indicate policy cycling.

The results are written to CSV for reproducibility.  Figures are generated
separately (figures/plot_results.py) so the data only needs to be collected
once, even if the plots are restyled many times.

--- Why deterministic=True for evaluation but False for training ---
During training, stochastic actions maintain exploration.  During evaluation
we want to measure the policy's best behaviour, not an average over random
action noise.  deterministic=True disables the policy's entropy and returns
the mode of the action distribution.
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from env.tag_env import TaggerEnv, RunnerEnv, GRID_SIZE


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Run historical evaluation and write results to CSV."
    )
    p.add_argument(
        "--snapshots_dir", type=str, default="snapshots",
        help="Directory containing runner_NNNN.zip and tagger_NNNN.zip (default: snapshots/).",
    )
    p.add_argument(
        "--results_dir", type=str, default="results",
        help="Directory to write CSV results (default: results/).",
    )
    p.add_argument(
        "--n_episodes", type=int, default=100,
        help="Episodes per snapshot pair (default: 100). "
             "50–100 gives stable mean estimates; more = slower.",
    )
    p.add_argument(
        "--deterministic", action="store_true", default=True,
        help="Use deterministic policy predictions (default: True).",
    )
    p.add_argument(
        "--stochastic", dest="deterministic", action="store_false",
        help="Use stochastic policy predictions instead.",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="Base random seed for evaluation episodes (default: 0).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_snapshots(snapshots_dir: str, prefix: str) -> dict:
    """
    Scan snapshots_dir for files matching '{prefix}_NNNN.zip'.
    Returns an OrderedDict mapping cycle_number (int) → file_path (str).
    """
    pattern = os.path.join(snapshots_dir, f"{prefix}_*.zip")
    files   = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No {prefix} snapshots found in '{snapshots_dir}'. "
            f"Run agents/train.py first."
        )

    snapshots = {}
    for f in files:
        basename  = os.path.basename(f)
        cycle_str = basename.replace(f"{prefix}_", "").replace(".zip", "")
        try:
            cycle = int(cycle_str)
        except ValueError:
            print(f"  [warn] Could not parse cycle from filename: {basename} — skipping")
            continue
        snapshots[cycle] = f

    return dict(sorted(snapshots.items()))


def load_model(path: str, env) -> PPO:
    """
    Load a PPO snapshot from path (with or without .zip extension).
    The env is used only to set up the model's action/observation spaces
    for prediction; it does not affect the loaded weights.
    """
    # SB3 appends .zip if it is missing
    return PPO.load(path, env=env)


def run_episodes_runner_vs_tagger(
    runner_model: PPO,
    tagger_model: PPO,
    n_episodes:   int,
    deterministic: bool,
    seed:         int,
) -> list:
    """
    Run n_episodes where runner_model is the active agent and tagger_model
    is the frozen opponent (injected via RunnerEnv.set_opponent).

    Returns a list of episode durations (number of steps until done).

    Using RunnerEnv here means the runner is the "foreground" agent:
    obs is the runner's observation, and the env internally queries
    tagger_model to get the tagger's action on each step.
    """
    env = RunnerEnv(seed=seed)
    env.set_opponent(tagger_model)

    durations = []
    for ep in range(n_episodes):
        # Use different seeds per episode for varied starting positions,
        # but deterministic given the overall seed so results are reproducible.
        obs, _ = env.reset(seed=seed + ep)
        done   = False
        steps  = 0
        while not done:
            action, _ = runner_model.predict(obs, deterministic=deterministic)
            obs, _, terminated, truncated, _ = env.step(action)
            done   = terminated or truncated
            steps += 1
        durations.append(steps)

    return durations


def run_episodes_tagger_vs_runner(
    tagger_model: PPO,
    runner_model: PPO,
    n_episodes:   int,
    deterministic: bool,
    seed:         int,
) -> list:
    """
    Run n_episodes where tagger_model is the active agent and runner_model
    is the frozen opponent (injected via TaggerEnv.set_opponent).

    Returns a list of episode durations.
    """
    env = TaggerEnv(seed=seed)
    env.set_opponent(runner_model)

    durations = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done   = False
        steps  = 0
        while not done:
            action, _ = tagger_model.predict(obs, deterministic=deterministic)
            obs, _, terminated, truncated, _ = env.step(action)
            done   = terminated or truncated
            steps += 1
        durations.append(steps)

    return durations


def write_csv(path: str, rows: list) -> None:
    """
    Write evaluation results to CSV.
    Each row: (snapshot_cycle, mean_duration, std_duration, n_episodes).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["snapshot_cycle", "mean_duration", "std_duration", "n_episodes"])
        for row in rows:
            writer.writerow(row)
    print(f"  Wrote {len(rows)} rows → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # --- Discover snapshots ---
    print(f"\nScanning '{args.snapshots_dir}/' for snapshots...")
    runner_snaps = find_snapshots(args.snapshots_dir, "runner")
    tagger_snaps = find_snapshots(args.snapshots_dir, "tagger")

    print(f"  Found {len(runner_snaps)} runner snapshots: "
          f"cycles {min(runner_snaps)}-{max(runner_snaps)}")
    print(f"  Found {len(tagger_snaps)} tagger snapshots: "
          f"cycles {min(tagger_snaps)}-{max(tagger_snaps)}")

    # Latest snapshot = highest cycle number
    latest_runner_cycle = max(runner_snaps.keys())
    latest_tagger_cycle = max(tagger_snaps.keys())
    print(f"\n  Latest runner: cycle {latest_runner_cycle}")
    print(f"  Latest tagger: cycle {latest_tagger_cycle}")

    # --- Load latest models ---
    # Dummy envs used only to supply obs/action space metadata to PPO.load().
    dummy_runner_env = RunnerEnv(seed=0)
    dummy_tagger_env = TaggerEnv(seed=0)

    print(f"\nLoading latest runner (cycle {latest_runner_cycle})...")
    latest_runner = load_model(runner_snaps[latest_runner_cycle], dummy_runner_env)

    print(f"Loading latest tagger (cycle {latest_tagger_cycle})...")
    latest_tagger = load_model(tagger_snaps[latest_tagger_cycle], dummy_tagger_env)

    # -----------------------------------------------------------------------
    # Experiment A: latest runner vs each historical tagger
    # Expected result: duration INCREASES as tagger snapshot age increases
    # (latest runner beats older, weaker taggers for longer)
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Experiment A: latest runner vs historical tagger snapshots")
    print(f"  n_episodes={args.n_episodes}, deterministic={args.deterministic}")
    print(f"{'='*60}")

    rows_A = []
    for cycle, snap_path in sorted(tagger_snaps.items()):
        print(f"  Loading tagger cycle {cycle:4d}...", end=" ", flush=True)
        hist_tagger = load_model(snap_path, dummy_tagger_env)

        durations = run_episodes_runner_vs_tagger(
            runner_model  = latest_runner,
            tagger_model  = hist_tagger,
            n_episodes    = args.n_episodes,
            deterministic = args.deterministic,
            seed          = args.seed,
        )

        mean_d = float(np.mean(durations))
        std_d  = float(np.std(durations))
        print(f"mean_duration={mean_d:.1f} ± {std_d:.1f}")
        rows_A.append((cycle, mean_d, std_d, args.n_episodes))

    csv_path_A = os.path.join(args.results_dir, "runner_vs_historical_taggers.csv")
    write_csv(csv_path_A, rows_A)

    # -----------------------------------------------------------------------
    # Experiment B: latest tagger vs each historical runner
    # Expected result: duration DECREASES as runner snapshot age increases
    # (latest tagger catches older, weaker runners faster)
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Experiment B: latest tagger vs historical runner snapshots")
    print(f"  n_episodes={args.n_episodes}, deterministic={args.deterministic}")
    print(f"{'='*60}")

    rows_B = []
    for cycle, snap_path in sorted(runner_snaps.items()):
        print(f"  Loading runner cycle {cycle:4d}...", end=" ", flush=True)
        hist_runner = load_model(snap_path, dummy_runner_env)

        durations = run_episodes_tagger_vs_runner(
            tagger_model  = latest_tagger,
            runner_model  = hist_runner,
            n_episodes    = args.n_episodes,
            deterministic = args.deterministic,
            seed          = args.seed,
        )

        mean_d = float(np.mean(durations))
        std_d  = float(np.std(durations))
        print(f"mean_duration={mean_d:.1f} ± {std_d:.1f}")
        rows_B.append((cycle, mean_d, std_d, args.n_episodes))

    csv_path_B = os.path.join(args.results_dir, "tagger_vs_historical_runners.csv")
    write_csv(csv_path_B, rows_B)

    print(f"\nEvaluation complete.")
    print(f"Results written to '{args.results_dir}/'")
    print(f"  {csv_path_A}")
    print(f"  {csv_path_B}")
    print(f"\nNext step: python figures/plot_results.py")


if __name__ == "__main__":
    main()
