"""
eval/evaluate_fixed_opponent.py — Fixed-opponent evaluation.

Inverse of evaluate.py: instead of testing the LATEST agent against historical
opponents, this tests HISTORICAL agents against the LATEST (fixed) opponent.

Because the opponent does not change across the x-axis, any trend in the curve
reflects only the historical agent's improving policy — not co-adaptation. This
produces a cleaner, less noisy learning curve showing each agent's solo progress.

Experiment C — historical runners vs latest tagger (fixed):
  runner_cycle_X plays against tagger_0500 (best tagger).
  Duration should INCREASE over cycles — runner getting better over time.

Experiment D — historical taggers vs latest runner (fixed):
  tagger_cycle_X plays against runner_0500 (best runner).
  Duration should DECREASE over cycles — tagger getting faster over time.

Usage:
    python eval/evaluate_fixed_opponent.py [--n_episodes N] [--snapshots_dir DIR]
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from env.tag_env import TaggerEnv, RunnerEnv


def parse_args():
    p = argparse.ArgumentParser(
        description="Fixed-opponent evaluation: historical agents vs latest opponent."
    )
    p.add_argument("--snapshots_dir", type=str, default="snapshots")
    p.add_argument("--results_dir",   type=str, default="results")
    p.add_argument("--n_episodes",    type=int, default=100)
    p.add_argument("--seed",          type=int, default=0)
    p.add_argument("--deterministic", action="store_true", default=True)
    return p.parse_args()


def find_snapshots(snapshots_dir, prefix):
    files = sorted(glob.glob(os.path.join(snapshots_dir, f"{prefix}_*.zip")))
    snaps = {}
    for f in files:
        base = os.path.basename(f)
        try:
            cycle = int(base.replace(f"{prefix}_", "").replace(".zip", ""))
            snaps[cycle] = f
        except ValueError:
            pass
    return dict(sorted(snaps.items()))


def run_episodes(active_model, opponent_model, role, n_episodes, deterministic, seed):
    """
    Run n_episodes with active_model playing against frozen opponent_model.
    role = 'runner' or 'tagger' — determines which env wraps the active agent.
    Returns list of episode durations.
    """
    if role == "runner":
        env = RunnerEnv(seed=seed)
        env.set_opponent(opponent_model)
    else:
        env = TaggerEnv(seed=seed)
        env.set_opponent(opponent_model)

    durations = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        steps = 0
        while not done:
            action, _ = active_model.predict(obs, deterministic=deterministic)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        durations.append(steps)
    return durations


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["snapshot_cycle", "mean_duration", "std_duration", "n_episodes"])
        for row in rows:
            writer.writerow(row)
    print(f"  Wrote {len(rows)} rows → {path}")


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    runner_snaps = find_snapshots(args.snapshots_dir, "runner")
    tagger_snaps = find_snapshots(args.snapshots_dir, "tagger")

    latest_runner_cycle = max(runner_snaps)
    latest_tagger_cycle = max(tagger_snaps)

    dummy_runner_env = RunnerEnv(seed=0)
    dummy_tagger_env = TaggerEnv(seed=0)

    print(f"Loading latest tagger (cycle {latest_tagger_cycle}) as fixed opponent...")
    latest_tagger = PPO.load(tagger_snaps[latest_tagger_cycle], env=dummy_tagger_env)

    print(f"Loading latest runner (cycle {latest_runner_cycle}) as fixed opponent...")
    latest_runner = PPO.load(runner_snaps[latest_runner_cycle], env=dummy_runner_env)

    # -----------------------------------------------------------------------
    # Experiment C: historical runners vs fixed latest tagger
    # Duration should INCREASE — runner improving over training
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Experiment C: historical runners vs fixed latest tagger")
    print(f"{'='*60}")

    rows_C = []
    for cycle, snap_path in sorted(runner_snaps.items()):
        print(f"  Runner cycle {cycle:4d}...", end=" ", flush=True)
        hist_runner = PPO.load(snap_path, env=dummy_runner_env)
        durations = run_episodes(
            active_model   = hist_runner,
            opponent_model = latest_tagger,
            role           = "runner",
            n_episodes     = args.n_episodes,
            deterministic  = args.deterministic,
            seed           = args.seed,
        )
        mean_d = float(np.mean(durations))
        std_d  = float(np.std(durations))
        print(f"mean_duration={mean_d:.1f} ± {std_d:.1f}")
        rows_C.append((cycle, mean_d, std_d, args.n_episodes))

    write_csv(os.path.join(args.results_dir, "historical_runners_vs_fixed_tagger.csv"), rows_C)

    # -----------------------------------------------------------------------
    # Experiment D: historical taggers vs fixed latest runner
    # Duration should DECREASE — tagger improving (catching faster) over training
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Experiment D: historical taggers vs fixed latest runner")
    print(f"{'='*60}")

    rows_D = []
    for cycle, snap_path in sorted(tagger_snaps.items()):
        print(f"  Tagger cycle {cycle:4d}...", end=" ", flush=True)
        hist_tagger = PPO.load(snap_path, env=dummy_tagger_env)
        durations = run_episodes(
            active_model   = hist_tagger,
            opponent_model = latest_runner,
            role           = "tagger",
            n_episodes     = args.n_episodes,
            deterministic  = args.deterministic,
            seed           = args.seed,
        )
        mean_d = float(np.mean(durations))
        std_d  = float(np.std(durations))
        print(f"mean_duration={mean_d:.1f} ± {std_d:.1f}")
        rows_D.append((cycle, mean_d, std_d, args.n_episodes))

    write_csv(os.path.join(args.results_dir, "historical_taggers_vs_fixed_runner.csv"), rows_D)

    print(f"\nDone. Next step: python figures/plot_fixed_opponent.py")


if __name__ == "__main__":
    main()
