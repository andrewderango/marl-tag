"""
figures/plot_action_frequency.py — Behavioral evolution through action distribution.

Usage:
    python figures/plot_action_frequency.py --snapshots_dir snapshots --output_dir figures/output --n_episodes 10

This script quantifies the shift in agent "preferences" by sampling deterministic 
actions across the training curriculum. It produces a longitudinal study of 
decision-making for both the Runner and Tagger.

Key Insights provided:
  - Movement Bias: Detects if agents favor specific directions (e.g., wall-hugging 
    via excessive 'Left' or 'Up' actions).
  - Degree of Freedom Utilization: Specifically tracks the Tagger's adoption of 
    diagonal movement (UL, UR, DL, DR) vs. cardinal movement as it matures.
  - Strategy Convergence: A flattening of the action lines indicates the agents 
    have settled into a stable, specialized movement policy.
  - Inactivity Monitoring: Monitors the 'Stay' action frequency to see if 
    stationary "hiding" or "camping" strategies emerge over time.

Output:
    - action_frequencies.png/pdf: Line plots showing total action counts per 
      training cycle for both agents.
"""

import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from stable_baselines3 import PPO
import gymnasium as gym

from env.tag_env import GridState, N_ACTIONS, TAGGER_N_ACTIONS


def collect_action_frequencies(snapshots_dir: str, n_episodes: int = 10) -> dict:
    """
    For each snapshot cycle, load the models, run n_episodes, collect action counts.

    Returns dict with cycles, runner actions, tagger actions.
    """
    grid_state = GridState(seed=42)

    cycles = []
    runner_actions_over_cycles = []
    tagger_actions_over_cycles = []

    # Find all runner snapshots
    runner_files = [f for f in os.listdir(snapshots_dir) if f.startswith('runner_') and f.endswith('.zip')]
    runner_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    for runner_file in runner_files:
        cycle = int(runner_file.split('_')[1].split('.')[0])
        tagger_file = f"tagger_{cycle:04d}.zip"

        runner_path = os.path.join(snapshots_dir, runner_file)
        tagger_path = os.path.join(snapshots_dir, tagger_file)

        if not os.path.exists(tagger_path):
            print(f"  Skipping cycle {cycle}: tagger model not found")
            continue

        print(f"Processing cycle {cycle}...")

        # Load models
        runner_model = PPO.load(runner_path)
        tagger_model = PPO.load(tagger_path)

        runner_actions = np.zeros(N_ACTIONS, dtype=int)
        tagger_actions = np.zeros(TAGGER_N_ACTIONS, dtype=int)

        for ep in range(n_episodes):
            grid_state.reset()
            done = False

            while not done:
                runner_obs = grid_state.get_runner_obs()
                tagger_obs = grid_state.get_tagger_obs()

                # Get actions
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    runner_action, _ = runner_model.predict(runner_obs, deterministic=True)
                    tagger_action, _ = tagger_model.predict(tagger_obs, deterministic=True)

                runner_action = int(runner_action)
                tagger_action = int(tagger_action)

                runner_actions[runner_action] += 1
                tagger_actions[tagger_action] += 1

                # Step environment
                _, _, terminated, truncated, _ = grid_state.step(
                    tagger_action, runner_action
                )
                done = terminated or truncated

        cycles.append(cycle)
        runner_actions_over_cycles.append(runner_actions)
        tagger_actions_over_cycles.append(tagger_actions)

    return {
        'cycles': cycles,
        'runner': np.array(runner_actions_over_cycles),
        'tagger': np.array(tagger_actions_over_cycles)
    }


def plot_action_frequencies(data: dict, output_dir: str) -> None:
    """
    Plot action frequencies over cycles for both agents.
    """
    cycles = data['cycles']
    runner_actions = data['runner']
    tagger_actions = data['tagger']

    runner_action_labels = ['Up', 'Down', 'Left', 'Right', 'Stay']
    tagger_action_labels = ['Up', 'Down', 'Left', 'Right', 'Stay', 'UL', 'UR', 'DL', 'DR']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Runner
    ax = axes[0]
    for i in range(N_ACTIONS):
        ax.plot(cycles, runner_actions[:, i], label=runner_action_labels[i], color=colors[i], marker='o')
    ax.set_title('Runner Action Frequencies')
    ax.set_xlabel('Training Cycle')
    ax.set_ylabel('Action Count')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Tagger
    ax = axes[1]
    for i in range(TAGGER_N_ACTIONS):
        ax.plot(cycles, tagger_actions[:, i], label=tagger_action_labels[i], color=colors[i], marker='s')
    ax.set_title('Tagger Action Frequencies')
    ax.set_xlabel('Training Cycle')
    ax.set_ylabel('Action Count')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    for ext in ('png', 'pdf'):
        path = os.path.join(output_dir, f'action_frequencies.{ext}')
        plt.savefig(path)
        print(f"Saved: {path}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot action frequency evolution.")
    parser.add_argument('--snapshots_dir', type=str, default='snapshots',
                        help='Directory containing model snapshots.')
    parser.add_argument('--output_dir', type=str, default='figures/output',
                        help='Directory to save plots.')
    parser.add_argument('--n_episodes', type=int, default=10,
                        help='Number of episodes to run per snapshot for action collection.')

    args = parser.parse_args()

    print("Collecting action frequencies...")
    data = collect_action_frequencies(args.snapshots_dir, args.n_episodes)

    if not data['cycles']:
        print("No valid snapshots found.")
        return

    print("Plotting...")
    plot_action_frequencies(data, args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()