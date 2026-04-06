"""
figures/plot_reward_entropy.py — Reward and policy entropy over training cycle.

Reads snapshots from a given snapshots directory, runs short evaluation episodes
between matched tagger/runner pairs, and computes policy entropy analytically.

Usage:
    python figures/plot_reward_entropy.py [--snapshots_dir snapshots3]
                                          [--output_dir figures/output]
                                          [--n_eval_episodes 20]
                                          [--n_entropy_obs 256]
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch

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

TAGGER_COLOR  = "#d6604d"
RUNNER_COLOR  = "#2166ac"
ENTROPY_COLOR = "#4dac26"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshots_dir",   type=str, default="snapshots3")
    p.add_argument("--output_dir",      type=str, default="figures/output")
    p.add_argument("--n_eval_episodes", type=int, default=20,
                   help="Episodes per cycle to estimate mean reward (default: 20).")
    p.add_argument("--n_entropy_obs",   type=int, default=256,
                   help="Observations to average entropy over (default: 256).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_cycles(snapshots_dir: str) -> list[int]:
    files = sorted(glob.glob(os.path.join(snapshots_dir, "tagger_*.zip")))
    cycles = []
    for f in files:
        base = os.path.basename(f)
        try:
            c = int(base.replace("tagger_", "").replace(".zip", ""))
            runner_path = os.path.join(snapshots_dir, f"runner_{c:04d}.zip")
            if os.path.exists(runner_path):
                cycles.append(c)
        except ValueError:
            pass
    return sorted(cycles)


def collect_entropy_obs(n_obs: int, env_cls, seed: int = 0) -> np.ndarray:
    """
    Collect a fixed batch of observations from a short random rollout.
    These are reused for every model to make entropy estimates comparable.
    """
    env = env_cls(seed=seed)
    obs_list = []
    obs, _ = env.reset(seed=seed)
    while len(obs_list) < n_obs:
        obs_list.append(obs.copy())
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    return np.array(obs_list[:n_obs], dtype=np.float32)


def compute_entropy(model, obs_batch: np.ndarray) -> float:
    """
    Compute mean policy entropy over a batch of observations.
    Works for both discrete (Categorical) and continuous (Gaussian) policies.
    """
    obs_tensor = torch.as_tensor(obs_batch).to(model.policy.device)
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_tensor)
        entropy = dist.entropy().mean().item()
    return entropy


def eval_role(agent_model, opponent_model, env_cls, n_episodes: int, seed: int = 0):
    """
    Run n_episodes in env_cls with agent_model acting and opponent_model frozen.
    Returns (mean_episode_reward, mean_episode_length).
    Each env handles the opponent's action internally via set_opponent().
    """
    env = env_cls(seed=seed)
    env.set_opponent(opponent_model)

    ep_rewards, ep_lengths = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = agent_model.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, _ = env.step(action)
            ep_reward += rew
            steps += 1
        ep_rewards.append(ep_reward)
        ep_lengths.append(steps)

    env.close()
    return float(np.mean(ep_rewards)), float(np.mean(ep_lengths))


def evaluate_pair(tagger_model, runner_model, n_episodes: int, seed: int = 0):
    """
    Returns (mean_tagger_reward, mean_runner_reward, mean_episode_length).
    Tagger reward measured with TaggerEnv (runner frozen as opponent).
    Runner reward measured with RunnerEnv (tagger frozen as opponent).
    """
    from env.tag_env import TaggerEnv, RunnerEnv
    t_rew, ep_len = eval_role(tagger_model, runner_model, TaggerEnv, n_episodes, seed)
    r_rew, _      = eval_role(runner_model, tagger_model, RunnerEnv, n_episodes, seed)
    return t_rew, r_rew, ep_len


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_metrics(snapshots_dir: str, cycles: list[int],
                    n_eval_episodes: int, n_entropy_obs: int):
    from stable_baselines3 import PPO
    from env.tag_env import TaggerEnv, RunnerEnv
    print(f"  Collecting fixed entropy observation batch ({n_entropy_obs} obs)...")
    entropy_obs_tagger = collect_entropy_obs(n_entropy_obs, TaggerEnv, seed=0)
    entropy_obs_runner = collect_entropy_obs(n_entropy_obs, RunnerEnv, seed=100)

    # Create dummy envs for PPO.load
    dummy_tagger_env = TaggerEnv(seed=0)
    dummy_runner_env = RunnerEnv(seed=0)

    records = []
    total = len(cycles)

    for i, cycle in enumerate(cycles):
        tagger_path = os.path.join(snapshots_dir, f"tagger_{cycle:04d}")
        runner_path = os.path.join(snapshots_dir, f"runner_{cycle:04d}")

        tagger_model = PPO.load(tagger_path, env=dummy_tagger_env, device="cpu")
        runner_model = PPO.load(runner_path, env=dummy_runner_env, device="cpu")

        t_entropy = compute_entropy(tagger_model, entropy_obs_tagger)
        r_entropy = compute_entropy(runner_model, entropy_obs_runner)

        t_reward, r_reward, ep_len = evaluate_pair(
            tagger_model, runner_model, n_eval_episodes, seed=42
        )

        records.append({
            "cycle":          cycle,
            "tagger_reward":  t_reward,
            "runner_reward":  r_reward,
            "tagger_entropy": t_entropy,
            "runner_entropy": r_entropy,
            "ep_length":      ep_len,
        })

        if (i + 1) % 10 == 0 or i == total - 1:
            print(f"  [{i+1:3d}/{total}] cycle={cycle:4d}  "
                  f"t_rew={t_reward:6.2f}  r_rew={r_reward:6.2f}  "
                  f"t_ent={t_entropy:.3f}  r_ent={r_entropy:.3f}  "
                  f"ep_len={ep_len:.1f}")

    dummy_tagger_env.close()
    dummy_runner_env.close()

    return records


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def smooth(x, w: int = 5):
    """Simple centered moving average for visual clarity."""
    if w <= 1 or len(x) <= w:
        return np.array(x)
    kernel = np.ones(w) / w
    padded = np.pad(x, w // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(x)]


def plot_dual_axis(cycles, reward, entropy, role: str,
                   output_dir: str, smooth_w: int = 7) -> None:
    reward_color  = TAGGER_COLOR if role == "tagger" else RUNNER_COLOR
    entropy_color = ENTROPY_COLOR

    cycles_arr  = np.array(cycles)
    reward_arr  = np.array(reward)
    entropy_arr = np.array(entropy)

    reward_smooth  = smooth(reward_arr,  smooth_w)
    entropy_smooth = smooth(entropy_arr, smooth_w)

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax2 = ax1.twinx()

    # Raw data as faint scatter; smoothed line on top
    ax1.scatter(cycles_arr, reward_arr,
                color=reward_color, alpha=0.25, s=8, zorder=2)
    ax1.plot(cycles_arr, reward_smooth,
             color=reward_color, linewidth=2, label="Reward (smoothed)", zorder=3)

    ax2.scatter(cycles_arr, entropy_arr,
                color=entropy_color, alpha=0.20, s=8, zorder=2)
    ax2.plot(cycles_arr, entropy_smooth,
             color=entropy_color, linewidth=2, linestyle="--",
             label="Entropy (smoothed)", zorder=3)

    ax1.set_xlabel("Training cycle")
    ax1.set_ylabel("Mean episode reward", color=reward_color)
    ax2.set_ylabel("Policy entropy (nats)", color=entropy_color)
    ax1.tick_params(axis="y", labelcolor=reward_color)
    ax2.tick_params(axis="y", labelcolor=entropy_color)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    role_label = role.capitalize()
    ax1.set_title(f"{role_label}: reward and policy entropy over training")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    ax1.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.5)
    ax1.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    name = f"{role}_reward_entropy"
    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"{name}.{ext}")
        fig.savefig(path)
        print(f"  Saved: {path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"\nGenerating reward/entropy plots from '{args.snapshots_dir}/'")

    cycles = discover_cycles(args.snapshots_dir)
    if not cycles:
        print(f"  [error] No matching snapshot pairs found in '{args.snapshots_dir}'.")
        return

    print(f"  Found {len(cycles)} snapshot pairs: cycles {cycles[0]}–{cycles[-1]}")
    print(f"  Eval episodes per cycle: {args.n_eval_episodes}")

    records = collect_metrics(
        args.snapshots_dir, cycles,
        args.n_eval_episodes, args.n_entropy_obs,
    )

    cycle_list     = [r["cycle"]          for r in records]
    tagger_rewards = [r["tagger_reward"]  for r in records]
    runner_rewards = [r["runner_reward"]  for r in records]
    tagger_entropy = [r["tagger_entropy"] for r in records]
    runner_entropy = [r["runner_entropy"] for r in records]

    print("\nPlotting tagger figure...")
    plot_dual_axis(cycle_list, tagger_rewards, tagger_entropy,
                   role="tagger", output_dir=args.output_dir)

    print("Plotting runner figure...")
    plot_dual_axis(cycle_list, runner_rewards, runner_entropy,
                   role="runner", output_dir=args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
