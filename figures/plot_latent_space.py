"""
figures/plot_latent_space.py — t-SNE / UMAP projection of policy latent space.

Usage:
    python figures/plot_latent_space.py [--snapshots_dir DIR] [--output_dir DIR]
                                        [--n_episodes N] [--method {tsne,umap}]
                                        [--perplexity P] [--n_neighbors N]

This script generates 2D projections of the hidden states from the agents' neural
networks, colored by training cycle. Separate plots for seeker (tagger) and
hider (runner).

Clusters in the latent space indicate emergent strategies. For example, if the
hider forms two distinct clusters, it has learned two different behaviors
(e.g., hiding vs. running).

Requires: scikit-learn (for t-SNE) and umap-learn (for UMAP).
Install with: pip install scikit-learn umap-learn
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import warnings
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
        description="Generate latent space projections for seeker and hider policies."
    )
    p.add_argument(
        "--snapshots_dir", type=str, default="snapshots",
        help="Directory containing runner_NNNN.zip and tagger_NNNN.zip snapshots.",
    )
    p.add_argument(
        "--output_dir", type=str, default="figures/output",
        help="Directory to save latent space plots.",
    )
    p.add_argument(
        "--n_episodes", type=int, default=50,
        help="Episodes to run per snapshot for collecting hidden states (default: 50).",
    )
    p.add_argument(
        "--method", type=str, default="tsne", choices=["tsne", "umap"],
        help="Dimensionality reduction method (default: tsne).",
    )
    p.add_argument(
        "--perplexity", type=float, default=30.0,
        help="t-SNE perplexity parameter (default: 30.0).",
    )
    p.add_argument(
        "--n_neighbors", type=int, default=15,
        help="UMAP n_neighbors parameter (default: 15).",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for reproducibility.",
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


def load_model(path: str, env):
    return PPO.load(path, env=env)


def collect_hidden_states(
    model: PPO,
    env_class,
    set_opponent_method_name: str,
    opponent_model: PPO,
    n_episodes: int,
    seed: int,
    cycle: int,
) -> tuple:
    """
    Run episodes and collect hidden states from the policy network.
    Returns (hidden_states, labels) where labels are the cycle numbers.
    """
    env = env_class(seed=seed)
    getattr(env, set_opponent_method_name)(opponent_model)

    hidden_states = []
    labels = []

    model.policy.eval()  # Set to eval mode

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            # Get hidden states from the policy's MLP extractor
            # policy.mlp_extractor returns (shared_latent, policy_latent, value_latent)
            # We'll use the shared latent as the "hidden state"
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                latent = model.policy.mlp_extractor(obs_tensor)
                hidden_states.append(latent[0].cpu().numpy().flatten())  # shared latent
                labels.append(cycle)

            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    return np.array(hidden_states), np.array(labels)


def reduce_dimensionality(
    hidden_states: np.ndarray,
    method: str,
    perplexity: float = 30.0,
    n_neighbors: int = 15,
) -> np.ndarray:
    if method == "tsne":
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        return tsne.fit_transform(hidden_states)
    elif method == "umap":
        import umap
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="n_jobs value 1 overridden to 1 by setting random_state",
            )
            warnings.filterwarnings(
                "ignore",
                message="Spectral initialisation failed! The eigenvector solver failed.*",
            )
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                init="random",
                random_state=42,
            )
            return reducer.fit_transform(hidden_states)
    else:
        raise ValueError(f"Unknown method: {method}")


def plot_latent_space(
    reduced_data: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_dir: str,
    filename: str,
):
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        reduced_data[:, 0],
        reduced_data[:, 1],
        c=labels,
        cmap="viridis",
        alpha=0.7,
        s=10,
    )
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Training Cycle")
    fig.tight_layout()
    save_fig(fig, output_dir, filename)
    plt.close(fig)


def save_fig(fig: plt.Figure, output_dir: str, name: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"{name}.{ext}")
        fig.savefig(path)
        print(f"  Saved: {path}")


def main():
    args = parse_args()

    tagger_snapshots = find_snapshots(args.snapshots_dir, "tagger")
    runner_snapshots = find_snapshots(args.snapshots_dir, "runner")

    # Collect data for seeker (tagger)
    seeker_hidden_states = []
    seeker_labels = []
    for cycle, path in tagger_snapshots.items():
        print(f"Processing seeker cycle {cycle}...")
        tagger_model = load_model(path, TaggerEnv(seed=args.seed))
        # Use the same cycle's runner if available, else the closest
        runner_cycle = min(runner_snapshots.keys(), key=lambda c: abs(c - cycle))
        runner_model = load_model(runner_snapshots[runner_cycle], RunnerEnv(seed=args.seed + 1))
        hidden, labels = collect_hidden_states(
            tagger_model, TaggerEnv, "set_opponent", runner_model, args.n_episodes, args.seed, cycle
        )
        seeker_hidden_states.append(hidden)
        seeker_labels.append(labels)

    if seeker_hidden_states:
        seeker_all = np.vstack(seeker_hidden_states)
        seeker_labels_all = np.concatenate(seeker_labels)
        print(f"Seeker: collected {len(seeker_all)} hidden states")
        reduced_seeker = reduce_dimensionality(
            seeker_all, args.method, perplexity=args.perplexity, n_neighbors=args.n_neighbors
        )
        method_name = "t-SNE" if args.method == "tsne" else "UMAP"
        plot_latent_space(
            reduced_seeker, seeker_labels_all, f"Seeker Latent Space ({method_name})", args.output_dir, "seeker_latent_space"
        )

    # Collect data for hider (runner)
    hider_hidden_states = []
    hider_labels = []
    for cycle, path in runner_snapshots.items():
        print(f"Processing hider cycle {cycle}...")
        runner_model = load_model(path, RunnerEnv(seed=args.seed))
        # Use the same cycle's tagger if available, else the closest
        tagger_cycle = min(tagger_snapshots.keys(), key=lambda c: abs(c - cycle))
        tagger_model = load_model(tagger_snapshots[tagger_cycle], TaggerEnv(seed=args.seed + 1))
        hidden, labels = collect_hidden_states(
            runner_model, RunnerEnv, "set_opponent", tagger_model, args.n_episodes, args.seed, cycle
        )
        hider_hidden_states.append(hidden)
        hider_labels.append(labels)

    if hider_hidden_states:
        hider_all = np.vstack(hider_hidden_states)
        hider_labels_all = np.concatenate(hider_labels)
        print(f"Hider: collected {len(hider_all)} hidden states")
        reduced_hider = reduce_dimensionality(
            hider_all, args.method, perplexity=args.perplexity, n_neighbors=args.n_neighbors
        )
        method_name = "t-SNE" if args.method == "tsne" else "UMAP"
        plot_latent_space(
            reduced_hider, hider_labels_all, f"Hider Latent Space ({method_name})", args.output_dir, "hider_latent_space"
        )

    print("Done.")


if __name__ == "__main__":
    main()
