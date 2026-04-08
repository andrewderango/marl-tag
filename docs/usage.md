# Usage

## 1. Train

```bash
python agents/train.py
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--num_cycles` | 500 | Number of alternating training cycles |
| `--steps_per_cycle` | 2048 | Env steps per agent per cycle |
| `--snapshot_freq` | 10 | Save snapshots every N cycles |
| `--seed` | 42 | Random seed |
| `--resume_from` | 0 | Resume from this cycle number |

Outputs: `snapshots/` (policy `.zip` files) and `tensorboard_logs/` (training curves).

Resume from a checkpoint:

```bash
python agents/train.py --resume_from 200 --num_cycles 500
```

Monitor training live:

```bash
tensorboard --logdir tensorboard_logs/
```

Open `http://localhost:6006/` — watch `ep_len_mean` and `ep_rew_mean` for both agents.

---

## 2. Evaluate — co-evolution (main result)

Tests the latest trained policy against every historical snapshot. Run after training completes.

```bash
python eval/evaluate.py --n_episodes 100
```

Outputs two CSVs to `results/`:
- `runner_vs_historical_taggers.csv` — latest runner vs each historical tagger
- `tagger_vs_historical_runners.csv` — latest tagger vs each historical runner

Both curves diverging simultaneously confirms genuine co-evolution.

> **Note:** If you have snapshots from multiple training runs with different observation space sizes, old incompatible snapshots will cause a load error. Delete them or move them to a separate directory before running.

---

## 3. Evaluate — solo learning curves

Tests each historical snapshot against the latest (fixed) opponent. Since the opponent is held constant, any trend reflects only the historical agent's own improvement.

```bash
python eval/evaluate_fixed_opponent.py --n_episodes 100
```

Outputs to `results/`:
- `historical_runners_vs_fixed_tagger.csv`
- `historical_taggers_vs_fixed_runner.csv`

---

## 4. Plot

Generate all figures from CSVs and TensorBoard logs:

```bash
python figures/plot_results.py
python figures/plot_elo_rating.py
python figures/plot_heatmaps.py
python figures/plot_action_frequency.py
python figures/plot_reward_entropy.py
python figures/plot_anticipation_heatmap.py
python figures/plot_fixed_opponent.py
```

Saves `.png` and `.pdf` to `figures/output/`.

---

## 5. Render

Requires a display (not headless). Load any snapshot pair and watch a live episode:

```bash
python render/visualize.py --tagger snapshots/tagger_0500 --runner snapshots/runner_0500
```

Render and save GIFs for early / mid / late training stages:

```bash
python render/visualize.py --stages --snapshots_dir snapshots --save_dir output/
```

Replay all snapshots in cycle order to watch agents evolve:

```bash
python render/visualize.py --replay --snapshots_dir snapshots
```

---

## Full pipeline (fresh run)

```bash
source .venv/bin/activate
python agents/train.py --num_cycles 2000
python eval/evaluate.py --n_episodes 100
python eval/evaluate_fixed_opponent.py --n_episodes 100
python figures/plot_results.py
python figures/plot_elo_rating.py
python figures/plot_heatmaps.py
python figures/plot_action_frequency.py
```

---

## Project structure

```
marl-tag/
├── env/tag_env.py                    # Gymnasium environments (TaggerEnv, RunnerEnv, GridState)
├── agents/train.py                   # Alternating PPO training loop
├── eval/evaluate.py                  # Historical co-evolution evaluation
├── eval/evaluate_fixed_opponent.py   # Fixed-opponent evaluation
├── figures/                          # Plotting scripts
├── render/visualize.py               # Pygame renderer (never imported during training)
├── snapshots/                        # Saved .zip policy files
├── results/                          # CSV outputs from evaluators
├── figures/output/                   # Generated figures (.png and .pdf)
└── docs/                             # This documentation
```
