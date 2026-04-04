# tagRL
Emergent Evasion Strategies via Multi-Agent Self-Play modelled by 2D Apex Predator–Prey interactions.

Two agents — a **tagger** (red) and a **runner** (blue) — are trained via alternating PPO self-play on a 20×20 grid. The tagger tries to catch the runner as fast as possible; the runner tries to survive as long as possible. The core research contribution is a historical evaluation scheme that verifies genuine co-evolution rather than policy cycling.

---

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Deactivate when done:
```bash
deactivate
```

---

## Workflow

### 1. Train

```bash
python agents/train.py
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--num_cycles` | 500 | Number of alternating training cycles |
| `--steps_per_cycle` | 2048 | Env steps per agent per cycle |
| `--snapshot_freq` | 20 | Save snapshots every N cycles (gives ~25 snapshots at 500 cycles) |
| `--seed` | 42 | Random seed |
| `--resume_from` | 0 | Resume from this cycle (loads snapshots and continues from cycle+1) |

Outputs: `snapshots/` (policy `.zip` files) and `tensorboard_logs/` (training curves).

**Resume from a checkpoint:**
```bash
python agents/train.py --resume_from 200 --num_cycles 500
```

**Monitor training live:**
```bash
tensorboard --logdir tensorboard_logs/
```
Open `http://localhost:6006/` — watch `ep_len_mean` (runner should increase, tagger should decrease) and `ep_rew_mean` (both should trend upward).

---

### 2. Evaluate — co-evolution (main result)

Tests the **latest** trained policy against every historical snapshot. Run after training completes.

```bash
python eval/evaluate.py --n_episodes 100
```

Outputs two CSVs to `results/`:
- `runner_vs_historical_taggers.csv` — latest runner vs each historical tagger
- `tagger_vs_historical_runners.csv` — latest tagger vs each historical runner

Both curves diverging simultaneously confirms genuine co-evolution.

> **Note:** If you have snapshots from multiple training runs (different observation space sizes), old incompatible snapshots will cause a load error. Delete them first:
> ```bash
> # Find and remove incompatible snapshots
> python -c "
> import glob
> from stable_baselines3 import PPO
> from env.tag_env import TaggerEnv
> dummy = TaggerEnv(seed=0)
> for f in sorted(glob.glob('snapshots/tagger_*.zip')):
>     try: PPO.load(f, env=dummy)
>     except: print(f'rm {f}')
> " 2>/dev/null
> ```

---

### 3. Evaluate — solo learning curves (supplementary)

Tests each **historical** snapshot against the latest (fixed) opponent. Since the opponent doesn't change across the x-axis, any trend reflects only the historical agent's own improvement — less noisy than the co-evolution plot.

```bash
python eval/evaluate_fixed_opponent.py --n_episodes 100
```

Outputs to `results/`:
- `historical_runners_vs_fixed_tagger.csv` — runner skill over training vs best tagger
- `historical_taggers_vs_fixed_runner.csv` — tagger skill over training vs best runner

---

### 4. Plot

Generate all figures from CSVs and TensorBoard logs:

```bash
python figures/plot_results.py
```

Saves `.png` and `.pdf` to `figures/output/`:
- `co_evolution` — main result: two diverging duration curves
- `training_curves` — episode reward for both agents over training
- `episode_length` — mean episode length over training (sanity check)

Generate solo learning curve figures (requires step 3 first):

```bash
python figures/plot_fixed_opponent.py
```

Saves to `figures/output/`:
- `runner_solo_learning` — runner improvement vs fixed best tagger
- `tagger_solo_learning` — tagger improvement vs fixed best runner

---

### 5. Render demo

Requires a display. Load any saved snapshot pair and watch a live episode:

```bash
python render/visualize.py --tagger snapshots/tagger_0500 --runner snapshots/runner_0500
```

Render and save GIFs for early / mid / late training stages:

```bash
python render/visualize.py --stages --snapshots_dir snapshots --save_dir output/
```

Replay all snapshots in cycle order (watch agents evolve):

```bash
python render/visualize.py --replay --snapshots_dir snapshots
```

---

### 6. Environment sanity check

```bash
python env/tag_env.py
```

Steps both envs with random actions and prints observations, rewards, and termination flags.

---

## Full pipeline (fresh run)

```bash
source .venv/bin/activate
python agents/train.py                              # ~2–3 hours
python eval/evaluate.py --n_episodes 100            # ~15 min
python eval/evaluate_fixed_opponent.py --n_episodes 100  # ~15 min
python figures/plot_results.py
python figures/plot_fixed_opponent.py
open figures/output/co_evolution.png
open figures/output/runner_solo_learning.png
open figures/output/tagger_solo_learning.png
```

---

## Project structure

```
tagRL/
├── env/tag_env.py                    # Gymnasium environments (TaggerEnv, RunnerEnv, GridState)
├── agents/train.py                   # Alternating PPO training loop
├── eval/evaluate.py                  # Historical evaluation → co-evolution CSVs
├── eval/evaluate_fixed_opponent.py   # Fixed-opponent evaluation → solo learning CSVs
├── figures/plot_results.py           # Co-evolution + training curve figures
├── figures/plot_fixed_opponent.py    # Solo learning curve figures
├── render/visualize.py               # Pygame renderer (never imported during training)
├── snapshots/                        # Saved .zip policy files (one per agent per checkpoint)
├── results/                          # CSV outputs from evaluators
└── figures/output/                   # Generated figures (.png and .pdf)
```

## Key environment parameters

| Parameter | Value | Description |
|---|---|---|
| Grid size | 20×20 | Border walls + interior obstacles |
| MAX_STEPS | 200 | Episode length cap; runner wins if reached |
| Tagger actions | 9 | 4 cardinal + 4 diagonal + STAY |
| Runner actions | 5 | 4 cardinal + STAY |
| Tagger obs dim | 15 | Position, last-known opponent, velocity, 8 wall flags, visibility |
| Runner obs dim | 11 | Position, last-known opponent, velocity, 4 wall flags, visibility |
| Movement | Simultaneous | Both agents move at the same time each step |

## Key training parameters

| Parameter | Value | Description |
|---|---|---|
| Algorithm | PPO (MlpPolicy) | Proximal Policy Optimization, flat vector observations |
| num_cycles | 500 | Alternating training cycles (~1M steps per agent total) |
| steps_per_cycle | 2048 | Env steps per agent per cycle (2 PPO updates) |
| snapshot_freq | 20 | Save every 20 cycles → 25 snapshots |
| learning_rate | 3e-4 | Adam optimizer |
| ent_coef | 0.03 | Entropy bonus — sustains exploration in self-play |
| gamma | 0.99 | Discount factor |
