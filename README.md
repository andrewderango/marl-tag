# tagRL
Emergent Evasion Strategies via Multi-Agent Self-Play modelled by 2D Apex Predator–Prey interactions.

Two agents — a **tagger** (red) and a **runner** (blue) — are trained via alternating PPO self-play on a 12×12 grid. The tagger tries to catch the runner as fast as possible; the runner tries to survive as long as possible. The core research contribution is a historical evaluation scheme that verifies genuine co-evolution rather than policy cycling.

## Install

**Using a virtual environment (recommended):**

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

To deactivate the environment when you're done:
```bash
deactivate
```

> For GIF export in the renderer, also install `imageio` or `Pillow` (either works):
> ```bash
> pip install imageio
> ```

## Workflow

### 1. Train

```bash
python agents/train.py
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--num_cycles` | 200 | Number of alternating training cycles |
| `--steps_per_cycle` | 2048 | Env steps per agent per cycle |
| `--snapshot_freq` | 8 | Save snapshots every N cycles |
| `--seed` | 42 | Random seed |

Outputs: `snapshots/` (policy files) and `tensorboard_logs/` (training curves).

Monitor training live:
```bash
tensorboard --logdir tensorboard_logs/
```

### 2. Evaluate

Run after training completes. Tests the latest policy against all historical snapshots.

```bash
python eval/evaluate.py
```

Outputs two CSVs to `results/`:
- `runner_vs_historical_taggers.csv` — latest runner vs old taggers (duration should increase)
- `tagger_vs_historical_runners.csv` — latest tagger vs old runners (duration should decrease)

Both trends diverging simultaneously confirms genuine co-evolution.

### 3. Plot

```bash
python figures/plot_results.py
```

Saves `.png` and `.pdf` figures to `figures/output/`:
- `co_evolution` — main result: two diverging duration curves
- `training_curves` — episode reward for both agents over training
- `episode_length` — mean episode length over training

### 4. Render demo

Requires a display. Load any saved snapshot pair and watch a live episode:

```bash
python render/visualize.py --tagger snapshots/tagger_0200 --runner snapshots/runner_0200
```

Render and save GIFs for early / mid / late training stages:

```bash
python render/visualize.py --stages --snapshots_dir snapshots --save_dir output/
```

### Environment sanity check

```bash
python env/tag_env.py
```

Steps both envs with random actions and prints observations, rewards, and termination flags.

## Project structure

```
tagRL/
├── env/tag_env.py          # Gymnasium environments (TaggerEnv, RunnerEnv, GridState)
├── agents/train.py         # Alternating PPO training loop
├── eval/evaluate.py        # Historical evaluation → CSVs
├── figures/plot_results.py # Figures from CSVs and TensorBoard logs
├── render/visualize.py     # Pygame renderer (never imported during training)
├── snapshots/              # Saved .zip policy files
└── results/                # CSV outputs from evaluator
```
