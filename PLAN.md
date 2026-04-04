# Improvement Plan — tagRL Model Performance

**Goal:** Achieve stable co-evolution where both evaluation curves diverge monotonically
with low variance, confirming genuine mutual improvement rather than policy cycling.

**Current state:** Evaluation shows wild oscillation (tags range from 0 to 126 across
snapshots) with standard deviations exceeding means. No monotonic trend in either direction.

---

## Phase 1: Environment Fixes (do first, re-train before anything else)

### 1.1 Return to simultaneous movement

**File:** `env/tag_env.py` — `GridState.step()`

**Current:** Tagger moves first, catch is checked, then runner moves. Tagger has a
structural first-mover advantage — runner can never dodge the final approach step.

**Change:** Compute both new positions before checking catch. Replace the sequential
move-check-move-check logic with:

```python
new_tagger, t_delta = self._try_move(self.tagger_pos, tagger_action)
new_runner, r_delta = self._try_move(self.runner_pos, runner_action)

self.tagger_pos = new_tagger
self.runner_pos = new_runner
self.tagger_last_d = t_delta
self.runner_last_d = r_delta
self.step_count += 1

tagged = bool(np.array_equal(self.tagger_pos, self.runner_pos))
```

Remove the early-return catch check that currently sits between tagger move and runner move.
This collapses the two separate catch checks into one after both agents have moved.

**Why:** Fairer game. Runner can dodge. Creates richer pursuit/evasion dynamics where
both agents must predict each other's movement.

---

### 1.2 Increase MAX_STEPS to 200

**File:** `env/tag_env.py` — constant at top

**Change:** `MAX_STEPS = 200`

**Why:** 100 steps on a 20x20 grid with obstacles is too short. Max Manhattan distance
is ~36 but actual pathfinding distance through obstacles can be 50+ steps. Agents need
room to develop multi-step strategies — approach, feint, corner, escape. At 100 steps
the tagger either catches immediately or the episode times out with no learning signal.

200 is a good middle ground — long enough for strategy, short enough to keep training
fast. Don't go higher until this is working.

**Side effect:** Update the comment in `agents/train.py` `make_ppo()` docstring that
references MAX_STEPS (currently says 200, should match actual value).

---

### 1.3 Rebalance rewards

**File:** `env/tag_env.py` — `GridState.step()`

**Changes:**

| Reward | Current | New | Rationale |
|--------|---------|-----|-----------|
| Tagger catch bonus | +20 | +10 | Large terminal rewards dominate per-step shaping and create noisy gradients |
| Runner catch penalty | -20 | -10 | Same — smoother learning when terminal events don't flip entire episode sign |
| Runner survival/step | +0.5 | +1.0 | At 200 steps, total survival = 200. Catch penalty (-10) = 10 steps lost. Good ratio. |
| Runner timeout bonus | +15 | **Remove** | Runner already gets +1/step for surviving. Lump sum at step 200 creates a value function discontinuity at step 199→200. The per-step signal is sufficient. |
| Tagger time penalty | -0.1 | -0.1 | Keep — this is already well-scaled |
| STAY penalty (tagger) | -0.5 | -0.3 | Slightly soften — current value is 5x the time penalty, which makes STAY catastrophic even when it's strategically correct (e.g., waiting around a corner) |

In `step()`, remove the `if truncated: runner_reward += 15.0` block entirely.

Update the reward values in the early-return catch block (tagger catches before runner
moves — though this block goes away if 1.1 is implemented) and in the main reward
computation section.

---

### 1.4 Reduce distance shaping weight

**File:** `env/tag_env.py` — constants at top

**Change:** `SHAPING_SCALE = 0.3` (from 0.5)

**Why:** The potential-based shaping uses Manhattan distance, which ignores walls. Two
agents 3 cells apart in Manhattan distance may be 15 steps apart through obstacles.
At 0.5 weight this misleading gradient dominates early learning. Reducing to 0.3 lets
the agent rely more on terminal rewards and learn wall-aware navigation from experience
rather than being pulled through walls.

**Future (optional, not in this phase):** Replace Manhattan distance with BFS shortest
path distance. Precompute a distance matrix at env init (20x20 grid = 400 cells, BFS
from each cell is cheap). This makes shaping accurate but adds complexity — only do
this if 0.3 weight doesn't stabilise training.

---

## Phase 2: Training Configuration Changes

### 2.1 Increase training budget

**File:** `agents/train.py` — CLI defaults

**Change:** Increase default `--num_cycles` from 200 to 500.

**Why:** 200 cycles x 2048 steps = ~410K steps per agent. Most multi-agent RL results
need millions of steps. 500 cycles = ~1M steps per agent — a reasonable target that
should still complete in a few hours on a laptop.

If 500 cycles shows clear trends but hasn't converged, go to 800-1000.

**How to run:**
```bash
python agents/train.py --num_cycles 500 --snapshot_freq 20
```

Snapshot every 20 cycles gives 25 snapshots across 500 cycles (same density as before).

---

### 2.2 Increase entropy coefficient

**File:** `agents/train.py` — `make_ppo()`

**Change:** `ent_coef=0.03` (from 0.01)

**Why:** In self-play, the opponent changes every cycle. The agent needs sustained
exploration to handle novel strategies. At 0.01 the policy collapses to near-deterministic
behaviour early, then can't adapt when the opponent shifts. 0.03 keeps the policy
stochastic longer, which is critical for adversarial robustness.

If policies still collapse (entropy_loss drops to near-zero in TensorBoard), try 0.05.

---

### 2.3 Adjust snapshot frequency for longer training

**File:** `agents/train.py` — CLI defaults

**Change:** Default `--snapshot_freq` from 8 to 20 (for 500 cycles → 25 snapshots).

**Why:** Keep ~25 snapshots for the evaluation plot regardless of total cycles. Too
many snapshots = slow evaluation. Too few = coarse co-evolution picture.

---

## Phase 3: Simplify Environment (if Phase 1+2 don't stabilise)

### 3.1 Reduce obstacle density

**File:** `env/tag_env.py` — `INTERIOR_OBSTACLES` list

**Change:** Comment out all but the middle island block (the 5x5 cluster at rows 8-10,
cols 9-13). This leaves one significant obstacle for LOS blocking and hiding, but
removes dead-ends, tight corridors, and trap geometries.

```python
INTERIOR_OBSTACLES = [
    # Middle island only — single obstacle for LOS and hiding
    (8, 9), (9, 9), (10, 9),
    (8, 10), (9, 10), (10, 10),
    (8, 11), (9, 11), (10, 11),
    (8, 12), (9, 12), (10, 12),
    (8, 13), (9, 13), (10, 13),
]
```

**Why:** Dense obstacles create degenerate strategies — runner hides in a dead-end,
tagger can't navigate. This is hard to learn from and hard to interpret. Get co-evolution
working on a simpler map first, then add complexity in a separate training run.

**Only do this if Phase 1+2 results still show cycling.** The current map is more
interesting for the report if it works.

---

### 3.2 Larger network (optional)

**File:** `agents/train.py` — `make_ppo()`

**Change:**
```python
policy_kwargs = dict(net_arch=[128, 128])
PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, ...)
```

**Why:** Default is [64, 64]. The partially observable environment with 15-dim input
might benefit from more capacity to learn wall-navigation heuristics and opponent
prediction. Low priority — architecture matters less than reward design.

---

## Implementation Order

```
Phase 1 (environment fixes):
  1.1 Simultaneous movement     ← biggest impact on fairness
  1.2 MAX_STEPS = 200           ← gives room for strategy
  1.3 Rebalance rewards         ← smoother learning signal
  1.4 SHAPING_SCALE = 0.3       ← less misleading wall gradients

Phase 2 (training config):
  2.1 num_cycles = 500          ← more training time
  2.2 ent_coef = 0.03           ← sustained exploration
  2.3 snapshot_freq = 20        ← match new cycle count

── Re-train and evaluate here ──

Phase 3 (only if still unstable):
  3.1 Simplify obstacles        ← reduce environment complexity
  3.2 Larger network            ← more model capacity
```

---

## Validation Checklist

After each re-training run, check:

- [ ] `ep_rew_mean` trends upward for both agents (TensorBoard)
- [ ] `ep_len_mean` is not stuck at 0 (tagger wins instantly) or MAX_STEPS (tagger never catches)
- [ ] `entropy_loss` decreases slowly, not instantly (policy exploring, not collapsed)
- [ ] Co-evolution plot: runner curve (tags conceded) decreases left-to-right
- [ ] Co-evolution plot: tagger curve (tags scored) increases left-to-right
- [ ] Standard deviations in eval CSVs are reasonable (< mean, not 2-3x mean)
- [ ] Both curves are smooth (no wild oscillation between adjacent snapshots)

---

## Files Modified

| File | Changes |
|------|---------|
| `env/tag_env.py` | Simultaneous movement, MAX_STEPS=200, reward rebalance, SHAPING_SCALE=0.3 |
| `agents/train.py` | num_cycles=500, ent_coef=0.03, snapshot_freq=20 |

All changes are in exactly two files. No new files needed.
