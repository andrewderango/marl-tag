"""
env/tag_env.py — Custom Gymnasium environments for the multi-agent tag game.

Architecture: two separate Gymnasium envs (TaggerEnv, RunnerEnv) that each wrap
a shared GridState object. SB3 drives one env at a time; the frozen opponent's
action is computed inside step() so SB3 only sees a standard single-agent MDP.

--- Report-relevant design decisions ---

1. Two-env / shared-state architecture
   Each env is a standard Gymnasium env, so SB3 can plug in without modification.
   The GridState object holds ground-truth positions and is owned by the env that
   is currently being trained; the other env re-uses that same object during eval.
   This avoids synchronisation bugs that arise when two separate envs would drift.

2. Flat observation vector (MlpPolicy, not CNN)
   We encode all task-relevant information explicitly:
     [own_row_n, own_col_n, opp_row_n, opp_col_n, own_dr, own_dc, wall_patch...]
   A CNN over raw 12×12 pixels would need many more samples to learn positional
   features from scratch, and the grid is small enough that explicit coordinates
   are unambiguous. Flat vector → MlpPolicy → faster training convergence.

3. Partial observability via visibility radius (Chebyshev distance)
   If the opponent is more than VISIBILITY_RADIUS cells away (L∞ norm), their
   position is masked to −1 in the observation. −1 is outside the normal [0, 1]
   normalised range, so the policy can learn to distinguish "I see them" from
   "I don't" without a separate binary flag. This forces the runner to learn to
   hide and the tagger to learn to search, making evasion/pursuit non-trivial.
   Chebyshev distance is chosen because it matches how the 5×5 wall patch is
   computed and aligns with the 4-directional movement model.

4. Velocity (last displacement) in observation
   own_dr, own_dc ∈ {−1, 0, 1} encode the direction the agent moved last step.
   This gives the policy a one-step momentum signal without requiring recurrence.
   Without it, the policy cannot distinguish "I'm moving towards the opponent"
   from "I just bounced off a wall and am now stationary".

5. Reward design — sparse + small tagger time penalty
   Runner: +1/step (survival), −10 on catch (distinguishes losing from winning).
   Tagger: +10 on catch, −0.1/step (discourages stalling — without this, a tagger
   that never catches the runner still gets reward 0, which is the same as the
   initial untrained baseline and provides no gradient to improve from).

6. Simultaneous movement
   Both agents move at the same time each step. Sequential movement would give a
   first-mover advantage (the first-moving agent sees the updated position of the
   other and can react). Simultaneous movement is fairer and closer to real pursuit.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRID_SIZE         = 12   # 12×12 grid; border cells are walls; interior is 10×10
MAX_STEPS         = 200  # episode length cap; runner "wins" if this is reached
VISIBILITY_RADIUS = 5    # Chebyshev radius beyond which opponent pos is masked

PATCH_RADIUS = 2                          # wall patch half-width
PATCH_SIZE   = 2 * PATCH_RADIUS + 1      # 5×5 = 25 cells
N_ACTIONS    = 5
OBS_DIM      = 6 + PATCH_SIZE * PATCH_SIZE   # 6 scalars + 25 wall values = 31

# Action index → (delta_row, delta_col)
# Row increases downward; UP means row−1, DOWN means row+1.
ACTIONS = {
    0: (-1,  0),   # UP
    1: ( 1,  0),   # DOWN
    2: ( 0, -1),   # LEFT
    3: ( 0,  1),   # RIGHT
    4: ( 0,  0),   # STAY
}


# ---------------------------------------------------------------------------
# Shared grid state
# ---------------------------------------------------------------------------

class GridState:
    """
    Ground-truth game state shared between TaggerEnv and RunnerEnv.

    Owns the static wall map and the mutable agent positions. Both env classes
    hold a reference to the same GridState so their views are always consistent.
    Keeping state here (rather than duplicating it in each env) is the canonical
    way to implement shared-state multi-agent envs without a full MARL framework.
    """

    def __init__(self, grid_size: int = GRID_SIZE, seed: int = None):
        self.grid_size = grid_size
        self.rng = np.random.default_rng(seed)

        # Static wall map: 1 = wall, 0 = passable.
        # Simple baseline: only border walls, open interior.
        # Adding internal walls is straightforward later but unnecessary for
        # a working baseline (see CLAUDE.md notes on keeping env simple first).
        self.walls = np.zeros((grid_size, grid_size), dtype=np.int8)
        self.walls[0, :]  = 1   # top border
        self.walls[-1, :] = 1   # bottom border
        self.walls[:, 0]  = 1   # left border
        self.walls[:, -1] = 1   # right border

        # Agent positions — initialised properly by reset()
        self.tagger_pos    = np.array([1, 1], dtype=np.int32)
        self.runner_pos    = np.array([grid_size - 2, grid_size - 2], dtype=np.int32)
        self.tagger_last_d = np.zeros(2, dtype=np.int32)
        self.runner_last_d = np.zeros(2, dtype=np.int32)

        self.step_count = 0
        self.done       = False
        self.tagger_won = False
        self.runner_won = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _passable_cells(self):
        """List of all (row, col) positions that are not walls."""
        rows, cols = np.where(self.walls == 0)
        return list(zip(rows.tolist(), cols.tolist()))

    def _chebyshev(self, a: np.ndarray, b: np.ndarray) -> int:
        """L∞ distance between two positions — natural metric for grid movement."""
        return int(np.max(np.abs(a.astype(int) - b.astype(int))))

    def _try_move(self, pos: np.ndarray, action: int):
        """
        Attempt to move from pos using action. Returns (new_pos, actual_delta).
        Movement into a wall is blocked; agent stays in place and delta is (0,0).
        """
        dr, dc = ACTIONS[action]
        candidate = pos + np.array([dr, dc], dtype=np.int32)
        candidate = np.clip(candidate, 0, self.grid_size - 1)
        if self.walls[candidate[0], candidate[1]]:
            return pos.copy(), np.zeros(2, dtype=np.int32)
        return candidate, np.array([dr, dc], dtype=np.int32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """
        Randomise agent starting positions (non-overlapping), reset counters.
        Both agents are placed uniformly at random among all passable cells.
        """
        passable = self._passable_cells()
        idx = self.rng.choice(len(passable), size=2, replace=False)
        self.tagger_pos    = np.array(passable[idx[0]], dtype=np.int32)
        self.runner_pos    = np.array(passable[idx[1]], dtype=np.int32)
        self.tagger_last_d = np.zeros(2, dtype=np.int32)
        self.runner_last_d = np.zeros(2, dtype=np.int32)
        self.step_count    = 0
        self.done          = False
        self.tagger_won    = False
        self.runner_won    = False

    def step(self, tagger_action: int, runner_action: int):
        """
        Apply both actions simultaneously, update positions, check termination.
        Returns (tagger_reward, runner_reward, terminated, truncated, info).

        Simultaneous movement: both intended moves are resolved at the same time.
        This prevents first-mover advantages and matches the spirit of the game.
        """
        assert not self.done, "step() called on a finished episode — call reset() first"

        new_tagger, t_delta = self._try_move(self.tagger_pos, tagger_action)
        new_runner, r_delta = self._try_move(self.runner_pos, runner_action)

        self.tagger_pos    = new_tagger
        self.runner_pos    = new_runner
        self.tagger_last_d = t_delta
        self.runner_last_d = r_delta
        self.step_count   += 1

        tagged    = bool(np.array_equal(self.tagger_pos, self.runner_pos))
        timed_out = self.step_count >= MAX_STEPS

        # Per-step rewards
        tagger_reward = -0.1   # time penalty: tagger must actively pursue
        runner_reward =  1.0   # survival bonus: runner wants to last as long as possible

        if tagged:
            tagger_reward += 10.0
            runner_reward += -10.0
            self.tagger_won = True
            self.done = True

        terminated = tagged
        truncated  = (not tagged) and timed_out
        if truncated:
            self.runner_won = True
            self.done = True

        info = {
            "tagged":      tagged,
            "timed_out":   timed_out,
            "step_count":  self.step_count,
            "tagger_won":  self.tagger_won,
            "runner_won":  self.runner_won,
        }
        return tagger_reward, runner_reward, terminated, truncated, info

    def get_local_patch(self, pos: np.ndarray) -> np.ndarray:
        """
        Return a flattened (PATCH_SIZE²,) float32 binary array of walls in the
        (2*PATCH_RADIUS+1)² neighbourhood centred on pos.  Cells outside the
        grid boundary are treated as walls (value 1).

        Design note: a 5×5 patch (radius 2) lets the agent see 2 cells in every
        direction.  This is enough to avoid walking into walls while keeping the
        observation vector small (25 extra floats).  A larger patch would give
        more look-ahead but diminishing returns — the border walls are the main
        obstacle in the baseline environment.
        """
        pr = PATCH_RADIUS
        patch = np.ones((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)  # default: wall
        r0, c0 = int(pos[0]), int(pos[1])
        for dr in range(-pr, pr + 1):
            for dc in range(-pr, pr + 1):
                r, c = r0 + dr, c0 + dc
                if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                    patch[dr + pr, dc + pr] = float(self.walls[r, c])
        return patch.flatten()

    def get_tagger_obs(self) -> np.ndarray:
        """
        Tagger's observation vector (float32, shape (OBS_DIM,) = (31,)):
          [own_row_n, own_col_n, runner_row_n, runner_col_n, own_dr, own_dc,
           wall_patch (25 values)]

        Positions are normalised to [0, 1] by dividing by (grid_size − 1).
        Velocity components are in {−1, 0, 1} (raw deltas, already small).
        Masked positions are −1 (outside the [0,1] range, so the policy can
        learn to distinguish visible from invisible opponents).
        """
        norm = float(self.grid_size - 1)
        own_r = self.tagger_pos[0] / norm
        own_c = self.tagger_pos[1] / norm

        if self._chebyshev(self.tagger_pos, self.runner_pos) <= VISIBILITY_RADIUS:
            opp_r = self.runner_pos[0] / norm
            opp_c = self.runner_pos[1] / norm
        else:
            opp_r, opp_c = -1.0, -1.0   # partial observability: runner not visible

        dr = float(self.tagger_last_d[0])
        dc = float(self.tagger_last_d[1])
        patch = self.get_local_patch(self.tagger_pos)
        return np.array([own_r, own_c, opp_r, opp_c, dr, dc], dtype=np.float32)  \
               if False else \
               np.concatenate([[own_r, own_c, opp_r, opp_c, dr, dc], patch]).astype(np.float32)

    def get_runner_obs(self) -> np.ndarray:
        """
        Runner's observation vector (float32, shape (OBS_DIM,) = (31,)):
          [own_row_n, own_col_n, tagger_row_n, tagger_col_n, own_dr, own_dc,
           wall_patch (25 values)]

        Same partial observability logic: tagger position is masked to −1 when
        outside VISIBILITY_RADIUS.  This forces the runner to learn to evade
        even when the tagger is not visible — e.g. by moving to corners or by
        remembering last-known tagger position implicitly through the value fn.
        """
        norm = float(self.grid_size - 1)
        own_r = self.runner_pos[0] / norm
        own_c = self.runner_pos[1] / norm

        if self._chebyshev(self.runner_pos, self.tagger_pos) <= VISIBILITY_RADIUS:
            opp_r = self.tagger_pos[0] / norm
            opp_c = self.tagger_pos[1] / norm
        else:
            opp_r, opp_c = -1.0, -1.0   # partial observability: tagger not visible

        dr = float(self.runner_last_d[0])
        dc = float(self.runner_last_d[1])
        patch = self.get_local_patch(self.runner_pos)
        return np.concatenate([[own_r, own_c, opp_r, opp_c, dr, dc], patch]).astype(np.float32)


# ---------------------------------------------------------------------------
# Gymnasium environments
# ---------------------------------------------------------------------------

class TaggerEnv(gym.Env):
    """
    Single-agent Gymnasium env for the Tagger.

    From SB3's perspective this is a standard MDP: the tagger picks an action,
    the env returns (obs, reward, terminated, truncated, info).  Internally the
    env also computes the runner's action from the frozen opponent model and
    applies it to the shared GridState.

    set_opponent(model) is called before each training cycle to swap in the
    latest frozen runner snapshot.  If no opponent is set, the runner acts
    randomly — useful for the very first training cycle.
    """

    metadata = {"render_modes": []}

    def __init__(self, grid_size: int = GRID_SIZE, seed: int = None):
        super().__init__()
        self.grid_state     = GridState(grid_size=grid_size, seed=seed)
        self.opponent_model = None   # frozen RunnerModel; None → uniform random

        self.action_space = spaces.Discrete(N_ACTIONS)

        # Bounds: normalised positions ∈ [0,1], masked to −1; velocity ∈ {−1,0,1};
        # wall patch ∈ {0,1}.  Using [−1, 1] as a universal safe bound.
        low  = np.full(OBS_DIM, -1.0, dtype=np.float32)
        high = np.full(OBS_DIM,  1.0, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def set_opponent(self, model) -> None:
        """Swap in a new frozen runner model.  Called between training cycles."""
        self.opponent_model = model

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.grid_state.rng = np.random.default_rng(seed)
        self.grid_state.reset()
        return self.grid_state.get_tagger_obs(), {}

    def step(self, tagger_action: int):
        # Compute the runner's action from the frozen opponent model.
        runner_obs = self.grid_state.get_runner_obs()
        if self.opponent_model is not None:
            runner_action, _ = self.opponent_model.predict(runner_obs, deterministic=False)
            runner_action = int(runner_action)
        else:
            runner_action = self.action_space.sample()

        tagger_reward, _, terminated, truncated, info = self.grid_state.step(
            int(tagger_action), runner_action
        )
        obs = self.grid_state.get_tagger_obs()
        return obs, float(tagger_reward), terminated, truncated, info


class RunnerEnv(gym.Env):
    """
    Single-agent Gymnasium env for the Runner.

    Mirror of TaggerEnv: the tagger's action is computed from the frozen
    opponent model inside step().  From SB3's perspective, a standard MDP.
    """

    metadata = {"render_modes": []}

    def __init__(self, grid_size: int = GRID_SIZE, seed: int = None):
        super().__init__()
        self.grid_state     = GridState(grid_size=grid_size, seed=seed)
        self.opponent_model = None   # frozen TaggerModel; None → uniform random

        self.action_space = spaces.Discrete(N_ACTIONS)

        low  = np.full(OBS_DIM, -1.0, dtype=np.float32)
        high = np.full(OBS_DIM,  1.0, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def set_opponent(self, model) -> None:
        """Swap in a new frozen tagger model.  Called between training cycles."""
        self.opponent_model = model

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.grid_state.rng = np.random.default_rng(seed)
        self.grid_state.reset()
        return self.grid_state.get_runner_obs(), {}

    def step(self, runner_action: int):
        # Compute the tagger's action from the frozen opponent model.
        tagger_obs = self.grid_state.get_tagger_obs()
        if self.opponent_model is not None:
            tagger_action, _ = self.opponent_model.predict(tagger_obs, deterministic=False)
            tagger_action = int(tagger_action)
        else:
            tagger_action = self.action_space.sample()

        _, runner_reward, terminated, truncated, info = self.grid_state.step(
            tagger_action, int(runner_action)
        )
        obs = self.grid_state.get_runner_obs()
        return obs, float(runner_reward), terminated, truncated, info


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("TaggerEnv sanity check")
    print("=" * 60)

    tagger_env = TaggerEnv(seed=42)
    obs, info = tagger_env.reset()
    print(f"obs shape : {obs.shape}  dtype: {obs.dtype}")
    print(f"obs[:6]   : {obs[:6]}  (own_r, own_c, opp_r, opp_c, dr, dc)")
    print(f"obs[6:]   : {obs[6:]}  (25-cell wall patch)")
    assert obs.shape == (OBS_DIM,), f"Expected ({OBS_DIM},), got {obs.shape}"

    total_r = 0.0
    for step_i in range(1, MAX_STEPS + 2):
        action = tagger_env.action_space.sample()
        obs, reward, terminated, truncated, info = tagger_env.step(action)
        total_r += reward
        print(f"  step {step_i:3d}: action={action}  reward={reward:+.2f}  "
              f"terminated={terminated}  truncated={truncated}  info={info}")
        if terminated or truncated:
            break
    print(f"Episode ended at step {info['step_count']}.  "
          f"tagger_won={info['tagger_won']}  runner_won={info['runner_won']}")
    print(f"Total tagger reward: {total_r:.2f}\n")

    print("=" * 60)
    print("RunnerEnv sanity check")
    print("=" * 60)

    runner_env = RunnerEnv(seed=0)
    obs, info = runner_env.reset()
    print(f"obs shape : {obs.shape}  dtype: {obs.dtype}")
    print(f"obs[:6]   : {obs[:6]}  (own_r, own_c, opp_r, opp_c, dr, dc)")
    assert obs.shape == (OBS_DIM,), f"Expected ({OBS_DIM},), got {obs.shape}"

    total_r = 0.0
    for step_i in range(1, MAX_STEPS + 2):
        action = runner_env.action_space.sample()
        obs, reward, terminated, truncated, info = runner_env.step(action)
        total_r += reward
        print(f"  step {step_i:3d}: action={action}  reward={reward:+.2f}  "
              f"terminated={terminated}  truncated={truncated}  info={info}")
        if terminated or truncated:
            break
    print(f"Episode ended at step {info['step_count']}.  "
          f"tagger_won={info['tagger_won']}  runner_won={info['runner_won']}")
    print(f"Total runner reward: {total_r:.2f}")

    print("\n[PASS] obs shape = (31,), rewards are floats, "
          "termination flags correct, episode terminates.")
    sys.exit(0)
