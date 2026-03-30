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
   We encode all task-relevant information explicitly in 11 dimensions:
     [own_row_n, own_col_n, lk_opp_row_n, lk_opp_col_n, own_dr, own_dc,
      can_move_up, can_move_down, can_move_left, can_move_right, opp_visible]
   A CNN over raw 12×12 pixels would need many more samples to learn positional
   features from scratch, and the grid is small enough that explicit coordinates
   are unambiguous. Flat vector → MlpPolicy → faster training convergence.
   Movement flags directly encode whether a cardinal action would be blocked by a wall,
   so the agent learns navigation constraints without trial-and-error wall collisions.

3. Partial observability via line-of-sight (LOS) occlusion with last-known position
   If a wall cell lies on the ray between the two agents, the opponent's *current*
   position is not known. The observation reports the last-known opponent position
   (the most recent step where LOS was clear) plus an explicit binary `opp_visible`
   flag (1.0 = currently visible, 0.0 = using stale last-known position). Before the
   opponent is ever seen, the last-known position is −1 (a sentinel for "never seen").
   This gives the policy a principled search target when the opponent hides, rather
   than a useless −1 that conveys no spatial information. The explicit visibility flag
   lets the policy learn to distinguish "I see them now" from "I last saw them there".
   LOS is computed with Bresenham's line algorithm using integer arithmetic —
   conservative on diagonal corner touches.

4. Velocity (last displacement) in observation
   own_dr, own_dc ∈ {−1, 0, 1} encode the direction the agent moved last step.
   This gives the policy a one-step momentum signal without requiring recurrence.
   Without it, the policy cannot distinguish "I'm moving towards the opponent"
   from "I just bounced off a wall and am now stationary".

5. Reward design — shaped tagger reward for dense learning signal
   Runner: +1/step (survival), −10 on catch.
   Tagger: +10 on catch, −0.1/step time penalty, plus three shaping components:
     (a) Potential-based distance shaping: F = γ·Φ(s') − Φ(s), Φ(s) = −dist.
         Provides a dense gradient toward the runner each step without altering
         the optimal policy (Ng et al., 1999).
     (b) STAY action penalty (−0.05): breaks standstill equilibria where the
         tagger avoids negative expected reward by doing nothing.
     (c) Revisit penalty (−0.05): penalises returning to a recently-visited cell,
         breaking 2-step loop traps common in early self-play training.

6. Simultaneous movement
   Both agents move at the same time each step. Sequential movement would give a
   first-mover advantage (the first-moving agent sees the updated position of the
   other and can react). Simultaneous movement is fairer and closer to real pursuit.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRID_SIZE = 12   # 12×12 grid; border cells are walls; interior is 10×10
MAX_STEPS = 200  # episode length cap; runner "wins" if this is reached

N_ACTIONS = 5
OBS_DIM   = 11   # 6 scalars (own pos, lk opp pos, velocity) + 4 movement flags + 1 visibility flag

# Tagger reward shaping constants
SHAPING_GAMMA   = 0.99   # must match PPO gamma
SHAPING_SCALE   = 0.5    # weight on potential-based distance shaping
STAY_PENALTY    = -0.05  # extra penalty for tagger choosing STAY (action 4)
REVISIT_PENALTY = -0.05  # penalty for tagger returning to a recently-visited cell
REVISIT_WINDOW  = 8      # number of recent tagger positions tracked for loop detection

# Interior obstacles: 11 cells.
# Creates 2 corner hiding spots (L-blocks) and 1 horizontal chokepoint bar.
INTERIOR_OBSTACLES = [
    # SW L-block
    (8, 2), (9, 2), (9, 3),
    # SE L-block
    (8, 9), (9, 8), (9, 9),
    # North chokepoint bar
    (4, 4), (4, 5), (4, 6), (4, 7),
]

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
        # Border walls + interior obstacles for line-of-sight occlusion and hiding.
        self.walls = np.zeros((grid_size, grid_size), dtype=np.int8)
        self.walls[0, :]  = 1   # top border
        self.walls[-1, :] = 1   # bottom border
        self.walls[:, 0]  = 1   # left border
        self.walls[:, -1] = 1   # right border
        for r, c in INTERIOR_OBSTACLES:
            self.walls[r, c] = 1

        # Agent positions — initialised properly by reset()
        self.tagger_pos    = np.array([1, 1], dtype=np.int32)
        self.runner_pos    = np.array([grid_size - 2, grid_size - 2], dtype=np.int32)
        self.tagger_last_d = np.zeros(2, dtype=np.int32)
        self.runner_last_d = np.zeros(2, dtype=np.int32)

        # Last-known opponent positions (updated only when LOS is clear).
        # −1 = never seen yet (episode start sentinel, outside [0,1] range).
        self.tagger_last_known_runner = np.array([-1.0, -1.0], dtype=np.float32)
        self.runner_last_known_tagger = np.array([-1.0, -1.0], dtype=np.float32)

        self.step_count = 0
        self.done       = False
        self.tagger_won = False
        self.runner_won = False

        self._tagger_pos_history: deque = deque(maxlen=REVISIT_WINDOW)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _passable_cells(self):
        """List of all (row, col) positions that are not walls."""
        rows, cols = np.where(self.walls == 0)
        return list(zip(rows.tolist(), cols.tolist()))

    def _has_los(self, a: np.ndarray, b: np.ndarray) -> bool:
        """
        Bresenham line-of-sight: returns True if no wall cell lies on the
        integer rasterisation of the ray from a to b (exclusive of endpoint b,
        inclusive of start a). Both positions are assumed to be passable.
        """
        r0, c0 = int(a[0]), int(a[1])
        r1, c1 = int(b[0]), int(b[1])
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        r, c = r0, c0
        n = 1 + dr + dc           # total cells visited (including start)
        r_inc = 1 if r1 > r0 else -1
        c_inc = 1 if c1 > c0 else -1
        error = dr - dc
        dr *= 2
        dc *= 2
        for _ in range(n - 1):    # iterate n-1 steps, stopping before endpoint
            if self.walls[r, c]:  # wall at current cell blocks sight
                return False
            if error > 0:
                r += r_inc
                error -= dc
            else:
                c += c_inc
                error += dr
        return True

    def _get_movement_flags(self, pos: np.ndarray) -> np.ndarray:
        """
        Return a 4-element float32 array indicating passability in cardinal directions.
        [can_move_up, can_move_down, can_move_left, can_move_right]
        Each value is 1.0 (passable) or 0.0 (wall blocks movement).
        """
        flags = []
        for action in range(4):  # UP, DOWN, LEFT, RIGHT
            dr, dc = ACTIONS[action]
            candidate = pos + np.array([dr, dc], dtype=np.int32)
            candidate = np.clip(candidate, 0, self.grid_size - 1)
            can_move = 0.0 if self.walls[candidate[0], candidate[1]] else 1.0
            flags.append(can_move)
        return np.array(flags, dtype=np.float32)

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
        self.tagger_last_known_runner = np.array([-1.0, -1.0], dtype=np.float32)
        self.runner_last_known_tagger = np.array([-1.0, -1.0], dtype=np.float32)
        self.step_count    = 0
        self.done          = False
        self.tagger_won    = False
        self.runner_won    = False
        self._tagger_pos_history.clear()

    def step(self, tagger_action: int, runner_action: int):
        """
        Apply both actions simultaneously, update positions, check termination.
        Returns (tagger_reward, runner_reward, terminated, truncated, info).

        Simultaneous movement: both intended moves are resolved at the same time.
        This prevents first-mover advantages and matches the spirit of the game.

        Tagger reward = −0.1 (time penalty)
                      + γ·Φ(s') − Φ(s)  where Φ(s) = −dist  (potential-based shaping)
                      + STAY_PENALTY if action == STAY  (break standstill equilibria)
                      + REVISIT_PENALTY if new cell was recently visited  (break loops)
                      + 10.0 on catch
        """
        assert not self.done, "step() called on a finished episode — call reset() first"

        dist_before = float(np.sum(np.abs(self.tagger_pos - self.runner_pos)))

        new_tagger, t_delta = self._try_move(self.tagger_pos, tagger_action)
        new_runner, r_delta = self._try_move(self.runner_pos, runner_action)

        self.tagger_pos    = new_tagger
        self.runner_pos    = new_runner
        self.tagger_last_d = t_delta
        self.runner_last_d = r_delta
        self.step_count   += 1

        tagged    = bool(np.array_equal(self.tagger_pos, self.runner_pos))
        timed_out = self.step_count >= MAX_STEPS

        # Base per-step rewards
        tagger_reward = -0.1   # time penalty: tagger must actively pursue
        runner_reward =  1.0   # survival bonus: runner wants to last as long as possible

        # Fix 1: Potential-based distance shaping — dense chase signal.
        # F(s, s') = γ·Φ(s') − Φ(s), Φ(s) = −dist preserves the optimal policy.
        dist_after = float(np.sum(np.abs(self.tagger_pos - self.runner_pos)))
        tagger_reward += (SHAPING_GAMMA * (-dist_after) - (-dist_before)) * SHAPING_SCALE

        # Fix 3: Extra STAY penalty — discourages standstill equilibria.
        if tagger_action == 4:
            tagger_reward += STAY_PENALTY

        # Fix 2: Revisit penalty — discourages 2-step loops.
        tagger_pos_key = (int(self.tagger_pos[0]), int(self.tagger_pos[1]))
        if tagger_pos_key in self._tagger_pos_history:
            tagger_reward += REVISIT_PENALTY
        self._tagger_pos_history.append(tagger_pos_key)

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

    def get_tagger_obs(self) -> np.ndarray:
        """
        Tagger's observation vector (float32, shape (OBS_DIM,) = (11,)):
          [own_row_n, own_col_n, lk_runner_row_n, lk_runner_col_n, own_dr, own_dc,
           can_move_up, can_move_down, can_move_left, can_move_right, opp_visible]

        Positions are normalised to [0, 1] by dividing by (grid_size − 1).
        Velocity components are in {−1, 0, 1} (raw deltas, already small).
        lk_runner_{row,col}_n is the *last-known* runner position: updated each step
        when LOS is clear, held fixed when LOS is blocked. Before the runner has ever
        been seen it is −1 (episode-start sentinel). opp_visible is 1.0 when the runner
        is currently visible (LOS clear), 0.0 when using the stale last-known position.
        Movement flags are 1.0 (passable) or 0.0 (blocked by wall) for each cardinal
        direction, allowing the agent to learn immediate navigation constraints.
        """
        norm = float(self.grid_size - 1)
        own_r = self.tagger_pos[0] / norm
        own_c = self.tagger_pos[1] / norm

        if self._has_los(self.tagger_pos, self.runner_pos):
            lk_r = self.runner_pos[0] / norm
            lk_c = self.runner_pos[1] / norm
            self.tagger_last_known_runner[:] = [lk_r, lk_c]
            visible = 1.0
        else:
            lk_r, lk_c = float(self.tagger_last_known_runner[0]), float(self.tagger_last_known_runner[1])
            visible = 0.0

        dr = float(self.tagger_last_d[0])
        dc = float(self.tagger_last_d[1])
        move_flags = self._get_movement_flags(self.tagger_pos)
        return np.concatenate([[own_r, own_c, lk_r, lk_c, dr, dc], move_flags, [visible]]).astype(np.float32)

    def get_runner_obs(self) -> np.ndarray:
        """
        Runner's observation vector (float32, shape (OBS_DIM,) = (11,)):
          [own_row_n, own_col_n, lk_tagger_row_n, lk_tagger_col_n, own_dr, own_dc,
           can_move_up, can_move_down, can_move_left, can_move_right, opp_visible]

        Same partial observability logic as the tagger: the last-known tagger position
        is reported when LOS is blocked, together with an opp_visible flag (1.0 = tagger
        currently in sight, 0.0 = using stale last-known position). Before the tagger
        has ever been seen, the last-known position is −1 (episode-start sentinel).
        Movement flags are 1.0 (passable) or 0.0 (blocked by wall) for each cardinal
        direction, allowing the agent to learn immediate navigation constraints.
        """
        norm = float(self.grid_size - 1)
        own_r = self.runner_pos[0] / norm
        own_c = self.runner_pos[1] / norm

        if self._has_los(self.runner_pos, self.tagger_pos):
            lk_r = self.tagger_pos[0] / norm
            lk_c = self.tagger_pos[1] / norm
            self.runner_last_known_tagger[:] = [lk_r, lk_c]
            visible = 1.0
        else:
            lk_r, lk_c = float(self.runner_last_known_tagger[0]), float(self.runner_last_known_tagger[1])
            visible = 0.0

        dr = float(self.runner_last_d[0])
        dc = float(self.runner_last_d[1])
        move_flags = self._get_movement_flags(self.runner_pos)
        return np.concatenate([[own_r, own_c, lk_r, lk_c, dr, dc], move_flags, [visible]]).astype(np.float32)


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
    print(f"obs[:6]   : {obs[:6]}  (own_r, own_c, lk_opp_r, lk_opp_c, dr, dc)")
    print(f"obs[6:10] : {obs[6:10]}  (movement flags: up, down, left, right)")
    print(f"obs[10]   : {obs[10]}  (opp_visible)")
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
    print(f"obs[:6]   : {obs[:6]}  (own_r, own_c, lk_opp_r, lk_opp_c, dr, dc)")
    print(f"obs[6:10] : {obs[6:10]}  (movement flags: up, down, left, right)")
    print(f"obs[10]   : {obs[10]}  (opp_visible)")
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

    print("\n[PASS] obs shape = (11,), rewards are floats, "
          "termination flags correct, episode terminates.")
    sys.exit(0)
