from __future__ import annotations

import contextlib
import io
import logging
from dataclasses import dataclass
from typing import Any

import gym
import numpy as np
from ChefsHatGym.rewards.only_winning import RewardOnlyWinning
from ChefsHatGym.rewards.reward import Reward


DISCOUNT_FACTOR = 0.99


@contextlib.contextmanager
def _mute_external_output():
    buffer = io.StringIO()
    root = logging.getLogger()
    previous_level = root.level
    root.setLevel(logging.CRITICAL + 1)

    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        try:
            yield
        finally:
            root.setLevel(previous_level)


@dataclass
class StepResult:
    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class SingleAgentWrapper(gym.Env):
    """
    Converts Chef’s Hat into a single-agent environment.
    Seat 0 is controlled by RL; others act randomly.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_id: str = "chefshat-v1",
        learning_seat: int = 0,
        seed: int = 42,
        reward_fn: Reward | None = None,
    ):
        super().__init__()

        self.learning_seat = learning_seat
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._reward_logic = reward_fn if reward_fn else RewardOnlyWinning()

        self.base_env = gym.make(env_id)

        with _mute_external_output():
            self.base_env.startExperiment(
                playerNames=["RL", "Random1", "Random2", "Random3"],
                logDirectory="log",
                verbose=False,
            )

        self.observation_space = self.base_env.observation_space
        self.action_space = self.base_env.action_space

        self._last_obs = None
        self._last_info = {}

    # ==============================
    # Potential-based shaping
    # ==============================

    def _phi(self, obs):
        hand_section = obs[11:28]
        remaining_cards = np.count_nonzero(hand_section)
        return -(remaining_cards / 17.0)

    # ==============================
    # Core Gym API
    # ==============================

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)

        with _mute_external_output():
            result = self.base_env.reset(seed=self._seed, options=options)

        obs, info = result if isinstance(result, tuple) else (result, {})

        obs, _, terminated, truncated, info = \
            self._advance_to_learning_turn(obs, info)

        self._last_obs = obs
        self._last_info = info

        return obs, info

    def step(self, action: int):
        if not self._is_valid(action):
            raise ValueError(f"Invalid action {action}")

        phi_prev = self._phi(self._last_obs)

        result = self._base_step(action)
        total_reward = result.reward

        if result.terminated or result.truncated:
            bonus = self._terminal_reward(result.info)
            return (
                result.observation,
                total_reward + bonus,
                result.terminated,
                result.truncated,
                result.info,
            )

        obs, extra_reward, terminated, truncated, info = \
            self._advance_to_learning_turn(result.observation, result.info)

        total_reward += extra_reward

        phi_next = self._phi(obs)
        shaped = total_reward + DISCOUNT_FACTOR * phi_next - phi_prev + self._step_penalty()

        if terminated or truncated:
            shaped += self._terminal_reward(info)

        self._last_obs = obs
        self._last_info = info

        return obs, shaped, terminated, truncated, info

    # ==============================
    # Reward helpers
    # ==============================

    def _terminal_reward(self, info):
        scores = info.get("Match_Score", [])
        if not scores:
            return 0.0

        match_score = int(scores[self.learning_seat])
        position = 3 - match_score
        return float(self._reward_logic.getReward(position, True))

    def _step_penalty(self):
        return float(self._reward_logic.getReward(0, False))

    # ==============================
    # Turn management
    # ==============================

    def _advance_to_learning_turn(self, obs, info):
        total_reward = 0.0
        terminated = truncated = False
        current_obs = obs
        current_info = info

        while (
            not (terminated or truncated)
            and self._current_player(current_info) != self.learning_seat
        ):
            self._last_obs = current_obs
            valid_actions = np.flatnonzero(self.action_masks())
            action = int(self._rng.choice(valid_actions))

            result = self._base_step(action)
            total_reward += result.reward

            current_obs = result.observation
            current_info = result.info
            terminated = result.terminated
            truncated = result.truncated

        return current_obs, float(total_reward), terminated, truncated, current_info

    def _base_step(self, action: int) -> StepResult:
        action_vec = np.zeros(self.action_space.n, dtype=np.float32)
        action_vec[int(action)] = 1.0

        output = self.base_env.step(action_vec)

        if len(output) == 5:
            obs, reward, terminated, truncated, info = output
        else:
            obs, reward, done, info = output
            terminated, truncated = done, False

        return StepResult(obs, float(reward), bool(terminated), bool(truncated), dict(info))

    # ==============================
    # Utilities
    # ==============================

    def action_masks(self):
        mask = np.asarray(self._last_obs)[28:]
        return mask.astype(bool)

    def get_action_mask(self):
        return self.action_masks()

    def _is_valid(self, action):
        mask = self.action_masks()
        return 0 <= int(action) < mask.size and bool(mask[int(action)])

    def _current_player(self, info):
        keys = [
            info.get("current_player"),
            info.get("currentPlayer"),
            getattr(self.base_env, "current_player", None),
            getattr(self.base_env, "currentPlayer", None),
        ]
        for k in keys:
            if k is not None:
                return int(k)
        raise RuntimeError("Unable to determine current player")

    def render(self):
        return self.base_env.render() if hasattr(self.base_env, "render") else None

    def close(self):
        self.base_env.close()