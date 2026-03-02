from __future__ import annotations

from pathlib import Path
import numpy as np
import gym
import ChefsHatGym.env

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)

from single_agent_wrapper import SingleAgentWrapper


# ==============================
# Configuration
# ==============================

SEED_VALUE = 42
DISCOUNT_FACTOR = 0.99
TRAINING_STEPS = 200_000
SAVE_LOCATION = Path("models/ppo_chefhats_masked")


# ==============================
# Custom Callback
# ==============================

class EpisodeWinTracker(BaseCallback):
    """
    Logs the percentage of matches won (1st place finishes).
    A win corresponds to Match_Score == 3 for the learning seat.
    """

    def __init__(self, learning_seat: int = 0, verbose: int = 0):
        super().__init__(verbose)
        self.learning_seat = learning_seat
        self._results_buffer: list[int] = []

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                scores = info.get("Match_Score", [])
                if scores:
                    win_flag = int(int(scores[self.learning_seat]) == 3)
                    self._results_buffer.append(win_flag)
        return True

    def _on_rollout_end(self) -> None:
        if self._results_buffer:
            win_ratio = float(np.mean(self._results_buffer))
            self.logger.record("rollout/win_rate", win_ratio)
            self._results_buffer.clear()


# ==============================
# Training Pipeline
# ==============================

def main() -> None:
    env = SingleAgentWrapper(
        env_id="chefshat-v1",
        learning_seat=0,
        seed=SEED_VALUE
    )

    policy_type = (
        "MultiInputPolicy"
        if isinstance(env.observation_space, gym.spaces.Dict)
        else "MlpPolicy"
    )

    model = MaskablePPO(
        policy=policy_type,
        env=env,
        gamma=DISCOUNT_FACTOR,
        n_steps=2048,
        batch_size=64,
        seed=SEED_VALUE,
        verbose=1,
    )

    SAVE_LOCATION.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=str(SAVE_LOCATION.parent),
        name_prefix=SAVE_LOCATION.name,
    )

    callbacks = CallbackList([
        EpisodeWinTracker(learning_seat=0),
        checkpoint_cb
    ])

    model.learn(total_timesteps=TRAINING_STEPS, callback=callbacks)
    model.save(str(SAVE_LOCATION))

    print(f"Training complete. Model stored at {SAVE_LOCATION}.zip")
    env.close()


if __name__ == "__main__":
    main()