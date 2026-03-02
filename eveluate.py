from __future__ import annotations

from pathlib import Path
import numpy as np
from sb3_contrib import MaskablePPO
from single_agent_wrapper import SingleAgentWrapper


MODEL_FILE = Path("models/ppo_chefhats_masked")
EPISODES = 100
SEAT = 0
SEED_BASE = 42


def locate_model(path: Path) -> Path:
    if Path(str(path) + ".zip").exists():
        return path

    checkpoints = sorted(path.parent.glob(f"{path.name}_*_steps.zip"))
    if checkpoints:
        latest = checkpoints[-1]
        print(f"Using latest checkpoint: {latest.name}")
        return latest.with_suffix("")

    raise FileNotFoundError("No trained model found.")


def is_first_place(info: dict) -> bool:
    scores = info.get("Match_Score", [])
    return bool(scores) and int(scores[SEAT]) == 3


def main():
    env = SingleAgentWrapper("chefshat-v1", SEAT, SEED_BASE)
    model_path = locate_model(MODEL_FILE)
    model = MaskablePPO.load(str(model_path), env=env)

    print(f"Loaded model from: {model_path}")

    wins = 0
    rewards = []

    for ep in range(EPISODES):
        obs, info = env.reset(seed=SEED_BASE + ep)
        done = truncated = False
        total = 0.0

        while not (done or truncated):
            action, _ = model.predict(
                obs,
                deterministic=True,
                action_masks=env.action_masks()
            )
            obs, reward, done, truncated, info = env.step(int(action))
            total += reward

        if is_first_place(info):
            wins += 1

        rewards.append(total)

    print(f"Episodes: {EPISODES}")
    print(f"Wins: {wins}")
    print(f"Win Rate: {(wins / EPISODES) * 100:.1f}%")
    print(f"Average Reward: {np.mean(rewards):.4f}")

    env.close()


if __name__ == "__main__":
    main()