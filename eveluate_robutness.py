from __future__ import annotations

from pathlib import Path
import numpy as np
from sb3_contrib import MaskablePPO
from single_agent_wrapper import SingleAgentWrapper


MODEL_PATH = Path("models/ppo_chefhats_masked")
TEST_SEEDS = [10, 99, 1234, 5555, 9999]
EPISODES_PER_SEED = 50
SEAT = 0


def load_model(path: Path) -> Path:
    if Path(str(path) + ".zip").exists():
        return path
    raise FileNotFoundError("Model not found. Train first.")


def run_robustness_evaluation():
    model_path = load_model(MODEL_PATH)
    print(f"Evaluating model: {model_path}")

    results = {}

    for seed in TEST_SEEDS:
        print(f"\nTesting seed {seed}")
        env = SingleAgentWrapper("chefshat-v1", SEAT, seed)
        model = MaskablePPO.load(str(model_path), env=env)

        wins = 0
        reward_list = []

        for ep in range(EPISODES_PER_SEED):
            obs, info = env.reset(seed=seed + ep)
            done = truncated = False
            total_reward = 0.0

            while not (done or truncated):
                action, _ = model.predict(
                    obs,
                    deterministic=True,
                    action_masks=env.action_masks()
                )
                obs, reward, done, truncated, info = env.step(int(action))
                total_reward += reward

            scores = info.get("Match_Score", [])
            if scores and int(scores[SEAT]) == 3:
                wins += 1

            reward_list.append(total_reward)

        win_rate = (wins / EPISODES_PER_SEED) * 100
        avg_reward = float(np.mean(reward_list))

        results[seed] = (win_rate, avg_reward)

        print(f"Win Rate: {win_rate:.1f}% | Avg Reward: {avg_reward:.2f}")

        env.close()

    print("\n===== ROBUSTNESS SUMMARY =====")
    win_rates = []

    for seed, (wr, ar) in results.items():
        win_rates.append(wr)
        print(f"Seed {seed} → Win Rate: {wr:.1f}% | Avg Reward: {ar:.2f}")

    print(f"\nOverall Mean Win Rate: {np.mean(win_rates):.1f}%")
    print(f"Win Rate Std Dev: {np.std(win_rates):.2f}% (Lower = More Robust)")


if __name__ == "__main__":
    run_robustness_evaluation()