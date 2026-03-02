from __future__ import annotations

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sb3_contrib import MaskablePPO
from single_agent_wrapper import SingleAgentWrapper

# Configuration
MODELS_DIR = Path("models")
MODEL_PREFIX = "ppo_chefhats_masked"
EVAL_EPISODES = 20  # Keep this relatively low so the script doesn't take hours
SEAT = 0
SEED = 42

def extract_step_number(filepath: Path) -> int:
    """Extracts the step number from the checkpoint filename."""
    match = re.search(r"_(\d+)_steps", filepath.name)
    return int(match.group(1)) if match else 0

def generate_learning_curve():
    print("Finding checkpoints...")
    # Find all checkpoint files matching the prefix
    checkpoints = list(MODELS_DIR.glob(f"{MODEL_PREFIX}_*_steps.zip"))
    
    if not checkpoints:
        print("No checkpoints found! Make sure you have trained the model.")
        return

    # Sort checkpoints chronologically by step number
    checkpoints.sort(key=extract_step_number)
    
    env = SingleAgentWrapper("chefshat-v1", SEAT, SEED)
    
    steps = []
    win_rates = []
    avg_rewards = []

    print(f"Evaluating {len(checkpoints)} checkpoints...")
    
    for ckpt in checkpoints:
        step_num = extract_step_number(ckpt)
        print(f"Evaluating checkpoint at {step_num} steps...")
        
        # Load the specific checkpoint
        model = MaskablePPO.load(str(ckpt.with_suffix("")), env=env)
        
        wins = 0
        rewards = []
        
        for ep in range(EVAL_EPISODES):
            obs, info = env.reset(seed=SEED + ep)
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
                
            rewards.append(total_reward)
            
        # Calculate metrics
        win_rate = (wins / EVAL_EPISODES) * 100
        avg_reward = float(np.mean(rewards))
        
        steps.append(step_num)
        win_rates.append(win_rate)
        avg_rewards.append(avg_reward)
        
    env.close()

    # Plotting the results
    print("Generating plots...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot Win Rate
    ax1.plot(steps, win_rates, marker='o', color='b', linestyle='-')
    ax1.set_title('Agent Win Rate Over Training Steps')
    ax1.set_ylabel('Win Rate (%)')
    ax1.grid(True)

    # Plot Average Reward
    ax2.plot(steps, avg_rewards, marker='s', color='g', linestyle='-')
    ax2.set_title('Average Reward Over Training Steps')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Average Reward')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('learning_curve.png')
    print("Plot saved as 'learning_curve.png'")
    plt.show()

if __name__ == "__main__":
    generate_learning_curve()