from __future__ import annotations

import time
from pathlib import Path
from sb3_contrib import MaskablePPO
from single_agent_wrapper import SingleAgentWrapper

MODEL_FILE = Path("models/ppo_chefhats_masked")
SEAT = 0
SEED_BASE = 42

def locate_model(path: Path) -> Path:
    if Path(str(path) + ".zip").exists():
        return path
    checkpoints = sorted(path.parent.glob(f"{path.name}_*_steps.zip"))
    if checkpoints:
        return checkpoints[-1].with_suffix("")
    raise FileNotFoundError("No trained model found.")

def run_demo():
    print("=========================================")
    print("👨‍🍳 Starting Chef's Hat Demonstration 👨‍🍳")
    print("=========================================")
    
    env = SingleAgentWrapper("chefshat-v1", SEAT, SEED_BASE)
    
    model_path = locate_model(MODEL_FILE)
    model = MaskablePPO.load(str(model_path), env=env)
    print(f"[INFO] Loaded trained MaskablePPO model from: {model_path}\n")

    obs, info = env.reset(seed=SEED_BASE)
    done = truncated = False
    turn_counter = 1
    
    time.sleep(2) # Pause before starting the match

    while not (done or truncated):
        print(f"--- Turn {turn_counter} ---")
        
        # Agent predicts the next action
        action, _ = model.predict(
            obs,
            deterministic=True,
            action_masks=env.action_masks()
        )
        
        # Step the environment
        obs, reward, done, truncated, info = env.step(int(action))
        
        print(f"🤖 RL Agent (Seat 0) played Action: {action}")
        print(f"💰 Reward received: {reward:.3f}\n")
        
        turn_counter += 1
        time.sleep(0.8) # Slows down the output so you can talk over it in the video

    # Check the final score
    print("=========================================")
    print("🏁 MATCH FINISHED 🏁")
    scores = info.get("Match_Score", [])
    if scores:
        print(f"Final Scores: {scores}")
        if int(scores[SEAT]) == 3:
            print("Result: The RL Agent WON the match! 🏆")
        else:
            print("Result: The RL Agent lost the match. 😔")
    print("=========================================")

    env.close()

if __name__ == "__main__":
    run_demo()