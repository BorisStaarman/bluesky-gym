# standard imports
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import bluesky as bs

# MARL ray imports
import ray
from ray.rllib.algorithms.algorithm import Algorithm

# Make sure these imports point to your custom environment files
from bluesky_gym.envs.ma_env_SAC import SectorEnv  # Use SAC version
from bluesky_gym import register_envs
from run_config import RUN_ID


# Register your custom environment with Gymnasium
register_envs()


# Conversion factor from meters per second to knots
MpS2Kt = 1.94384

# --- Parameters for Evaluation ---
N_AGENTS = 6  # The number of agents the model was trained with
NUM_EVAL_EPISODES = 10  # How many episodes to run for evaluation
RENDER = True # Set to True to watch the agent play
# NUM_EVAL_EPISODES = 20  # How many episodes to run for evaluation
# RENDER = True # Set to True to watch the agent play

# This path MUST match the checkpoint directory from your main.py training script
script_dir = os.path.dirname(os.path.abspath(__file__))
BASE_CHECKPOINT_DIR = os.path.join(script_dir, "models/sectorcr_ma_sac")
METRICS_DIR = os.path.join(script_dir, "metrics")

# --- CHOOSE WHICH CHECKPOINT TO EVALUATE ---
USE_BEST_CHECKPOINT = True  # Set to True to use best checkpoint, False for final checkpoint

def find_best_checkpoint(base_dir):
    """Find the best checkpoint (best_iter_XXXXX) in the checkpoint directory."""
    if not os.path.exists(base_dir):
        return None
    
    # Look for best_iter_* subdirectories
    best_checkpoints = [
        d for d in os.listdir(base_dir) 
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("best_iter_")
    ]
    
    if not best_checkpoints:
        return None
    
    # Sort by iteration number (extract from best_iter_XXXXX)
    best_checkpoints.sort(key=lambda x: int(x.split("_")[-1]), reverse=True)
    
    # Return the most recent best checkpoint
    return os.path.join(base_dir, best_checkpoints[0])

# Determine which checkpoint to use
if USE_BEST_CHECKPOINT:
    best_checkpoint = find_best_checkpoint(BASE_CHECKPOINT_DIR)
    if best_checkpoint:
        CHECKPOINT_DIR = best_checkpoint
        print(f"🌟 Using BEST checkpoint: {os.path.basename(CHECKPOINT_DIR)}")
    else:
        CHECKPOINT_DIR = BASE_CHECKPOINT_DIR
        print(f"⚠️  No best checkpoint found, using final checkpoint")
else:
    CHECKPOINT_DIR = BASE_CHECKPOINT_DIR
    print(f"📁 Using FINAL checkpoint")


if __name__ == "__main__":
    # Initialize Ray
    os.environ["TENSORBOARD"] = "0"   # ✅ Prevent auto-launch
    ray.shutdown()
    ray.init(include_dashboard=False)

    # --- Check if a checkpoint exists ---
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"❌ Checkpoint directory not found at: {CHECKPOINT_DIR}")
        print("Please run the `main.py` script first to train and save a model.")
        ray.shutdown()
        exit()

    print(f"\n🎯 Evaluating policy from checkpoint:")
    print(f"   {CHECKPOINT_DIR}\n")

    # --- Load the trained algorithm and policy ---
    algo = Algorithm.from_checkpoint(CHECKPOINT_DIR)
    
    # OLD API: Get policy from workers, not module
    # module = algo.get_module("shared_policy")  # This is NEW API only
    policy = algo.get_policy("shared_policy")
    
    env = SectorEnv(
        render_mode="human" if RENDER else None, 
        n_agents=N_AGENTS,
        run_id=RUN_ID,
        metrics_base_dir=METRICS_DIR
    )

    # --- Lists to store metrics from the evaluation run ---
    episode_rewards = []
    episode_steps_list = []
    episode_intrusions = []
    total_waypoints_reached = 0
    episode_witout_intrusion = 0
    velocity_agent_1 = []

    # --- Main Evaluation Loop ---
    for episode in range(1, NUM_EVAL_EPISODES + 1):
        print(f"\n--- Starting Evaluation Episode {episode}/{NUM_EVAL_EPISODES} ---")

        obs, info = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        
        # Run the episode until it's done
        while env.agents:
            # OLD API: Use policy.compute_actions instead of module.forward_inference
            agent_ids = list(obs.keys())
            obs_array = np.stack(list(obs.values()))
            
            # Compute deterministic actions (no exploration)
            actions_np = policy.compute_actions(obs_array, explore=False)[0]
            
            # Map actions back to agent IDs
            actions = {agent_id: action for agent_id, action in zip(agent_ids, actions_np)}

            # Step the environment
            obs, rewards, terminateds, truncateds, infos = env.step(actions)
            
            ac_idx = bs.traf.id2idx("KL001")
            airspeed_kts = bs.traf.tas[ac_idx] * MpS2Kt
            velocity_agent_1.append(airspeed_kts)
            # print(airspeed_kts)
            
            if rewards:
                episode_reward += sum(rewards.values())
            episode_steps += 1
            
            
            
            # Slow down rendering to make it watchable
            if RENDER:
                time.sleep(0.1)
                
        if env.total_intrusions == 0:
                episode_witout_intrusion +=1
        # After the episode is finished, collect and store the final stats
        print(f"-> Episode finished in {episode_steps} steps.")
        print(f"   - Total Reward: {episode_reward:.3f}")
        print(f"   - Intrusions: {env.total_intrusions}")
        print(f"   - Waypoints Reached: {len(env.waypoint_reached_agents)}/{N_AGENTS}")
        

        episode_rewards.append(episode_reward)
        episode_steps_list.append(episode_steps)
        episode_intrusions.append(env.total_intrusions)
        total_waypoints_reached += len(env.waypoint_reached_agents)

    # --- Print Final Summary Statistics ---
    max_intrusions = max(episode_intrusions)
    max_intrusion_episode = episode_intrusions.index(max_intrusions) + 1  # +1 because episodes start at 1
    
    print("\n" + "="*50)
    print("✅ EVALUATION COMPLETE")
    print(f"Ran {NUM_EVAL_EPISODES} episodes.")
    print(f"  - Average Reward: {np.mean(episode_rewards):.3f}")
    print(f"  - Average Episode Length: {np.mean(episode_steps_list):.1f} steps")
    print(f"  - Average Intrusions per Episode: {np.mean(episode_intrusions):.2f}")
    print(f"  - Maximum Intrusions: {max_intrusions} (occurred in Episode {max_intrusion_episode})")
    
    waypoint_rate = (total_waypoints_reached / (NUM_EVAL_EPISODES * N_AGENTS)) * 100
    print(f"  - Overall Waypoint Reached Rate: {waypoint_rate:.1f}%")
    print(f"   - episode without Intrusion: {episode_witout_intrusion}")
    print("="*50 + "\n")

    # # --- Plot the results ---
    # plt.figure(figsize=(12, 6))
    # plt.plot(range(1, NUM_EVAL_EPISODES + 1), episode_rewards, marker='o', linestyle='-')
    # plt.title("Total Reward per Evaluation Episode")
    # plt.xlabel("Episode Number")
    # plt.ylabel("Total Reward")
    # plt.xticks(range(1, NUM_EVAL_EPISODES + 1))
    # plt.grid(True)
    # plt.show()
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(velocity_agent_1) + 1), velocity_agent_1, marker='o', linestyle='-')
    plt.title("Velocity of Agent KL001 During Evaluation Episodes")
    plt.xlabel("Time Step")
    plt.ylabel("Velocity (knots)")
    plt.grid(True)
    plt.show()


    # --- Clean up ---
    env.close()
    ray.shutdown()
