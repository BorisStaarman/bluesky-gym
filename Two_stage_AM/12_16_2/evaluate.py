# standard imports
import os
import sys
import shutil
import csv
import matplotlib.pyplot as plt
import numpy as np
import time
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import io

# Add the script directory to Python path so Ray workers can find attention_model_A
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# MARL ray imports
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from attention_model_A import AttentionSACModel # additive method

from bluesky_gym.envs.ma_env_two_stage_AM import SectorEnv
from ray.tune.registry import register_env

import torch
import torch.nn.functional as F

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.sample_batch import SampleBatch


from run_config import RUN_ID

# Register your custom environment with Gymnasium
# Register your custom environment directly for RLlib
register_env("sector_env", lambda config: SectorEnv(**config))
ModelCatalog.register_custom_model("attention_sac", AttentionSACModel)

# Register your custom environment with Gymnasium

# Conversion factor from meters per second to knots
MpS2Kt = 1.94384
# Conversion factor from nautical miles to kilometers
NM2KM = 1.852

# --- Parameters for Evaluation ---
N_AGENTS = 20  # The number of agents the model was trained with
# NUM_EVAL_EPISODES = 100  # How many episodes to run for evaluation
# RENDER = False # Set to True to watch the agent play
NUM_EVAL_EPISODES = 10  # How many episodes to run for evaluation
RENDER = True # Set to True to watch the agent play

# This path MUST match the checkpoint directory from your main.py training script
script_dir = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.join(script_dir, "metrics")

# --- CHOOSE WHICH CHECKPOINT TO EVALUATE ---
# Set to True to use stage1_best_weights, False to use stage1_weights (last iteration)
USE_BEST_STAGE1_WEIGHTS = True

# --- CHOOSE WHICH CHECKPOINT TO EVALUATE ---
# Set to True to use stage1_best_weights, False to use stage1_weights (last iteration)
USE_BEST_STAGE1_WEIGHTS = True

# Determine which checkpoint to use based on the boolean
if USE_BEST_STAGE1_WEIGHTS:
    CHECKPOINT_DIR = os.path.join(script_dir, "models/sectorcr_ma_sac/stage1_best_weights")
    print(f"🌟 Using BEST Stage 1 weights: stage1_best_weights")
else:
    CHECKPOINT_DIR = os.path.join(script_dir, "models/sectorcr_ma_sac/stage1_weights")
    print(f"📁 Using LAST Stage 1 weights: stage1_weights")


if __name__ == "__main__":
    # Initialize Ray
    os.environ["TENSORBOARD"] = "0"   # ✅ Prevent auto-launch
    ray.shutdown()
    # ray.init(include_dashboard=False)
    ray.init(runtime_env={
        "working_dir": script_dir,
        "py_modules": [os.path.join(script_dir, "attention_model_A.py")],
    })

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
    episode_aircraft_with_intrusions = []  # Track number of unique aircraft with intrusions
    total_waypoints_reached = 0
    episode_witout_intrusion = 0
    # velocity_agent_1 = []
    # `polygon`_areas_km2 = []  # Store polygon area in km² for each episode

    # --- Main Evaluation Loop ---
    for episode in range(1, NUM_EVAL_EPISODES + 1):
        print(f"\n--- Starting Evaluation Episode {episode}/{NUM_EVAL_EPISODES} ---")

        obs, info = env.reset()
        
        # Calculate and store polygon area for this episode
        # polygon_area = calculate_polygon_area_km2(env.poly_points)
        # polygon_areas_km2.append(polygon_area)
        # print(f"   - Polygon Area: {polygon_area:.4f} km²")
        
        episode_reward = 0.0
        episode_steps = 0
        
        # Run the episode until it's done
        while env.agents:
            # OLD API: Use policy.compute_actions instead of module.forward_inference
            agent_ids = list(obs.keys())
            obs_array = np.stack(list(obs.values()))
            
            # Compute deterministic actions (no exploration)
            actions_np = policy.compute_actions(obs_array, explore=False)[0]
            # print(actions_np)
            
            
            # Map actions back to agent IDs
            actions = {agent_id: action for agent_id, action in zip(agent_ids, actions_np)}

            # Step the environment
            obs, rewards, terminateds, truncateds, infos = env.step(actions)
            
            # ac_idx = bs.traf.id2idx("KL001")
            # airspeed_kts = bs.traf.tas[ac_idx] * MpS2Kt
            # velocity_agent_1.append(airspeed_kts)
            # print(airspeed_kts)
            
            if rewards:
                episode_reward += sum(rewards.values())
            episode_steps += 1
            
            
            
            # Slow down rendering to make it watchable
            if RENDER:
                time.sleep(0.1)
                
        if env.total_intrusions == 0:
                episode_witout_intrusion +=1
        
        # Count number of aircraft that had at least one intrusion
        aircraft_with_intrusions = sum(1 for count in env._intrusions_acc.values() if count > 0)
        
        # After the episode is finished, collect and store the final stats
        print(f"-> Episode finished in {episode_steps} steps.")
        print(f"   - Total Reward: {episode_reward:.3f}")
        print(f"   - Intrusions: {env.total_intrusions}")
        print(f"   - Aircraft with Intrusions: {aircraft_with_intrusions}/{N_AGENTS}")
        print(f"   - Waypoints Reached: {len(env.waypoint_reached_agents)}/{N_AGENTS}")
        
        
        

        episode_rewards.append(episode_reward)
        episode_steps_list.append(episode_steps)
        episode_intrusions.append(env.total_intrusions)
        episode_aircraft_with_intrusions.append(aircraft_with_intrusions)
        total_waypoints_reached += len(env.waypoint_reached_agents)

    # --- Print Final Summary Statistics ---
    max_intrusions = max(episode_intrusions)
    max_intrusion_episode = episode_intrusions.index(max_intrusions) + 1  # +1 because episodes start at 1
    
    # Calculate polygon area statistics
    # avg_polygon_area = np.mean(polygon_areas_km2)
    # min_polygon_area = np.min(polygon_areas_km2)
    # max_polygon_area = np.max(polygon_areas_km2)
    # std_polygon_area = np.std(polygon_areas_km2)
    
    print("\n" + "="*50)
    print("✅ EVALUATION COMPLETE")
    print(f"Ran {NUM_EVAL_EPISODES} episodes.")
    print(f"  - Average Reward: {np.mean(episode_rewards):.3f}")
    print(f"  - Average Episode Length: {np.mean(episode_steps_list):.1f} steps")
    print(f"  - Average Intrusions per Episode: {np.mean(episode_intrusions):.2f}")
    print(f"  - Maximum Intrusions: {max_intrusions} (occurred in Episode {max_intrusion_episode})")
    print(f"  - Average Aircraft with Intrusions: {np.mean(episode_aircraft_with_intrusions):.2f} ({np.mean(episode_aircraft_with_intrusions)/N_AGENTS*100:.1f}%)")
    
    waypoint_rate = (total_waypoints_reached / (NUM_EVAL_EPISODES * N_AGENTS)) * 100
    print(f"  - Overall Waypoint Reached Rate: {waypoint_rate:.1f}%")
    print(f"  - Episodes without Intrusion: {episode_witout_intrusion}")
    
    print('average density created', N_AGENTS / np.mean(env.areas_km2))
    
    # print(f"\n📐 Polygon Area Statistics:")
    # print(f"  - Average Area: {avg_polygon_area:.4f} km²")
    # print(f"  - Min Area: {min_polygon_area:.4f} km²")
    # print(f"  - Max Area: {max_polygon_area:.4f} km²")
    # print(f"  - Std Dev: {std_polygon_area:.4f} km²")
    # print("="*50 + "\n")

    # # --- Plot the results ---
    # plt.figure(figsize=(12, 6))
    # plt.plot(range(1, NUM_EVAL_EPISODES + 1), episode_rewards, marker='o', linestyle='-')
    # plt.title("Total Reward per Evaluation Episode")
    # plt.xlabel("Episode Number")
    # plt.ylabel("Total Reward")
    # plt.xticks(range(1, NUM_EVAL_EPISODES + 1))
    # plt.grid(True)
    # plt.show()
    # plt.figure(figsize=(12, 6))
    # plt.plot(range(1, len(velocity_agent_1) + 1), velocity_agent_1, marker='o', linestyle='-')
    # plt.title("Velocity of Agent KL001 During Evaluation Episodes")
    # plt.xlabel("Time Step")
    # plt.ylabel("Velocity (knots)")
    # plt.grid(True)
    # plt.show()


    # --- Clean up ---
    env.close()
    ray.shutdown()
