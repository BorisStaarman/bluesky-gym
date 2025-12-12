# standard imports
import os
import numpy as np
import sys
import torch
import matplotlib.pyplot as plt
import time
import bluesky as bs

# Add current directory to Python path so Ray workers can find attention_model
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
    
# MARL ray imports
import ray
from ray.rllib.algorithms.algorithm import Algorithm

# Make sure these imports point to your custom environment files
from bluesky_gym.envs.ma_env_SAC_AM import SectorEnv  # Use SAC version
from run_config import RUN_ID
from ray.tune.registry import register_env

from ray.rllib.models import ModelCatalog
from attention_model_A import AttentionSACModel


# Conversion factor from meters per second to knots
MpS2Kt = 1.94384
# Conversion factor from nautical miles to kilometers
NM2KM = 1.852

def calculate_polygon_area_km2(poly_points):
    """
    Calculate the area of a polygon in km² given vertices in nautical miles.
    
    Parameters
    ----------
    poly_points : np.array
        Array of polygon vertices in nautical miles (x, y coordinates)
    
    Returns
    -------
    float
        Area of the polygon in square kilometers
    """
    from bluesky_gym.envs.common import functions as fn
    
    # Calculate area in NM² using the Shoelace formula
    area_nm2 = fn.polygon_area(poly_points)
    
    # Convert from NM² to km²
    area_km2 = area_nm2 * (NM2KM ** 2)
    
    return area_km2

# --- Parameters for Evaluation ---
N_AGENTS = 20  # The number of agents the model was trained with
# NUM_EVAL_EPISODES = 100  # How many episodes to run for evaluation
# RENDER = False # Set to True to watch the agent play
NUM_EVAL_EPISODES = 1  # How many episodes to run for evaluation
RENDER = True # Set to True to watch the agent play

# --- Visualization Settings ---
SHOW_ALPHA_VALUES = True  # Set to False to hide attention weight visualization (faster rendering)

# This path MUST match the checkpoint directory from your main.py training script
script_dir = os.path.dirname(os.path.abspath(__file__))
BASE_CHECKPOINT_DIR = os.path.join(script_dir, "models/sectorcr_ma_sac")
METRICS_DIR = os.path.join(script_dir, "metrics")

# --- ALWAYS USE FINAL MODEL ---
CHECKPOINT_DIR = os.path.join(BASE_CHECKPOINT_DIR, "final_model")
print(f"📁 Using FINAL MODEL checkpoint")


if __name__ == "__main__":
    # Initialize Ray with runtime environment so workers can find attention_model
    os.environ["TENSORBOARD"] = "0"   # ✅ Prevent auto-launch
    ray.shutdown()
    ray.init(
        include_dashboard=False,
        runtime_env={
            "env_vars": {"PYTHONPATH": script_dir},  # Add script directory to PYTHONPATH for all workers
            "excludes": [
                "models/",       # Exclude trained model checkpoints
                "metrics/",      # Exclude metrics data
                "*.pkl",         # Exclude pickle files
                "__pycache__/",  # Exclude Python cache
            ]
        }
    )
    
    # Register environment and model AFTER ray.init so workers can access them
    register_env("sector_env", lambda config: SectorEnv(**config))
    ModelCatalog.register_custom_model("attention_sac", AttentionSACModel)

    # --- Check if a checkpoint exists ---
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"❌ Final model checkpoint not found at: {CHECKPOINT_DIR}")
        print("Please run the `main.py` script first with SAVE_FINAL_MODEL=True to train and save the final model.")
        ray.shutdown()
        exit()

    print(f"\n🎯 Evaluating FINAL MODEL from checkpoint:")
    print(f"   {CHECKPOINT_DIR}\n")

    # --- Load the trained algorithm and policy ---
    algo = Algorithm.from_checkpoint(CHECKPOINT_DIR)
    
    # OLD API: Get policy from workers, not module
    policy = algo.get_policy("shared_policy")
    
    # Set model to eval mode
    if hasattr(policy, 'model'):
        policy.model.eval()
        print(f"[OK] Model set to eval mode")
    
    # DEBUG: Check what the model was initialized with
    if hasattr(policy.model, 'action_model'):
        actor_model = policy.model.action_model
        print(f"\n[DEBUG] Model Architecture Info (from loaded checkpoint):")
        print(f"  - ownship_dim: {actor_model.ownship_dim}")
        print(f"  - intruder_dim: {actor_model.intruder_dim}")
        print(f"  - num_intruders: {actor_model.num_intruders}")
        print(f"  - expected_intruder_size: {actor_model.expected_intruder_size}")
        print(f"  - Total obs space model expects: {actor_model.ownship_dim + actor_model.expected_intruder_size}")
        
        # Calculate what this means
        expected_neighbors = actor_model.num_intruders
        print(f"\n[ANALYSIS]:")
        print(f"  - This model was trained to track {expected_neighbors} neighbors")
        print(f"  - Current environment NUM_AC_STATE=24 (should create obs size {7 + 7*24} = 171)")
        if expected_neighbors != 24:
            print(f"  ⚠️  MISMATCH! Model expects {expected_neighbors} but env provides 24!")
            print(f"  ⚠️  This model was trained with NUM_AC_STATE={expected_neighbors}, not 24")
        print()
    
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
        
        # Create figure once per episode if rendering AND showing alpha values
        if RENDER and SHOW_ALPHA_VALUES:
            plt.ion()
            # Create a grid of subplots for all agents (5x5 grid for 25 agents)
            fig, axes = plt.subplots(5, 5, figsize=(20, 16))
            axes = axes.flatten()  # Flatten to 1D array for easy indexing
            fig.suptitle('Attention Weights for All Agents', fontsize=16, fontweight='bold')
        
        # Run the episode until it's done
        while env.agents:
            # OLD API: Use policy.compute_actions instead of module.forward_inference
            agent_ids = list(obs.keys())
            obs_array = np.stack(list(obs.values()))
            
            # Compute deterministic actions (no exploration)
            actions_np = policy.compute_actions(obs_array, explore=False)[0]
            
            # After compute_actions, get attention weights from the action_model
            attention_model = None
            if hasattr(policy.model, 'action_model') and hasattr(policy.model.action_model, '_last_attn_weights'):
                attention_model = policy.model.action_model
            
            # Store attention weights in environment for visualization and plot for first agent
            if RENDER and SHOW_ALPHA_VALUES:
                if attention_model and hasattr(attention_model, '_last_attn_weights'):
                    attn_weights = attention_model._last_attn_weights
                    
                    # Only plot for the first agent in the list
                    target_idx = 0
                    target_agent = agent_ids[target_idx]  # KL001
                    agent_attn = attn_weights[target_idx, 0, :]  # Shape: (Num_Neighbors,)
                    num_neighbors = len(agent_attn)
                    
                    # Map attention weights to actual neighbor agents using the neighbor mapping from env
                    env.attention_weights = {}  # Clear previous weights
                    if target_agent in env.neighbor_mapping:
                        neighbor_ids = env.neighbor_mapping[target_agent]
                        for idx, neighbor_id in enumerate(neighbor_ids):
                            if idx < len(agent_attn):
                                env.attention_weights[neighbor_id] = agent_attn[idx]
                    
                    if episode_steps == 0:
                        print(f"\n[INFO] Observation and Attention Info:")
                        print(f"  - Plotting attention weights for agent: {agent_ids[target_idx]}")
                        print(f"  - Actual observation shape from env: {obs_array.shape}")
                        print(f"  - Attention weights shape: {attn_weights.shape}")
                        print(f"  - Number of neighbors: {num_neighbors}")
                        print()
                    plt.clf()
                    fig.suptitle(f'Attention Weights for {agent_ids[target_idx]} | Step {episode_steps}', fontsize=16, fontweight='bold')
                    ax = plt.gca()
                    indices = range(num_neighbors)
                    bars = ax.bar(indices, agent_attn, color='steelblue', edgecolor='black', width=0.7)
                    ax.set_ylim(0, 1.0)
                    ax.set_xlim(-0.5, num_neighbors - 0.5)
                    ax.set_xticks(range(num_neighbors))
                    ax.set_ylabel('Attention Weight', fontsize=12)
                    ax.set_xlabel('Neighbor Index (Sorted by Distance)', fontsize=12)
                    ax.set_title(f'Agent: {agent_ids[target_idx]}', fontsize=14, fontweight='bold')
                    ax.grid(axis='y', alpha=0.3, linestyle='--')
                    ax.tick_params(labelsize=10)
                    # Add value labels for high attention weights
                    for idx, val in enumerate(agent_attn):
                        if val > 0.05:
                            ax.text(idx, val + 0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=10)
                    plt.tight_layout(rect=[0, 0, 1, 0.97])
                    plt.draw()
                    plt.pause(0.01)
            elif RENDER and not SHOW_ALPHA_VALUES:
                # Clear attention weights so they don't display on aircraft
                env.attention_weights = {}
            
            # Map actions back to agent IDs
            actions = {agent_id: action for agent_id, action in zip(agent_ids, actions_np)}

            # Step the environment
            obs, rewards, terminateds, truncateds, infos = env.step(actions)
            
            ac_idx = bs.traf.id2idx("KL001")
            airspeed_kts = bs.traf.tas[ac_idx] * MpS2Kt
            velocity_agent_1.append(airspeed_kts)
            
            if rewards:
                episode_reward += sum(rewards.values())
            episode_steps += 1
            
            # Slow down rendering to make it watchable
            if RENDER:
                time.sleep(0.1)
        
        # Close the figure after episode ends (only if we created one)
        if RENDER and SHOW_ALPHA_VALUES:
            plt.close(fig)
                
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
    print("✅ EVALUATION COMPLETE (FINAL MODEL)")
    print(f"Ran {NUM_EVAL_EPISODES} episodes.")
    print(f"  - Average Reward: {np.mean(episode_rewards):.3f}")
    print(f"  - Average Episode Length: {np.mean(episode_steps_list):.1f} steps")
    print(f"  - Average Intrusions per Episode: {np.mean(episode_intrusions):.2f}")
    print(f"  - Maximum Intrusions: {max_intrusions} (occurred in Episode {max_intrusion_episode})")
    
    waypoint_rate = (total_waypoints_reached / (NUM_EVAL_EPISODES * N_AGENTS)) * 100
    print(f"  - Overall Waypoint Reached Rate: {waypoint_rate:.1f}%")
    print(f"   - episode without Intrusion: {episode_witout_intrusion}")
    print("="*50 + "\n")

    # Plot velocity of agent 1
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(velocity_agent_1) + 1), velocity_agent_1, marker='o', linestyle='-')
    plt.title("Velocity of Agent KL001 During Evaluation Episodes (FINAL MODEL)")
    plt.xlabel("Time Step")
    plt.ylabel("Velocity (knots)")
    plt.grid(True)
    plt.show()


    # --- Clean up ---
    env.close()
    ray.shutdown()
