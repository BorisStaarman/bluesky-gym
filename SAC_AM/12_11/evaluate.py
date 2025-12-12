# from rich.traceback import install
# install(show_locals=True)

# standard imports
import os
import sys
import numpy as np
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
from bluesky_gym.envs.ma_env_SAC_AM import SectorEnv
from run_config import RUN_ID
from ray.tune.registry import register_env

from ray.rllib.models import ModelCatalog
from attention_model_A import AttentionSACModel
# from attention_model_M import AttentionSACModel

# Conversion factor from meters per second to knots
MpS2Kt = 1.94384    
# Conversion factor from nautical miles to kilometers
NM2KM = 1.852

# --- Parameters for Evaluation ---
N_AGENTS = 20  # The number of agents the model was trained with (MUST match training!)
NUM_EVAL_EPISODES = 5 # How many episodes to run for evaluation
RENDER = True # Set to True to watch the agent play (keep False for faster evaluation)

# --- Visualization Settings ---
SHOW_ALPHA_VALUES = True  # Set to False to hide attention weight visualization (faster rendering)

# This path MUST match the checkpoint directory from your main.py training script
# (script_dir already defined above)
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
        print(f"[BEST] Using checkpoint: {os.path.basename(CHECKPOINT_DIR)}")
    else:
        CHECKPOINT_DIR = BASE_CHECKPOINT_DIR
        print(f"[WARN] No best checkpoint found, using final checkpoint")
else:
    CHECKPOINT_DIR = BASE_CHECKPOINT_DIR
    print(f"[FINAL] Using checkpoint")


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
        print(f"❌ Checkpoint directory not found at: {CHECKPOINT_DIR}")
        print("Please run the `main.py` script first to train and save a model.")
        ray.shutdown()
        exit()

    print(f"\n[EVAL] Evaluating policy from checkpoint:")
    print(f"   {CHECKPOINT_DIR}\n")

    # --- Load the trained algorithm and policy ---
    algo = Algorithm.from_checkpoint(CHECKPOINT_DIR)
    
    # OLD API: Get policy from workers, not module
    # module = algo.get_module("shared_policy")  # This is NEW API only
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
        
        # Create figure once per episode if rendering AND showing alpha values
        if RENDER and SHOW_ALPHA_VALUES:
            plt.ion()
            # Create a grid of subplots for all agents (5x5 grid for 25 agents)
            fig, axes = plt.subplots(5, 5, figsize=(20, 16))
            axes = axes.flatten()  # Flatten to 1D array for easy indexing
            fig.suptitle('Attention Weights for All Agents', fontsize=16, fontweight='bold')
        
        # Run the episode until it's done
        while env.agents:
            agent_ids = list(obs.keys())
            obs_array = np.stack(list(obs.values()))
            
            # Compute actions - this will internally call the attention model forward
            actions_np = policy.compute_actions(obs_array, explore=True)[0]
            
            # After compute_actions, get attention weights from the model
            attention_model = None
            attn_weights = None
            
            # Try different model structures to find attention weights
            if hasattr(policy, 'model'):
                if hasattr(policy.model, '_last_attn_weights'):
                    attention_model = policy.model
                    attn_weights = policy.model._last_attn_weights
                elif hasattr(policy.model, 'action_model') and hasattr(policy.model.action_model, '_last_attn_weights'):
                    attention_model = policy.model.action_model
                    attn_weights = policy.model.action_model._last_attn_weights
            
            # Debug: Print model structure on first step
            if episode_steps == 0 and RENDER and SHOW_ALPHA_VALUES:
                print(f"\n[DEBUG] Policy model structure:")
                print(f"  hasattr(policy, 'model'): {hasattr(policy, 'model')}")
                if hasattr(policy, 'model'):
                    print(f"  hasattr(policy.model, '_last_attn_weights'): {hasattr(policy.model, '_last_attn_weights')}")
                    print(f"  hasattr(policy.model, 'action_model'): {hasattr(policy.model, 'action_model')}")
                    if hasattr(policy.model, 'action_model'):
                        print(f"  hasattr(policy.model.action_model, '_last_attn_weights'): {hasattr(policy.model.action_model, '_last_attn_weights')}")
                print(f"  Attention weights found: {attn_weights is not None}")
                if attn_weights is not None:
                    print(f"  Attention weights shape: {attn_weights.shape}")
            
            # Debug: Check attention weights every step
            if RENDER and SHOW_ALPHA_VALUES:
                print(f"\n[DEBUG Step {episode_steps}] Attention weights check:")
                print(f"  attn_weights is None: {attn_weights is None}")
                print(f"  env.agents: {env.agents}")
                print(f"  agent_ids: {agent_ids}")
            
            # Store attention weights in environment for visualization and plot for first agent
            if RENDER and SHOW_ALPHA_VALUES:
                if attn_weights is not None and env.agents:
                    print(f"[DEBUG Step {episode_steps}] INSIDE visualization block")
                    print(f"  attn_weights shape: {attn_weights.shape}")
                    print(f"  attn_weights min/max/mean: {attn_weights.min():.4f} / {attn_weights.max():.4f} / {attn_weights.mean():.4f}")
                    
                    # Track the first agent consistently (the one shown in GREEN)
                    green_agent = env.agents[0]  # First active agent (shown in green)
                    print(f"  Green agent: {green_agent}")
                    
                    # Map attention weights for ALL agents (not just green agent) for rendering
                    env.attention_weights = {}  # Clear previous weights
                    
                    # Store weights for all agents in the batch
                    for batch_idx, agent_id in enumerate(agent_ids):
                        if batch_idx < len(attn_weights):
                            agent_neighbors = env.neighbor_mapping.get(agent_id, [])
                            agent_attn_full = attn_weights[batch_idx, 0, :]  # Shape: (Num_Neighbors,)
                            
                            # Map each neighbor's attention weight
                            for neigh_idx, neighbor_id in enumerate(agent_neighbors):
                                if neigh_idx < len(agent_attn_full):
                                    # Store with key as neighbor_id so render can display it
                                    env.attention_weights[neighbor_id] = agent_attn_full[neigh_idx]
                    
                    # Find green agent's position in current observation batch for plotting
                    if green_agent in agent_ids:
                        print(f"[DEBUG Step {episode_steps}] Green agent FOUND in agent_ids")
                        target_idx = agent_ids.index(green_agent)
                        target_agent = green_agent
                        print(f"  target_idx: {target_idx}, target_agent: {target_agent}")
                        
                        agent_attn = attn_weights[target_idx, 0, :]  # Shape: (Num_Neighbors,)
                        print(f"  agent_attn shape: {agent_attn.shape}")
                        print(f"  agent_attn values: {agent_attn}")
                        
                        # Get actual neighbor IDs and attention weights from environment
                        neighbor_ids = env.neighbor_mapping.get(target_agent, [])
                        num_actual_neighbors = len(neighbor_ids)
                        print(f"  neighbor_ids: {neighbor_ids}")
                        print(f"  num_actual_neighbors: {num_actual_neighbors}")
                        
                        # DEBUG: Print neighbor order on first step
                        if episode_steps == 1:
                            print(f"\n[DEBUG] Step {episode_steps} - Agent {target_agent} observing neighbors:")
                            print(f"  Neighbor order (x-axis): {neighbor_ids}")
                            print(f"  All active agents: {env.agents}")
                            print(f"  Observing agent index in list: {env.agents.index(target_agent) if target_agent in env.agents else 'N/A'}")
                        
                        # Only use attention weights for actual neighbors (trim padding)
                        agent_attn_active = agent_attn[:num_actual_neighbors]
                        print(f"  agent_attn_active: {agent_attn_active}")
                        print(f"[DEBUG Step {episode_steps}] About to plot...")
                        
                        # Plot with agent IDs on x-axis
                        plt.clf()
                        fig.suptitle(f'Attention Weights for {target_agent} | Step {episode_steps} | Active Neighbors: {num_actual_neighbors}', fontsize=16, fontweight='bold')
                        ax = plt.gca()
                        
                        # Create x positions and labels with actual agent IDs
                        x_positions = range(num_actual_neighbors)
                        x_labels = neighbor_ids  # Use actual agent IDs as labels
                        
                        # Create bar plot with agent IDs
                        bars = ax.bar(x_positions, agent_attn_active, color='steelblue', edgecolor='black', width=0.7)
                        
                        ax.set_ylim(0, 1.0)
                        ax.set_xlim(-0.5, num_actual_neighbors - 0.5)
                        ax.set_xticks(x_positions)
                        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)  # Rotate labels for readability
                        ax.set_ylabel('Attention Weight (α)', fontsize=12, fontweight='bold')
                        ax.set_xlabel('Agent ID', fontsize=12, fontweight='bold')
                        ax.set_title(f'Observing Agent: {target_agent}', fontsize=14, fontweight='bold')
                        ax.grid(axis='y', alpha=0.3, linestyle='--')
                        ax.tick_params(axis='y', labelsize=10)
                        
                        # Add value labels for high attention weights
                        for idx, (neighbor_id, val) in enumerate(zip(neighbor_ids, agent_attn_active)):
                            if val > 0.05:
                                ax.text(idx, val + 0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
                        
                        # Highlight top 3 attended neighbors with different colors
                        if num_actual_neighbors > 0:
                            top_3_indices = np.argsort(agent_attn_active)[-3:][::-1]  # Get indices of top 3
                            colors = ['#d62728', '#ff7f0e', '#2ca02c']  # Red, Orange, Green
                            for rank, idx in enumerate(top_3_indices):
                                if idx < len(bars) and agent_attn_active[idx] > 0.01:  # Only highlight if significant
                                    bars[idx].set_color(colors[rank])
                                    bars[idx].set_edgecolor('black')
                                    bars[idx].set_linewidth(2)
                        
                            plt.tight_layout(rect=[0, 0, 1, 0.97])
                            plt.draw()
                            plt.pause(0.01)
                            print(f"[DEBUG Step {episode_steps}] Plot updated successfully!")
                    else:
                        print(f"[DEBUG Step {episode_steps}] Green agent NOT in agent_ids")
            elif RENDER and not SHOW_ALPHA_VALUES:
                # Clear attention weights so they don't display on aircraft
                env.attention_weights = {}
                
            # Optional: Print top attended neighbor index
            # if hasattr(policy.model, '_last_attn_weights'):
            #     attn_weights = policy.model._last_attn_weights
            #     agent_attn = attn_weights[0, 0, :]s
            #     max_attn_idx = np.argmax(agent_attn)
            #     print(f"Agent {agent_ids[0]} focuses most on Neighbor #{max_attn_idx} (Val: {agent_attn[max_attn_idx]:.2f})")

            
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
    
    # Calculate polygon area statistics
    # avg_polygon_area = np.mean(polygon_areas_km2)
    # min_polygon_area = np.min(polygon_areas_km2)
    # max_polygon_area = np.max(polygon_areas_km2)
    # std_polygon_area = np.std(polygon_areas_km2)
    
    print("\n" + "="*50)
    print("[OK] EVALUATION COMPLETE")
    print(f"Ran {NUM_EVAL_EPISODES} episodes.")
    print(f"  - Average Reward: {np.mean(episode_rewards):.3f}")
    print(f"  - Average Episode Length: {np.mean(episode_steps_list):.1f} steps")
    print(f"  - Average Intrusions per Episode: {np.mean(episode_intrusions):.2f}")
    print(f"  - Maximum Intrusions: {max_intrusions} (occurred in Episode {max_intrusion_episode})")
    
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
