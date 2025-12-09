from rich.traceback import install
install(show_locals=True)

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
from bluesky_gym import register_envs
from run_config import RUN_ID

from ray.rllib.models import ModelCatalog
from attention_model import AttentionSACModel

# Register your custom environment with Gymnasium
register_envs()
ModelCatalog.register_custom_model("attention_sac", AttentionSACModel)

# Conversion factor from meters per second to knots
MpS2Kt = 1.94384
# Conversion factor from nautical miles to kilometers
NM2KM = 1.852

# --- Parameters for Evaluation ---
N_AGENTS = 25  # The number of agents the model was trained with (MUST match training!)
NUM_EVAL_EPISODES = 1  # How many episodes to run for evaluation
RENDER = True # Set to True to watch the agent play (keep False for faster evaluation)

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
        
        # Create figure once per episode if rendering
        if RENDER:
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
            
            # After compute_actions, search for where attention weights were stored
            attention_model = None
            
            # Check all possible locations in the model hierarchy
            for search_obj in [policy.model, policy]:
                for attr_name in dir(search_obj):
                    if attr_name.startswith('_'):
                        continue
                    try:
                        attr = getattr(search_obj, attr_name, None)
                        if attr is not None and hasattr(attr, '_last_attn_weights'):
                            attention_model = attr
                            if episode_steps == 0:
                                print(f"[DEBUG] Found _last_attn_weights at: {search_obj.__class__.__name__}.{attr_name}")
                                print(f"[DEBUG] Model type: {type(attention_model)}")
                            break
                    except:
                        continue
                if attention_model:
                    break
            
            # Visualize attention weights if rendering and available
            if RENDER:
                if attention_model and hasattr(attention_model, '_last_attn_weights'):
                    attn_weights = attention_model._last_attn_weights
                    
                    # Debug: Print info on first step
                    if episode_steps == 0:
                        num_neighbors = attn_weights.shape[2]
                        print(f"[INFO] Model tracks {num_neighbors} closest neighbors (not all {len(agent_ids)-1} agents)")
                        print(f"[INFO] Attention weights shape: {attn_weights.shape}")
                        print(f"[INFO] Number of agents: {len(agent_ids)}")
                    
                    num_neighbors = attn_weights.shape[2]
                    
                    # Plot attention weights for ALL agents
                    for agent_idx in range(min(len(agent_ids), len(axes))):
                        ax = axes[agent_idx]
                        ax.clear()
                        
                        agent_attn = attn_weights[agent_idx, 0, :]  # Shape: (Num_Neighbors,)
                        
                        # Create a bar chart for this agent
                        indices = range(num_neighbors)
                        bars = ax.bar(indices, agent_attn, color='steelblue', edgecolor='black', width=0.7)
                        
                        ax.set_ylim(0, 1.0)
                        ax.set_xlim(-0.5, num_neighbors - 0.5)
                        ax.set_xticks(range(num_neighbors))
                        ax.set_ylabel('Weight', fontsize=7)
                        ax.set_xlabel('Neighbor', fontsize=7)
                        ax.set_title(f'{agent_ids[agent_idx]}', fontsize=8, fontweight='bold')
                        ax.grid(axis='y', alpha=0.2, linestyle='--')
                        ax.tick_params(labelsize=6)
                        
                        # Add value labels for high attention weights
                        for idx, val in enumerate(agent_attn):
                            if val > 0.15:  # Only show significant values
                                ax.text(idx, val + 0.03, f'{val:.2f}', 
                                       ha='center', va='bottom', fontsize=6)
                    
                    # Add step counter at the top
                    fig.text(0.5, 0.98, f'Step: {episode_steps} | Tracking Top {num_neighbors} Neighbors', 
                            ha='center', va='top', fontsize=12, fontweight='bold')
                    
                    plt.tight_layout(rect=[0, 0, 1, 0.97])
                    plt.draw()
                    plt.pause(0.01) # Short pause to update plot
                
            # Optional: Print top attended neighbor index
            # if hasattr(policy.model, '_last_attn_weights'):
            #     attn_weights = policy.model._last_attn_weights
            #     agent_attn = attn_weights[0, 0, :]
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
        
        # Close the figure after episode ends
        if RENDER:
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
