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
import re

# Add current directory to Python path so Ray workers can find attention_model
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# MARL ray imports
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.sac import SACConfig
import json

# Make sure these imports point to your custom environment files
from bluesky_gym.envs.ma_env_SAC_AM import SectorEnv
from run_config import RUN_ID
from ray.tune.registry import register_env

from ray.rllib.models import ModelCatalog
from attention_model_A import AttentionSACModel
# from attention_model_M import AttentionSACModel

from attention_visualization import plot_attention_combined

# Conversion factor from meters per second to knots
MpS2Kt = 1.94384    
# Conversion factor from nautical miles to kilometers
NM2KM = 1.852

# --- Parameters for Evaluation ---
N_AGENTS = 20  # The number of agents the model was trained with (MUST match training!)
NUM_EVAL_EPISODES = 40  # 600 episodes: 300 probe + 300 reference for bootstrap convergence analysis
RENDER = False # Set to True to watch the agent play (keep False for faster evaluation)

# --- Visualization Settings ---
SHOW_ALPHA_VALUES = False  # Set to False to hide attention weight visualization (faster rendering)

# --- Attention snapshot settings ---
SNAPSHOT_STEP = 30          # save a static figure when episode_steps reaches this value
MAX_SNAPSHOTS = 5           # maximum figures to save across all episodes
FIGS_DIR = os.path.join(script_dir, "figures")

# This path MUST match the checkpoint directory from your main.py training script
# (script_dir already defined above)
BASE_CHECKPOINT_DIR = os.path.join(script_dir, "models/sectorcr_ma_sac")
METRICS_DIR = os.path.join(script_dir, "metrics")

# --- CHOOSE WHICH CHECKPOINT TO EVALUATE ---
# USE_BEST_CHECKPOINT = True  # Set to True to use best checkpoint, False for final checkpoint
MANUAL_CHECKPOINT_ITER = 23465 # Best checkpoint from 2_4_2 run (best_iter_19705_low_i)

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
    # Sort by iteration number (extract digits robustly, handle suffixes like '_low_i')
    def iter_from_name(name):
        m = re.search(r"best_iter_(\d+)", name)
        if m:
            return int(m.group(1))
        nums = re.findall(r"(\d+)", name)
        if nums:
            return int(nums[-1])
        return -1

    best_checkpoints.sort(key=iter_from_name, reverse=True)
    # Return the most recent best checkpoint directory
    return os.path.join(base_dir, best_checkpoints[0])

# Determine which checkpoint to use
# Check if manual override is set first
if 'MANUAL_CHECKPOINT_ITER' in globals():
    # User manually specified an iteration number
    manual_checkpoint = os.path.join(BASE_CHECKPOINT_DIR, f"best_iter_{MANUAL_CHECKPOINT_ITER:05d}")
    print('CHECKPOIN USED', manual_checkpoint)
    # Also check for _low_i variant
    if not os.path.exists(manual_checkpoint):
        manual_checkpoint_low_i = os.path.join(BASE_CHECKPOINT_DIR, f"best_iter_{MANUAL_CHECKPOINT_ITER:05d}_low_i")
        print('CHECKPOIN USED', manual_checkpoint_low_i)
        if os.path.exists(manual_checkpoint_low_i):
            manual_checkpoint = manual_checkpoint_low_i
    
    if os.path.exists(manual_checkpoint):
        CHECKPOINT_DIR = manual_checkpoint
        print(f"[MANUAL] Using checkpoint: {os.path.basename(CHECKPOINT_DIR)}")
    else:
        print(f"[ERROR] Manual checkpoint not found: best_iter_{MANUAL_CHECKPOINT_ITER:05d}")
        print(f"[ERROR] Falling back to automatic selection")
        CHECKPOINT_DIR = BASE_CHECKPOINT_DIR
elif USE_BEST_CHECKPOINT:
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
    
    # Get number of CPUs available
    import multiprocessing
    num_cpus = multiprocessing.cpu_count()
    print(f"[INFO] Detected {num_cpus} CPUs available")
    
    # Initialize Ray with limited resources for local evaluation
    ray.init(
        num_cpus=num_cpus,  # Use all available CPUs
        num_gpus=0,  # No GPUs
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
    # Ensure absolute path is used for metrics to avoid permission errors
    register_env("sector_env", lambda config: SectorEnv(**{**config, "metrics_base_dir": METRICS_DIR}))
    ModelCatalog.register_custom_model("attention_sac", AttentionSACModel)
    
    # print the model path for checking
    print(f"[INFO] Looking for checkpoints in: {CHECKPOINT_DIR}")
    print(f"[INFO] Looking for model in: {script_dir}")

    # --- Check if a checkpoint exists ---
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"❌ Checkpoint directory not found at: {CHECKPOINT_DIR}")
        print("Please run the `main.py` script first to train and save a model.")
        ray.shutdown()
        exit()

    print(f"\n[EVAL] Evaluating policy from checkpoint:")
    print(f"   {CHECKPOINT_DIR}\n")

    # --- Modify checkpoint config to reduce resource requirements ---
    # CHECKPOINT_DIR = 'C:\\Users\\boris\\Documents\\bsgym\\bluesky-gym\\SAC_AM_PreTrain\\2_3\\models\\sectorcr_ma_sac\\best_iter_15992_low_i'
    checkpoint_config_file = os.path.join(CHECKPOINT_DIR, "rllib_checkpoint.json")
    checkpoint_config_backup = os.path.join(CHECKPOINT_DIR, "rllib_checkpoint.json.backup")
    
    # Backup and modify the config file
    import shutil
    if not os.path.exists(checkpoint_config_backup):
        shutil.copy(checkpoint_config_file, checkpoint_config_backup)
        print("[INFO] Backed up original checkpoint config")
    
    # Read and modify config
    with open(checkpoint_config_file, 'r') as f:
        checkpoint_data = json.load(f)
    
    # Reduce workers to fit local machine
    if 'num_workers' in checkpoint_data:
        original_workers = checkpoint_data['num_workers']
        checkpoint_data['num_workers'] = 0  # No parallel workers
        print(f"[INFO] Reduced num_workers: {original_workers} -> 0")
    
    if 'num_envs_per_worker' in checkpoint_data:
        checkpoint_data['num_envs_per_worker'] = 1
    
    if 'num_gpus' in checkpoint_data:
        checkpoint_data['num_gpus'] = 0
        
    if 'evaluation_num_workers' in checkpoint_data:
        checkpoint_data['evaluation_num_workers'] = 0
    
    # Write modified config back
    with open(checkpoint_config_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    print("[INFO] Modified checkpoint config for local evaluation")
    print("[INFO] Loading checkpoint (this may take a moment)...")
    
    # Now load with modified config
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
        # Some older model variants exposed `expected_intruder_size`; compute if missing
        if hasattr(actor_model, 'expected_intruder_size'):
            expected_intruder_size = actor_model.expected_intruder_size
        else:
            expected_intruder_size = actor_model.num_intruders * actor_model.intruder_dim
        print(f"  - expected_intruder_size: {expected_intruder_size}")
        print(f"  - Total obs space model expects: {actor_model.ownship_dim + expected_intruder_size}")
        
        # Calculate what this means
        expected_neighbors = actor_model.num_intruders
        actual_neighbors = N_AGENTS - 1  # Environment uses n_agents-1 neighbors
        actual_obs_size = 7 + 5 * actual_neighbors
        print(f"\n[ANALYSIS]:")
        print(f"  - This model was trained to track {expected_neighbors} neighbors")
        print(f"  - Current environment: N_AGENTS={N_AGENTS} → {actual_neighbors} neighbors (obs size {actual_obs_size})")
        if expected_neighbors != actual_neighbors:
            print(f"  ⚠️  MISMATCH! Model expects {expected_neighbors} but env provides {actual_neighbors}!")
            print(f"  ⚠️  Update N_AGENTS in evaluate.py to {expected_neighbors + 1} to match training!")
        else:
            print(f"  ✅ MATCH! Model and environment both use {actual_neighbors} neighbors.")
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
    episode_aircraft_with_intrusions = []  # Track how many aircraft had intrusions per episode
    total_waypoints_reached = 0
    episode_witout_intrusion = 0
    
    # Track attention weight statistics across all episodes
    all_attention_weights = []  # Store all attention weight vectors for later analysis

    # velocity_agent_1 = []
    # `polygon`_areas_km2 = []  # Store polygon area in km² for each episode

    total_snapshots_saved = 0  # number of combined attention figures saved this run
    os.makedirs(FIGS_DIR, exist_ok=True)

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
        snapshot_saved = False   # reset: one snapshot per episode

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
            
                # Collect attention weights for statistics (regardless of rendering)
                if attn_weights is not None:
                    # Flatten and store all attention weights from this step
                    all_attention_weights.append(attn_weights.flatten())
                    
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

            # ── Unconditional attention statistics collection ──────────────────────
            if attn_weights is not None:
                all_attention_weights.append(attn_weights.flatten())

            # ── Thesis visualization snapshot ─────────────────────────────────────
            # Saved once per episode at SNAPSHOT_STEP; works regardless of RENDER.
            if (
                not snapshot_saved
                and attn_weights is not None
                and env.agents
                and total_snapshots_saved < MAX_SNAPSHOTS
                and episode_steps == SNAPSHOT_STEP
            ):
                try:
                    target_agent = env.agents[0]
                    if target_agent in agent_ids:
                        target_idx   = agent_ids.index(target_agent)
                        agent_attn   = attn_weights[target_idx, 0, :]   # (num_neighbors,)
                        neigh_ids_snap = env.neighbor_mapping.get(target_agent, [])
                        active_set   = set(env.agents)

                        def _get_km_pos(aid):
                            try:
                                hidx = bs.traf.id2idx(aid)
                                qdr, dis = bs.tools.geo.kwikqdrdist(
                                    env.center[0], env.center[1],
                                    bs.traf.lat[hidx], bs.traf.lon[hidx])
                                # East = sin(qdr)*dis, North = cos(qdr)*dis  (standard convention)
                                return (
                                    float(np.sin(np.deg2rad(qdr)) * dis * NM2KM),
                                    float(np.cos(np.deg2rad(qdr)) * dis * NM2KM),
                                )
                            except Exception:
                                return (0.0, 0.0)

                        own_km   = _get_km_pos(target_agent)
                        neigh_km = {
                            nid: _get_km_pos(nid)
                            for nid in neigh_ids_snap if nid in active_set
                        }
                        attn_map = {
                            nid: float(agent_attn[i])
                            for i, nid in enumerate(neigh_ids_snap)
                            if nid in neigh_km
                        }

                        snap_path = os.path.join(
                            FIGS_DIR,
                            f"attention_ep{episode:03d}_step{episode_steps:04d}.png"
                        )
                        plot_attention_combined(
                            own_km, neigh_km, attn_map,
                            ownship_id=target_agent,
                            step=episode_steps,
                            save_path=snap_path,
                        )
                        plt.close("all")   # avoid figure accumulation
                        snapshot_saved = True
                        total_snapshots_saved += 1
                except Exception as exc:
                    print(f"[viz] Snapshot failed at ep={episode} step={episode_steps}: {exc}")

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
        print(f"   - Aircraft with Intrusions: {len(env.aircraft_with_intrusions)}/{N_AGENTS}")
        print(f"   - Waypoints Reached: {len(env.waypoint_reached_agents)}/{N_AGENTS}")
        
        
        

        episode_rewards.append(episode_reward)
        episode_steps_list.append(episode_steps)
        episode_intrusions.append(env.total_intrusions)
        episode_aircraft_with_intrusions.append(len(env.aircraft_with_intrusions))
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
    print(f"  - Average Aircraft with Intrusions: {np.mean(episode_aircraft_with_intrusions):.2f}/{N_AGENTS} ({np.mean(episode_aircraft_with_intrusions)/N_AGENTS*100:.1f}%)")
    
    waypoint_rate = (total_waypoints_reached / (NUM_EVAL_EPISODES * N_AGENTS)) * 100
    print(f"  - Overall Waypoint Reached Rate: {waypoint_rate:.1f}%")
    print(f"  - Episodes without Intrusion: {episode_witout_intrusion}")
    
    print('average density created', N_AGENTS / np.mean(env.areas_km2))

    # --- Save episode rewards for bootstrap convergence analysis ---
    rewards_path = os.path.join(script_dir, "episode_rewards.npy")
    np.save(rewards_path, np.array(episode_rewards))
    print(f"\n💾 Episode rewards saved to: {rewards_path}")
    print("   Run ks_analysis.py to perform the bootstrap convergence analysis.")
    if all_attention_weights:
        # Concatenate all collected attention weights
        all_attn = np.concatenate(all_attention_weights)
        mean_alpha = np.mean(all_attn)
        std_alpha = np.std(all_attn)
        min_alpha = np.min(all_attn)
        max_alpha = np.max(all_attn)
        
        print(f"\n📊 Attention Weight Statistics:")
        print(f"  - Mean (α): {mean_alpha:.4f} ± {std_alpha:.4f}")
        print(f"  - Min: {min_alpha:.4f}")
        print(f"  - Max: {max_alpha:.4f}")
        print(f"  - Total samples: {len(all_attn)}")
    
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
