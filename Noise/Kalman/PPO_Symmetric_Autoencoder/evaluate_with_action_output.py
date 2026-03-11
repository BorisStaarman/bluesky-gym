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

from bluesky_gym.envs.ma_env_two_stage_AM_PPO_NOISE_autoencoder import SectorEnv, D_HEADING, D_VELOCITY
from ray.tune.registry import register_env

import torch
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
NUM_EVAL_EPISODES = 100  # How many episodes to run for evaluation
RENDER = False # Set to True to watch the agent play

# This path MUST match the checkpoint directory from your main.py training script
script_dir = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.join(script_dir, "metrics")

# --- CHOOSE WHICH CHECKPOINT TO EVALUATE ---
# Set to True to use stage1_best_weights, False to use stage1_weights (last iteration)
USE_BEST_STAGE1_WEIGHTS = False

# Determine which checkpoint to use based on the boolean
if USE_BEST_STAGE1_WEIGHTS:
    CHECKPOINT_DIR = os.path.join(script_dir, "models/sectorcr_ma_sac/stage1_best_weights")
    print(f"🌟 Using BEST Stage 1 weights from: {CHECKPOINT_DIR}")
else:
    CHECKPOINT_DIR = os.path.join(script_dir, "models/sectorcr_ma_sac/best_iter_00130")
    print(f"📁 Using stage 2 weights {CHECKPOINT_DIR}")


if __name__ == "__main__":
    # Initialize Ray
    os.environ["TENSORBOARD"] = "0"   # ✅ Prevent auto-launch
    ray.shutdown()
    # ray.init(include_dashboard=False)
    ray.init(runtime_env={
        "working_dir": script_dir,
        "py_modules": [os.path.join(script_dir, "attention_model_A.py")],
        "excludes": [
        "metrics/",         # Excludes all CSVs and logs
        # "models/",          # Excludes large model weights
        "*.csv",            # Excludes all CSV files
        "*.zip"             # Excludes any zip files
    ]
    })

    # --- Check if a checkpoint exists ---
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"❌ Checkpoint directory not found at: {CHECKPOINT_DIR}")
        print("Please run the `main.py` script first to train and save a model.")
        ray.shutdown()
        exit()

    print(f"\n🎯 Evaluating policy from checkpoint:")
    print(f"   {CHECKPOINT_DIR}\n")

    # --- Build algorithm configuration and restore weights ---
    # Define policy mapping
    def policy_map(agent_id, *_, **__):
        return "shared_policy"
    
    # Build PPO config (should match training configuration)
    from ray.rllib.algorithms.ppo import PPOConfig
    
    cfg = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            "sector_env",
            env_config={
                "n_agents": N_AGENTS,
                "run_id": RUN_ID,
                "metrics_base_dir": METRICS_DIR,
            },
            disable_env_checking=True,
        )
        .framework("torch")
        .env_runners(
            num_env_runners=0,  # No workers needed for evaluation
            num_envs_per_env_runner=1,
        )
        .training(
            lr=1e-4,  # Doesn't matter for evaluation but needs to match structure
            train_batch_size=16000,
            minibatch_size=1024,
            model={
                "custom_model": "attention_sac",
                "custom_model_config": {
                    "hidden_dims": [512, 512],
                    "is_critic": False,
                    "n_agents": N_AGENTS,
                    "embed_dim": 128,
                },
                "free_log_std": True,
                "vf_share_layers": False,
            },
        )
        .multi_agent(
            policies={"shared_policy": (None, None, None, {})},
            policy_mapping_fn=policy_map,
        )
        .resources(num_gpus=0)
    )
    
    # Build algorithm
    algo = cfg.build()
    
    # Get policy (before restoring)
    policy = algo.get_policy("shared_policy")
    
    # --- Manually load ONLY model weights (skip optimizer state) ---
    print(f"📥 Loading model weights from: {CHECKPOINT_DIR}")
    
    import pickle
    
    # RLlib checkpoint structure: policies/shared_policy/policy_state.pkl
    policy_state_path = os.path.join(CHECKPOINT_DIR, "policies", "shared_policy", "policy_state.pkl")
    
    if os.path.exists(policy_state_path):
        print(f"   Found policy state at: {policy_state_path}")
        
        with open(policy_state_path, "rb") as f:
            policy_state = pickle.load(f)
        
        # Load only model weights, not optimizer state
        if "weights" in policy_state:
            model_weights = policy_state["weights"]
            
            # Convert numpy arrays to torch tensors
            model_weights_converted = {}
            for key, value in model_weights.items():
                if isinstance(value, np.ndarray):
                    model_weights_converted[key] = torch.from_numpy(value)
                else:
                    model_weights_converted[key] = value
            
            # Load with strict=False to allow missing keys (log_std, value_branch)
            # These were added after the checkpoint was saved
            missing_keys, unexpected_keys = policy.model.load_state_dict(model_weights_converted, strict=False)
            
            if missing_keys:
                print(f"⚠️  Missing keys (will use randomly initialized values): {missing_keys}")
            if unexpected_keys:
                print(f"⚠️  Unexpected keys (ignored): {unexpected_keys}")
            
            print(f"✅ Model weights loaded successfully (skipped optimizer state)")
        else:
            print(f"⚠️  Warning: 'weights' key not found in policy state")
            print(f"   Available keys: {list(policy_state.keys())}")
            
            # Sometimes weights are stored directly in the state dict
            try:
                # Convert all numpy arrays to tensors
                converted_state = {}
                for key, value in policy_state.items():
                    if isinstance(value, np.ndarray):
                        converted_state[key] = torch.from_numpy(value)
                    else:
                        converted_state[key] = value
                
                policy.model.load_state_dict(converted_state, strict=False)
                print(f"✅ Loaded weights directly from policy_state")
            except Exception as e:
                print(f"❌ Failed to load weights: {e}")
                ray.shutdown()
                exit()
    else:
        print(f"❌ Could not find policy state file at: {policy_state_path}")
        print(f"   Checkpoint directory contents:")
        for root, dirs, files in os.walk(CHECKPOINT_DIR):
            level = root.replace(CHECKPOINT_DIR, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"   {indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"   {subindent}{file}")
        ray.shutdown()
        exit()
    
    # DEBUG: Check the model's final layer output dimension
    model = policy.model
    print(f"\n[DEBUG Model Architecture]")
    print(f"  Model type: {type(model)}")
    if hasattr(model, 'final_layer'):
        print(f"  final_layer input dim: {model.final_layer.in_features}")
        print(f"  final_layer output dim: {model.final_layer.out_features}")
    if hasattr(model, 'action_dim'):
        print(f"  model.action_dim: {model.action_dim}")
    if hasattr(model, '_last_output_dim'):
        print(f"  model._last_output_dim: {model._last_output_dim}")
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
    episode_aircraft_with_intrusions = []  # Track number of unique aircraft with intrusions
    total_waypoints_reached = 0
    episode_witout_intrusion = 0
    # velocity_agent_1 = []
    # `polygon`_areas_km2 = []  # Store polygon area in km² for each episode

    # --- Main Evaluation Loop ---
    # Collect NN outputs across episodes for plotting (heading change [deg], speed change [kt])
    all_heading_changes = []
    all_speed_changes = []
    # MVP (teacher) action collections
    all_mvp_heading_changes = []
    all_mvp_speed_changes = []

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
            
            # Get raw model output directly (bypassing policy's action distribution)
            with torch.no_grad():
                input_dict = {"obs": torch.from_numpy(obs_array).float()}
                model_out, _ = policy.model(input_dict, [], None)
                actions_np = model_out.numpy()  # Use model output directly
            
            # Compare teacher vs model actions every 20 steps
            # if episode_steps % 20 == 0 and episode_steps < 60:  # Only first 3 comparisons to reduce spam
            #     print(f"\n[Episode {episode}, Step {episode_steps}] Teacher vs Model Actions:")
            #     for i, agent_id in enumerate(agent_ids[:3]):  # Show first 3 agents
            #         teacher_action = env._calculate_mvp_action(agent_id)
            #         model_action = actions_np[i]
                    
                    # # Handle both 1D and 2D actions
                    # if len(model_action.shape) == 0:
                    #     print(f"  {agent_id}: [ERROR] Scalar action")
                    #     continue
                    # elif model_action.shape[0] == 1:
                    #     # Model only outputs 1 value - show it alongside teacher's 2 values
                    #     print(f"  {agent_id}:")
                    #     print(f"    Teacher: [{teacher_action[0]:+.3f}, {teacher_action[1]:+.3f}]")
                    #     print(f"    Model:   [{model_action[0]:+.3f}] (only 1D output - INCORRECT)")
                    # elif model_action.shape[0] >= 2:
                    #     print(f"  {agent_id}:")
                    #     print(f"    Teacher: [{teacher_action[0]:+.3f}, {teacher_action[1]:+.3f}]")
                    #     print(f"    Model:   [{model_action[0]:+.3f}, {model_action[1]:+.3f}]")
                    #     # Calculate action difference
                    #     diff = np.abs(teacher_action - model_action[:2])
                    #     print(f"    Diff:    [{diff[0]:.3f}, {diff[1]:.3f}]")
            

            # Map actions back to agent IDs
            actions = {agent_id: action for agent_id, action in zip(agent_ids, actions_np)}

            # Collect model actions for later plotting (convert to physical units)
            try:
                for action in actions.values():
                    # action expected shape (2,) -> [dh_normalized, dv_normalized]
                    if hasattr(action, '__len__') and len(action) >= 2:
                        dh_norm = float(action[0])
                        dv_norm = float(action[1])
                        heading_deg = dh_norm * D_HEADING
                        speed_kt = dv_norm * D_VELOCITY
                        all_heading_changes.append(heading_deg)
                        all_speed_changes.append(speed_kt)
            except Exception:
                pass

            # Collect MVP (teacher) actions for the same agents (raw physical units, unclipped)
            try:
                for agent_id in agent_ids:
                    try:
                        hdg_diff, spd_diff = env._calculate_mvp_action(agent_id, return_physical=True)
                        all_mvp_heading_changes.append(hdg_diff)
                        all_mvp_speed_changes.append(spd_diff)
                    except Exception:
                        continue
            except Exception:
                pass

            # Share actions with env so _render_frame can draw policy command arrows
            env.last_actions = actions

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

    # ------------------------------
    # Plot 2D histogram + optional KDE of NN outputs
    # ------------------------------
    try:
        figs_dir = os.path.join(script_dir, "figures")
        os.makedirs(figs_dir, exist_ok=True)

        x = np.array(all_heading_changes)
        y = np.array(all_speed_changes)

        if x.size == 0 or y.size == 0:
            print("No actions collected — skipping action distribution plot.")
        else:
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))

            # 2D histogram (fixed axis ranges)
            x_range = (-45.0, 45.0)
            y_range = (-10.0/3.0, 10.0/3.0)
            hb = ax[0].hist2d(x, y, bins=80, range=[x_range, y_range], cmap="Blues")
            fig.colorbar(hb[3], ax=ax[0])
            ax[0].set_xlim(x_range)
            ax[0].set_ylim(y_range)
            ax[0].set_xlabel("Heading change (deg)")
            ax[0].set_ylabel("Speed change (kt)")
            ax[0].set_title("2D histogram of NN outputs (all agents, all episodes)")

            # KDE contour if scipy available
            try:
                from scipy.stats import gaussian_kde
                xy = np.vstack([x, y])
                kde = gaussian_kde(xy)
                xmin, xmax = x_range
                ymin, ymax = y_range
                X, Y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
                Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
                ax[1].contourf(X, Y, Z, levels=20, cmap="viridis")
                ax[1].set_title("KDE of NN outputs")
                ax[1].set_xlabel("Heading change (deg)")
                ax[1].set_ylabel("Speed change (kt)")
                ax[1].set_xlim(xmin, xmax)
                ax[1].set_ylim(ymin, ymax)
            except Exception:
                # Fallback: hexbin
                ax[1].hexbin(x, y, gridsize=60, cmap="viridis")
                ax[1].set_title("Hexbin (KDE unavailable)")
                ax[1].set_xlabel("Heading change (deg)")
                ax[1].set_ylabel("Speed change (kt)")
                ax[1].set_xlim(x_range)
                ax[1].set_ylim(y_range)

            out_path = os.path.join(figs_dir, f"action_distribution_{NUM_EVAL_EPISODES}eps.png")
            fig.tight_layout()
            fig.savefig(out_path, dpi=200)
            print(f"Saved action distribution figure to: {out_path}")
            plt.close(fig)
            # Also plot MVP (teacher) distribution using same axis limits
            fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))
            xm = np.array(all_mvp_heading_changes)
            ym = np.array(all_mvp_speed_changes)
            if xm.size == 0 or ym.size == 0:
                print("No MVP actions collected — skipping MVP action distribution plot.")
            else:
                # Use data-driven axis limits (min/max of the MVP data)
                xmin_m, xmax_m = float(xm.min()), float(xm.max())
                ymin_m, ymax_m = float(ym.min()), float(ym.max())
                # Add small padding
                pad_x = max(1e-3, 0.05 * (xmax_m - xmin_m))
                pad_y = max(1e-3, 0.05 * (ymax_m - ymin_m))
                ax2[0].hist2d(xm, ym, bins=80, cmap="Reds")
                ax2[0].set_xlim(xmin_m - pad_x, xmax_m + pad_x)
                ax2[0].set_ylim(ymin_m - pad_y, ymax_m + pad_y)
                ax2[0].set_xlabel("Heading change (deg)")
                ax2[0].set_ylabel("Speed change (kt)")
                ax2[0].set_title("2D histogram of MVP (teacher) outputs")

                try:
                    from scipy.stats import gaussian_kde
                    X2, Y2 = np.mgrid[xmin_m - pad_x:xmax_m + pad_x:200j, ymin_m - pad_y:ymax_m + pad_y:200j]
                    kde2 = gaussian_kde(np.vstack([xm, ym]))
                    Z2 = kde2(np.vstack([X2.ravel(), Y2.ravel()])).reshape(X2.shape)
                    ax2[1].contourf(X2, Y2, Z2, levels=20, cmap="magma")
                    ax2[1].set_xlim(xmin_m - pad_x, xmax_m + pad_x)
                    ax2[1].set_ylim(ymin_m - pad_y, ymax_m + pad_y)
                    ax2[1].set_title("KDE of MVP outputs")
                    ax2[1].set_xlabel("Heading change (deg)")
                    ax2[1].set_ylabel("Speed change (kt)")
                except Exception:
                    ax2[1].hexbin(xm, ym, gridsize=60, cmap="magma")
                    ax2[1].set_xlim(xmin_m - pad_x, xmax_m + pad_x)
                    ax2[1].set_ylim(ymin_m - pad_y, ymax_m + pad_y)
                    ax2[1].set_title("Hexbin (KDE unavailable)")

                out_path2 = os.path.join(figs_dir, f"mvp_action_distribution_{NUM_EVAL_EPISODES}eps.png")
                fig2.tight_layout()
                fig2.savefig(out_path2, dpi=200)
                print(f"Saved MVP action distribution figure to: {out_path2}")
                plt.close(fig2)
    except Exception as e:
        print(f"Failed to create/save action distribution figure: {e}")
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
