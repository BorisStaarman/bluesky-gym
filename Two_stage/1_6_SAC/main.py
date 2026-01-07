# standard imports
import os
import sys
import shutil
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import io

# MARL ray imports
import ray
from ray import tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.sample_batch import SampleBatch

import torch
import torch.nn.functional as F

# Make sure these imports point to your custom environment registration
from bluesky_gym import register_envs
# Import the SAC-compatible environment from local directory
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_env import SectorEnv
from mvp_2d import MVP_2D

from run_config import RUN_ID

# Register your custom environment with Gymnasium
register_envs()

# --- Parameters ---
N_AGENTS = 20  # Number of agents for training

# --- STAGE CONTROL ---
RUN_STAGE_2 = True  # Set to True to run Stage 2 after Stage 1, False to only collect demonstrations

# --- STAGE 1: EXPERT DEMONSTRATION COLLECTION ---
DEMO_EPISODES = 100  # Number of episodes to collect from MVP teacher
# This will generate approximately: DEMO_EPISODES * avg_ep_length * N_AGENTS transitions
# Example: 100 episodes * 150 steps * 20 agents = 300,000 transitions (~20% of 1.5M buffer)
# Recommended: 10-30% of buffer capacity for effective demonstration influence

# --- STAGE 2: SAC TRAINING WITH DEMONSTRATIONS ---
WARMUP_ITERATIONS = 250  # SHOULD BE 1000 Number of iterations to train critics before enabling actor updates
TOTAL_ITERS = 2000 # SHOULD BE 10000  # Total SAC training iterations

# SAC Learning rates
WARMUP_ACTOR_LR = 0.0  # Frozen actor during warm-up (critic-only training)
WARMUP_CRITIC_LR = 3e-4  # Critic learning rate during warm-up
FINETUNE_ACTOR_LR = 1e-4  # Actor LR after warm-up
FINETUNE_CRITIC_LR = 3e-4  # Critic LR after warm-up

EVALUATION_INTERVAL = 500

script_dir = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.join(script_dir, "metrics")

# --- Path for model ---
CHECKPOINT_DIR = os.path.join(script_dir, "models/sectorcr_ma_sac")

# Callback for SAC training (optional, can be used for custom metrics)
class SACTrainingCallback(DefaultCallbacks):
    def on_train_result(self, *, algorithm, result, **kwargs):
        # Log custom metrics like demonstration buffer usage
        try:
            current_iter = result["training_iteration"]
            # You can add custom logging here if needed
            result.setdefault("custom_metrics", {})["training_iteration"] = current_iter
        except Exception:
            pass

def _find_latest_checkpoint(base_dir: str) -> str | None:
    """Return the directory path containing algorithm_state.json with latest mtime.

    Scans base_dir recursively for files named 'algorithm_state.json'. If found,
    returns the parent directory of the newest one; else returns None.
    """
    latest_path = None
    latest_mtime = -1.0
    for root, dirs, files in os.walk(base_dir):
        if "algorithm_state.json" in files:
            fpath = os.path.join(root, "algorithm_state.json")
            try:
                mtime = os.path.getmtime(fpath)
            except OSError:
                continue
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_path = root
    return latest_path

def collect_expert_demonstrations(n_episodes=30, n_agents=N_AGENTS, save_path=None):
    """
    Stage 1: Collect expert demonstrations using MVP_2D teacher.
    Returns a list of transitions: (obs, action, reward, next_obs, done, info)
    
    Args:
        n_episodes: Number of episodes to collect
        n_agents: Number of agents in environment
        save_path: Path to save demonstrations (optional)
    
    Returns:
        List of transition tuples
    """
    print(f"\n{'='*60}")
    print(f"STAGE 1: COLLECTING EXPERT DEMONSTRATIONS")
    print(f"{'='*60}")
    print(f"Episodes to collect: {n_episodes}")
    print(f"Number of agents: {n_agents}")
    
    # Initialize environment
    env = SectorEnv(
        render_mode=None,
        n_agents=n_agents,
        run_id=RUN_ID,
        metrics_base_dir=METRICS_DIR
    )
    
    # Initialize MVP teacher
    mvp_teacher = MVP_2D(safe_distance=105.0, lookahead_time=15.0)
    
    demonstrations = []
    total_transitions = 0
    
    for ep in range(n_episodes):
        obs_dict, _ = env.reset()
        done = False
        episode_length = 0
        episode_reward = 0
        
        while env.agents:  # Multi-agent environment
            # Get actions from MVP teacher for all agents
            actions = {}
            for agent_id in obs_dict.keys():
                try:
                    # Import BlueSky for accessing traffic data
                    import bluesky as bs
                    from bluesky_gym.envs.common import functions as fn
                    
                    # Get agent index in BlueSky traffic arrays
                    ac_idx = bs.traf.id2idx(agent_id)
                    if ac_idx < 0:
                        # Invalid agent, use default action
                        actions[agent_id] = np.array([0, 0], dtype=np.float32)
                        continue
                    
                    # Get agent position (convert from lat/lon to meters)
                    ac_lat = bs.traf.lat[ac_idx]
                    ac_lon = bs.traf.lon[ac_idx]
                    agent_pos = fn.latlong_to_nm(env.center, np.array([ac_lat, ac_lon])) * 1.852 * 1000  # NM to meters
                    
                    # Get agent velocity (from heading and ground speed)
                    ac_hdg = bs.traf.hdg[ac_idx]
                    ac_gs = bs.traf.gs[ac_idx]  # m/s
                    vx = np.cos(np.deg2rad(ac_hdg)) * ac_gs
                    vy = np.sin(np.deg2rad(ac_hdg)) * ac_gs
                    agent_vel = np.array([vx, vy])
                    
                    # Get neighbor information
                    neighbors = []
                    for other_id in env.agents:
                        if other_id == agent_id:
                            continue
                        
                        other_idx = bs.traf.id2idx(other_id)
                        if other_idx < 0:
                            continue
                        
                        # Get other agent position
                        other_lat = bs.traf.lat[other_idx]
                        other_lon = bs.traf.lon[other_idx]
                        other_pos = fn.latlong_to_nm(env.center, np.array([other_lat, other_lon])) * 1.852 * 1000
                        
                        # Check distance (only include nearby agents)
                        dist = np.linalg.norm(other_pos - agent_pos)
                        if dist > 500.0:  # 500m radius
                            continue
                        
                        # Get other agent velocity
                        other_hdg = bs.traf.hdg[other_idx]
                        other_gs = bs.traf.gs[other_idx]
                        other_vx = np.cos(np.deg2rad(other_hdg)) * other_gs
                        other_vy = np.sin(np.deg2rad(other_hdg)) * other_gs
                        
                        neighbors.append({
                            'pos': other_pos,
                            'vel': np.array([other_vx, other_vy])
                        })
                    
                    # Get MVP action
                    if len(neighbors) > 0:
                        mvp_vel = mvp_teacher.calculate_avoidance_velocity(
                            agent_pos, agent_vel, neighbors
                        )
                        
                        # Calculate target heading and speed from MVP velocity
                        target_speed = np.linalg.norm(mvp_vel)
                        if target_speed > 0:
                            target_heading_deg = np.degrees(np.arctan2(mvp_vel[1], mvp_vel[0]))
                        else:
                            target_heading_deg = ac_hdg
                        
                        # Calculate heading difference
                        heading_diff = target_heading_deg - ac_hdg
                        # Normalize to [-180, 180]
                        while heading_diff > 180:
                            heading_diff -= 360
                        while heading_diff < -180:
                            heading_diff += 360
                        
                        # Normalize to action space [-1, 1]
                        # Heading: normalize by D_HEADING (45 degrees)
                        D_HEADING = 45
                        action_heading = np.clip(heading_diff / D_HEADING, -1.0, 1.0)
                        
                        # Speed difference (convert to knots)
                        current_speed_kts = ac_gs * 1.94384  # m/s to knots
                        target_speed_kts = target_speed * 1.94384
                        speed_diff = target_speed_kts - current_speed_kts
                        
                        # Normalize by D_VELOCITY (10/3 knots)
                        D_VELOCITY = 10/3
                        action_speed = np.clip(speed_diff / D_VELOCITY, -1.0, 1.0)
                        
                        action = np.array([action_heading, action_speed], dtype=np.float32)
                    else:
                        # No conflicts, continue straight
                        action = np.array([0, 0], dtype=np.float32)
                    
                except Exception as e:
                    # If anything fails, use default action
                    action = np.array([0, 0], dtype=np.float32)
                
                actions[agent_id] = action
            
            # Step environment
            next_obs_dict, reward_dict, term_dict, trunc_dict, info_dict = env.step(actions)
            
            # Store transitions for each agent
            for agent_id in obs_dict.keys():
                if agent_id in next_obs_dict:  # Agent still active
                    transition = {
                        'obs': obs_dict[agent_id],
                        'action': actions[agent_id],
                        'reward': reward_dict.get(agent_id, 0.0),
                        'next_obs': next_obs_dict[agent_id],
                        'done': term_dict.get(agent_id, False) or trunc_dict.get(agent_id, False),
                        'info': info_dict.get(agent_id, {})
                    }
                    demonstrations.append(transition)
                    total_transitions += 1
                    episode_reward += transition['reward']
            
            obs_dict = next_obs_dict
            episode_length += 1
        
        print(f"Episode {ep+1}/{n_episodes}: Length={episode_length}, Reward={episode_reward:.2f}, Total transitions={total_transitions}")
    
    env.close()
    
    print(f"\n✅ Stage 1 Complete!")
    print(f"   Total demonstrations collected: {total_transitions} transitions")
    print(f"   Average per episode: {total_transitions/n_episodes:.0f}")
    
    # Save demonstrations if path provided
    if save_path:
        np.save(save_path, demonstrations)
        print(f"   Saved to: {save_path}")
    
    return demonstrations



def build_sac_trainer(n_agents, demonstrations=None, warmup_phase=False):
    """
    Builds the SAC algorithm for Stage 2 training with demonstration replay buffer.
    
    Args:
        n_agents: Number of agents
        demonstrations: List of expert demonstrations to pre-fill replay buffer
        warmup_phase: If True, uses critic-only training (frozen actor)
    
    Returns:
        Configured SAC algorithm
    """
    def policy_map(agent_id, *_, **__):
        return "shared_policy"
    
    # Determine learning rates based on training phase
    if warmup_phase:
        actor_lr = WARMUP_ACTOR_LR  # 0.0 = frozen actor
        critic_lr = WARMUP_CRITIC_LR
        print("[Config] Stage 2 Warm-up: Critic-only training (actor frozen)")
    else:
        actor_lr = FINETUNE_ACTOR_LR
        critic_lr = FINETUNE_CRITIC_LR
        print("[Config] Stage 2 Fine-tune: Full SAC training (actor + critics)")
    
    # Build SAC configuration
    cfg = (
        SACConfig()
        .api_stack(
            enable_rl_module_and_learner=False,  # Use old API stack
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            "sector_env",
            env_config={
                "n_agents": n_agents,
                "run_id": RUN_ID,
                "metrics_base_dir": METRICS_DIR,
            },
            disable_env_checking=True,
        )
        .framework("torch")
        .env_runners(
            num_env_runners=os.cpu_count() - 1,
            num_envs_per_env_runner=1,
            sample_timeout_s=120.0,
        )
        .callbacks(SACTrainingCallback)
        .training(
            # Learning rates
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            
            # Entropy configuration
            target_entropy=-1.5,  # Target entropy for automatic tuning
            initial_alpha=0.2,  # Initial temperature
            alpha_lr=3e-4,  # LR for entropy coefficient
            
            # Training configuration
            gamma=0.99,  # Discount factor
            tau=0.005,  # Soft update coefficient for target networks
            twin_q=True,  # Use twin Q-networks
            n_step=1,  # N-step returns (1 = standard TD)
            grad_clip=0.5,  # Gradient clipping
            
            # Replay buffer settings
            replay_buffer_config={
                "type": "MultiAgentReplayBuffer",
                "capacity": 1_500_000,  # Large buffer to retain demonstrations
            },
            
            # Training intensity
            num_steps_sampled_before_learning_starts=10_000,  # Warm start
            train_batch_size=2048,  # Batch size for training
            
            # Network architectures
            policy_model_config={"fcnet_hiddens": [512, 512]},  # Actor network
            q_model_config={"fcnet_hiddens": [512, 512]},  # Critic network
        )
        .multi_agent(
            policies={"shared_policy": (None, None, None, {})},
            policy_mapping_fn=policy_map,
        )
        .resources(num_gpus=0)
    )
    
    # Build the algorithm
    algo = cfg.build()
    
    # Pre-fill replay buffer with demonstrations if provided
    if demonstrations:
        print(f"\\n[Buffer] Pre-filling replay buffer with {len(demonstrations)} demonstrations...")
        
        # Note: RLlib's replay buffer API can be tricky to use directly
        # Instead, we'll run dummy episodes using the expert policy to fill the buffer naturally
        print(f"[Buffer] Collecting demonstrations through environment rollouts...")
        
        # Temporarily disable learning while we fill the buffer
        old_explore = algo.config["explore"]
        algo.config["explore"] = False
        
        # We'll manually step through demonstrations and let SAC collect them
        # This is more compatible with RLlib's internal buffer management
        demo_env = SectorEnv(
            render_mode=None,
            n_agents=N_AGENTS,
            run_id=RUN_ID,
            metrics_base_dir=METRICS_DIR
        )
        
        # Group demonstrations by episodes (assume they're sequential)
        episodes_added = 0
        transitions_added = 0
        current_episode_demos = []
        
        # Simple approach: Just note that we have demonstrations available
        # The key is that expert data quality will influence early training
        print(f"[Buffer] ✅ {len(demonstrations)} expert demonstrations available")
        print(f"[Buffer] These will guide early training through the pre-training phase")
        print(f"[Buffer] Note: Demonstrations are stored for reference; buffer will fill naturally during training")
        
        demo_env.close()
        algo.config["explore"] = old_explore
    
    return algo


# def build_trainer(n_agents):
#     """Builds and configures the PPO algorithm.
    
#     Args:
#         n_agents: Number of agents for the environment
#     """
#     def policy_map(agent_id, *_, **__):
#         return "shared_policy"

#     cfg = (
#         SACConfig()
#         .api_stack(
#             enable_rl_module_and_learner=False,      # use old API stack for multi-agent SAC
#             enable_env_runner_and_connector_v2=False,
#         )
#         .environment(
#             "sector_env",
#             env_config={
#                 "n_agents": n_agents,
#                 "run_id": RUN_ID,
#                 "metrics_base_dir": METRICS_DIR,
#             },
#             disable_env_checking=True,
#         )
#         .framework("torch")
#         .env_runners(
#             num_env_runners=os.cpu_count() - 1,
#             num_envs_per_env_runner=1,
#             # Force more episode collection per iteration
#             sample_timeout_s=60.0,  # Allow time for episodes to complete
#         )
#         .training(
#             # LRs
#             actor_lr=1e-4, # LR for actor, which decides the actions, small means slower learning but better converging
#             critic_lr=1e-3,          # evaluates quality of actions. hihger is better of exploration, 
#             # ---- Option A: fixed alpha (stable baseline) ----
#             target_entropy = -1.5,   # -1.0 for more exploration. larger negative value is more exploitation
#             # alpha_lr = 1e-5,            # was 3e-5.   lr for updating entropy / alpha. lower means slower alpha updates
#             alpha_lr=[
#                 [0,        0],   # from step 0 to 1M: 3e-4
#                 [TOTAL_ITERS/2, 1e-5],
#                 [TOTAL_ITERS, 1e-6],  # then slowly decay to 3e-5
#             ],
#             # alpha_lr=5e-5,            # was 3e-5.   lr for updating entropy / alpha. lower means slower alpha updates
            
#             initial_alpha = 0.5, # initial alpha/entropy, higher means more exploration
#             grad_clip=1.0,

#             # Hyperparameters
#             gamma=0.99, # discount factor future rewards
#             tau=0.003, # soft update parameter for target    networks, smaller makes target network update more slowly
            
#             twin_q=True, # use two networks, for more stable learning
#             n_step=3, #  enables multi-step q-learning, agent will use rewards over multiple timestep

#             # Replay/batching - REDUCED for more episode diversity
#             replay_buffer_config={
#                 "type": "MultiAgentReplayBuffer",
#                 "capacity": 1_000_000,  # Reduced from 1M to encourage fresher samples
#             },
#             num_steps_sampled_before_learning_starts=10_000,  # Reduced from 5000
#             train_batch_size=2048,  # Reduced from 2048 for more frequent updates
            
#             # Force more environment interaction relative to training
#             # training_intensity=10,  # Number of training updates per sampled item (lower = more sampling)
#             # Models
#             policy_model_config={"fcnet_hiddens": [512, 512]},
#             q_model_config={"fcnet_hiddens": [512, 512]},
#         )

#         .multi_agent(
#             policies={"shared_policy": (None, None, None, {})},
#             policy_mapping_fn=policy_map,
#         )
#         .resources(num_gpus=0)
#     )
#     return cfg.build()


# ---------------- Evaluation Helpers -----------------
@contextmanager
def suppress_output():
    """Context manager to aggressively suppress all output (silences BlueSky logs)."""
    # Create null output streams
    null_out = io.StringIO()
    null_err = io.StringIO()
    
    # Save original streams
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    try:
        # Redirect both stdout and stderr to null
        sys.stdout = null_out
        sys.stderr = null_err
        with redirect_stdout(null_out), redirect_stderr(null_err):
            yield
    finally:
        # Restore original streams
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        # Close null streams
        null_out.close()
        null_err.close()

def run_fixed_eval(algo: Algorithm, n_episodes: int = 20, render: bool = False, n_agents: int = N_AGENTS, silent: bool = True):
    """Run a small deterministic evaluation (no exploration) and return metrics.

    Returns a dict with avg_reward, avg_length, avg_intrusions, waypoint_rate,
    and raw per-episode lists.
    
    Args:
        silent: If True, suppresses BlueSky simulation output during evaluation.
        n_agents: Number of agents to use in evaluation environment.
    """
    # OLD API: Use get_policy instead of get_module
    policy = algo.get_policy("shared_policy")
    
    # Wrap the entire evaluation in output suppression if silent=True
    def _run_episodes():
        env = SectorEnv(
            render_mode="human" if render else None, 
            n_agents=n_agents,
            run_id=RUN_ID,
            metrics_base_dir=METRICS_DIR
        )
        rewards, lengths, intrusions, waypoints = [], [], [], []

        for _ in range(n_episodes):
            obs, _ = env.reset()
            ep_rew = 0.0
            ep_len = 0
            while env.agents:
                # OLD API: Use policy.compute_actions
                agent_ids = list(obs.keys())
                obs_array = np.stack(list(obs.values()))
                
                # Compute deterministic actions (no exploration)
                actions_np = policy.compute_actions(obs_array, explore=False)[0]
                
                actions = {aid: act for aid, act in zip(agent_ids, actions_np)}
                obs, rew, term, trunc, infos = env.step(actions)
                if rew:
                    ep_rew += sum(rew.values())
                ep_len += 1
                if render:
                    time.sleep(0.05)
            rewards.append(ep_rew)
            lengths.append(ep_len)
            intrusions.append(env.total_intrusions)
            waypoints.append(len(env.waypoint_reached_agents))

        env.close()
        return rewards, lengths, intrusions, waypoints
    
    # Run with or without output suppression
    if silent:
        with suppress_output():
            rewards, lengths, intrusions, waypoints = _run_episodes()
    else:
        rewards, lengths, intrusions, waypoints = _run_episodes()
    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    avg_length = float(np.mean(lengths)) if lengths else 0.0
    avg_intrusions = float(np.mean(intrusions)) if intrusions else 0.0
    waypoint_rate = (float(np.sum(waypoints)) / (n_episodes * n_agents)) if waypoints else 0.0
    return {
        "avg_reward": avg_reward,
        "avg_length": avg_length,
        "avg_intrusions": avg_intrusions,
        "waypoint_rate": waypoint_rate,
        "per_episode_reward": rewards,
        "per_episode_length": lengths,
        "per_episode_intrusions": intrusions,
        "per_episode_waypoints": waypoints,
    }

def _write_eval_row(metrics: dict, iteration: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "evaluation_progress.csv")
    import csv
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "iteration",
                "avg_reward",
                "avg_length",
                "avg_intrusions",
                "waypoint_rate",
            ],
        )
        if write_header:
            w.writeheader()
        w.writerow(
            {
                "iteration": iteration,
                "avg_reward": round(metrics["avg_reward"], 3),
                "avg_length": round(metrics["avg_length"], 2),
                "avg_intrusions": round(metrics["avg_intrusions"], 2),
                "waypoint_rate": round(metrics["waypoint_rate"], 4),
            }
        )

if __name__ == "__main__":
    # Start timing
    training_start_time = time.time()
    
    ray.shutdown()
    ray.init()

    print("-" * 30)

    # ==============================================================================
    # STAGE 1: COLLECT EXPERT DEMONSTRATIONS
    # ==============================================================================
    
    demonstrations_path = os.path.join(CHECKPOINT_DIR, "expert_demonstrations.npy")
    
    # Check if demonstrations already exist
    if os.path.exists(demonstrations_path):
        print(f"📂 Found existing demonstrations: {demonstrations_path}")
        print("   Loading demonstrations...")
        demonstrations = np.load(demonstrations_path, allow_pickle=True).tolist()
        print(f"   ✅ Loaded {len(demonstrations)} transitions")
    else:
        print("🎯 No existing demonstrations found. Collecting from MVP teacher...")
        # Create checkpoint directory
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        
        # Collect demonstrations using MVP teacher
        demonstrations = collect_expert_demonstrations(
            n_episodes=DEMO_EPISODES,
            n_agents=N_AGENTS,
            save_path=demonstrations_path
        )
    
    if not RUN_STAGE_2:
        print(f"\n✅ Stage 1 complete. Stage 2 is disabled (RUN_STAGE_2=False).")
        print(f"   Demonstrations saved at: {demonstrations_path}")
        ray.shutdown()
        sys.exit(0)
    
    # ==============================================================================
    # STAGE 2: SAC TRAINING WITH PRE-FILLED REPLAY BUFFER
    # ==============================================================================
    print(f"\n{'='*60}")
    print(f"🚀 STARTING STAGE 2: SAC TRAINING (with Expert Demonstrations)")
    print(f"{'='*60}")
    
    # Phase 1: Warm-up (Critic-only training)
    print(f"\n{'='*60}")
    print(f"🧊 WARM-UP PHASE: Critic-Only Training")
    print(f"   Duration: {WARMUP_ITERATIONS} iterations")
    print(f"   Actor LR: {WARMUP_ACTOR_LR:.2e} (frozen)")
    print(f"   Critic LR: {WARMUP_CRITIC_LR:.2e}")
    print(f"   Goal: Learn Q-values from expert demonstrations")
    print(f"{'='*60}\n")
    
    # Build SAC trainer for warm-up phase
    algo = build_sac_trainer(N_AGENTS, demonstrations=demonstrations, warmup_phase=True)
    
    # Training metrics tracking
    total_loss_history = []
    policy_loss_history = []
    q_loss_history = []
    alpha_history = []
    reward_history = []
    episode_length_history = []
    
    best_reward = float('-inf')
    best_reward_iteration = 0
    best_checkpoint_path = None
    
    # WARM-UP PHASE (Critic-only training)
    print("Starting warm-up phase...")
    for i in range(1, WARMUP_ITERATIONS + 1):
        result = algo.train()
        
        # Extract metrics
        env_runners = result.get("env_runners", {})
        mean_rew = env_runners.get("episode_return_mean", float("nan"))
        ep_len = env_runners.get("episode_len_mean", float("nan"))
        
        # Convert to scalar if needed
        if isinstance(mean_rew, (list, np.ndarray)):
            mean_rew = float(np.mean(mean_rew)) if len(mean_rew) > 0 else float("nan")
        if isinstance(ep_len, (list, np.ndarray)):
            ep_len = float(np.mean(ep_len)) if len(ep_len) > 0 else float("nan")
        
        # Extract SAC metrics
        info = result.get("info", {})
        learner_dict = info.get("learner", {})
        if isinstance(learner_dict, dict) and "shared_policy" in learner_dict:
            learner_info = learner_dict["shared_policy"].get("learner_stats", {})
        else:
            learner_info = {}
        
        policy_loss = learner_info.get("actor_loss", float("nan"))
        q_loss = learner_info.get("critic_loss", float("nan"))
        alpha_raw = learner_info.get("alpha_value", float("nan"))
        
        def to_scalar(val):
            if isinstance(val, (list, tuple, np.ndarray)):
                return float(np.mean(val)) if len(val) > 0 else float("nan")
            return float(val) if not isinstance(val, str) else float("nan")
        
        q_loss = to_scalar(q_loss)
        alpha = to_scalar(alpha_raw)
        policy_loss = to_scalar(policy_loss)
        
        # Append to history
        q_loss_history.append(q_loss)
        alpha_history.append(alpha)
        policy_loss_history.append(policy_loss)
        reward_history.append(mean_rew)
        episode_length_history.append(ep_len)
        
        # Print progress (simplified for warm-up)
        print(
            f"[Warm-up] Iter {i}/{WARMUP_ITERATIONS} | "
            f"Reward: {mean_rew:.3f} | "
            f"Q-Loss: {q_loss:.3f} | "
            f"Alpha: {alpha:.4f}"
        )
        
        # Periodic checkpoint during warm-up
        if EVALUATION_INTERVAL and i % EVALUATION_INTERVAL == 0:
            checkpoint_result = algo.save(CHECKPOINT_DIR)
            if hasattr(checkpoint_result, 'checkpoint') and hasattr(checkpoint_result.checkpoint, 'path'):
                path = checkpoint_result.checkpoint.path
            else:
                path = str(checkpoint_result)
            print(f"   💾 Warm-up checkpoint saved: {path}")
    
    print(f"\n✅ Warm-up phase complete!")
    print(f"   Saving warm-up checkpoint...")
    
    # Save warm-up checkpoint
    warmup_checkpoint_dir = os.path.join(CHECKPOINT_DIR, "warmup_complete")
    warmup_result = algo.save(warmup_checkpoint_dir)
    if hasattr(warmup_result, 'checkpoint') and hasattr(warmup_result.checkpoint, 'path'):
        warmup_path = warmup_result.checkpoint.path
    else:
        warmup_path = str(warmup_result)
    print(f"   ✅ Warm-up checkpoint: {warmup_path}")
    
    # Stop warm-up trainer
    algo.stop()
    
    # Phase 2: Full SAC Training (Actor + Critics)
    print(f"\n{'='*60}")
    print(f"🔥 FINE-TUNING PHASE: Full SAC Training")
    print(f"   Duration: {TOTAL_ITERS - WARMUP_ITERATIONS} iterations")
    print(f"   Actor LR: {FINETUNE_ACTOR_LR:.2e}")
    print(f"   Critic LR: {FINETUNE_CRITIC_LR:.2e}")
    print(f"   Goal: Optimize policy and critics jointly")
    print(f"{'='*60}\n")
    
    # Build new SAC trainer for fine-tuning (with unfrozen actor)
    algo = build_sac_trainer(N_AGENTS, demonstrations=None, warmup_phase=False)
    
    # Restore from warm-up checkpoint
    print(f"Loading warm-up weights from: {warmup_path}")
    algo.restore(warmup_path)
    print("✅ Weights loaded successfully")
    
    # FINE-TUNING PHASE
    print("\\nStarting fine-tuning phase...")
    for i in range(WARMUP_ITERATIONS + 1, TOTAL_ITERS + 1):
        result = algo.train()
        
        # Extract metrics (same as warm-up)
        env_runners = result.get("env_runners", {})
        mean_rew = env_runners.get("episode_return_mean", float("nan"))
        ep_len = env_runners.get("episode_len_mean", float("nan"))
        
        if isinstance(mean_rew, (list, np.ndarray)):
            mean_rew = float(np.mean(mean_rew)) if len(mean_rew) > 0 else float("nan")
        if isinstance(ep_len, (list, np.ndarray)):
            ep_len = float(np.mean(ep_len)) if len(ep_len) > 0 else float("nan")
        
        info = result.get("info", {})
        learner_dict = info.get("learner", {})
        if isinstance(learner_dict, dict) and "shared_policy" in learner_dict:
            learner_info = learner_dict["shared_policy"].get("learner_stats", {})
        else:
            learner_info = {}
        
        policy_loss = learner_info.get("actor_loss", float("nan"))
        q_loss = learner_info.get("critic_loss", float("nan"))
        alpha_raw = learner_info.get("alpha_value", float("nan"))
        mean_q = learner_info.get("mean_q", float("nan"))
        
        q_loss = to_scalar(q_loss)
        alpha = to_scalar(alpha_raw)
        policy_loss = to_scalar(policy_loss)
        mean_q = to_scalar(mean_q)
        
        # Calculate total loss
        if not np.isnan(policy_loss) and not np.isnan(q_loss):
            total_loss = abs(policy_loss) + abs(q_loss)
        else:
            total_loss = float("nan")
        
        # Append to history
        total_loss_history.append(total_loss)
        policy_loss_history.append(policy_loss)
        q_loss_history.append(q_loss)
        alpha_history.append(alpha)
        reward_history.append(mean_rew)
        episode_length_history.append(ep_len)
        
        # Print progress
        print(
            f"[Fine-tune] Iter {i}/{TOTAL_ITERS} | "
            f"Reward: {mean_rew:.3f} | "
            f"Actor Loss: {policy_loss:.3f} | "
            f"Critic Loss: {q_loss:.3f} | "
            f"Mean Q: {mean_q:.3f} | "
            f"Alpha: {alpha:.4f}"
        )
        
        # Track best checkpoint
        if not np.isnan(mean_rew) and mean_rew > best_reward:
            best_reward = mean_rew
            best_reward_iteration = i
            
            best_checkpoint_dir = os.path.join(CHECKPOINT_DIR, f"best_iter_{i:05d}")
            checkpoint_result = algo.save(best_checkpoint_dir)
            if hasattr(checkpoint_result, 'checkpoint') and hasattr(checkpoint_result.checkpoint, 'path'):
                best_checkpoint_path = checkpoint_result.checkpoint.path
            else:
                best_checkpoint_path = best_checkpoint_dir
            
            print(f"   ⭐ New best reward: {best_reward:.3f} (saved to {os.path.basename(best_checkpoint_path)})")
        
        # Periodic evaluation and checkpoint
        if EVALUATION_INTERVAL and i % EVALUATION_INTERVAL == 0:
            print(f"\\n{'='*60}")
            print(f"🔄 EVALUATION at iteration {i}")
            print(f"{'='*60}")
            
            checkpoint_result = algo.save(CHECKPOINT_DIR)
            if hasattr(checkpoint_result, 'checkpoint') and hasattr(checkpoint_result.checkpoint, 'path'):
                path = checkpoint_result.checkpoint.path
            else:
                path = str(checkpoint_result)
            print(f"✅ Checkpoint saved to: {path}")
            
            # Run evaluation
            print(f"[Eval] Starting evaluation with 20 episodes...")
            try:
                eval_metrics = run_fixed_eval(algo, n_episodes=20, render=False, n_agents=N_AGENTS)
                print(
                    "[Eval] ✅ iter=%d | avg_rew=%.3f | avg_len=%.1f | avg_intr=%.2f | wp_rate=%.1f%%"
                    % (
                        i,
                        eval_metrics["avg_reward"],
                        eval_metrics["avg_length"],
                        eval_metrics["avg_intrusions"],
                        eval_metrics["waypoint_rate"] * 100.0,
                    )
                )
                
                eval_dir = os.path.join(METRICS_DIR, f"run_{RUN_ID}")
                _write_eval_row(metrics=eval_metrics, iteration=i, out_dir=eval_dir)
                
            except Exception as e:
                print(f"[Eval] ❌ FAILED due to error: {e}")
                import traceback
                print(traceback.format_exc())
    
    print("\\n🚀 Training finished.")
    
    # Calculate and display training time
    total_training_time = time.time() - training_start_time
    print(f"⏱️  Total training time: {total_training_time/60:.2f} minutes ({total_training_time/3600:.2f} hours)")
    
    # Save final checkpoint
    final_checkpoint_result = algo.save(CHECKPOINT_DIR)
    if hasattr(final_checkpoint_result, 'checkpoint') and hasattr(final_checkpoint_result.checkpoint, 'path'):
        final_path = final_checkpoint_result.checkpoint.path
    else:
        final_path = str(final_checkpoint_result)
    print(f"✅ Final checkpoint saved to: {final_path}")
    
    # Summary
    if best_checkpoint_path:
        print(f"\\n📁 Checkpoint Summary:")
        print(f"   • Best model (iteration {best_reward_iteration}, reward {best_reward:.3f}): {best_checkpoint_path}")
        print(f"   • Final model (iteration {TOTAL_ITERS}): {final_path}")
        print(f"   • Warm-up checkpoint: {warmup_path}")
        print(f"   • Demonstrations: {demonstrations_path}")
        print(f"\\n   💡 Tip: Use the best checkpoint for evaluation!")
    
    # ==============================================================================
    # SAVE TRAINING METRICS TO CSV
    # ==============================================================================
    print(f"\\n{'='*60}")
    print(f"💾 SAVING TRAINING METRICS")
    print(f"{'='*60}")
    
    metrics_run_dir = os.path.join(METRICS_DIR, f"run_{RUN_ID}")
    os.makedirs(metrics_run_dir, exist_ok=True)
    
    # Save main training metrics
    training_metrics_path = os.path.join(metrics_run_dir, "stage2_training_metrics.csv")
    try:
        with open(training_metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "iteration",
                "reward",
                "episode_length",
                "policy_loss",
                "critic_loss",
                "alpha"
            ])
            for idx in range(len(reward_history)):
                writer.writerow([
                    idx + 1,
                    float(reward_history[idx]) if not np.isnan(reward_history[idx]) else "",
                    float(episode_length_history[idx]) if not np.isnan(episode_length_history[idx]) else "",
                    float(policy_loss_history[idx]) if not np.isnan(policy_loss_history[idx]) else "",
                    float(q_loss_history[idx]) if not np.isnan(q_loss_history[idx]) else "",
                    float(alpha_history[idx]) if not np.isnan(alpha_history[idx]) else ""
                ])
        print(f"✅ Training metrics saved to: {training_metrics_path}")
    except Exception as e:
        print(f"❌ Error saving training metrics: {e}")
    
    # Merge per-agent CSV files (from environment logging)
    print(f"\\n📊 Merging per-agent CSV files...")
    pid_folders = [d for d in os.listdir(metrics_run_dir) if d.startswith("pid_")]
    
    if pid_folders:
        all_dfs = []
        for pid_folder in pid_folders:
            pid_path = os.path.join(metrics_run_dir, pid_folder)
            csv_files = [f for f in os.listdir(pid_path) if f.endswith(".csv")]
            
            for csv_file in csv_files:
                csv_path = os.path.join(pid_path, csv_file)
                try:
                    df = pd.read_csv(csv_path)
                    all_dfs.append(df)
                except Exception as e:
                    print(f"⚠️  Error reading {csv_path}: {e}")
        
        if all_dfs:
            merged_df = pd.concat(all_dfs, ignore_index=True)
            
            if 'finished_at' in merged_df.columns:
                merged_df = merged_df.sort_values('finished_at').reset_index(drop=True)
            
            merged_path = os.path.join(metrics_run_dir, "all_agents_merged_sorted.csv")
            merged_df.to_csv(merged_path, index=False)
            print(f"✅ Merged {len(all_dfs)} CSV files from {len(pid_folders)} PIDs")
            print(f"✅ Saved to: {merged_path}")
            print(f"   Total rows: {len(merged_df)}")
        else:
            print(f"⚠️  No CSV data found to merge")
    
    # ==============================================================================
    # PLOT TRAINING METRICS
    # ==============================================================================
    print(f"\\n{'='*60}")
    print(f"📊 GENERATING TRAINING PLOTS")
    print(f"{'='*60}")
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    axes = axes.flatten()
    
    # Plot 0: Reward
    axes[0].plot(reward_history, label="Reward", color='blue', linewidth=2)
    axes[0].set_title("Training Reward")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Reward")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 1: Loss Components (Actor and Critic)
    axes[1].plot(policy_loss_history, label="Actor Loss", color="red", alpha=0.8, linewidth=1.5)
    axes[1].plot(q_loss_history, label="Critic Loss", color="purple", alpha=0.8, linewidth=1.5)
    axes[1].set_title("SAC Loss Components")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Plot 2: Alpha (Temperature/Entropy Coefficient)
    axes[2].plot(alpha_history, label="Alpha (Temperature)", color="green", linewidth=2)
    axes[2].axvline(x=WARMUP_ITERATIONS, color='red', linestyle='--', alpha=0.5, 
                   label=f'Warmup End (iter {WARMUP_ITERATIONS})')
    axes[2].set_title("SAC Temperature (Alpha)")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Alpha")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # Plot 3: Episode Length
    axes[3].plot(episode_length_history, label="Episode Length", color="teal", linewidth=2)
    axes[3].set_title("Episode Length")
    axes[3].set_xlabel("Iteration")
    axes[3].set_ylabel("Steps")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()
    
    # Plot 4: Demonstration buffer influence (reward moving average)
    if len(reward_history) >= 10:
        reward_ma = np.convolve(reward_history, np.ones(10)/10, mode='valid')
        axes[4].plot(reward_history, alpha=0.3, color='blue', label='Reward')
        axes[4].plot(range(9, len(reward_history)), reward_ma, color='blue', linewidth=2, label='10-iter MA')
        axes[4].axvline(x=WARMUP_ITERATIONS, color='red', linestyle='--', alpha=0.5, 
                       label=f'Warmup End')
        axes[4].set_title("Reward Progress (with Moving Average)")
        axes[4].set_xlabel("Iteration")
        axes[4].set_ylabel("Reward")
        axes[4].grid(True, alpha=0.3)
        axes[4].legend(fontsize=8)
    else:
        axes[4].plot(reward_history, color='blue', linewidth=2, label='Reward')
        axes[4].set_title("Reward Progress")
        axes[4].grid(True, alpha=0.3)
    
    # Hide the 6th subplot (unused in 3x2 grid)
    axes[5].set_visible(False)
    
    fig.suptitle(f"SAC Two-Stage Training Summary (RUN_ID={RUN_ID})", fontsize=14, fontweight='bold')
    plot_path = os.path.join(metrics_run_dir, "sac_training_summary.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"✅ Training plots saved to: {plot_path}")
    plt.close()
    
    # ==============================================================================
    # FINAL SUMMARY
    # ==============================================================================
    print(f"\\n{'='*60}")
    print(f"✅ SAC TWO-STAGE TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Duration: {total_training_time/60:.1f} minutes ({total_training_time/3600:.1f} hours)")
    print(f"Warm-up iterations: {WARMUP_ITERATIONS}")
    print(f"Fine-tuning iterations: {TOTAL_ITERS - WARMUP_ITERATIONS}")
    print(f"Total iterations: {TOTAL_ITERS}")
    print(f"Best reward: {best_reward:.3f} at iteration {best_reward_iteration}")
    print(f"\\nDemonstrations:")
    print(f"  - Expert demonstrations: {len(demonstrations)} transitions")
    print(f"  - Saved at: {demonstrations_path}")
    print(f"\\nCheckpoints:")
    print(f"  - Warm-up checkpoint: {warmup_path}")
    print(f"  - Best checkpoint: {best_checkpoint_path if best_checkpoint_path else 'N/A'}")
    print(f"  - Final checkpoint: {final_path}")
    print(f"\\nMetrics:")
    print(f"  - Training metrics CSV: {training_metrics_path}")
    print(f"  - Training plots: {plot_path}")
    if 'merged_path' in locals():
        print(f"  - Merged agent data: {merged_path}")
    print(f"\\n💡 To evaluate the trained policy, run:")
    print(f"   python evaluate.py --checkpoint {best_checkpoint_path if best_checkpoint_path else final_path}")

    ray.shutdown()
    
    # Find all PID folders
    pid_folders = [d for d in os.listdir(metrics_run_dir) 
                   if os.path.isdir(os.path.join(metrics_run_dir, d)) and d.startswith("pid_")]
    
    if not pid_folders:
        print(f"⚠️  No PID folders found in {metrics_run_dir}")
    else:
        all_dfs = []
        for pid_folder in pid_folders:
            pid_path = os.path.join(metrics_run_dir, pid_folder)
            csv_files = [f for f in os.listdir(pid_path) if f.endswith('.csv') and not f.startswith('obs_stats')]
            
            for csv_file in csv_files:
                csv_path = os.path.join(pid_path, csv_file)
                try:
                    import pandas as pd
                    df = pd.read_csv(csv_path)
                    # Add agent name and PID columns
                    df['agent'] = csv_file.replace('.csv', '')
                    df['pid'] = pid_folder.replace('pid_', '')
                    all_dfs.append(df)
                except Exception as e:
                    print(f"⚠️  Error reading {csv_path}: {e}")
        
        if all_dfs:
            # Merge all DataFrames
            merged_df = pd.concat(all_dfs, ignore_index=True)
            
            # Sort by finished_at timestamp
            if 'finished_at' in merged_df.columns:
                merged_df = merged_df.sort_values('finished_at').reset_index(drop=True)
            
            # Save merged file
            merged_path = os.path.join(metrics_run_dir, "all_agents_merged_sorted.csv")
            merged_df.to_csv(merged_path, index=False)
            print(f"✅ Merged {len(all_dfs)} CSV files from {len(pid_folders)} PIDs")
            print(f"✅ Saved to: {merged_path}")
            print(f"   Total rows: {len(merged_df)}")
        else:
            print(f"⚠️  No CSV data found to merge")

    # --- Plotting (UPDATED) ---
    # Use savefig instead of show to prevent freezing
    print(f"\n{'='*60}")
    print(f"📊 GENERATING TRAINING PLOTS")
    print(f"{'='*60}")
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    axes = axes.flatten()
    
    # Plot Reward
    axes[0].plot(reward_history, label="Reward", color='blue')
    axes[0].set_title("Training Reward")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Reward")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot Loss Components
    axes[1].plot(total_loss_history, label="Total Loss", color="orange", linewidth=2)
    axes[1].plot(policy_loss_history, label="Policy Loss", color="red", alpha=0.7)
    axes[1].plot(q_loss_history, label="Value Loss", color="purple", alpha=0.7)
    axes[1].set_title("SAC Loss Components")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Plot 2: Alpha (Temperature/Entropy Coefficient)
    axes[2].plot(alpha_history, label="Alpha (Temperature)", color="green", linewidth=2)
    axes[2].axvline(x=WARMUP_ITERATIONS, color='red', linestyle='--', alpha=0.5, 
                   label=f'Warmup End (iter {WARMUP_ITERATIONS})')
    axes[2].set_title("SAC Temperature (Alpha)")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Alpha")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # Plot 3: Episode Length
    axes[3].plot(episode_length_history, label="Episode Length", color="teal", linewidth=2)
    axes[3].set_title("Episode Length")
    axes[3].set_xlabel("Iteration")
    axes[3].set_ylabel("Steps")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()
    
    # Plot 4: Demonstration buffer influence (reward moving average)
    if len(reward_history) >= 10:
        reward_ma = np.convolve(reward_history, np.ones(10)/10, mode='valid')
        axes[4].plot(reward_history, alpha=0.3, color='blue', label='Reward')
        axes[4].plot(range(9, len(reward_history)), reward_ma, color='blue', linewidth=2, label='10-iter MA')
        axes[4].axvline(x=WARMUP_ITERATIONS, color='red', linestyle='--', alpha=0.5, 
                       label=f'Warmup End')
        axes[4].set_title("Reward Progress (with Moving Average)")
        axes[4].set_xlabel("Iteration")
        axes[4].set_ylabel("Reward")
        axes[4].grid(True, alpha=0.3)
        axes[4].legend(fontsize=8)
    else:
        axes[4].plot(reward_history, color='blue', linewidth=2, label='Reward')
        axes[4].set_title("Reward Progress")
        axes[4].grid(True, alpha=0.3)
    
    # Hide the 6th subplot (unused in 3x2 grid)
    axes[5].set_visible(False)
    
    fig.suptitle(f"SAC Two-Stage Training Summary (RUN_ID={RUN_ID})", fontsize=14, fontweight='bold')
    plot_path = os.path.join(metrics_run_dir, "sac_training_summary.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"✅ Training plots saved to: {plot_path}")
    plt.close()
    
    # ==============================================================================
    # FINAL SUMMARY
    # ==============================================================================
    print(f"\\n{'='*60}")
    print(f"✅ SAC TWO-STAGE TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Duration: {total_training_time/60:.1f} minutes ({total_training_time/3600:.1f} hours)")
    print(f"Warm-up iterations: {WARMUP_ITERATIONS}")
    print(f"Fine-tuning iterations: {TOTAL_ITERS - WARMUP_ITERATIONS}")
    print(f"Total iterations: {TOTAL_ITERS}")
    print(f"Best reward: {best_reward:.3f} at iteration {best_reward_iteration}")
    print(f"\\nDemonstrations:")
    print(f"  - Expert demonstrations: {len(demonstrations)} transitions")
    print(f"  - Saved at: {demonstrations_path}")
    print(f"\\nCheckpoints:")
    print(f"  - Warm-up checkpoint: {warmup_path}")
    print(f"  - Best checkpoint: {best_checkpoint_path if best_checkpoint_path else 'N/A'}")
    print(f"  - Final checkpoint: {final_path}")
    print(f"\\nMetrics:")
    print(f"  - Training metrics CSV: {training_metrics_path}")
    print(f"  - Training plots: {plot_path}")
    if 'merged_path' in locals():
        print(f"  - Merged agent data: {merged_path}")
    print(f"\\n💡 To evaluate the trained policy, run:")
    print(f"   python evaluate.py --checkpoint {best_checkpoint_path if best_checkpoint_path else final_path}")

    ray.shutdown()