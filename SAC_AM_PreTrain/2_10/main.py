# standard imports
import os
import sys
import shutil
import matplotlib.pyplot as plt
import numpy as np
import time
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import io
import copy

import bluesky as bs
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch
from mvp_2d import MVP_2D
import bluesky_gym.envs.common.functions as fn

# MARL ray imports
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.sac import SACConfig

from ray.rllib.models import ModelCatalog
# from attention_model_M import AttentionSACModel # Multiplicative method
from attention_model_A import AttentionSACModel # additive method

# Make sure these imports point to your custom environment registration
from bluesky_gym.envs.ma_env_SAC_AM import SectorEnv
from ray.rllib.policy.sample_batch import SampleBatch

from ray.tune.registry import register_env

import torch
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from run_config import RUN_ID

# Register custom environment with dynamic intrusion penalty
def make_env(config):
    # Read current intrusion penalty from config (updated by curriculum)
    return SectorEnv(**config)

register_env("sector_env", make_env)
ModelCatalog.register_custom_model("attention_sac", AttentionSACModel)

# --- Parameters ---
N_AGENTS = 20  # Number of agents for training
TOTAL_ITERS = 16000   # 18000 Maximum total iterations
EXTRA_ITERS = 50           # When resuming, run this many more iterations
FORCE_RETRAIN = True       # Start fresh with new hyperparameters
# Optional: Only useful if you want periodic checkpoints mid-training.
EVALUATION_INTERVAL = 500  # e.g., set to 1 or 5 to save during training

START_ALPHA = 0.15  # Initial alpha value for SAC entropy term

# IMPORTNAT PARAMETERS FOR PRETRAINING AND BURN-IN
PRETRAIN_EPISODES = 200 # 150
BURN_IN_ITERATIONS = 2000     # UPGRADED: 1500→2000 to prevent collapse

# --- Expert Mixing Parameters (Linear Decay) ---
EXPERT_MIX_START = 0.30        # Start with 30% expert data at iteration 0
EXPERT_MIX_END = 0.0           # End with 0% exp    ert data
EXPERT_MIX_DECAY_UNTIL = 0.1  # Decay to 0% by 75% of total iterations (375/500)

# --- Burn-in Phase Parameters (Offline Learning from Expert Buffer) ---
BURN_IN_BATCH_SIZE = 4096     # Batch size for burn-in sampling
ENABLE_BURN_IN = True         # Set to False to skip burn-in phase

# --- KL Divergence Penalty (Keep Policy Close to Expert During Burn-in) ---
ENABLE_KL_PENALTY = True      # Add KL penalty during burn-in
KL_COEFF_START = 1.0          # Initial KL coefficient (strong constraint)
KL_COEFF_END = 0.0            # Final KL coefficient (no constraint)
KL_ANNEAL_UNTIL = 3000        # Anneal KL penalty from start to 0 by this iteration

# # --- Burn-in learning rates (separate for actor, attention, temperature, and critic) ---
# # Actor is kept very small for stability; critic can be slightly larger.
# BURN_IN_TEMPERATURE_LR = 1e-4  # For temperature parameter (controls attention sharpness)
# BURN_IN_ACTOR_LR = 1e-6 # 7.5e-7
# BURN_IN_ATTENTION_LR = 5e-5  # For attention mechanism (W_q, W_k, W_v)
# BURN_IN_CRITIC_LR = 1.0e-4


# --- VEILIGE LEARNING RATE AANPASSING VOOR BURN-IN ---
# RESTORED: Parameters that achieved 74% WP success in burn-in
BURN_IN_TEMPERATURE_LR_LOCAL = 1e-4  # Stable temperature growth (worked at 74% run)
BURN_IN_ACTOR_LR_LOCAL = 5e-7       # Keep stable for navigation
BURN_IN_ATTENTION_LR_LOCAL = 3e-5   # RESTORED: 8e-5 was too high, causing instability
BURN_IN_CRITIC_LR_LOCAL = 1e-4      # Keep critic stable


# --- Main Training Phase Learning Rates (Separate Control) ---
# Actor LR: constant throughout training
MAIN_ACTOR_LR_SCHEDULE = [
    [0, 1e-6],        # Stabilization after burn-in
    [500, 5e-6],      # Gradual ramp-up
    [2000, 2e-5],     # Peak: collision avoidance mastery
    [6000, 1e-5],     # Maintain for extended learning
    [14000, 5e-6],    # Final refinement
]

# Attention/Temperature LR Schedule (for attention mechanism focus)
# PROGRESSIVE SHARPENING: Gradual increase over 12K iterations
# UPGRADED: Extended high-LR phase from 6K to 8K iterations
MAIN_ATTENTION_LR_SCHEDULE = [
    [0, 1e-4],        # Start moderate after burn-in
    [2000, 2e-4],     # Ramp up during collision learning phase
    [8000, 1e-4],   # UPGRADED: 6K→8K maintain high for extended mastery
    [14000, 8e-5],    # Final refinement
]

# Temperature Parameter LR Schedule (for attention sharpness control)
# UPGRADED: Extended high-LR phase from 6K to 8K iterations
MAIN_TEMPERATURE_LR_SCHEDULE = [
    [0, 2e-4],        # Start moderate
    [2000, 3e-4],     # Increase for sharper softmax
    [8000, 1.5e-4],     # UPGRADED: 6K→8K maintain for extended sharpness
    [14000, 1e-4],    # Final refinement
]

# Critic LR Schedule (for Q-value learning)
# UPGRADED: Boosted LR during crisis phase (iter 2K-6K) for faster intrusion learning
MAIN_CRITIC_LR_SCHEDULE = [
    [0, 3e-4],        # Aggressive penalty structure learning
    [2000, 5e-4],     # UPGRADED: 2e-4→5e-4 (2.5x boost during crisis phase)
    [6000, 2e-4],     # Back to moderate after collision mastery
    [8000, 1e-4],     # Stabilization
    [14000, 5e-5],    # Minimal noise in final Q-values
]

# --- Early Stopping Parameters ---
ENABLE_EARLY_STOPPING = True    # Set to False to disable early stopping
EARLY_STOP_PATIENCE = 18001        # Number of iterations without improvement before stopping
EARLY_STOP_MIN_DELTA = 0.5      # Minimum improvement in smoothed reward to count as progress
EARLY_STOP_USE_SMOOTHED = True  # Use moving average of last 5 rewards for stability

# --- Final Model Saving ---
SAVE_FINAL_MODEL = True  # Set cto True to save the model from the last iteration as "final_model"

# --- Metrics Directory ---
# When copying to a new folder, update this to match your folder name!

script_dir = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.join(script_dir, "metrics")

# --- Path for model ---
CHECKPOINT_DIR = os.path.join(script_dir, "models/sectorcr_ma_sac")

# --- Expert Buffer Storage (for mixing during training) ---
expert_buffer_storage = []  # Global list to store expert demonstrations separately

# --- Global Curriculum State (shared across all environments) ---
# This allows environments to dynamically read the current intrusion penalty
CURRENT_INTRUSION_PENALTY = -100  # Start with Stage 1 (curriculum)

# --- INTRUSION PENALTY CURRICULUM SCHEDULE ---
# 3-stage progressive penalty increase to balance waypoint learning + intrusion avoidance
# REVISED: 4-stage curriculum with delayed Stage 3
INTRUSION_CURRICULUM_STAGES = [
    (0, -100),       # Stage 1 (0-4K): Learn navigation
    (4000, -150),    # Stage 2 (4K-8K): Light collision awareness  
    (8000, -250),    # Stage 3 (8K-12K): Moderate penalties - KEEP PRACTICING
    (11000, -350),   # Stage 4 (12K-16K): Full penalties AFTER attention sharpens
]

def apply_dynamic_learning_rates(algorithm, iteration):
    """
    Apply separate learning rates for actor base, attention/temperature, and critic
    based on the current training iteration.
    
    Args:
        algorithm: The RLlib algorithm instance
        iteration: Current training iteration
    """
    # Helper function to interpolate learning rate from schedule
    def get_lr_from_schedule(schedule, iteration):
        """Linear interpolation between schedule points"""
        if iteration <= schedule[0][0]:
            return schedule[0][1]
        if iteration >= schedule[-1][0]:
            return schedule[-1][1]
        
        for i in range(len(schedule) - 1):
            iter_start, lr_start = schedule[i]
            iter_end, lr_end = schedule[i + 1]
            if iter_start <= iteration <= iter_end:
                # Linear interpolation
                progress = (iteration - iter_start) / (iter_end - iter_start)
                return lr_start + (lr_end - lr_start) * progress
        return schedule[-1][1]
    
    # Get target learning rates (actor is constant, others from schedules)
    actor_lr = get_lr_from_schedule(MAIN_ACTOR_LR_SCHEDULE, iteration)
    attention_lr = get_lr_from_schedule(MAIN_ATTENTION_LR_SCHEDULE, iteration)
    temperature_lr = get_lr_from_schedule(MAIN_TEMPERATURE_LR_SCHEDULE, iteration)
    critic_lr = get_lr_from_schedule(MAIN_CRITIC_LR_SCHEDULE, iteration)
    
    # Debug: Print LRs for first few iterations to verify
    if iteration <= 5 or iteration == 1000 or iteration == 1001 or iteration == 2000:
        print(f"   [LR Debug] Iter {iteration}: Actor={actor_lr:.2e}, Attention={attention_lr:.2e}, Temp={temperature_lr:.2e}, Critic={critic_lr:.2e}")
    
    policy = algorithm.get_policy("shared_policy")
    
    if hasattr(policy, 'optimizers'):
        for idx, opt in enumerate(policy.optimizers()):
            if idx == 0:  # Actor optimizer
                for group in opt.param_groups:
                    # Identify parameter group type
                    params_in_group = set(group['params'])
                    
                    # Check if this group contains temperature parameter (most specific)
                    is_temperature_group = any(
                        "temperature" in n.lower()
                        for n, p in policy.model.named_parameters() 
                        if p in params_in_group
                    )
                    
                    # Check if this group contains other attention params (W_q, W_k, W_v)
                    is_attention_group = any(
                        ("attention" in n.lower() or "w_q" in n.lower() or "w_k" in n.lower() or "w_v" in n.lower())
                        for n, p in policy.model.named_parameters() 
                        if p in params_in_group
                    )
                    
                    if is_temperature_group:
                        group['lr'] = temperature_lr
                    elif is_attention_group:
                        group['lr'] = attention_lr
                    else:
                        group['lr'] = actor_lr
            else:  # Critic optimizers (idx >= 1)
                for group in opt.param_groups:
                    group['lr'] = critic_lr

        # Diagnostic: print actual optimizer param groups and parameter name samples
        if iteration <= 10 or iteration in (1000, 1001, 2000, 2001):
            try:
                name_map = {p: n for n, p in policy.model.named_parameters()}
            except Exception:
                name_map = {}
            if hasattr(policy, 'optimizers'):
                for idx, opt in enumerate(policy.optimizers()):
                    print(f"   [Opt Debug] optimizer {idx}:")
                    for gi, group in enumerate(opt.param_groups):
                        lr = group.get('lr', None)
                        param_names = []
                        for p in list(group.get('params', []))[:6]:
                            param_names.append(name_map.get(p, '<unnamed>'))
                        print(f"      group[{gi}].lr={lr:.3e} params={param_names}")
    
    # Return current LRs for logging
    return {
        'actor_lr': actor_lr,
        'attention_lr': attention_lr,
        'temperature_lr': temperature_lr,
        'critic_lr': critic_lr
    }

class ForceAlphaCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.episode_count = 0  # Track episode count for debug
        self.success_streak = 0
        # For performance-gated burn-in LR reduction
        self.wp_history = []
        self.gated_lr_triggered = False
        self.gated_lr_threshold = 0.6  # Trigger when smoothed WP >= 60%
        self.gated_lr_window = 20      # Moving average window (episodes)
        self.gated_lr_factor = 0.5     # Multiply actor LR by this factor when triggered
        
    def on_episode_end(self, *args, **kwargs):
        """Track custom metrics per episode: waypoint reach rate and intrusions"""
        self.episode_count += 1
        episode = kwargs.get("episode") or (len(args) > 3 and args[3])
        worker = kwargs.get("worker")
        
        # if episode is None:
        #     if self.episode_count <= 3:
        #         print(f"   [Callback Debug {self.episode_count}] episode is None")
        #     return
        
        # Try to access environment from worker
        env = None
        if worker is not None and hasattr(worker, 'env'):
            env = worker.env
        else:
            env = kwargs.get("env")
        
        if env is not None:
            # Get the base environment (unwrap if needed)
            base_env = env
            while hasattr(base_env, 'env'):
                base_env = base_env.env
            
            # Extract episode-level metrics from environment
            waypoints_reached = len(base_env.waypoint_reached_agents) if hasattr(base_env, 'waypoint_reached_agents') else 0
            total_intrusions = base_env.total_intrusions if hasattr(base_env, 'total_intrusions') else 0
            n_agents = base_env.num_ac if hasattr(base_env, 'num_ac') else 20
            
            # Calculate waypoint success rate for this episode
            waypoint_rate = waypoints_reached / n_agents if n_agents > 0 else 0.0
            
            # Debug output for first few episodes
            # if self.episode_count <= 3:
            #     print(f"   [Callback Debug {self.episode_count}] WP: {waypoint_rate:.2f}, Intrusions: {total_intrusions}")
            
            # Log metrics using the new API
            metrics_logger = kwargs.get("metrics_logger")
            if metrics_logger is not None:
                metrics_logger.log_value("waypoint_rate", float(waypoint_rate))
                metrics_logger.log_value("intrusions", float(total_intrusions))
                # if self.episode_count <= 3:
                #     print(f"   [Callback Debug {self.episode_count}] Logged via metrics_logger")
            else:
                # Fallback for older API
                if hasattr(episode, "custom_metrics"):
                    episode.custom_metrics["waypoint_rate"] = float(waypoint_rate)
                    episode.custom_metrics["intrusions"] = float(total_intrusions)
                    # if self.episode_count <= 3:
                    #     print(f"   [Callback Debug {self.episode_count}] Logged via episode.custom_metrics")
                # else:
                #     if self.episode_count <= 3:
                #         print(f"   [Callback Debug {self.episode_count}] WARNING: No way to log metrics!")
    
    def on_train_result(self, *, algorithm, result, **kwargs):
        global CURRENT_INTRUSION_PENALTY  # Allow modification of global variable
        current_iter = result["training_iteration"]

        # ========================================
        # 0. APPLY DYNAMIC LEARNING RATES
        # ========================================
        lr_info = apply_dynamic_learning_rates(algorithm, current_iter)
        result["custom_metrics"]["actor_lr"] = lr_info['actor_lr']
        result["custom_metrics"]["attention_lr"] = lr_info['attention_lr']
        result["custom_metrics"]["temperature_lr"] = lr_info['temperature_lr']
        result["custom_metrics"]["critic_lr"] = lr_info['critic_lr']

        # ========================================
        # 1. CURRICULUM LEARNING: INTRUSION PENALTY
        # ========================================
        # 3-stage progressive penalty: -100 → -175 → -250
        # This prevents policy collapse by learning navigation first, then adding collision avoidance
        
        # Determine current stage based on iteration
        target_intrusion_penalty = INTRUSION_CURRICULUM_STAGES[0][1]  # Default to Stage 1
        stage_name = "Stage 1 (Navigation)"
        stage_number = 1
        
        for i, (threshold_iter, penalty) in enumerate(INTRUSION_CURRICULUM_STAGES):
            if current_iter >= threshold_iter:
                target_intrusion_penalty = penalty
                stage_number = i + 1
                if i == 0:
                    stage_name = "Stage 1 (Navigation)"
                elif i == 1:
                    stage_name = "Stage 2 (Collision Awareness)"
                elif i == 2:
                    stage_name = "Stage 3 (Mastery)"
        
        # Update global penalty (for new environments)
        CURRENT_INTRUSION_PENALTY = target_intrusion_penalty
        
        # Update all active environments (for running workers)
        try:
            # Access env_runner workers and update their environments
            if hasattr(algorithm, 'env_runner_group') and algorithm.env_runner_group is not None:
                for worker in algorithm.env_runner_group.remote_workers():
                    worker.foreach_env.remote(lambda env: env.update_intrusion_penalty(target_intrusion_penalty))
        except Exception:
            pass  # Silently continue if environment doesn't support dynamic updates
        
        # Log curriculum state
        result["custom_metrics"]["intrusion_penalty"] = float(target_intrusion_penalty)
        result["custom_metrics"]["curriculum_stage"] = float(stage_number)
        
        # Print stage transitions (only when changing)
        if current_iter in [s[0] for s in INTRUSION_CURRICULUM_STAGES if s[0] > 0]:
            print(f"\n🎯 CURRICULUM STAGE CHANGE at iteration {current_iter}")
            print(f"   {stage_name}: Intrusion Penalty = {target_intrusion_penalty}")
            print(f"   Goal: {'Learn waypoint navigation' if stage_number == 1 else 'Add collision awareness' if stage_number == 2 else 'Master both skills'}\n")
        
        # Get current performance metrics
        custom_metrics_top = result.get("custom_metrics", {})
        custom_metrics_env = result.get("env_runners", {}).get("custom_metrics", {})
        custom_metrics = {**custom_metrics_top, **custom_metrics_env}
        
        # Try both possible keys for waypoint rate
        wp_rate = custom_metrics.get("waypoint_rate_mean", 
                                     custom_metrics.get("waypoint_rate", 0.0))

        # --- PERFORMANCE-GATED BURN-IN LR REDUCTION ---
        # Maintain a short moving-average of waypoint_rate (episodes) and
        # trigger a one-time actor-LR reduction when smoothed WP >= threshold
        try:
            self.wp_history.append(float(wp_rate))
        except Exception:
            pass

        if len(self.wp_history) > 0:
            smoothed_wp = float(np.mean(self.wp_history[-self.gated_lr_window:]))
        else:
            smoothed_wp = 0.0
        result["custom_metrics"]["waypoint_rate_smoothed"] = float(smoothed_wp)

        # Only apply gating once and only during burn-in period
        burnin_limit = globals().get('BURN_IN_ITERATIONS', TOTAL_ITERS)
        if (not self.gated_lr_triggered) and (current_iter <= burnin_limit) and (smoothed_wp >= self.gated_lr_threshold):
            try:
                policy = algorithm.get_policy("shared_policy")
                if hasattr(policy, 'optimizers'):
                    for idx, opt in enumerate(policy.optimizers()):
                        if idx == 0:  # Actor optimizer groups
                            for group in opt.param_groups:
                                params_in_group = set(group.get('params', []))
                                is_temperature_group = any(
                                    "temperature" in n.lower()
                                    for n, p in policy.model.named_parameters()
                                    if p in params_in_group
                                )
                                is_attention_group = any(
                                    ("attention" in n.lower() or "w_q" in n.lower() or "w_k" in n.lower() or "w_v" in n.lower())
                                    for n, p in policy.model.named_parameters()
                                    if p in params_in_group
                                )
                                # Only reduce base-actor groups (not attention/temperature)
                                if not is_temperature_group and not is_attention_group:
                                    old_lr = group.get('lr', None)
                                    if old_lr is not None:
                                        group['lr'] = old_lr * self.gated_lr_factor
                self.gated_lr_triggered = True
                print(f"\n🔒 Performance-gated LR reduction triggered at iter {current_iter}: smoothed WP={smoothed_wp:.3f}. Actor LR x{self.gated_lr_factor}\n")
                result["custom_metrics"]["gated_lr_triggered"] = 1.0
            except Exception:
                # Don't let this crash training; just continue
                pass
        else:
            # Ensure metric exists
            result["custom_metrics"].setdefault("gated_lr_triggered", 0.0)

        # --- 2. BETA ANNEALING (van 0.4 naar 1.0) ---
        progress = min(1.0, current_iter / TOTAL_ITERS)
        new_beta = 0.4 + (1.0 - 0.4) * progress
        if hasattr(algorithm.local_replay_buffer, "prioritized_replay_beta"):
            algorithm.local_replay_buffer.prioritized_replay_beta = new_beta
        elif hasattr(algorithm.local_replay_buffer, "beta"):
            algorithm.local_replay_buffer.beta = new_beta

        # --- 3. LOGGING VOOR MONITORING ---
        result["custom_metrics"]["per_beta"] = float(new_beta)

        # ========================================
        # 2. ALPHA (ENTROPY) SCHEDULE - EXTENDED FOR 14K ITERATIONS
        # ========================================
        FREEZE_UNTIL = 2000   # Iteration where Actor wakes up
        DECAY_UNTIL  = 14000  # Extended decay window for longer exploration

        # Define target values - SLOWER DECAY for longer exploration
        ALPHA_FREEZE = 0.05   # Safe start to keep expert behavior (~5% randomness)
        ALPHA_FINAL  = 0.008  # UPGRADED: 0.01→0.015 (50% more exploration)

        if current_iter < FREEZE_UNTIL:
            # PHASE 1: FREEZE (Safety First)
            # Keep alpha low while Critic learns penalty structure
            target_alpha = ALPHA_FREEZE

        elif current_iter < DECAY_UNTIL:
            # PHASE 2: REFINEMENT (Linear Decay)
            # Smoothly reduce randomness to eliminate jittering into neighbors
            progress = (current_iter - FREEZE_UNTIL) / (DECAY_UNTIL - FREEZE_UNTIL)
            target_alpha = ALPHA_FREEZE - (ALPHA_FREEZE - ALPHA_FINAL) * progress

        else:
            # PHASE 3: STABLE
            # Run almost deterministically for maximum safety
            target_alpha = ALPHA_FINAL


        
        # Beta van de Prioritized Replay Experience
        BETA_START = 0.4
        BETA_END = 1.0
        # calculate new beta    
        progress_beta = min(1.0, current_iter/TOTAL_ITERS)
        new_beta = BETA_START + (BETA_END - BETA_START)*progress_beta
        # update buffer parameters
        if hasattr(algorithm, "local_replay_buffer"):
            algorithm.local_replay_buffer.prioritized_replay_beta = new_beta
            result["custom_metrics"]["prioritized_replay_beta"]=new_beta
            
        # # Voor een run van 12.000
        # START_ALPHA = 0.15
        # END_ALPHA = 0.04
        # FREEZE_UNTIL = 1000 
        # TOTAL_ITERS_VAL = 6000  # <--- Verlengd naar 12k
        
        # if current_iter < FREEZE_UNTIL:
        #     target_alpha = START_ALPHA
        # else:
        #     decay_steps = TOTAL_ITERS_VAL - FREEZE_UNTIL
        #     progress = min(1.0, (current_iter - FREEZE_UNTIL) / decay_steps)
        #     target_alpha = START_ALPHA * (END_ALPHA / START_ALPHA) ** progress  

        # 3. Apply the force (Same logic as before)
        target_log_alpha = np.log(target_alpha)
        policy = algorithm.get_policy("shared_policy")
        
        
        
        alpha_param = None
        if hasattr(policy, "model") and hasattr(policy.model, "log_alpha"):
            alpha_param = policy.model.log_alpha
        elif hasattr(policy, "log_alpha"):
            alpha_param = policy.log_alpha
            
        if alpha_param is not None:
            with torch.no_grad():
                alpha_param.fill_(target_log_alpha)
            
            # 4. Log it!
            # This is super important so you can verify the decay in TensorBoard
            result["custom_metrics"]["forced_alpha"] = target_alpha
        
        # 5. FIX FOR NaN ENTROPY: Clamp log_std to prevent numerical instability
        # This prevents log_std from becoming too negative, which causes entropy calculation to fail
        try:
            # Access the actor model (policy network)
            actor_model = None
            if hasattr(policy, 'model'):
                # For SAC, the policy model might wrap the action model
                if hasattr(policy.model, 'action_model'):
                    actor_model = policy.model.action_model
                elif not hasattr(policy.model, 'is_critic') or not policy.model.is_critic:
                    actor_model = policy.model
            
            # If the actor model has log_std parameters, clamp them
            if actor_model is not None:
                # Check for log_std as a parameter (common in SAC)
                if hasattr(actor_model, 'log_std'):
                    with torch.no_grad():
                        # Clamp to reasonable range:
                        # log_std ∈ [-5, 2] → std ∈ [0.0067, 7.39]
                        # This prevents both: 
                        # - Too deterministic (log_std < -5 causes entropy NaN)
                        # - Too random (log_std > 2 is excessive exploration)
                        actor_model.log_std.clamp_(-5.0, 2.0)
                        
                        # Log current std for monitoring (only occasionally to avoid spam)
                        if current_iter % 100 == 1:
                            current_log_std = actor_model.log_std.detach().cpu().numpy()
                            current_std = np.exp(current_log_std)
                            result["custom_metrics"]["policy_log_std_mean"] = float(np.mean(current_log_std))
                            result["custom_metrics"]["policy_std_mean"] = float(np.mean(current_std))
        except Exception as e:
            # Silently continue if clamping fails
            pass



class SACExpert:
    def __init__(self, safe_dist_m=100.0, lookahead_s=15.0):
        # We gebruiken de MVP logica die ook in Stage 1 PPO werkte
        self.mvp = MVP_2D(safe_distance=safe_dist_m, lookahead_time=lookahead_s)

    def get_action(self, env, agent_id):
        try:
            ac_idx = bs.traf.id2idx(agent_id)
            # 1. Huidige staat (meters t.o.v. center)
            pos = env.compute_relative_position(env.center, bs.traf.lat[ac_idx], bs.traf.lon[ac_idx])
            vel = np.array([
                np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * bs.traf.gs[ac_idx],
                np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * bs.traf.gs[ac_idx]
            ])

            # 2. Buren verzamelen
            neighbors = []
            for other in env.agents:
                if other == agent_id: continue
                o_idx = bs.traf.id2idx(other)
                neighbors.append({
                    'pos': env.compute_relative_position(env.center, bs.traf.lat[o_idx], bs.traf.lon[o_idx]),
                    'vel': np.array([np.cos(np.deg2rad(bs.traf.hdg[o_idx])) * bs.traf.gs[o_idx],
                                   np.sin(np.deg2rad(bs.traf.hdg[o_idx])) * bs.traf.gs[o_idx]])
                })

            # 3. MVP Berekening
            target_vel = self.mvp.calculate_avoidance_velocity(pos, vel, neighbors)
            
            # Indien geen conflict: direct naar waypoint (zoals in PPO Stage 1)
            if np.array_equal(target_vel, vel):
                wpt_lat, wpt_lon = env.agent_waypoints[agent_id]
                wpt_qdr, _ = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], wpt_lat, wpt_lon)
                target_hdg, target_spd = wpt_qdr, 35.0 
            else:
                target_hdg = np.rad2deg(np.arctan2(target_vel[1], target_vel[0]))
                target_spd = np.linalg.norm(target_vel) * 1.94384 # m/s -> kts
            
            # 4. Schalen naar de [-1, 1] actie-ruimte van SectorEnv
            dh = fn.bound_angle_positive_negative_180(target_hdg - bs.traf.hdg[ac_idx])
            dv = target_spd - (bs.traf.tas[ac_idx] * 1.94384)
            
            # Schaling: D_HEADING=45, D_VELOCITY=3.33 (10/3)
            return np.array([np.clip(dh / 45.0, -1, 1), np.clip(dv / (10/3), -1, 1)], dtype=np.float32)
        except Exception:
            return np.array([0.0, 0.0], dtype=np.float32)

def prefill_sac_buffer(algo, n_episodes=30):
    print(f"\n🚀 Start Buffer Pre-fill met {n_episodes} expert episodes...")
    expert = SACExpert()
    # Maak lokale omgeving aan om data te genereren
    env = SectorEnv(n_agents=20, run_id="prefill") 
    
    total_samples, waypoints_hit, intrusions_total = 0, 0, 0
    episode_id = 0  # Track episode IDs
    
    # --- DYNAMISCHE PRIORITISERING LOGICA ---
    HEADING_THRESHOLD = 0.06  # Drempelwaarde voor 'actieve' stuuracties
    OVERSAMPLE_FACTOR = 2    # Hoeveel vaker we stuuracties opslaan
    maneuvers_count = 0       # Teller voor unieke stuurmomenten
    # ----------------------------------------

    for ep in range(n_episodes):
        episode_id += 1
        timestep = 0  
        obs, _ = env.reset()
        while env.agents:
            timestep += 1
            active_agents = set(obs.keys())
            agent_actions = {aid: expert.get_action(env, aid) for aid in active_agents}
            current_obs_snapshot = {aid: o.copy() for aid, o in obs.items()}
            
            obs, rewards, terms, truncs, infos = env.step(agent_actions)
            
            for aid, action in agent_actions.items():
                if aid not in current_obs_snapshot:
                    continue
                
                next_obs = obs.get(aid, np.zeros_like(current_obs_snapshot[aid]))
                if terms.get(aid, False) and not np.allclose(next_obs, 0.0):
                    next_obs = np.zeros_like(current_obs_snapshot[aid])
                
                batch = SampleBatch({
                    SampleBatch.OBS: [current_obs_snapshot[aid]],
                    SampleBatch.ACTIONS: [action],
                    SampleBatch.REWARDS: [rewards.get(aid, 0.0)], 
                    SampleBatch.NEXT_OBS: [next_obs],
                    SampleBatch.TERMINATEDS: [terms.get(aid, False)],
                    SampleBatch.TRUNCATEDS: [truncs.get(aid, False)],
                    SampleBatch.INFOS: [infos.get(aid, {})],
                    "weights": [1.0],  # Prioritized Experience Replay startgewicht
                    SampleBatch.EPS_ID: [episode_id],
                    SampleBatch.AGENT_INDEX: [aid],
                    SampleBatch.UNROLL_ID: [episode_id],
                    SampleBatch.T: [timestep],
                })
                
                ma_batch = MultiAgentBatch({"shared_policy": batch}, env_steps=1)

                # --- LOGICA: OVERSAMPLING VOOR BEHAVIOR CLONING BALANS ---
                # Check of de expert een stuuractie (heading) onderneemt
                is_stuur_actie = abs(action[0]) > HEADING_THRESHOLD
                repeat_count = OVERSAMPLE_FACTOR if is_stuur_actie else 1
                
                if is_stuur_actie:
                    maneuvers_count += 1

                # Voeg de ervaring herhaaldelijk toe aan de buffers
                for _ in range(repeat_count):
                    # 1. Direct in de replay buffer voor de burn-in
                    algo.local_replay_buffer.add(ma_batch)
                    
                    # 2. In de expert storage voor mixing tijdens actieve training
                    expert_buffer_storage.append(ma_batch)
                    
                    total_samples += 1
                # ------------------------------------------------------
        
        waypoints_hit += len(env.waypoint_reached_agents)
        intrusions_total += env.total_intrusions
        
        if (ep + 1) % 10 == 0:
            avg_wp = waypoints_hit / (ep + 1)
            avg_intr = intrusions_total / (ep + 1)
            print(f"   Ep {ep+1}/{n_episodes} | Samples: {total_samples} | Avg WP: {avg_wp:.2f} | Avg Intr: {avg_intr:.2f}")
            
    print(f"✅ Buffer gevuld met {total_samples} samples.")
    print(f"   Focus: {maneuvers_count} unieke stuuracties zijn {OVERSAMPLE_FACTOR}x vaker opgeslagen.")
    print(f"   Expert WP Success Rate: {(waypoints_hit/(n_episodes*20))*100:.1f}%")
    print(f"   Expert Totaal Intrusies: {intrusions_total}")
    
    try:
        buffer_size = algo.local_replay_buffer._num_added
        print(f"   Buffer size after pre-fill: {buffer_size}")
    except AttributeError:
        pass
    print()
    env.close()

def calculate_expert_mix_ratio(iteration, total_iters):
    """
    Calculate the expert mixing ratio with linear decay.
    
    Args:
        iteration: Current training iteration (0-indexed after burn-in)
        total_iters: Total number of training iterations
        
    Returns:
        float: Ratio of expert data to sample (0.0 to EXPERT_MIX_START)
    """
    # Calculate the iteration at which mixing should end
    mix_end_iter = int(total_iters * EXPERT_MIX_DECAY_UNTIL)
    
    if iteration >= mix_end_iter:
        return EXPERT_MIX_END
    
    # Linear decay from START to END
    progress = iteration / mix_end_iter
    ratio = EXPERT_MIX_START + (EXPERT_MIX_END - EXPERT_MIX_START) * progress
    
    return ratio

def inject_expert_samples_before_training(algo, expert_ratio, batch_size=4096):
    """
    Before training step, inject expert samples into the replay buffer based on ratio.
    This ensures that when algo.train() samples batches, they will include expert data.
    
    Args:
        algo: The SAC algorithm instance
        expert_ratio: Fraction of expert data to include (0.0 to 1.0)
        batch_size: Size of training batch (used to determine how many expert samples to inject)
    """
    if expert_ratio <= 0.0 or len(expert_buffer_storage) == 0:
        return 0
    
    # Calculate how many expert samples to inject
    # We inject them into the buffer so they get sampled naturally
    n_expert_samples = int(batch_size * expert_ratio)
    n_expert_samples = min(n_expert_samples, len(expert_buffer_storage))
    
    if n_expert_samples > 0:
        # Randomly sample from expert buffer and add to replay buffer
        import random
        expert_samples = random.sample(expert_buffer_storage, n_expert_samples)
        for sample in expert_samples:
            algo.local_replay_buffer.add(sample)
    
    return n_expert_samples

def burn_in_on_expert_buffer(algo, n_iterations=2000, batch_size=4096):
    print(f"\n🔥 Starting Burn-in Phase: {n_iterations} iterations on expert buffer...")
    print(f"   Goal: Calibrate Critic to recognize expert-level Q-values")
    if ENABLE_KL_PENALTY:
        print(f"   🔗 KL Penalty ENABLED: β={KL_COEFF_START:.2f}→{KL_COEFF_END:.2f} (anneal until iter {KL_ANNEAL_UNTIL})")
        print(f"      Keeps policy close to expert behavior during burn-in")
    
    original_lr = None
    original_lrs = None

    policy = algo.get_policy("shared_policy")
    
    # 1. Zoek de juiste SAC Alpha parameter (NIET de model temperature)
    alpha_param = None
    if hasattr(policy, "model") and hasattr(policy.model, "log_alpha"):
        alpha_param = policy.model.log_alpha
    elif hasattr(policy, "log_alpha"):
        alpha_param = policy.log_alpha
        
    # Sla de originele log_alpha op om te kunnen herstellen
    original_log_alpha = alpha_param.detach().item() if alpha_param is not None else np.log(START_ALPHA)

    # --- BEST MODEL TRACKING (Anti-Collapse Mechanism) ---
    best_wp_rate = -1.0 
    best_intrusions = float('inf')  # Track intrusions for models with >80% WP
    best_weights = None
    best_iteration = 0
    wp_history = []  # Voor moving average (smoothing)
    print(f"   🛡️  Anti-collapse protection enabled: tracking best model")
    print(f"   Strategy: WP >80% → minimize intrusions, else maximize WP rate")
    
    

    if hasattr(policy, 'optimizers'):
        for idx, opt in enumerate(policy.optimizers()):
            # 1. De Actor/Policy optimizer (meestal index 0)
            if idx == 0:
                for group in opt.param_groups:
                    # We identificeren parameters op basis van hun naam in de model-state
                    params_in_group = set(group['params'])
                    
                    # Check for temperature parameter first (most specific)
                    is_temperature_group = any(
                        "temperature" in n.lower()
                        for n, p in policy.model.named_parameters() 
                        if p in params_in_group
                    )
                    
                    # Check for other attention parameters (W_q, W_k, W_v)
                    is_attention_group = any(
                        ("attention" in n.lower() or "w_q" in n.lower() or "w_k" in n.lower() or "w_v" in n.lower()) 
                        for n, p in policy.model.named_parameters() 
                        if p in params_in_group
                    )
                    
                    if is_temperature_group:
                        group['lr'] = BURN_IN_TEMPERATURE_LR_LOCAL
                        label = "TEMPERATURE"
                    elif is_attention_group:
                        group['lr'] = BURN_IN_ATTENTION_LR_LOCAL
                        label = "ATTENTION"
                    else:
                        group['lr'] = BURN_IN_ACTOR_LR_LOCAL
                        label = "ACTOR-BASE"
                
                print(f"   ✅ Actor LRs aangepast: {label} groepen herkend.")
            
            # 2. De Critic optimizers (index 1 en verder)
            else:
                for group in opt.param_groups:
                    group['lr'] = BURN_IN_CRITIC_LR_LOCAL
                print(f"   ✅ Critic {idx} set to {BURN_IN_CRITIC_LR_LOCAL:.2e}")
    
    mean_q_history = []
    critic_loss_history = []
    actor_loss_history = []
    actual_alpha_history = []
    attention_sharpness_history = []  # Track attention sharpness
    mse_history = []  # Track MSE (from critic loss as proxy)
    kl_loss_history = []  # Track KL penalty loss
    kl_coeff_history = []  # Track KL coefficient over time
    
    # Track episode metrics during burn-in
    episode_waypoint_rates = []  # Waypoint success rate per episode
    episode_intrusions = []      # Total intrusions per episode
    episode_rewards = []         # Average episode rewards
    
    # Run a few evaluation episodes every N iterations to track learning progress
    EVAL_INTERVAL = 100  # Evaluate every 100 burn-in iterations
    N_EVAL_EPISODES = 10  # Use 5 episodes per evaluation (quick check)
    
    # Helper function: fixed burn-in actor LR
    def get_actor_lr_for_burnin(iteration):
        return BURN_IN_ACTOR_LR_LOCAL
    
    # KL penalty coefficient annealing schedule
    def get_kl_coeff(iteration):
        """Linearly anneal KL coefficient from start to end"""
        if not ENABLE_KL_PENALTY:
            return 0.0
        if iteration >= KL_ANNEAL_UNTIL:
            return KL_COEFF_END
        progress = iteration / KL_ANNEAL_UNTIL
        return KL_COEFF_START - (KL_COEFF_START - KL_COEFF_END) * progress
    
    for i in range(1, n_iterations + 1):
        # Apply dynamic actor LR based on burn-in schedule
        current_actor_lr = get_actor_lr_for_burnin(i)
        if hasattr(policy, 'optimizers'):
            for idx, opt in enumerate(policy.optimizers()):
                if idx == 0:  # Actor optimizer
                    for group in opt.param_groups:
                        params_in_group = set(group['params'])
                        
                        # Check parameter type (same logic as initial setup)
                        is_temperature_group = any(
                            "temperature" in n.lower()
                            for n, p in policy.model.named_parameters() 
                            if p in params_in_group
                        )
                        
                        is_attention_group = any(
                            ("attention" in n.lower() or "w_q" in n.lower() or "w_k" in n.lower() or "w_v" in n.lower()) 
                            for n, p in policy.model.named_parameters() 
                            if p in params_in_group
                        )
                        
                        # Apply scheduled LR only to base actor params
                        if is_temperature_group:
                            group['lr'] = BURN_IN_TEMPERATURE_LR_LOCAL
                        elif is_attention_group:
                            group['lr'] = BURN_IN_ATTENTION_LR_LOCAL
                        else:
                            group['lr'] = current_actor_lr  # Use scheduled LR
        
        batch = algo.local_replay_buffer.sample(batch_size)
        
        if isinstance(batch, MultiAgentBatch):
            train_batch = batch.policy_batches.get("shared_policy")
        else:
            train_batch = batch

        if train_batch is None or len(train_batch) == 0:
            continue

        # --- 2. SAC ALPHA ONDERDRUKKING (Voor pure imitatie) ---
        if alpha_param is not None:
            with torch.no_grad():
                # We zetten SAC Alpha op 0.01 (log(0.01) ≈ -4.6)
                alpha_param.fill_(np.log(0.01))
        
        # --- 3. KL PENALTY: Add imitation loss to keep policy close to expert ---
        current_kl_coeff = get_kl_coeff(i)
        kl_loss_value = 0.0
        
        if ENABLE_KL_PENALTY and current_kl_coeff > 0.0:
            try:
                # Get expert actions from batch
                expert_actions = torch.FloatTensor(train_batch['actions'])
                obs = torch.FloatTensor(train_batch['obs'])
                
                # Get current policy's action distribution
                with torch.no_grad():
                    policy_actions = policy.compute_actions(train_batch['obs'], explore=False)[0]
                    policy_actions = torch.FloatTensor(policy_actions)
                
                # Compute MSE between policy and expert actions as KL proxy
                # (For Gaussian policies, minimizing MSE ≈ maximizing log prob of expert actions)
                kl_loss = torch.mean((policy_actions - expert_actions) ** 2)
                kl_loss_value = kl_loss.item()
                
                # Add penalty to actor loss by manually adjusting policy gradients
                # We do this by adding a small "fake reward" bonus for matching expert actions
                # Convert KL penalty to reward bonus: more matching = higher reward
                reward_bonus = -current_kl_coeff * kl_loss_value
                
                # Apply reward shaping to batch (add bonus to rewards)
                train_batch['rewards'] = train_batch['rewards'] + reward_bonus
                
            except Exception as e:
                print(f"[WARN] KL penalty computation failed at iter {i}: {e}")
                kl_loss_value = 0.0
        
        # --- DIAGNOSTIC: Compute actions and track statistics ---
        action_mean, action_std, action_min, action_max = None, None, None, None
        reward_mean, reward_std, reward_min, reward_max = None, None, None, None
        if i % 100 == 0 or i <= 3:
            try:
                # Sample a subset for efficiency
                sample_obs = train_batch['obs'][:100]
                actions_diagnostic = policy.compute_actions(sample_obs, explore=False)[0]
                actions_tensor = torch.FloatTensor(actions_diagnostic)
                action_mean = actions_tensor.mean().item()
                action_std = actions_tensor.std().item()
                action_min = actions_tensor.min().item()
                action_max = actions_tensor.max().item()
                
                # Analyze reward distribution
                rewards = train_batch['rewards']
                reward_mean = float(np.mean(rewards))
                reward_std = float(np.std(rewards))
                reward_min = float(np.min(rewards))
                reward_max = float(np.max(rewards))
            except Exception as e:
                pass  # Skip diagnostics if error
        
        # 3. Voer de update uit (but manually to capture gradients)
        # For gradient diagnostics, we need to manually do forward + backward
        compute_grads = (i % 100 == 0 or i <= 3)
        grad_norms = {'actor_base': 0.0, 'attention': 0.0, 'temperature': 0.0, 'critic': 0.0}
        
        if compute_grads:
            # Capture gradients BEFORE optimizer step
            # Save current state
            saved_weights = {k: v.clone() for k, v in policy.model.state_dict().items()}
        
        train_results = policy.learn_on_batch(train_batch)
        learner_stats = train_results.get('learner_stats', {})
        
        # --- DIAGNOSTIC: Compute parameter changes (after update) ---
        if compute_grads:
            param_changes = {}
            for k, v in policy.model.state_dict().items():
                if k in saved_weights:
                    diff = (v - saved_weights[k]).norm().item()
                    param_changes[k] = diff
            
            # Categorize changes
            actor_base_change = sum(v for k, v in param_changes.items() 
                                   if 'attention' not in k.lower() and 'temperature' not in k.lower() 
                                   and 'w_q' not in k.lower() and 'w_k' not in k.lower() and 'w_v' not in k.lower())
            attention_change = sum(v for k, v in param_changes.items() 
                                  if 'attention' in k.lower() or 'w_q' in k.lower() or 'w_k' in k.lower() or 'w_v' in k.lower())
            temperature_change = sum(v for k, v in param_changes.items() if 'temperature' in k.lower())
            
            grad_norms['actor_base'] = actor_base_change
            grad_norms['attention'] = attention_change
            grad_norms['temperature'] = temperature_change
        
        # --- 4. HERSTEL ORIGINELE ALPHA ---
        if alpha_param is not None:
            with torch.no_grad():
                alpha_param.fill_(original_log_alpha)
        
        # Convert metrics to scalars
        def to_scalar(val):
            if isinstance(val, (list, tuple, np.ndarray)):
                if len(val) > 0:
                    clean_val = np.array(val)
                    clean_val = clean_val[~np.isnan(clean_val)]
                    if len(clean_val) > 0:
                        return float(np.mean(clean_val))
                return float('nan')
            return float(val) if not isinstance(val, str) else float('nan')
        
        mean_q = to_scalar(learner_stats.get('mean_q', float('nan')))
        critic_loss = to_scalar(learner_stats.get('critic_loss', float('nan')))
        actor_loss = to_scalar(learner_stats.get('actor_loss', float('nan')))
        alpha_value = to_scalar(learner_stats.get('alpha_value', float('nan')))
        # Record the actual used alpha for this update (scalar)
        actual_used_alpha = alpha_value
        actual_alpha_history.append(actual_used_alpha)
        
        # Track history
        mean_q_history.append(mean_q)
        critic_loss_history.append(critic_loss)
        actor_loss_history.append(actor_loss)
        kl_loss_history.append(kl_loss_value)
        kl_coeff_history.append(current_kl_coeff)
        
        # Extract attention sharpness from actor model metrics (if available)
        try:
            actor_model = None
            if hasattr(policy, 'model'):
                if hasattr(policy.model, 'action_model'):
                    actor_model = policy.model.action_model
                elif not hasattr(policy.model, 'is_critic') or not policy.model.is_critic:
                    actor_model = policy.model
            
            if actor_model is not None and hasattr(actor_model, 'metrics'):
                attention_metrics = actor_model.metrics()
                attention_sharpness = attention_metrics.get('attention_sharpness', float('nan'))
            else:
                attention_sharpness = float('nan')
        except Exception:
            attention_sharpness = float('nan')
        
        attention_sharpness_history.append(attention_sharpness)
        
        # MSE: use critic_loss as proxy (TD error squared)
        mse_history.append(critic_loss if not np.isnan(critic_loss) else float('nan'))
        
        # Run evaluation episodes to track performance during burn-in
        if i % EVAL_INTERVAL == 0 or i == 1:
            # Quick evaluation to check waypoint rate, intrusions, and rewards (suppress verbose output)
            with suppress_output():
                env = SectorEnv(n_agents=20, run_id=f"burnin_eval_{i}")
                wp_counts = []
                intr_counts = []
                ep_rewards = []
                
                for _ in range(N_EVAL_EPISODES):
                    obs, _ = env.reset()
                    episode_reward = 0.0
                    while env.agents:
                        agent_ids = list(obs.keys())
                        obs_array = np.stack(list(obs.values()))
                        actions_np = policy.compute_actions(obs_array, explore=False)[0]
                        actions = {aid: act for aid, act in zip(agent_ids, actions_np)}
                        obs, rew, term, trunc, infos = env.step(actions)
                        # Sum rewards across all agents for this step
                        episode_reward += sum(rew.values()) if rew else 0.0
                    
                    wp_counts.append(len(env.waypoint_reached_agents))
                    intr_counts.append(env.total_intrusions)
                    ep_rewards.append(episode_reward)
                
                env.close()
            
            # Calculate averages
            avg_wp_rate = np.mean(wp_counts) / 20.0  # 20 agents
            avg_intrusions = np.mean(intr_counts)
            avg_reward = np.mean(ep_rewards)
            
            episode_waypoint_rates.append(avg_wp_rate)
            episode_intrusions.append(avg_intrusions)
            episode_rewards.append(avg_reward)
            wp_history.append(avg_wp_rate)
            
            # --- EVALUATION SMOOTHING: Voortschrijdend gemiddelde van laatste 2 evaluaties ---
            if len(wp_history) >= 2:
                smoothed_wp = np.mean(wp_history[-2:])
            else:
                smoothed_wp = avg_wp_rate
            
            # --- BEST MODEL TRACKING: Save based on WP >80% → intrusions, else WP rate ---
            is_better = False
            
            # Both above 80%: compare intrusions (lower is better)
            if avg_wp_rate > 0.80 and best_wp_rate > 0.80:
                if avg_intrusions < best_intrusions:
                    is_better = True
                    reason = f"Lower intrusions: {avg_intrusions:.1f} < {best_intrusions:.1f}"
            # Current above 80%, best below: current is better
            elif avg_wp_rate > 0.80 and best_wp_rate <= 0.80:
                is_better = True
                reason = f"Reached 80% WP threshold ({avg_wp_rate*100:.1f}%)"
            # Neither above 80%: compare WP rate
            elif avg_wp_rate > best_wp_rate:
                is_better = True
                reason = f"Higher WP rate: {avg_wp_rate*100:.1f}% > {best_wp_rate*100:.1f}%"
            
            if is_better:
                prev_best_wp = best_wp_rate
                prev_best_intr = best_intrusions
                best_wp_rate = avg_wp_rate
                best_intrusions = avg_intrusions
                best_iteration = i
                best_weights = copy.deepcopy(policy.get_weights())
                print(f"   ⭐ NEW BEST MODEL saved! {reason}")
                print(f"      WP: {avg_wp_rate*100:.1f}% | Intrusions: {avg_intrusions:.1f}")
            
            # Calculate moving averages for stability
            recent_q = np.nanmean(mean_q_history[-10:]) if len(mean_q_history) >= 10 else mean_q
            recent_critic_loss = np.nanmean(critic_loss_history[-10:]) if len(critic_loss_history) >= 10 else critic_loss
            recent_actor_loss = np.nanmean(actor_loss_history[-10:]) if len(actor_loss_history) >= 10 else actor_loss
            
            # Print main metrics
            print(
                f"   Burn-in {i}/{n_iterations} | "
                f"ActorLR: {current_actor_lr:.2e} | KL_β: {current_kl_coeff:.3f} | "
                f"MeanQ: {recent_q:.4f} | "
                f"Critic Loss: {recent_critic_loss:.4f} | "
                f"Actor Loss: {recent_actor_loss:.4f} | "
                f"UsedAlpha: {actual_used_alpha:.4f} | "
                f"WP Rate: {avg_wp_rate*100:.1f}% (Smoothed: {smoothed_wp*100:.1f}%) | "
                f"Avg Intrusions: {avg_intrusions:.2f} | "
                f"Avg Reward: {avg_reward:.2f}"
            )
            
            # Print diagnostics if available
            if i % 100 == 0 or i <= 3:
                if action_mean is not None:
                    print(f"      [Actions] mean={action_mean:.4f}, std={action_std:.4f}, range=[{action_min:.3f}, {action_max:.3f}]")
                if reward_mean is not None:
                    print(f"      [Rewards] mean={reward_mean:.4f}, std={reward_std:.4f}, range=[{reward_min:.3f}, {reward_max:.3f}]")
                if grad_norms['actor_base'] > 0:
                    print(
                        f"      [ParamΔ] ActorBase={grad_norms['actor_base']:.6f}, "
                        f"Attn={grad_norms['attention']:.6f}, "
                        f"Temp={grad_norms['temperature']:.6f}"
                    )
            
            # --- EARLY STOPPING: Stop when WP > 90% AND Intrusions < 80 ---
            if smoothed_wp >= 0.90 and avg_intrusions < 80:
                print(f"\n🎯 Early stopping: Burn-in targets achieved!")
                print(f"   ✅ WP Rate: {avg_wp_rate*100:.1f}% (Smoothed: {smoothed_wp*100:.1f}%) >= 90%")
                print(f"   ✅ Intrusions: {avg_intrusions:.2f} < 80")
                print(f"   Stopping at iteration {i}/{n_iterations}")
                break
        elif i % 50 == 0:
            # Print progress without evaluation (faster)
            recent_q = np.nanmean(mean_q_history[-10:]) if len(mean_q_history) >= 10 else mean_q
            recent_critic_loss = np.nanmean(critic_loss_history[-10:]) if len(critic_loss_history) >= 10 else critic_loss
            recent_actor_loss = np.nanmean(actor_loss_history[-10:]) if len(actor_loss_history) >= 10 else actor_loss
            
            # Print main metrics
            print(
                f"   Burn-in {i}/{n_iterations} | "
                f"ActorLR: {current_actor_lr:.2e} | KL_β: {current_kl_coeff:.3f} | "
                f"MeanQ: {recent_q:.4f} | "
                f"Critic Loss: {recent_critic_loss:.4f} | "
                f"Actor Loss: {recent_actor_loss:.4f} | "
                f"UsedAlpha: {actual_used_alpha:.4f}"
            )
            
            # Print diagnostics every 100 iterations
            if i % 100 == 0:
                if action_mean is not None:
                    print(f"      [Actions] mean={action_mean:.4f}, std={action_std:.4f}, range=[{action_min:.3f}, {action_max:.3f}]")
                if reward_mean is not None:
                    print(f"      [Rewards] mean={reward_mean:.4f}, std={reward_std:.4f}, range=[{reward_min:.3f}, {reward_max:.3f}]")
                if grad_norms['actor_base'] > 0:
                    print(
                        f"      [ParamΔ] ActorBase={grad_norms['actor_base']:.6f}, "
                        f"Attn={grad_norms['attention']:.6f}, "
                        f"Temp={grad_norms['temperature']:.6f}"
                    )
    
    # --- GEGARANDEERD HERSTEL: Laad ALTIJD de beste gewichten ---
    if best_weights is not None:
        policy.set_weights(best_weights)
        print(f"\n🔄 RESTORING BEST MODEL from iteration {best_iteration}")
        print(f"   Best WP Rate: {best_wp_rate*100:.1f}%")
        print(f"   Best Intrusions: {best_intrusions:.1f}")
        if episode_waypoint_rates and episode_intrusions:
            final_wp = episode_waypoint_rates[-1]
            final_intr = episode_intrusions[-1]
            print(f"   Final evaluation: WP {final_wp*100:.1f}%, Intrusions {final_intr:.1f}")
            if final_wp < best_wp_rate or (final_wp > 0.80 and final_intr > best_intrusions):
                print(f"   ✅ Prevented collapse by restoring better checkpoint!")
    else:
        print(f"\n⚠️  Warning: No best model was saved during burn-in")
    
    # Final summary
    final_mean_q = np.nanmean(mean_q_history[-50:]) if len(mean_q_history) >= 50 else np.nanmean(mean_q_history)
    initial_mean_q = np.nanmean(mean_q_history[:10]) if len(mean_q_history) >= 10 else mean_q_history[0]
    
    print(f"\n✅ Burn-in Phase Complete!")
    print(f"   Initial MeanQ: {initial_mean_q:.4f}")
    print(f"   Final MeanQ: {final_mean_q:.4f}")
    print(f"   Change: {final_mean_q - initial_mean_q:+.4f}")
    print(f"   Best WP Rate: {best_wp_rate*100:.1f}% (iteration {best_iteration})")
    print(f"   Critic is now calibrated to expert demonstrations.")
    print(f"   Ready to start main training with environment interaction.\n")
    
    # Save and plot burn-in metrics
    if episode_waypoint_rates:
        import matplotlib.pyplot as plt
        
        # Create comprehensive burn-in analysis figure with 7 subplots (3x3 grid, with last empty)
        fig = plt.figure(figsize=(21, 15))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Prepare evaluation iterations
        eval_iterations = [EVAL_INTERVAL * (j+1) for j in range(len(episode_waypoint_rates))]
        if len(episode_waypoint_rates) > 0 and episode_waypoint_rates[0] != episode_waypoint_rates[0]:  # Check for NaN
            eval_iterations = eval_iterations[1:]
            episode_waypoint_rates_plot = episode_waypoint_rates[1:]
            episode_intrusions_plot = episode_intrusions[1:]
        else:
            episode_waypoint_rates_plot = episode_waypoint_rates
            episode_intrusions_plot = episode_intrusions
        
        # Prepare per-iteration data
        all_iterations = list(range(1, len(mean_q_history) + 1))
        
        # Plot 1: Actor and Critic Loss
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(all_iterations, actor_loss_history, label='Actor Loss', color='blue', linewidth=1.5, alpha=0.7)
        ax1.plot(all_iterations, critic_loss_history, label='Critic Loss', color='red', linewidth=1.5, alpha=0.7)
        ax1.set_title('Actor & Critic Loss During Burn-in', fontweight='bold', fontsize=12)
        ax1.set_xlabel('Burn-in Iteration')
        ax1.set_ylabel('Loss')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Mean Q-Value
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(all_iterations, mean_q_history, color='green', linewidth=1.5)
        ax2.set_title('Mean Q-Value During Burn-in', fontweight='bold', fontsize=12)
        ax2.set_xlabel('Burn-in Iteration')
        ax2.set_ylabel('Mean Q')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Attention Sharpness
        ax3 = fig.add_subplot(gs[1, 0])
        # Filter out NaN values for cleaner plot
        valid_attn = [(i+1, val) for i, val in enumerate(attention_sharpness_history) if not np.isnan(val)]
        if valid_attn:
            attn_iters, attn_vals = zip(*valid_attn)
            ax3.plot(attn_iters, attn_vals, color='purple', linewidth=1.5)
            ax3.set_title('Attention Mechanism Sharpness', fontweight='bold', fontsize=12)
        else:
            ax3.text(0.5, 0.5, 'No attention data available', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Attention Mechanism Sharpness (N/A)', fontweight='bold', fontsize=12)
        ax3.set_xlabel('Burn-in Iteration')
        ax3.set_ylabel('Sharpness')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Average MSE (using critic loss as proxy)
        ax4 = fig.add_subplot(gs[1, 1])
        valid_mse = [(i+1, val) for i, val in enumerate(mse_history) if not np.isnan(val)]
        if valid_mse:
            mse_iters, mse_vals = zip(*valid_mse)
            ax4.plot(mse_iters, mse_vals, color='orange', linewidth=1.5)
        ax4.set_title('Average MSE (Critic Loss Proxy)', fontweight='bold', fontsize=12)
        ax4.set_xlabel('Burn-in Iteration')
        ax4.set_ylabel('MSE (TD Error²)')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Waypoint Reach Rate
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(eval_iterations, [wr * 100 for wr in episode_waypoint_rates_plot], 
                marker='o', linewidth=2, color='teal', markersize=6)
        ax5.set_title('Waypoint Reach Rate During Burn-in', fontweight='bold', fontsize=12)
        ax5.set_xlabel('Burn-in Iteration')
        ax5.set_ylabel('Waypoint Success Rate (%)')
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=95, color='green', linestyle='--', alpha=0.5, label='95% target')
        ax5.legend(loc='lower right')
        
        # Plot 6: Average Intrusions
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(eval_iterations, episode_intrusions_plot, 
                marker='o', linewidth=2, color='crimson', markersize=6)
        ax6.set_title('Avg Intrusions During Burn-in', fontweight='bold', fontsize=12)
        ax6.set_xlabel('Burn-in Iteration')
        ax6.set_ylabel('Average Intrusions per Episode')
        ax6.grid(True, alpha=0.3)
        
        # Plot 7: Average Episode Rewards
        if episode_rewards:
            episode_rewards_plot = episode_rewards[1:] if len(episode_rewards) > 0 and episode_rewards[0] != episode_rewards[0] else episode_rewards
            ax7 = fig.add_subplot(gs[2, 2])
            ax7.plot(eval_iterations, episode_rewards_plot, 
                    marker='o', linewidth=2, color='darkblue', markersize=6)
            ax7.set_title('Avg Episode Reward During Burn-in', fontweight='bold', fontsize=12)
            ax7.set_xlabel('Burn-in Iteration')
            ax7.set_ylabel('Average Episode Reward')
            ax7.grid(True, alpha=0.3)
        
        # Add overall title
        fig.suptitle(f'Burn-in Phase Analysis - Run {RUN_ID}', fontsize=16, fontweight='bold', y=0.995)
        
        # Save the comprehensive plot
        plot_path = os.path.join(CHECKPOINT_DIR, f'burn_in_comprehensive_{RUN_ID}.png')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"   📊 Burn-in comprehensive analysis saved to: {plot_path}")
        plt.close()
    
    # --- HERSTEL ORIGINELE LEARNING RATE ---
    if original_lr is not None and hasattr(policy, '_optimizer'):
        for param_group in policy._optimizer.param_groups:
            param_group['lr'] = original_lr
        print(f"   Learning rate restored: {BURN_IN_LR:.2e} → {original_lr:.2e}")
    elif hasattr(policy, 'optimizers'):
        for idx, opt in enumerate(policy.optimizers()):
            if idx < len(original_lrs):
                for param_group in opt.param_groups:
                    param_group['lr'] = original_lrs[idx]
        print(f"   Learning rates restored to original values")
    
    # --- GEGARANDEERD HERSTEL: Laad ALTIJD de beste gewichten ---
    if best_weights is not None:
        policy.set_weights(best_weights)
        print(f"\n🔄 RESTORING BEST MODEL from iteration {best_iteration}")
        print(f"   Best WP Rate achieved: {best_wp_rate*100:.1f}%")
        if episode_waypoint_rates:
            print(f"   Final evaluation WP: {episode_waypoint_rates[-1]*100:.1f}%")
            if episode_waypoint_rates[-1] < best_wp_rate:
                improvement = (best_wp_rate - episode_waypoint_rates[-1]) * 100
                print(f"   ✅ Prevented collapse: +{improvement:.1f}% improvement restored!")
    else:
        print(f"\n⚠️  Warning: No best model was saved during burn-in")
    
    # Final summary
    final_mean_q = np.nanmean(mean_q_history[-50:]) if len(mean_q_history) >= 50 else np.nanmean(mean_q_history)
    initial_mean_q = np.nanmean(mean_q_history[:10]) if len(mean_q_history) >= 10 else mean_q_history[0]
    
    print(f"\n✅ Burn-in Phase Complete!")
    print(f"   Initial MeanQ: {initial_mean_q:.4f}")
    print(f"   Final MeanQ: {final_mean_q:.4f}")
    print(f"   Change: {final_mean_q - initial_mean_q:+.4f}")
    print(f"   Best WP Rate: {best_wp_rate*100:.1f}% (iteration {best_iteration})")
    print(f"   Critic is now calibrated to expert demonstrations.")
    print(f"   Ready to start main training with environment interaction.\n")
    
    return {
        'mean_q_history': mean_q_history,
        'critic_loss_history': critic_loss_history,
        'actor_loss_history': actor_loss_history,
        'actual_alpha_history': actual_alpha_history,
        'final_mean_q': final_mean_q,
        'initial_mean_q': initial_mean_q,
        'episode_waypoint_rates': episode_waypoint_rates,
        'episode_intrusions': episode_intrusions,
        'episode_rewards': episode_rewards,
        'best_wp_rate': best_wp_rate,
        'best_iteration': best_iteration,
    }
    
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

def verify_buffer_rewards(algo):
    # Haal een grote batch samples uit de buffer
    batch = algo.local_replay_buffer.sample(5000)
    
    if isinstance(batch, MultiAgentBatch):
        rewards = batch.policy_batches.get("shared_policy")["rewards"]
    else:
        rewards = batch["rewards"]
    
    avg_reward = np.mean(rewards)
    max_reward = np.max(rewards)
    min_reward = np.min(rewards)
    std_reward = np.std(rewards)
    
    print("\n🔍 BUFFER REWARD VERIFICATION")
    print("-" * 30)
    print(f"Gemiddelde beloning: {avg_reward:.4f}")
    print(f"Std dev beloning:    {std_reward:.4f}")
    print(f"Min beloning:        {min_reward:.4f}")
    print(f"Max beloning:        {max_reward:.4f}")
    
    # With /400 scaling:
    # - Perfect expert: +98/400 = +0.245
    # - Bad episode: -2000/400 = -5.0
    # - Typical step: -0.01/400 = -0.000025
    # Expected range: [-5.0, +0.25] for current scaling
    
    # Check if rewards are in reasonable range for /400 scaling
    if max_reward > 1.0:
        print("⚠️ WAARSCHUWING: Max reward > 1.0 suggereert OUDE scaling (bijv. /100 of geen divisor).")
        print("   Buffer is mogelijk gevuld met verkeerde reward schaal.")
    elif max_reward < 0.01:
        print("⚠️ WAARSCHUWING: Max reward < 0.01 suggereert TE KLEINE scaling (bijv. /2000).")
        print("   Gradient signals zijn te zwak voor goede learning.")
    elif min_reward < -10.0:
        print("⚠️ WAARSCHUWING: Min reward < -10 suggereert extreme intrusions in buffer.")
        print("   Expert data quality is zeer laag.")
    elif abs(avg_reward) < 0.0001 and std_reward < 0.0001:
        print("⚠️ WAARSCHUWING: Rewards zijn bijna allemaal ~0 - mogelijk lege/corrupte buffer.")
    else:
        print("✅ SUCCESS: Buffer rewards passen bij huidige scaling (/400).")
        print(f"   Expert data quality: {'uitstekend' if max_reward > 0.15 else 'goed' if avg_reward > -0.01 else 'matig'}")
    print("-" * 30 + "\n")

def build_trainer(n_agents):
    """Builds and configures the PPO algorithm.
    
    Args:
        n_agents: Number of agents for the environment
    """
    def policy_map(agent_id, *_, **__):
        return "shared_policy"

    cfg = (
        SACConfig()
        .api_stack(
            enable_rl_module_and_learner=False,      # use old API stack for multi-agent SAC
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            "sector_env",
            env_config={
                "n_agents": n_agents,
                "run_id": RUN_ID,
                "metrics_base_dir": METRICS_DIR,
                "intrusion_penalty": CURRENT_INTRUSION_PENALTY,  # Use curriculum starting value
            },
            disable_env_checking=True,
        )
        .framework("torch")
        .env_runners(
            num_env_runners=os.cpu_count() - 1,
            num_envs_per_env_runner=1,
            # Force more episode collection per iteration
            sample_timeout_s=60.0,  # Allow time for episodes to complete
        )
        .callbacks(ForceAlphaCallback)
        .training(
            # LRs
            # ⚠️ STRATEGY: Keep actor LR very low during freeze phase (iter 0-1000) 
            # so critic can stabilize first, then gradually ramp up
            # Note: Burn-in (2000 iters) happens BEFORE iteration 0, so doesn't affect this schedule
            actor_lr=[
                [0, 1e-6],        # Korte stabilisatiefase direct na burn-in
                [250, 1e-5],      # Snellere ramp-up: navigatie is immers al bekend
                [1000, 5e-5],     # Piek LR: hier moet hij de collision avoidance 'kraken'
                [3000, 2e-5],     # Begin verfijning nadat de streak waarschijnlijk is geactiveerd
                [4500, 1e-5],     # Fine-tune fase
            ],
            critic_lr=[
                [0, 3e-4],        # Aggressief leren van de nieuwe penalty-structuur
                [2500, 1e-4],     # Pas verlagen als de Actor een stabiele koers heeft gevonden
                [4500, 5e-5],     # Minimale ruis in de Q-waardes aan het eind
            ],
            
            train_batch_size=4096,
            # ---- Option A: fixed alpha (stable baseline) ----
            target_entropy = -2.0,   #was -2.0. -1.0 for more exploration. larger negative value is more exploitation
            # alpha_lr = 1e-5,            # was 3e-5.   lr for updating entropy / alpha. lower means slower alpha updates
            # alpha_lr=[
            #     [0,        0],   # from step 0 to 1M: 3e-4
            #     [TOTAL_ITERS/2, 1e-5],
            #     [TOTAL_ITERS, 1e-6],  # then slowly decay to 3e-5
            # ],
            # alpha_lr=[ # DIT DOET DUS HELEMAAL NIKS DOOR DIE CALLBACK BOVENIN DE CODE
            #     [0, 0],                            # Tot 70% van de tijd: maximale exploratie
            #     [TOTAL_ITERS * 0.5, 0],         # Daarna: heel voorzichtig de chaos afbouwen
            #     [TOTAL_ITERS * 0.7, 5e-6],         # Daarna: heel voorzichtig de chaos afbouwen
            #     [TOTAL_ITERS, 1e-6],               # Eindigen met subtiele verfijning
            # ], 
            # alpha_lr=5e-5,            # was 3e-5.   lr for updating entropy / alpha. lower means slower alpha updates
            
            initial_alpha = START_ALPHA, #  DIT DOET NIKS DOOR DIE CALLBACK. was 0.5. initial alpha/entropy, higher means more exploration
            grad_clip=0.5,
            
            # Add numerical stability for entropy calculation
            # This prevents log(0) in TanhGaussian entropy computation
            clip_actions=True,  # Ensure actions stay in valid range

            # Hyperparameters
            gamma=0.99, # discount factor future rewards
            tau=0.001, # soft update parameter for target    networks, smaller makes target network update more slowly
            
            twin_q=True, # use two networks, for more stable learning
            n_step=3, #  enables multi-step q-learning, agent will use rewards over multiple timestep

            # Replay/batching - REDUCED for more episode diversity
            replay_buffer_config={
                "type": "MultiAgentPrioritizedReplayBuffer",
                "capacity": 1_000_000,  # Reduced from 1M to encourage fresher samples
                "prioritized_replay": True,
                "prioritized_replay_alpha": 0.4, # exponent determines how much prioritization is used. 
                "prioritized_replay_beta": 0.4,
                "prioritized_replay_eps": 1e-6,    # value to be added to each prior value to avoid 0 probabilities            
            },
            num_steps_sampled_before_learning_starts=0,  # kan ook 1000 zijn ofzo, of gewoon weinig Reduced from 5000
            # train_batch_size=16_384,  # Reduced from 2048 for more frequent updates
            
            # Actor (Policy) Configuration 
            policy_model_config={
                "custom_model": "attention_sac",
                "custom_model_config": {
                    "hidden_dims": [512, 512], 
                    "is_critic": False
                },
                "fcnet_hiddens": [], # Ensure default MLP is disabled
            },

            # --- CRITIC (Q-Network) Configuration ---
            q_model_config={
                "custom_model": "attention_sac",
                "custom_model_config": {
                    "hidden_dims": [512, 512],
                    "is_critic": True
                },
                "fcnet_hiddens": [], # Ensure default MLP is disabled
            },
        )

        .multi_agent(
            policies={"shared_policy": (None, None, None, {})},
            policy_mapping_fn=policy_map,
        )
        .resources(num_gpus=1)
        # .learners(num_gpus_per_learner=1)
    )
    return cfg.build()


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
        rewards, lengths, intrusions, waypoints, aircraft_with_intrusions = [], [], [], [], []

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
            aircraft_with_intrusions.append(len(env.aircraft_with_intrusions))

        env.close()
        return rewards, lengths, intrusions, waypoints, aircraft_with_intrusions
    
    # Run with or without output suppression
    if silent:
        with suppress_output():
            rewards, lengths, intrusions, waypoints, aircraft_with_intrusions = _run_episodes()
    else:
        rewards, lengths, intrusions, waypoints, aircraft_with_intrusions = _run_episodes()
    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    avg_length = float(np.mean(lengths)) if lengths else 0.0
    avg_intrusions = float(np.mean(intrusions)) if intrusions else 0.0
    waypoint_rate = (float(np.sum(waypoints)) / (n_episodes * n_agents)) if waypoints else 0.0
    avg_aircraft_with_intrusions = float(np.mean(aircraft_with_intrusions)) if aircraft_with_intrusions else 0.0
    aircraft_with_intrusions_rate = (avg_aircraft_with_intrusions / n_agents) if n_agents > 0 else 0.0
    return {
        "avg_reward": avg_reward,
        "avg_length": avg_length,
        "avg_intrusions": avg_intrusions,
        "waypoint_rate": waypoint_rate,
        "avg_aircraft_with_intrusions": avg_aircraft_with_intrusions,
        "aircraft_with_intrusions_rate": aircraft_with_intrusions_rate,
        "per_episode_reward": rewards,
        "per_episode_length": lengths,
        "per_episode_intrusions": intrusions,
        "per_episode_waypoints": waypoints,
        "per_episode_aircraft_with_intrusions": aircraft_with_intrusions,
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
                "avg_aircraft_with_intrusions",
                "aircraft_with_intrusions_rate",
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
                "avg_aircraft_with_intrusions": round(metrics.get("avg_aircraft_with_intrusions", 0), 2),
                "aircraft_with_intrusions_rate": round(metrics.get("aircraft_with_intrusions_rate", 0), 4),
            }
        )

if __name__ == "__main__":
    # Start timing
    training_start_time = time.time()
    
    
    ray.shutdown()
    ray.init(
        runtime_env={
            "working_dir": os.path.dirname(os.path.abspath(__file__)),
            "excludes": [
                "models/",       # Exclude trained model checkpoints
                "metrics/",      # Exclude metrics data
                "*.pkl",         # Exclude pickle files
                "__pycache__/",  # Exclude Python cache
            ]
        }
    )

    # Clean up old checkpoints and metrics if force retraining
    if FORCE_RETRAIN:
        # Delete checkpoint directory
        if os.path.exists(CHECKPOINT_DIR):
            print(f"FORCE_RETRAIN is True. Deleting old checkpoint directory:\n{CHECKPOINT_DIR}")
            try:
                shutil.rmtree(CHECKPOINT_DIR)
                print("✅ Old checkpoint directory removed.")
            except OSError as e:
                print(f"Error: {e.strerror} - {CHECKPOINT_DIR}")
        
        # Delete metrics directory for this run to prevent appending
        run_metrics_dir = os.path.join(METRICS_DIR, f"run_{RUN_ID}")
        if os.path.exists(run_metrics_dir):
            print(f"FORCE_RETRAIN is True. Deleting old metrics directory:\n{run_metrics_dir}")
            try:
                shutil.rmtree(run_metrics_dir)
                print("✅ Old metrics directory removed.")
            except OSError as e:
                print(f"Error: {e.strerror} - {run_metrics_dir}")
    
    print("-" * 30)

    target_iters = None
    

    target_iters = None
    restored_from = None

    base_state = os.path.join(CHECKPOINT_DIR, "algorithm_state.json")
    if not FORCE_RETRAIN and os.path.exists(base_state):
        restored_from = CHECKPOINT_DIR
    elif not FORCE_RETRAIN:
        # Try to find a checkpoint in subfolders (e.g., iter_00050, final_YYYYMMDD...)
        cand = _find_latest_checkpoint(CHECKPOINT_DIR)
        if cand:
            restored_from = cand
            
    # NEW LOGIC: Pre-fill buffer before training starts
    if restored_from:
        algo = Algorithm.from_checkpoint(restored_from)
        target_iters = algo.iteration + max(1, int(EXTRA_ITERS))
    else:
        algo = build_trainer(N_AGENTS)
        target_iters = int(TOTAL_ITERS)
    
    # ⚠️ CRITICAL FIX: Always clear and refill buffer with fresh expert data
    # This ensures reward scale matches current environment (even after checkpoint load)
    print("\n🔄 Clearing old buffer and refilling with fresh expert data...")
    print(f"   (Ensures reward scale matches current environment: /400.0)")
    
    # Clear the replay buffer completely
    algo.local_replay_buffer._storage.clear()
    algo.local_replay_buffer._num_added = 0
    print("   ✅ Buffer cleared")
    
    # Refill with fresh expert demonstrations
    if not restored_from:  # Only on fresh run
        prefill_sac_buffer(algo, n_episodes=PRETRAIN_EPISODES)
    else:  # Also refill when loading checkpoint (critical!)
        print("   📝 Checkpoint loaded - refilling buffer with current environment")
        prefill_sac_buffer(algo, n_episodes=PRETRAIN_EPISODES)

    # Print curriculum schedule
    print("\n" + "=" * 60)
    print("🎯 INTRUSION PENALTY CURRICULUM SCHEDULE")
    print("=" * 60)
    for i, (threshold_iter, penalty) in enumerate(INTRUSION_CURRICULUM_STAGES):
        stage_names = ["Navigation", "Collision Awareness", "Moderate Penalties", "Full Mastery"]
        stage_name = stage_names[i] if i < len(stage_names) else f"Stage {i+1}"
        next_threshold = INTRUSION_CURRICULUM_STAGES[i+1][0] if i < len(INTRUSION_CURRICULUM_STAGES)-1 else target_iters
        print(f"   Stage {i+1} ({stage_name}): Iterations {threshold_iter:,}-{next_threshold:,}")
        print(f"      Intrusion Penalty: {penalty}")
    print("=" * 60 + "\n")

    # Pre-Burn-in evaluation disabled: removed 30-episode baseline evaluation

    # --- BURN-IN: Offline learning from expert buffer only ---
    if ENABLE_BURN_IN:
        burn_in_results = burn_in_on_expert_buffer(
            algo,
            n_iterations=BURN_IN_ITERATIONS,
            batch_size=BURN_IN_BATCH_SIZE,
        )
        # Summarize burn-in improvements
        try:
            im_init = burn_in_results.get('initial_mean_q', float('nan'))
            im_final = burn_in_results.get('final_mean_q', float('nan'))
            print(f"\n🔥 Burn-in summary: initial_mean_q={im_init:.4f} -> final_mean_q={im_final:.4f} (Δ={im_final-im_init:+.4f})")
        except Exception:
            pass
    
    # Buffer verification
    results_buffer_verification = verify_buffer_rewards(algo)
    print("-" * 30)
    print(f" ✅ BUFFER VERIFICATION: {results_buffer_verification}")
    

    # EVALUATE INITIAL PERFORMANCE
    print(f"\n🌟 Evaluating initial performance before training...")
    try:
        checkpoint_result = algo.save(CHECKPOINT_DIR)
        # Extract just the path from the result to avoid printing massive object
        if hasattr(checkpoint_result, 'checkpoint') and hasattr(checkpoint_result.checkpoint, 'path'):
            path = checkpoint_result.checkpoint.path
        else:
            path = str(checkpoint_result)
        print(f"✅ Checkpoint saved to: {path}")

        # --- Fixed-seed mini evaluation ---is 
        print(f"[Eval] Starting evaluation with {10} episodes...")
        try:
            eval_metrics = run_fixed_eval(
                algo, 
                n_episodes=10, 
                render=False, 
                n_agents=N_AGENTS
            )
            iter_label = f"INIT({getattr(algo, 'iteration', 0)})"
            wp_pct = eval_metrics.get("waypoint_rate", float("nan")) * 100.0 if eval_metrics.get("waypoint_rate") is not None else float("nan")
            ac_intr_pct = eval_metrics.get("aircraft_with_intrusions_rate", float("nan")) * 100.0 if eval_metrics.get("aircraft_with_intrusions_rate") is not None else float("nan")
            print(
                f"[Eval] ✅ {iter_label} | avg_rew={eval_metrics['avg_reward']:.3f} | "
                f"avg_len={eval_metrics['avg_length']:.1f} | avg_intr={eval_metrics['avg_intrusions']:.2f} | "
                f"wp_rate={wp_pct:.1f}% | ac_w_intr={ac_intr_pct:.1f}%"
            )
        except Exception as e:
            print(f"[Eval] ❌ Evaluation failed: {e}")
    except Exception as e:
        print(f"❌ Checkpoint saving failed: {e}")
                            
    # Loss history for different components
    total_loss_history = []
    policy_loss_history = []
    entropy_history = []
    alpha_history = []
    q_loss_history = []
    reward_history = []
    episode_length_history = []
    reward_entropy_ratio_history = []  # Track reward-to-entropy balance
    waypoint_rate_history = []  # Track waypoint success rate
    avg_intrusions_history = []  # Track mean intrusions per iteration
    
    # Attention metrics tracking
    attention_sharpness_history = []
    attention_temperature_history = []  # Track learnable temperature parameter
    wq_weight_norm_history = []
    wq_grad_norm_history = []
    wk_weight_norm_history = []
    wk_grad_norm_history = []
    wv_weight_norm_history = []
    wv_grad_norm_history = []
    
    # Gradient norm tracking per parameter group
    actor_base_grad_norm_history = []
    attention_grad_norm_history = []
    temperature_grad_norm_history = []
    critic_grad_norm_history = []
    
    # Training step tracking
    total_training_steps = 0  # Total environment steps used during training
    
    # Expert mixing tracking
    expert_mix_history = []  # Track expert mixing ratio over time
    expert_samples_injected_history = []  # Track how many expert samples were injected
    
    # Early stopping tracking
    best_reward = float('-inf')  # Best single-iteration reward (for saving checkpoints)
    best_reward_iteration = 0
    best_checkpoint_path = None
    best_smoothed_reward = float('-inf')  # Best smoothed reward (for early stopping)
    iterations_without_improvement = 0  # Based on smoothed reward (for stopping)
    early_stop_triggered = False
    
    # Lowest intrusion tracking (separate from best reward)
    lowest_intrusions = float('inf')  # Lowest intrusion count
    lowest_intrusions_iteration = 0
    lowest_intrusions_checkpoint_path = None
    
    # --- Main Training Loop ---
    for i in range(algo.iteration + 1, target_iters + 1):
        # Calculate expert mixing ratio for this iteration
        # Note: iterations are 1-indexed, but we want 0-indexed for the calculation
        expert_ratio = calculate_expert_mix_ratio(i - 1, target_iters)
        expert_mix_history.append(expert_ratio)
        
        # Inject expert samples before training if ratio > 0
        n_expert_injected = 0
        if expert_ratio > 0.0 and len(expert_buffer_storage) > 0:
            n_expert_injected = inject_expert_samples_before_training(algo, expert_ratio, batch_size=4096)
        expert_samples_injected_history.append(n_expert_injected)
        
        # Print mixing info on first few iterations and periodically
        if i <= 5 or i % 200 == 0:
            print(f"   [Mixing] Iteration {i}: Expert ratio = {expert_ratio:.1%}, Injected {n_expert_injected} expert samples")
        
        result = algo.train()
        
        # --- COMPUTE GRADIENT NORMS PER PARAMETER GROUP ---
        grad_norms = {'actor_base': 0.0, 'attention': 0.0, 'temperature': 0.0, 'critic': 0.0}
        
        # Only compute on selected iterations to avoid overhead
        compute_grads = (i <= 10 or i % 100 == 0)
        
        if compute_grads:
            policy = algo.get_policy("shared_policy")
            if hasattr(policy, 'optimizers'):
                for idx, opt in enumerate(policy.optimizers()):
                    if idx == 0:  # Actor optimizer
                        for group_idx, param_group in enumerate(opt.param_groups):
                            # Identify group by parameter count
                            params_with_grad = [p for p in param_group['params'] if p.grad is not None]
                            if len(params_with_grad) == 0:
                                continue
                            
                            # Compute gradient norm
                            grad_norm = torch.nn.utils.clip_grad_norm_(params_with_grad, float('inf'))
                            grad_norm_val = float(grad_norm.item()) if hasattr(grad_norm, 'item') else float(grad_norm)
                            
                            # Classify by parameter count (same logic as LR assignment)
                            n_params = len(param_group['params'])
                            if n_params == 1:  # Temperature parameter
                                grad_norms['temperature'] = grad_norm_val
                            elif any('attention' in str(p.shape) or p.numel() in (512*256, 256*256) 
                                    for p in param_group['params'][:3] if hasattr(p, 'shape')):
                                grad_norms['attention'] = grad_norm_val
                            else:  # Actor base
                                grad_norms['actor_base'] = grad_norm_val
                    
                    elif idx == 1:  # Critic optimizer
                        params_with_grad = [p for param_group in opt.param_groups 
                                          for p in param_group['params'] if p.grad is not None]
                        if len(params_with_grad) > 0:
                            grad_norm = torch.nn.utils.clip_grad_norm_(params_with_grad, float('inf'))
                            grad_norms['critic'] = float(grad_norm.item()) if hasattr(grad_norm, 'item') else float(grad_norm)
        
        # Store gradient norms
        actor_base_grad_norm_history.append(grad_norms['actor_base'])
        attention_grad_norm_history.append(grad_norms['attention'])
        temperature_grad_norm_history.append(grad_norms['temperature'])
        critic_grad_norm_history.append(grad_norms['critic'])

        # Extract metrics from env_runners (new location in hybrid API)
        env_runners = result.get("env_runners", {})
        mean_rew = env_runners.get("episode_return_mean", float("nan"))
        ep_len = env_runners.get("episode_len_mean", float("nan"))
        
        # Extract custom metrics for waypoint reach rate
        # Try both locations: top-level and env_runners
        custom_metrics_top = result.get("custom_metrics", {})
        custom_metrics_env = env_runners.get("custom_metrics", {})
        
        # Merge both (env_runners takes priority if both exist)
        custom_metrics = {**custom_metrics_top, **custom_metrics_env}
                
        # Try multiple possible keys for waypoint rate
        waypoint_rate = custom_metrics.get("waypoint_rate_mean", 
                                          custom_metrics.get("waypoint_rate", float("nan")))
        avg_intrusions = custom_metrics.get("intrusions_mean",
                                           custom_metrics.get("intrusions", float("nan")))
        
        
        
        # Notify when first episode completes
        if i > 1 and not np.isnan(waypoint_rate):
            # Check if previous iteration had NaN
            if len(waypoint_rate_history) > 0 and waypoint_rate_history[-1] == 0.0:
                print(f"\n   ✅ First episode completed! Waypoint rate: {waypoint_rate*100:.1f}%, Intrusions: {avg_intrusions:.1f}\n")
        
        # Notify when episodes start completing (based on EpLen)
        if i > 1 and not np.isnan(ep_len):
            if len(episode_length_history) > 0 and np.isnan(episode_length_history[-1]):
                print(f"\n   🎬 Episodes are now completing! Average episode length: {ep_len:.1f} steps\n")
        
        # Convert waypoint_rate to scalar if needed
        if isinstance(waypoint_rate, (list, np.ndarray)):
            waypoint_rate = float(np.mean(waypoint_rate)) if len(waypoint_rate) > 0 else float("nan")
        # Convert avg_intrusions to scalar if needed
        if isinstance(avg_intrusions, (list, np.ndarray)):
            avg_intrusions = float(np.mean(avg_intrusions)) if len(avg_intrusions) > 0 else float("nan")
        
        # Track total environment steps used this iteration
        timesteps_this_iter = result.get("num_env_steps_sampled_this_iter", 0)
        
        if isinstance(timesteps_this_iter, (list, np.ndarray)):
            timesteps_this_iter = int(np.sum(timesteps_this_iter))
        else:
            timesteps_this_iter = int(timesteps_this_iter)
        total_training_steps += timesteps_this_iter
        
        # Convert to scalar if needed
        if isinstance(mean_rew, (list, np.ndarray)):
            mean_rew = float(np.mean(mean_rew)) if len(mean_rew) > 0 else float("nan")
        if isinstance(ep_len, (list, np.ndarray)):
            ep_len = float(np.mean(ep_len)) if len(ep_len) > 0 else float("nan")
        
        # Extract SAC-specific metrics from learner stats
        info = result.get("info", {})
        learner_dict = info.get("learner", {})
        

        
        if isinstance(learner_dict, dict) and "shared_policy" in learner_dict:
            learner_info = learner_dict["shared_policy"].get("learner_stats", {})
        else:
            learner_info = {}
        
        # Check if we're in pre-learning phase
        total_timesteps = result.get('timesteps_total', 0)
        in_prelearning = total_timesteps < 10_000
        
        
        # SAC metrics from learner_stats
        policy_loss = learner_info.get("actor_loss", float("nan"))
        q_loss = learner_info.get("critic_loss", float("nan"))
        alpha_raw = learner_info.get("alpha_value", float("nan"))
        mean_q = learner_info.get("mean_q", float("nan"))
        real_entropy = learner_info.get("policy_entropy", float("nan"))
        
        # Convert arrays to scalars (SAC returns arrays sometimes)
        def to_scalar(val):
            if isinstance(val, (list, tuple, np.ndarray)):
                # Check for NaN values in arrays
                if len(val) > 0:
                    clean_val = np.array(val)
                    if np.all(np.isnan(clean_val)):
                        return float("nan")
                    # Filter out NaN values before taking mean
                    clean_val = clean_val[~np.isnan(clean_val)]
                    if len(clean_val) > 0:
                        return float(np.mean(clean_val))
                return float("nan")
            return float(val) if not isinstance(val, str) else float("nan")
        
        q_loss = to_scalar(q_loss)
        alpha = to_scalar(alpha_raw)
        policy_loss = to_scalar(policy_loss)
        current_mean_q = to_scalar(mean_q)  # Use mean_q as proxy for entropy
        entropy = to_scalar(real_entropy) # THIS IS NOW TRUE ENTROPY
        
        # If entropy is NaN, try to get log_std statistics as a diagnostic
        manual_entropy = None  # Will try to compute if RLlib's is NaN
        
        if np.isnan(entropy):
            
            # Try to compute entropy manually from the policy's distribution
            try:
                policy = algo.get_policy("shared_policy")
                
                # Deep inspection on iteration 42 to understand model structure
                if i == 42:
                    print(f"\n   [Deep Inspection] SAC Model Structure:")
                    if hasattr(policy, 'model'):
                        model = policy.model
                        print(f"     - policy.model type: {type(model).__name__}")
                        print(f"     - model attributes: {[a for a in dir(model) if not a.startswith('_')][:20]}")
                        
                        if hasattr(model, 'action_model'):
                            am = model.action_model
                            print(f"     - action_model type: {type(am).__name__}")
                            print(f"     - action_model attributes: {[a for a in dir(am) if not a.startswith('_')][:20]}")
                            
                            # Check for log_std
                            if hasattr(am, 'log_std'):
                                print(f"     - ✓ action_model.log_std exists!")
                                print(f"     - log_std type: {type(am.log_std).__name__}")
                            else:
                                print(f"     - ✗ action_model.log_std NOT found")
                                print(f"     - Checking for other std-related attributes...")
                                std_attrs = [a for a in dir(am) if 'std' in a.lower() or 'log' in a.lower()]
                                print(f"     - std-related attrs: {std_attrs}")
                    print(f"")
                
                # Try multiple methods to get entropy/log_std
                computed_entropy = False
                
                # Method 1: Direct log_std from action_model
                if hasattr(policy, 'model') and hasattr(policy.model, 'action_model'):
                    am = policy.model.action_model
                    if hasattr(am, 'log_std'):
                        log_std = am.log_std.detach().cpu().numpy()
                        action_dim = len(log_std)
                        manual_entropy = 0.5 * action_dim * (1 + np.log(2 * np.pi)) + np.sum(log_std)
                        computed_entropy = True
                        
                        # if i % 50 == 0 or i == 42:
                        #     std = np.exp(log_std)
                        #     print(f"   [Manual Entropy] Method 1: Gaussian H = {manual_entropy:.4f}")
                        #     print(f"     log_std: {log_std}, std: {std}")
                        #     print(f"     Note: True SAC entropy ≈ {manual_entropy - 0.7:.4f} (after tanh correction)")
                
                # Method 2: Check policy.model directly
                if not computed_entropy and hasattr(policy, 'model'):
                    if hasattr(policy.model, 'log_std'):
                        log_std = policy.model.log_std.detach().cpu().numpy()
                        action_dim = len(log_std)
                        manual_entropy = 0.5 * action_dim * (1 + np.log(2 * np.pi)) + np.sum(log_std)
                        computed_entropy = True
                        
                        if i % 50 == 0 or i == 42:
                            std = np.exp(log_std)
                            print(f"   [Manual Entropy] Method 2: Gaussian H = {manual_entropy:.4f}")
                            print(f"     log_std: {log_std}, std: {std}")
                
                # Method 3: Sample actions and compute std empirically
                if not computed_entropy and i % 100 == 42:
                    print(f"   [Manual Entropy] Method 3: SAC uses state-dependent std")
                    print(f"     Cannot compute entropy without forward pass")
                    print(f"     Entropy will remain NaN - use Alpha as proxy")
                    
            except Exception as e:
                if i == 42 or (i % 100 == 0 and i <= 200):
                    print(f"   [Manual Entropy] Exception: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Use manual entropy if available, otherwise keep NaN
            if manual_entropy is not None:
                entropy = manual_entropy
            else:
                # Keep NaN but stop spamming messages after iteration 100
                entropy = float("nan")
        
        # Calculate total loss as sum of components
        if not np.isnan(policy_loss) and not np.isnan(q_loss):
            total_loss = abs(policy_loss) + abs(q_loss)
        else:
            total_loss = float("nan")

        # Extract attention metrics from actor model
        policy = algo.get_policy("shared_policy")
        attention_metrics = {}
        
        # Try different ways to access the actor model in SAC
        actor_model = None
        try:
            # Debug info on first iteration
            if i == 1:
                print(f"\n   [Debug] Inspecting policy structure:")
                print(f"   - hasattr(policy, 'model'): {hasattr(policy, 'model')}")
                if hasattr(policy, 'model'):
                    print(f"   - type(policy.model): {type(policy.model).__name__}")
                    print(f"   - hasattr(policy.model, 'action_model'): {hasattr(policy.model, 'action_model')}")
                    print(f"   - hasattr(policy.model, 'is_critic'): {hasattr(policy.model, 'is_critic')}")
                    if hasattr(policy.model, 'is_critic'):
                        print(f"   - policy.model.is_critic: {policy.model.is_critic}")
                print(f"   - hasattr(policy, 'pi_model'): {hasattr(policy, 'pi_model')}")
            
            # New API: Try to get the actor model from the policy
            if hasattr(policy, 'model'):
                # For SAC, check if this is the action_model (actor)
                if hasattr(policy.model, 'action_model'):
                    actor_model = policy.model.action_model
                elif not hasattr(policy.model, 'is_critic') or not policy.model.is_critic:
                    # If not marked as critic, assume it's the actor
                    actor_model = policy.model
            
            # Alternative: Try to access via pi_model or action_model directly on policy
            if actor_model is None and hasattr(policy, 'pi_model'):
                actor_model = policy.pi_model
                
            # --- SUCCESS MONITOR (Expert vs Learner) ---
            if i % 10 == 0:
                # 1. Haal de belangrijkste metrics op
                current_mean_q = result.get("info", {}).get("learner", {}).get("shared_policy", {}).get("learner_stats", {}).get("mean_q", 0.0)
                explained_var = result.get("info", {}).get("learner", {}).get("shared_policy", {}).get("learner_stats", {}).get("vf_explained_var", 0.0)
                
                # 2. Probeer de waypoint rate uit custom_metrics te halen (indien beschikbaar)
                custom_metrics = result.get("custom_metrics", {})
                wp_rate = custom_metrics.get("waypoint_rate_mean", 0.0) * 100 # Als percentage
                avg_intr = custom_metrics.get("intrusions_mean", float("nan"))
                if isinstance(avg_intr, (list, np.ndarray)):
                    avg_intr = float(np.mean(avg_intr)) if len(avg_intr) > 0 else float("nan")
                
                # # 3. Print de vergelijking
                # print(f"Waypoint Rate   | {wp_rate:>6.1f}%")
                # print(f"Mean Q-Value    | {current_mean_q:>6.3f}")
                # print(f"Avg Intrusions  | {avg_intr:>6.1f}")
                # print(f"Critic accuracy | {explained_var:>6.3f}")
                
               
            
            if i == 1:
                print(f"   - Found actor_model: {actor_model is not None}")
                if actor_model is not None:
                    print(f"   - Actor model type: {type(actor_model).__name__}")
                    print(f"   - hasattr(actor_model, 'metrics'): {hasattr(actor_model, 'metrics')}")
            
            # If we found the actor model, get metrics
            if actor_model is not None and hasattr(actor_model, 'metrics'):
                attention_metrics = actor_model.metrics()
                if i == 1:
                    print(f"   - Metrics extracted: {list(attention_metrics.keys())}")
        except Exception as e:
            # Debug: print why metrics extraction failed
            if i <= 3:  # Only print for first few iterations
                print(f"   [Debug] Could not extract attention metrics: {e}")
                import traceback
                traceback.print_exc()
            pass
        
        # Append to history
        total_loss_history.append(total_loss)
        policy_loss_history.append(policy_loss)
        q_loss_history.append(q_loss)
        entropy_history.append(entropy)
        alpha_history.append(alpha)
        reward_history.append(mean_rew)
        episode_length_history.append(ep_len)
        waypoint_rate_history.append(waypoint_rate * 100.0 if not np.isnan(waypoint_rate) else 0.0)  # Convert to percentage
        avg_intrusions_history.append(avg_intrusions if not np.isnan(avg_intrusions) else 0.0)
        
        # Append attention metrics (use NaN if not available)
        attention_sharpness_history.append(attention_metrics.get('attention_sharpness', float('nan')))
        attention_temperature_history.append(attention_metrics.get('attention_temperature', float('nan')))
        wq_weight_norm_history.append(attention_metrics.get('wq_weight_norm', float('nan')))
        wq_grad_norm_history.append(attention_metrics.get('wq_grad_norm', float('nan')))
        wk_weight_norm_history.append(attention_metrics.get('wk_weight_norm', float('nan')))
        wk_grad_norm_history.append(attention_metrics.get('wk_grad_norm', float('nan')))
        wv_weight_norm_history.append(attention_metrics.get('wv_weight_norm', float('nan')))
        wv_grad_norm_history.append(attention_metrics.get('wv_grad_norm', float('nan')))

        # Calculate moving average for reward (5 iterations)
        if len(reward_history) >= 5:
            reward_ma5 = np.mean(reward_history[-5:])
        else:
            reward_ma5 = mean_rew
        
        # Calculate Reward-to-Entropy (R/E) Ratio
        # This measures if agent prioritizes reward over exploration
        # 
        # Formula: R/E = |mean_reward| / (α × H + ε)
        # where:
        #   - α (alpha) = temperature/scaling factor for entropy
        #   - H (entropy) = actual policy entropy (randomness of actions)
        #   - ε = 1e-6 (small constant to prevent division by zero)
        #
        # Interpretation:
        #   - High R/E ratio: Agent focuses on reward (good if learning mission)
        #   - Low R/E ratio: Agent focuses on exploration (may ignore mission)
        #   - If entropy is NaN: R/E ratio is INVALID and should be ignored
        
        # Zoek deze regel in main.py (rond regel 1010)
        if not np.isnan(alpha) and not np.isnan(entropy) and not np.isnan(mean_rew) and not np.isnan(ep_len):
            # Bereken de TOTALE entropy bonus van de hele episode
            total_episode_entropy = ep_len * (alpha * entropy)
            
            # De Echte R/E Ratio (Balans over de hele vlucht)
            reward_entropy_ratio = abs(mean_rew) / (abs(total_episode_entropy) + 1e-6)
                    
            # Warn if weighted entropy is suspiciously small (policy collapsed)
            if abs(weighted_entropy) < 0.001 and i % 50 == 0:
                print(f"   ⚠️  Weighted Entropy very low: {weighted_entropy:.6f} (α={alpha:.4f}, H={entropy:.4f})")
                print(f"      Policy may have collapsed - agent not exploring!")
        else:
            # Cannot compute valid R/E ratio - set to NaN for visibility
            weighted_entropy = float('nan')
            reward_entropy_ratio = float('nan')
            
            # Warn on first occurrence and periodically
            if i == 71 or (i % 100 == 0 and i <= 500):
                print(f"   ⚠️  R/E Ratio is INVALID (entropy is NaN)")
                print(f"      Cannot assess reward vs exploration balance")
                print(f"      Track Alpha (α={alpha:.4f}) as exploration proxy instead")
        
        reward_entropy_ratio_history.append(reward_entropy_ratio)
        
        # Enhanced training progress display with entropy ratio
        prelearning_note = " [PRE-LEARNING]" if in_prelearning else ""
        
        # Format R/E ratio display - show "INVALID" if NaN due to entropy
        if np.isnan(reward_entropy_ratio) and not np.isnan(mean_rew):
            re_ratio_str = "INVALID (H=nan)"
        else:
            re_ratio_str = f"{reward_entropy_ratio:.2f}"
        
        # Print only every 50th iteration (or in pre-learning phase)
        if i % 50 == 0 or in_prelearning:
            # Format waypoint rate as percentage
            if np.isnan(waypoint_rate):
                wp_rate_str = "N/A (no episodes yet)"
            else:
                wp_rate_str = f"{waypoint_rate*100:.1f}%"
            
            # Format intrusions
            if np.isnan(avg_intrusions):
                intrusions_str = "N/A"
            else:
                intrusions_str = f"{avg_intrusions:>6.1f}"
            
            # Format attention metrics
            attn_sharp = attention_metrics.get('attention_sharpness', float('nan'))
            attn_temp = attention_metrics.get('attention_temperature', float('nan'))
            if np.isnan(attn_sharp):
                attn_sharp_str = "N/A"
            else:
                attn_sharp_str = f"{attn_sharp:.4f}"
            if np.isnan(attn_temp):
                attn_temp_str = "N/A"
            else:
                attn_temp_str = f"{attn_temp:.3f}"
            
            # Get learning rates from custom metrics
            actor_lr = float(custom_metrics.get("actor_lr", 0.0))
            attention_lr = float(custom_metrics.get("attention_lr", 0.0))
            temperature_lr = float(custom_metrics.get("temperature_lr", 0.0))
            critic_lr = float(custom_metrics.get("critic_lr", 0.0))
            
            # Format gradient norms (display every 10 iterations or when computed)
            if compute_grads:
                grad_str = (f"GradNorm(A:{grad_norms['actor_base']:.2e} "
                           f"At:{grad_norms['attention']:.2e} "
                           f"T:{grad_norms['temperature']:.2e} "
                           f"C:{grad_norms['critic']:.2e})")
            else:
                grad_str = ""
            
            print(
                f"Iter {i}/{target_iters}{prelearning_note} | "
                f"Reward: {mean_rew:.3f} (MA5: {reward_ma5:.3f}) | "
                f"EpLen: {ep_len:.1f} | "
                f"WP: {wp_rate_str} | "
                f"Avg Intrusions: {intrusions_str} | "
                f"Loss: {total_loss:.3f} (Critic: {q_loss:.3f}, Actor: {policy_loss:.3f}) | "
                f"MeanQ: {current_mean_q:.4f} | Entropy: {entropy:.4f} | Alpha: {alpha:.4f} | "
                f"R/E Ratio: {re_ratio_str} | "
                f"LR(A:{actor_lr:.1e} At:{attention_lr:.1e} T:{temperature_lr:.1e} C:{critic_lr:.1e}) | "
                f"Attn Sharpness: {attn_sharp_str} | Attn Temp: {attn_temp_str}"
                f"{' | ' + grad_str if grad_str else ''}"
            )

        # --- Best Checkpoint Tracking (ALWAYS ACTIVE) ---
        # Check if we have both new best reward AND new lowest intrusions
        is_new_best_reward = i > 50 and not np.isnan(mean_rew) and mean_rew > best_reward
        is_new_lowest_intrusions = i > 50 and not np.isnan(avg_intrusions) and avg_intrusions < lowest_intrusions
        
        # Update tracking variables
        if is_new_best_reward:
            best_reward = mean_rew
            best_reward_iteration = i
        
        if is_new_lowest_intrusions:
            lowest_intrusions = avg_intrusions
            lowest_intrusions_iteration = i
        
        # Save checkpoint only once if both conditions are met
        if is_new_best_reward or is_new_lowest_intrusions:
            # Determine checkpoint naming and messages
            if is_new_best_reward and is_new_lowest_intrusions:
                # Both conditions met - save once with both indicators
                checkpoint_dir = os.path.join(CHECKPOINT_DIR, f"best_iter_{i:05d}_low_i")
                messages = [
                    f"⭐ New best reward: {best_reward:.3f}",
                    f"🛡️  New lowest intrusions: {lowest_intrusions:.2f}"
                ]
            elif is_new_best_reward:
                # Only best reward
                checkpoint_dir = os.path.join(CHECKPOINT_DIR, f"best_iter_{i:05d}")
                messages = [f"⭐ New best reward: {best_reward:.3f}"]
            else:
                # Only lowest intrusions
                checkpoint_dir = os.path.join(CHECKPOINT_DIR, f"best_iter_{i:05d}_low_i")
                messages = [f"🛡️  New lowest intrusions: {lowest_intrusions:.2f}"]
            
            # Save the checkpoint
            checkpoint_result = algo.save(checkpoint_dir)
            # Extract path from checkpoint result
            if hasattr(checkpoint_result, 'checkpoint') and hasattr(checkpoint_result.checkpoint, 'path'):
                checkpoint_path = checkpoint_result.checkpoint.path
            else:
                checkpoint_path = checkpoint_dir
            
            # Print all applicable messages
            for msg in messages:
                print(f"   {msg} (saved to {os.path.basename(checkpoint_path)})")

        # --- Early Stopping Check (OPTIONAL) ---
        if ENABLE_EARLY_STOPPING and not np.isnan(mean_rew):
            # Use smoothed reward (moving average of last 5 iterations) for early stopping decision
            if EARLY_STOP_USE_SMOOTHED and len(reward_history) >= 5:
                smoothed_reward = np.mean(reward_history[-5:])
            else:
                smoothed_reward = mean_rew
            
            # Check if smoothed reward has improved beyond minimum delta
            if smoothed_reward > best_smoothed_reward + EARLY_STOP_MIN_DELTA:
                best_smoothed_reward = smoothed_reward
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1
                if iterations_without_improvement >= EARLY_STOP_PATIENCE:
                    print(f"\n⏹️  Early stopping: No improvement in smoothed reward for {EARLY_STOP_PATIENCE} iterations")
                    print(f"   Best smoothed reward: {best_smoothed_reward:.3f}")
                    print(f"   Best single-iteration reward: {best_reward:.3f} at iteration {best_reward_iteration}")
                    if best_checkpoint_path:
                        print(f"   Best checkpoint saved at: {best_checkpoint_path}")
                    early_stop_triggered = True
        
        # Break if early stopping triggered
        if early_stop_triggered:
            print(f"   Stopping at iteration {i}/{target_iters}")
            break

        # Optional periodic checkpointing
        if EVALUATION_INTERVAL and i % EVALUATION_INTERVAL == 0:
            print(f"\n{'='*60}")
            print(f"🔄 EVALUATION at iteration {i}")
            print(f"{'='*60}")
            
            checkpoint_result = algo.save(CHECKPOINT_DIR)
            # Extract just the path from the result to avoid printing massive object
            if hasattr(checkpoint_result, 'checkpoint') and hasattr(checkpoint_result.checkpoint, 'path'):
                path = checkpoint_result.checkpoint.path
            else:
                path = str(checkpoint_result)
            print(f"✅ Checkpoint saved to: {path}")

            # --- Fixed-seed mini evaluation ---
            print(f"[Eval] Starting evaluation with {10} episodes...")
            try:
                eval_metrics = run_fixed_eval(
                    algo, 
                    n_episodes=10, 
                    render=False, 
                    n_agents=N_AGENTS
                )
                wp_pct = eval_metrics.get("waypoint_rate", float("nan")) * 100.0 if eval_metrics.get("waypoint_rate") is not None else float("nan")
                ac_intr_pct = eval_metrics.get("aircraft_with_intrusions_rate", float("nan")) * 100.0 if eval_metrics.get("aircraft_with_intrusions_rate") is not None else float("nan")
                print(
                    f"[Eval] ✅ iter={i} | avg_rew={eval_metrics['avg_reward']:.3f} | "
                    f"avg_len={eval_metrics['avg_length']:.1f} | avg_intr={eval_metrics['avg_intrusions']:.2f} | "
                    f"wp_rate={wp_pct:.1f}% | ac_w_intr={ac_intr_pct:.1f}%"
                )
                
                eval_dir = os.path.join(METRICS_DIR, f"run_{RUN_ID}")
                print(f"[Eval] Saving results to: {eval_dir}")
                _write_eval_row(metrics=eval_metrics, iteration=i, out_dir=eval_dir)
                print(f"[Eval] ✅ Results saved successfully")
                
            except Exception as e:
                print(f"[Eval] ❌ FAILED due to error: {e}")
                import traceback
                print(traceback.format_exc())

    print("\n🚀 Training finished.")
    
    # Early stopping summary and checkpoint handling
    if early_stop_triggered and best_checkpoint_path:
        print(f"   ✋ Early stopping was triggered")
        print(f"   📊 Best reward achieved: {best_reward:.3f} at iteration {best_reward_iteration}")
        print(f"   💾 Best checkpoint: {best_checkpoint_path}")
        print(f"\n   ℹ️  To use the best model, restore from: {best_checkpoint_path}")
    elif early_stop_triggered:
        print(f"   ✋ Early stopping was triggered")
        print(f"   📊 Best reward achieved: {best_reward:.3f}")
    
    # Calculate and display total training time and steps
    total_training_time = time.time() - training_start_time
    actual_iters = len(reward_history)
    print(f"⏱️  Total training time: {total_training_time/60:.2f} minutes ({total_training_time/3600:.2f} hours) for {actual_iters} iters.")
    
    # Try to get total steps from algorithm state if our counter didn't work
    if total_training_steps == 0:
        # Try alternative methods to get step count
        try:
            # Method 1: Check algorithm's internal counters
            if hasattr(algo, 'num_env_steps_sampled'):
                total_training_steps = algo.num_env_steps_sampled
            elif hasattr(algo, '_counters') and 'num_env_steps_sampled' in algo._counters:
                total_training_steps = algo._counters['num_env_steps_sampled']
            elif hasattr(algo, 'num_env_steps_trained'):
                total_training_steps = algo.num_env_steps_trained
        except:
            pass
    
    if total_training_steps > 0:
        print(f"📊 Total environment steps: {total_training_steps:,} steps")
        if actual_iters > 0:
            print(f"   Average steps per iteration: {total_training_steps/actual_iters:.0f}")
    else:
        print(f"⚠️  Warning: Could not track environment steps (counter remained at 0)")
        print(f"   This might be due to API differences in RLlib version")
    
    # Save final checkpoint (current state)
    final_checkpoint_result = algo.save(CHECKPOINT_DIR)
    # Extract just the path from the result to avoid printing massive object
    if hasattr(final_checkpoint_result, 'checkpoint') and hasattr(final_checkpoint_result.checkpoint, 'path'):
        final_path = final_checkpoint_result.checkpoint.path
    else:
        final_path = str(final_checkpoint_result)
    print(f"✅ Final checkpoint (last iteration) saved to: {final_path}")
    
    # Save final model with a special name if enabled
    if SAVE_FINAL_MODEL:
        final_model_dir = os.path.join(CHECKPOINT_DIR, "final_model")
        # Remove old final_model if it exists
        if os.path.exists(final_model_dir):
            shutil.rmtree(final_model_dir)
        final_model_result = algo.save(final_model_dir)
        if hasattr(final_model_result, 'checkpoint') and hasattr(final_model_result.checkpoint, 'path'):
            final_model_path = final_model_result.checkpoint.path
        else:
            final_model_path = str(final_model_result)
        print(f"💾 Final model (for deployment) saved to: {final_model_path}")
    
    # Summary of available checkpoints
    if best_checkpoint_path:
        print(f"\n📁 Checkpoint Summary:")
        print(f"   • Best model (iteration {best_reward_iteration}, reward {best_reward:.3f}): {best_checkpoint_path}")
        print(f"   • Final model (iteration {actual_iters}): {final_path}")
        if SAVE_FINAL_MODEL:
            print(f"   • Final model (for deployment): {final_model_path}")
        print(f"\n   💡 Tip: Use the best checkpoint for evaluation to get optimal performance!")
    
    # --- Plot Training Metrics in a Comprehensive Figure ---
    # Create 6 subplots (6 rows, 1 column) so the waypoint subplot (axes[5]) exists
    fig, axes = plt.subplots(6, 1, figsize=(14, 24))

    # Plot 1: Loss Components (SAC specific: Total, Actor, Critic)
    axes[0].plot(range(1, len(total_loss_history) + 1), total_loss_history, label="Total Loss", marker='o', linestyle='-')
    axes[0].plot(range(1, len(policy_loss_history) + 1), policy_loss_history, label="Actor Loss", marker='s', linestyle='--')
    axes[0].plot(range(1, len(q_loss_history) + 1), q_loss_history, label="Critic Loss", marker='^', linestyle='-.')
    axes[0].set_title("SAC Loss Components Over Training Iterations")
    axes[0].set_xlabel("Training Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Plot 2: Reward with Moving Average
    axes[1].plot(range(1, len(reward_history) + 1), reward_history, marker='o', linestyle='-', alpha=0.5, label='Mean Reward')
    
    # Calculate and plot 5-iteration moving average
    if len(reward_history) >= 5:
        reward_ma5 = np.convolve(reward_history, np.ones(5)/5, mode='valid')
        axes[1].plot(range(5, len(reward_history) + 1), reward_ma5, linewidth=2, label='5-Iter Moving Avg', color='red')
    
    # Mark the best checkpoint iteration if early stopping was used
    if best_checkpoint_path and best_reward_iteration > 0:
        axes[1].axvline(x=best_reward_iteration, color='green', linestyle='--', linewidth=2, 
                       label=f'Best Checkpoint (iter {best_reward_iteration})')
        axes[1].plot(best_reward_iteration, reward_history[best_reward_iteration-1], 
                    'g*', markersize=15, label=f'Best Reward: {best_reward:.2f}')
    
    axes[1].set_title("Mean Reward Over Training Iterations")
    axes[1].set_xlabel("Training Iteration")
    axes[1].set_ylabel("Mean Reward")
    axes[1].legend()
    axes[1].grid(True)

    # Plot 3: Episode Length
    axes[2].plot(range(1, len(episode_length_history) + 1), episode_length_history, marker='o', linestyle='-', color='purple')
    axes[2].set_title("Episode Length Over Training Iterations")
    axes[2].set_xlabel("Training Iteration")
    axes[2].set_ylabel("Episode Length")
    axes[2].grid(True)

    # Plot 4: SAC Mean Q-Value and Temperature (Alpha)
    ax4_twin = axes[3].twinx()  # Create twin axis for alpha
    axes[3].plot(range(1, len(entropy_history) + 1), entropy_history, marker='o', linestyle='-', color='orange', label='Mean Q-Value')
    ax4_twin.plot(range(1, len(alpha_history) + 1), alpha_history, marker='s', linestyle='--', color='blue', label='Alpha (Temperature)')
    axes[3].set_title("SAC Mean Q-Value and Temperature (Alpha) Over Training Iterations")
    axes[3].set_xlabel("Training Iteration")
    axes[3].set_ylabel("Mean Q-Value", color='orange')
    ax4_twin.set_ylabel("Alpha (Temperature)", color='blue')
    axes[3].tick_params(axis='y', labelcolor='orange')
    ax4_twin.tick_params(axis='y', labelcolor='blue')
    axes[3].legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    axes[3].grid(True)

    # Plot 5: Reward-to-Entropy Ratio
    axes[4].plot(range(1, len(reward_entropy_ratio_history) + 1), reward_entropy_ratio_history, 
                marker='o', linestyle='-', color='teal', linewidth=2)
    # Add reference lines for healthy ranges
    axes[4].axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Ratio = 1.0 (Entropy Dominant)')
    axes[4].axhline(y=10.0, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Ratio = 10.0 (Reward Dominant)')
    axes[4].set_title("Reward-to-Entropy Ratio Over Training Iterations\n(Low < 1.0: Entropy dominant | High > 10.0: Too greedy)")
    axes[4].set_xlabel("Training Iteration")
    axes[4].set_ylabel("Reward / (Alpha × Entropy)")
    axes[4].legend(loc='best')
    axes[4].grid(True)
    axes[4].set_yscale('log')  # Log scale for better visualization of wide ratio ranges
    
    # Plot 6: Waypoint Success Rate (left y-axis) and Mean Intrusions (right y-axis)
    axes[5].set_title("Waypoint Success Rate and Mean Intrusions Over Training")
    axes[5].set_xlabel("Training Iteration")
    # Plot waypoint rate as percentage on left axis
    if len(waypoint_rate_history) > 0:
        axes[5].plot(range(1, len(waypoint_rate_history) + 1), waypoint_rate_history, 
                     marker='o', linestyle='-', color='green', label='Waypoint Rate (%)')
    axes[5].set_ylabel("Waypoint Rate (%)", color='green')
    axes[5].tick_params(axis='y', labelcolor='green')
    axes[5].grid(True)

    # Plot mean intrusions on twin axis (right)
    if len(avg_intrusions_history) > 0:
        ax5_twin = axes[5].twinx()
        ax5_twin.plot(range(1, len(avg_intrusions_history) + 1), avg_intrusions_history, 
                      marker='s', linestyle='--', color='red', label='Mean Intrusions')
        ax5_twin.set_ylabel('Mean Intrusions', color='red')
        ax5_twin.tick_params(axis='y', labelcolor='red')
        # Combine legends if both series exist
        try:
            lines, labels = axes[5].get_legend_handles_labels()
            lines2, labels2 = ax5_twin.get_legend_handles_labels()
            axes[5].legend(lines + lines2, labels + labels2, loc='upper left')
        except Exception:
            pass

    # Adjust layout and show the figure
    plt.tight_layout()
    
    # Save figure to file instead of showing it (for headless servers)
    figure_path = os.path.join(script_dir, f"training_metrics_run_{RUN_ID}.png")
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Training metrics plot saved to: {figure_path}")
    plt.close()  # Close the figure to free memory
    
    # --- Create Separate Attention Metrics Plot ---
    fig_attn, axes_attn = plt.subplots(2, 1, figsize=(14, 10))
    
    # Filter out NaN values for plotting
    valid_sharpness = [(i+1, val) for i, val in enumerate(attention_sharpness_history) if not np.isnan(val)]
    valid_temperature = [(i+1, val) for i, val in enumerate(attention_temperature_history) if not np.isnan(val)]
    
    # Plot 1: Attention Sharpness over iterations
    if len(valid_sharpness) > 0:
        iters_sharp, vals_sharp = zip(*valid_sharpness)
        axes_attn[0].plot(iters_sharp, vals_sharp, marker='o', linestyle='-', color='blue', linewidth=2)
        axes_attn[0].set_title("Attention Sharpness Over Training Iterations\n(Higher = More focused on specific drones)")
        axes_attn[0].set_xlabel("Training Iteration")
        axes_attn[0].set_ylabel("Attention Sharpness (Max Attention Weight)")
        axes_attn[0].grid(True)
        axes_attn[0].axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, 
                           label='Threshold: 0.5 (Focused attention)')
        axes_attn[0].legend()
        
        # Add statistics
        mean_sharp = np.mean(vals_sharp)
        max_sharp = np.max(vals_sharp)
        min_sharp = np.min(vals_sharp)
        axes_attn[0].text(0.02, 0.98, 
                        f'Mean: {mean_sharp:.4f}\nMax: {max_sharp:.4f}\nMin: {min_sharp:.4f}',
                        transform=axes_attn[0].transAxes, 
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        axes_attn[0].text(0.5, 0.5, 'No valid attention sharpness data available',
                        ha='center', va='center', transform=axes_attn[0].transAxes)
        axes_attn[0].set_title("Attention Sharpness Over Training Iterations")
    
    # Plot 2: Attention Temperature over iterations
    if len(valid_temperature) > 0:
        iters_temp, vals_temp = zip(*valid_temperature)
        axes_attn[1].plot(iters_temp, vals_temp, marker='s', linestyle='-', color='orange', linewidth=2)
        axes_attn[1].set_title("Attention Temperature Over Training Iterations\n(Controls attention sharpness via softmax scaling)")
        axes_attn[1].set_xlabel("Training Iteration")
        axes_attn[1].set_ylabel("Attention Temperature (Learnable Parameter)")
        axes_attn[1].grid(True)
        
        # Add statistics
        mean_temp = np.mean(vals_temp)
        max_temp = np.max(vals_temp)
        min_temp = np.min(vals_temp)
        final_temp = vals_temp[-1]
        axes_attn[1].text(0.02, 0.98, 
                        f'Initial: {vals_temp[0]:.3f}\nFinal: {final_temp:.3f}\nMean: {mean_temp:.3f}\nMax: {max_temp:.3f}\nMin: {min_temp:.3f}',
                        transform=axes_attn[1].transAxes, 
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    else:
        axes_attn[1].text(0.5, 0.5, 'No valid attention temperature data available',
                        ha='center', va='center', transform=axes_attn[1].transAxes)
        axes_attn[1].set_title("Attention Temperature Over Training Iterations")
    
    plt.tight_layout()
    
    # Save attention plot
    attention_plot_path = os.path.join(script_dir, f"attention_metrics_plot_run_{RUN_ID}.png")
    plt.savefig(attention_plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Attention metrics plot saved to: {attention_plot_path}")
    plt.close()
    
    # --- Create Gradient Norm Plot ---
    fig_grad, axes_grad = plt.subplots(2, 1, figsize=(14, 10))
    
    # Filter out zero values (iterations where gradients weren't computed)
    valid_actor_grad = [(i+1, val) for i, val in enumerate(actor_base_grad_norm_history) if val > 0]
    valid_attention_grad = [(i+1, val) for i, val in enumerate(attention_grad_norm_history) if val > 0]
    valid_temperature_grad = [(i+1, val) for i, val in enumerate(temperature_grad_norm_history) if val > 0]
    valid_critic_grad = [(i+1, val) for i, val in enumerate(critic_grad_norm_history) if val > 0]
    
    # Plot 1: Actor-related gradient norms (actor base, attention, temperature)
    if len(valid_actor_grad) > 0 or len(valid_attention_grad) > 0 or len(valid_temperature_grad) > 0:
        if len(valid_actor_grad) > 0:
            iters, vals = zip(*valid_actor_grad)
            axes_grad[0].plot(iters, vals, marker='o', linestyle='-', label='Actor Base', linewidth=2)
        if len(valid_attention_grad) > 0:
            iters, vals = zip(*valid_attention_grad)
            axes_grad[0].plot(iters, vals, marker='s', linestyle='-', label='Attention', linewidth=2)
        if len(valid_temperature_grad) > 0:
            iters, vals = zip(*valid_temperature_grad)
            axes_grad[0].plot(iters, vals, marker='^', linestyle='-', label='Temperature', linewidth=2)
        
        axes_grad[0].set_title("Actor Components Gradient Norms Over Training\n(Measured every 100 iterations)")
        axes_grad[0].set_xlabel("Training Iteration")
        axes_grad[0].set_ylabel("Gradient Norm (L2)")
        axes_grad[0].set_yscale('log')
        axes_grad[0].legend()
        axes_grad[0].grid(True, alpha=0.3)
    else:
        axes_grad[0].text(0.5, 0.5, 'No gradient norm data available',
                        ha='center', va='center', transform=axes_grad[0].transAxes)
    
    # Plot 2: Critic gradient norms
    if len(valid_critic_grad) > 0:
        iters, vals = zip(*valid_critic_grad)
        axes_grad[1].plot(iters, vals, marker='D', linestyle='-', color='red', label='Critic', linewidth=2)
        axes_grad[1].set_title("Critic Gradient Norms Over Training\n(Measured every 100 iterations)")
        axes_grad[1].set_xlabel("Training Iteration")
        axes_grad[1].set_ylabel("Gradient Norm (L2)")
        axes_grad[1].set_yscale('log')
        axes_grad[1].legend()
        axes_grad[1].grid(True, alpha=0.3)
        
        # Add statistics
        mean_grad = np.mean(vals)
        max_grad = np.max(vals)
        min_grad = np.min(vals)
        axes_grad[1].text(0.02, 0.98, 
                        f'Mean: {mean_grad:.3e}\nMax: {max_grad:.3e}\nMin: {min_grad:.3e}',
                        transform=axes_grad[1].transAxes, 
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    else:
        axes_grad[1].text(0.5, 0.5, 'No critic gradient norm data available',
                        ha='center', va='center', transform=axes_grad[1].transAxes)
    
    plt.tight_layout()
    
    # Save gradient norm plot
    gradient_plot_path = os.path.join(script_dir, f"gradient_norms_plot_run_{RUN_ID}.png")
    plt.savefig(gradient_plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Gradient norms plot saved to: {gradient_plot_path}")
    plt.close()
    
    # --- Save Attention Metrics to CSV ---
    import csv
    attention_csv_path = os.path.join(script_dir, f"attention_metrics_run_{RUN_ID}.csv")
    with open(attention_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'iteration', 'attention_sharpness', 'attention_temperature',
            'wq_weight_norm', 'wq_grad_norm',
            'wk_weight_norm', 'wk_grad_norm',
            'wv_weight_norm', 'wv_grad_norm',
            'actor_base_grad_norm', 'attention_grad_norm', 
            'temperature_grad_norm', 'critic_grad_norm',
            'expert_mix_ratio', 'expert_samples_injected',  # New columns for expert mixing
        ])
        for idx in range(len(attention_sharpness_history)):
            writer.writerow([
                idx + 1,
                attention_sharpness_history[idx],
                attention_temperature_history[idx],
                wq_weight_norm_history[idx],
                wq_grad_norm_history[idx],
                wk_weight_norm_history[idx],
                wk_grad_norm_history[idx],
                wv_weight_norm_history[idx],
                wv_grad_norm_history[idx],
                actor_base_grad_norm_history[idx] if idx < len(actor_base_grad_norm_history) else 0.0,
                attention_grad_norm_history[idx] if idx < len(attention_grad_norm_history) else 0.0,
                temperature_grad_norm_history[idx] if idx < len(temperature_grad_norm_history) else 0.0,
                critic_grad_norm_history[idx] if idx < len(critic_grad_norm_history) else 0.0,
                expert_mix_history[idx] if idx < len(expert_mix_history) else '',
                expert_samples_injected_history[idx] if idx < len(expert_samples_injected) else '',
            ])
    print(f"📊 Attention + Expert Mixing metrics saved to: {attention_csv_path}")
    
    # Print summary of expert mixing
    if len(expert_mix_history) > 0:
        print(f"\n🎓 Expert Mixing Summary:")
        print(f"   • Start ratio: {expert_mix_history[0]:.1%}")
        print(f"   • End ratio: {expert_mix_history[-1]:.1%}")
        print(f"   • Total expert samples injected: {sum(expert_samples_injected_history):,}")
        # Calculate when mixing reached 0%
        zero_idx = next((i for i, r in enumerate(expert_mix_history) if r == 0.0), len(expert_mix_history))
        if zero_idx < len(expert_mix_history):
            print(f"   • Expert mixing ended at iteration {zero_idx + 1} ({zero_idx/len(expert_mix_history)*100:.1f}% of training)")
    
    

    ray.shutdown()
