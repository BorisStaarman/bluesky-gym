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
from bluesky_gym.envs.ma_env_two_stage_PPO import SectorEnv

from run_config import RUN_ID

# Register your custom environment with Gymnasium
register_envs()

# CALLBACK CLASS 
class MVPDataBridgeCallback(DefaultCallbacks):
    """
    Callback to extract teacher actions from environment info and inject them
    into the training batch for Stage 1 imitation learning.
    """
    def on_postprocess_trajectory(
        self, worker, episode, agent_id, policy_id, 
        policies, postprocessed_batch, original_batches, **kwargs
    ):
        # Check if we have data for this agent in the postprocessed batch
        # The teacher_action should already be in the infos
        if SampleBatch.INFOS in postprocessed_batch:
            original_infos = postprocessed_batch[SampleBatch.INFOS]
            
            # Extract the teacher_action you saved in the step function
            # Use a default [0,0] if it's missing to prevent crashes
            teacher_actions = []
            for info in original_infos:
                if "teacher_action" in info:
                    teacher_action = info["teacher_action"]
                    # Ensure it's a numpy array with correct dtype
                    if not isinstance(teacher_action, np.ndarray):
                        teacher_action = np.array(teacher_action, dtype=np.float32)
                    teacher_actions.append(teacher_action)
                else:
                    # Default action if missing
                    teacher_actions.append(np.zeros(2, dtype=np.float32))
            
            # Convert to numpy array for batch processing
            if teacher_actions:
                teacher_actions_array = np.array(teacher_actions, dtype=np.float32)
                
                # Write it into the batch so the Loss Function can see it
                postprocessed_batch["teacher_targets"] = teacher_actions_array
    
    def on_learn_on_batch(self, policy, train_batch, result, **kwargs):
        """
        Capture Stage 1 imitation loss and expose it in the result dict.
        This ensures the loss is available in trainer.train() results.
        """
        try:
            # Check if this is Stage 1 training (has teacher_targets)
            if "teacher_targets" not in train_batch:
                return
            
            # Get the loss from policy.loss_stats if available
            if hasattr(policy, 'loss_stats') and 'imitation_loss' in policy.loss_stats:
                loss_val = policy.loss_stats['imitation_loss']
                # Store in custom_metrics so RLlib aggregates it
                result.setdefault("custom_metrics", {})["imitation_loss"] = loss_val
        except Exception:
            # Don't break training if logging fails
            pass

# --- Parameters ---
N_AGENTS = 20  # Number of agents for training

# --- STAGE CONTROL ---
RUN_STAGE_2 = True  # Set to True to run Stage 2 after Stage 1, False to only train Stage 1

# --- STAGE 1: IMITATION LEARNING (PPO with custom loss) ---
iterations_stage1 = 75  # Number of iterations for Stage 1 imitation learning


# --- WARM-UP PHASE SETTINGS ---
WARMUP_ITERATIONS = 7  # Number of iterations to warm up critic with frozen policy
# --- STAGE 2: RL FINE-TUNING (PPO with standard loss) ---
TOTAL_ITERS = WARMUP_ITERATIONS + 100   # set to 300 or something for ppo

WARMUP_LR = 1e-5  # Low but not frozen - allows policy to adjust entropy while keeping actions stable
FINETUNE_LR = 5e-5  # Learning rate after warm-up for joint optimization

EVALUATION_INTERVAL = 10

script_dir = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.join(script_dir, "metrics")

# --- Path for model ---
CHECKPOINT_DIR = os.path.join(script_dir, "models/sectorcr_ma_sac")

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

def stage1_imitation_loss(policy, model, dist_class, train_batch):
    """
    Custom loss function for Stage 1 imitation learning with PPO.
    Minimizes MSE between NN predicted actions and teacher (MVP) target actions.
    """
    # Get the model's action logits
    logits, _ = model(train_batch)
    
    # Get action distribution
    action_dist = dist_class(logits, model)
    
    # Get deterministic action (mean of the distribution)
    predicted_actions = action_dist.deterministic_sample()
    
    # Get teacher targets from the batch (injected by callback)
    teacher_targets = train_batch["teacher_targets"]
    
    # Convert to tensor if needed
    if not isinstance(teacher_targets, torch.Tensor):
        teacher_targets = torch.tensor(
            teacher_targets, 
            dtype=torch.float32,
            device=predicted_actions.device
        )
    else:
        teacher_targets = teacher_targets.to(predicted_actions.device)
    
    # Ensure matching shapes
    if predicted_actions.shape != teacher_targets.shape:
        if predicted_actions.numel() == teacher_targets.numel():
            teacher_targets = teacher_targets.reshape(predicted_actions.shape)
        else:
            print(f"[LOSS ERROR] Cannot match shapes: {predicted_actions.shape} vs {teacher_targets.shape}")
            return torch.tensor(0.0, device=predicted_actions.device)
    
    # Compute MSE loss (pure imitation)
    imitation_loss = F.mse_loss(predicted_actions, teacher_targets)
    
    # Store the loss value in policy stats for logging
    loss_val = imitation_loss.item()
    policy.loss_stats = {"imitation_loss": loss_val}
    
    # Return as total loss (PPO will use this instead of policy gradient loss)
    return imitation_loss



def build_trainer(n_agents, stage=1, restore_path=None):
    """
    Builds the PPO algorithm for both stages with different configurations.
    Args:
        n_agents: Number of agents
        stage: 1 = MVP Imitation (with custom loss), 2 = PPO RL Fine-tuning 
        restore_path: Path to checkpoint to load (used for Stage 2)
    """
    
    # 1. Define Policy Mapping (Same for both stages)
    def policy_map(agent_id, *_, **__):
        return "shared_policy"

    # 2. Determine Stage-Specific Settings
    if stage == 1:
        # --- STAGE 1: IMITATION LEARNING (PPO with custom loss) ---
        print("[Config] Stage 1: Using PPO with custom imitation loss")
        current_callbacks = MVPDataBridgeCallback
        
        training_config = {
            "lr": 1e-4,  # Learning rate for imitation
            "train_batch_size": 8000, # how much data before a rounds of updates
            "minibatch_size": 1024, #  train batch is split into minibatches for training, of this size
            "num_sgd_iter": 10,  # how many times NN updates over each data batch
            "grad_clip": 1.0, # clipping of hte uypdate parameter, does have an influence here 
            "clip_param": 0.3,  # ignored
            "vf_loss_coeff": 0.01,  # ignored
            "entropy_coeff": 0.0,  # ignored
            "gamma": 0.99, # ignored
            "lambda_": 0.95, # ignored
            "model": {"fcnet_hiddens": [512, 512]}, # size of NN
        }
        
    else:
        # --- STAGE 2: RL FINE-TUNING (Standard PPO) ---
        print("[Config] Stage 2: Using standard PPO for RL")
        current_callbacks = None
        
        training_config = {
            "lr": WARMUP_LR,  # Start with very low LR for warm-up phase
            "train_batch_size": 32000,
            "minibatch_size": 2000,
            "num_sgd_iter": 12,
            "clip_param": 0.2,
            "vf_loss_coeff": 1.0,  # Value function learns during warm-up
            # Start with very low entropy coeff during warm-up
            # Will be increased programmatically after warm-up for exploration
            "entropy_coeff": 0.0,  # Very low → minimal entropy reward during warm-up
            "grad_clip": 0.5,
            "gamma": 0.99,
            "lambda_": 0.95,
            "model": {
                "fcnet_hiddens": [512, 512],
                # Allow log_std to be learned independently
                "free_log_std": True,
            },
        }

    # 3. Build the PPO Config
    cfg = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
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
            sample_timeout_s=120.0,
            num_envs_per_env_runner=1,
        )
        .callbacks(current_callbacks) 
        .training(**training_config)
        .multi_agent(
            policies={"shared_policy": (None, None, None, {})},
            policy_mapping_fn=policy_map,
        )
        .resources(num_gpus=0)
    )
    
    # Build the algorithm
    algo = cfg.build()
    
    # Inject custom loss function for Stage 1 (PPO supports this)
    if stage == 1:
        # Get the policy and override its loss function
        policy = algo.get_policy("shared_policy")
        # Monkey-patch the loss function with our imitation loss
        original_loss_fn = policy.loss
        
        def custom_loss_wrapper(model, dist_class, train_batch):
            return stage1_imitation_loss(policy, model, dist_class, train_batch)
        
        policy.loss = custom_loss_wrapper
        print("[Config] Custom imitation loss function injected into PPO policy")
    
    # For Stage 2, directly set log_std to control entropy
    if stage == 2:
        policy = algo.get_policy("shared_policy")
        model = policy.model
        
        # Set log_std to -2.5 for low entropy during warm-up
        state_dict = model.state_dict()
        if '_append_free_log_std.log_std' in state_dict:
            with torch.no_grad():
                old_val = state_dict['_append_free_log_std.log_std'].mean().item()
                state_dict['_append_free_log_std.log_std'].fill_(-2.5)
                new_val = state_dict['_append_free_log_std.log_std'].mean().item()
            expected_std = np.exp(new_val)
            expected_entropy = 0.5 * np.log(2 * np.pi * np.e * expected_std**2) * 2
            print(f"[Config] Set log_std for warm-up: {old_val:.3f} → {new_val:.3f} (std={expected_std:.4f}, entropy≈{expected_entropy:.2f})")
        else:
            print(f"[WARNING] Could not find _append_free_log_std.log_std parameter!")




    # 4. If loading from previous stage, restore weights
    if restore_path:
        print(f"Restoring weights from: {restore_path}")
        algo.restore(restore_path)
        
        # After restoring, re-apply log_std setting for Stage 2
        if stage == 2:
            policy = algo.get_policy("shared_policy")
            model = policy.model
            state_dict = model.state_dict()
            
            if '_append_free_log_std.log_std' in state_dict:
                with torch.no_grad():
                    old_val = state_dict['_append_free_log_std.log_std'].mean().item()
                    state_dict['_append_free_log_std.log_std'].fill_(-2.5)
                    new_val = state_dict['_append_free_log_std.log_std'].mean().item()
                print(f"[Config] Reset log_std after restore: {old_val:.3f} → {new_val:.3f}")
            else:
                print(f"[WARNING] Could not reset log_std after restore")

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
    # STAGE 1: TEACHER IMITATION (Supervised)
    # ==============================================================================
    # We only run this if we are NOT restoring from an existing Stage 2 checkpoint
    # and if we actually want to run stage 1.
    
    stage1_checkpoint = os.path.join(CHECKPOINT_DIR, "stage1_weights")
    stage1_best_checkpoint = os.path.join(CHECKPOINT_DIR, "stage1_best_weights")
    
    # important parameters
    run_stage1 = False
    restored_from = None
    
    # Check if we are trying to resume a Stage 2 run
    latest_checkpoint = _find_latest_checkpoint(CHECKPOINT_DIR)
    if latest_checkpoint:
        print(f"🔄 Found existing Stage 2 checkpoint: {latest_checkpoint}")
        print("⏭️  Skipping Stage 1 and resuming Stage 2 directly.")
        run_stage1 = False
        restored_from = latest_checkpoint
    elif not run_stage1:
        # If we're skipping Stage 1, we need to load from a Stage 1 checkpoint
        # Try best weights first, then regular stage1 weights
        if os.path.exists(stage1_best_checkpoint):
            restored_from = _find_latest_checkpoint(stage1_best_checkpoint)
            if restored_from:
                print(f"📂 Loading from Stage 1 best checkpoint: {restored_from}")
        
        if not restored_from and os.path.exists(stage1_checkpoint):
            restored_from = _find_latest_checkpoint(stage1_checkpoint)
            if restored_from:
                print(f"📂 Loading from Stage 1 checkpoint: {restored_from}")
        
        if not restored_from:
            print(f"⚠️  WARNING: No Stage 1 or Stage 2 checkpoint found!")
            print(f"   Checked locations:")
            print(f"   - {stage1_best_checkpoint}")
            print(f"   - {stage1_checkpoint}")
            print(f"   - {CHECKPOINT_DIR}")
            print(f"   Starting Stage 2 with random initialization (not recommended!)")
            restored_from = None
    
    if run_stage1:
        print(f"\n{'='*60}")
        print(f"🚀 STARTING STAGE 1: MVP IMITATION (Teacher Cloning)")
        print(f"{'='*60}")
        
        # Build Stage 1 Trainer (Custom Loss, No Critic)
        # Note: We use a smaller number of iterations for Stage 1 (e.g., 50)
        trainer_stage1 = build_trainer(N_AGENTS, stage=1)
        
        print("Training Stage 1...")
        # History for Stage 1 imitation loss
        stage1_loss_history = []
        best_stage1_loss = float('inf')
        best_stage1_iteration = 0
        best_stage1_checkpoint_path = None

        for i in range(1, iterations_stage1 + 1): # Run 50 iterations of cloning
            result = trainer_stage1.train()

            # Extract custom loss to print progress
            # Try multiple paths where the loss might be stored
            loss = "N/A"
            loss_val = None
            try:
                # 1. Check custom_metrics (set by callback - most reliable)
                if 'custom_metrics' in result and 'imitation_loss' in result['custom_metrics']:
                    loss_val = result['custom_metrics']['imitation_loss']
                
                # 2. Try to get from policy stats directly
                if loss_val is None:
                    policy = trainer_stage1.get_policy("shared_policy")
                    if hasattr(policy, 'loss_stats') and 'imitation_loss' in policy.loss_stats:
                        loss_val = policy.loss_stats['imitation_loss']
                
                # 3. Try standard RLlib learner stats paths
                if loss_val is None:
                    learner_stats = result.get('info', {}).get('learner', {}).get('shared_policy', {}).get('learner_stats', {})
                    loss_val = learner_stats.get('total_loss') or learner_stats.get('imitation_loss')
                
                # If we found a valid loss value, format and record it
                if loss_val is not None and loss_val != 'N/A':
                    try:
                        loss_val_float = float(loss_val)
                        loss = f"{loss_val_float:.6f}"
                        stage1_loss_history.append(loss_val_float)
                    except (ValueError, TypeError):
                        loss = "N/A"
            except Exception as e:
                # Debug: print available keys on first iteration
                if i == 1:
                    print(f"[DEBUG] Available result keys: {list(result.keys())}")
                    if 'custom_metrics' in result:
                        print(f"[DEBUG] custom_metrics keys: {list(result['custom_metrics'].keys())}")
                    if 'info' in result:
                        print(f"[DEBUG] Info keys: {list(result['info'].keys())}")
                loss = "N/A"

            print(f"Stage 1 - Iter {i}/{iterations_stage1} | Imitation Loss: {loss}")
            
            # Track best checkpoint based on lowest loss
            if loss_val is not None and isinstance(loss_val, (int, float)):
                if loss_val < best_stage1_loss:
                    best_stage1_loss = loss_val
                    best_stage1_iteration = i
                    # Save best checkpoint
                    best_stage1_checkpoint_dir = os.path.join(CHECKPOINT_DIR, "stage1_best_weights")
                    best_result = trainer_stage1.save(best_stage1_checkpoint_dir)
                    if hasattr(best_result, 'checkpoint') and hasattr(best_result.checkpoint, 'path'):
                        best_stage1_checkpoint_path = best_result.checkpoint.path
                    else:
                        best_stage1_checkpoint_path = str(best_result)
                    print(f"   ⭐ New best Stage 1 loss: {best_stage1_loss:.6f} (saved to stage1_best_weights)")

        # Save the "Safe" Policy
        print("💾 Saving Stage 1 (Teacher) weights...")
        stage1_result = trainer_stage1.save(stage1_checkpoint)
        
        # Handle different return types from .save()
        if hasattr(stage1_result, 'checkpoint') and hasattr(stage1_result.checkpoint, 'path'):
            stage1_path = stage1_result.checkpoint.path
        else:
            stage1_path = str(stage1_result)
            
        print(f"✅ Stage 1 Complete. Checkpoint saved: {stage1_path}")
        
        # Print best checkpoint information
        if best_stage1_checkpoint_path:
            print(f"⭐ Best Stage 1 checkpoint: Iteration {best_stage1_iteration} | Loss: {best_stage1_loss:.6f}")
            print(f"   Saved at: {best_stage1_checkpoint_path}")
        
        trainer_stage1.stop() # Free memory
        
        # Set this as the restore point for Stage 2 (use best weights if available)
        restored_from = best_stage1_checkpoint_path if best_stage1_checkpoint_path else stage1_path
        if RUN_STAGE_2:
            print(f"🔜 Transitioning to Stage 2 (Loading from: {restored_from})")
        else:
            print(f"\n✅ Stage 1 training complete. Stage 2 is disabled (RUN_STAGE_2=False).")
            print(f"   Checkpoint saved at: {stage1_path}")
            # Save recorded Stage 1 imitation loss history to CSV and PNG
            os.makedirs(METRICS_DIR, exist_ok=True)
            csv_path = os.path.join(METRICS_DIR, f"stage1_imitation_loss_{RUN_ID}.csv")
            try:
                with open(csv_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["iteration", "imitation_loss"])
                    for idx, val in enumerate(stage1_loss_history, start=1):
                        w.writerow([idx, float(val)])
                print(f"Stage1 imitation loss CSV saved to: {csv_path}")
            except Exception as e:
                print(f"Error saving Stage1 loss CSV: {e}")

            png_path = os.path.join(METRICS_DIR, f"stage1_imitation_loss_{RUN_ID}.png")
            try:
                if stage1_loss_history:
                    plt.figure(figsize=(8, 4))
                    plt.plot(stage1_loss_history, marker='o')
                    plt.title("Stage 1 Imitation Loss")
                    plt.xlabel("Iteration")
                    plt.ylabel("Imitation Loss")
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(png_path)
                    plt.close()
                    print(f"Stage1 imitation loss plot saved to: {png_path}")
                else:
                    print("No Stage1 imitation loss values recorded; skipping plot generation.")
            except Exception as e:
                print(f"Error saving Stage1 loss plot: {e}")

            ray.shutdown()
            sys.exit(0)


    # ==============================================================================
    # STAGE 2: RL FINE-TUNING (Standard PPO)
    # ==============================================================================
    if RUN_STAGE_2:
        print(f"\n{'='*60}")
        print(f"🚀 STARTING STAGE 2: PPO RL OPTIMIZATION (Maximizing Reward)")
        print(f"{'='*60}")
        
        target_iters = int(TOTAL_ITERS)
        
        # Build Stage 2 Trainer (Standard SAC Loss)
        # We pass 'restored_from' to load the Stage 1 weights!
        print(f"Building Stage 2 Trainer with {N_AGENTS} agents...")
        algo = build_trainer(N_AGENTS, stage=2, restore_path=restored_from)
        
        # Verify log_std was set correctly to reduce entropy
        policy = algo.get_policy("shared_policy")
        model = policy.model
        if hasattr(model, 'log_std'):
            log_std_val = model.log_std.mean().item()
            expected_std = np.exp(log_std_val)
            expected_entropy = 0.5 * np.log(2 * np.pi * np.e * expected_std**2) * 2  # For 2D action
            print(f"[Config] Initial log_std: {log_std_val:.3f} → std: {expected_std:.3f} → expected entropy: ~{expected_entropy:.2f}")
        elif hasattr(model, '_log_std'):
            log_std_val = model._log_std.mean().item()
            expected_std = np.exp(log_std_val)
            expected_entropy = 0.5 * np.log(2 * np.pi * np.e * expected_std**2) * 2
            print(f"[Config] Initial _log_std: {log_std_val:.3f} → std: {expected_std:.3f} → expected entropy: ~{expected_entropy:.2f}")
    else:
        print(f"\n⏭️  Stage 2 is disabled (RUN_STAGE_2=False). Exiting...")
        ray.shutdown()
        sys.exit(0)

    # Update starting iteration count if we restored from a Stage 2 checkpoint
    # (If we restored from Stage 1, algo.iteration is usually reset or 0)
    start_iter = algo.iteration + 1
    
    # Warm-up phase tracking
    warmup_complete = False
    
    print(f"\n{'='*60}")
    print(f"🧊 WARM-UP PHASE: Freezing Policy for {WARMUP_ITERATIONS} iterations")
    print(f"   Policy LR: {WARMUP_LR:.2e} (effectively frozen)")
    print(f"   Value Function will learn from pre-trained policy trajectories")
    print(f"{'='*60}\n")
    
    # ... [YOUR METRIC TRACKING VARIABLES REMAIN THE SAME] ...
    total_loss_history = []
    policy_loss_history = []
    entropy_history = []
    alpha_history = []
    q_loss_history = []
    vf_explained_var_history = []  # Track Value Function Explained Variance
    reward_history = []
    episode_length_history = []
    total_training_steps = 0
    best_reward = float('-inf')
    best_reward_iteration = 0
    best_checkpoint_path = None
    best_smoothed_reward = float('-inf')
    iterations_without_improvement = 0
    early_stop_triggered = False

    # --- Main Training Loop ---
    for i in range(1, target_iters+1):
        # Check if we need to unfreeze the policy after warm-up
        if i == WARMUP_ITERATIONS + 1 and not warmup_complete:
            warmup_complete = True
            print(f"\n{'='*60}")
            print(f"🔥 UNFREEZING POLICY: Warm-up complete!")
            print(f"   Updating learning rate: {WARMUP_LR:.2e} → {FINETUNE_LR:.2e}")
            print(f"   Increasing log_std: -2.5 → 0.0 (for exploration)")
            print(f"   Both policy and value function will now optimize jointly")
            print(f"{'='*60}\n")
            
            # Update learning rate by accessing optimizer directly
            policy = algo.get_policy("shared_policy")
            if hasattr(policy, '_optimizer'):
                for param_group in policy._optimizer.param_groups:
                    param_group['lr'] = FINETUNE_LR
                print(f"✅ Learning rate updated successfully")
            else:
                print(f"⚠️  WARNING: Could not find optimizer to update learning rate")
            
            # Reset log_std to enable exploration
            policy = algo.get_policy("shared_policy")
            model = policy.model
            state_dict = model.state_dict()
            
            if '_append_free_log_std.log_std' in state_dict:
                with torch.no_grad():
                    old_val = state_dict['_append_free_log_std.log_std'].mean().item()
                    state_dict['_append_free_log_std.log_std'].fill_(0.0)  # std = 1.0
                    new_val = state_dict['_append_free_log_std.log_std'].mean().item()
                expected_std = np.exp(new_val)
                expected_entropy = 0.5 * np.log(2 * np.pi * np.e * expected_std**2) * 2
                print(f"✅ Increased log_std: {old_val:.3f} → {new_val:.3f} (std={expected_std:.3f}, entropy≈{expected_entropy:.2f})")
            else:
                print(f"⚠️  WARNING: Could not find log_std to increase!")
            
            print()
        
        result = algo.train()

        # Extract metrics
        env_runners = result.get("env_runners", {})
        mean_rew = env_runners.get("episode_return_mean", float("nan"))
        ep_len = env_runners.get("episode_len_mean", float("nan"))
        
        # Debug: Print available keys on first iteration to verify data structure
        if i == 1:
            print(f"\n[DEBUG] Available result keys: {list(result.keys())}")
            if "info" in result:
                info_debug = result["info"]
                print(f"[DEBUG] Info keys: {list(info_debug.keys())}")
                if "learner" in info_debug and "shared_policy" in info_debug.get("learner", {}):
                    learner_debug = info_debug["learner"]["shared_policy"]
                    print(f"[DEBUG] Learner keys: {list(learner_debug.keys())}")
                    if "learner_stats" in learner_debug:
                        print(f"[DEBUG] Learner stats keys: {list(learner_debug['learner_stats'].keys())}")
            print()
        
        # ... [Keep your scalar conversion and history appending logic] ...
        # For brevity, I am assuming you paste your logic here
        
        # (Re-creating necessary variables for the print statement below)
        timesteps_this_iter = result.get("num_env_steps_sampled_this_iter", 0)
        total_training_steps += int(timesteps_this_iter) if isinstance(timesteps_this_iter, (int, float)) else 0
        
        # Extract PPO-specific loss components
        info = result.get("info", {})
        learner_stats = info.get("learner", {}).get("shared_policy", {}).get("learner_stats", {})
        
        # PPO uses different keys than SAC
        policy_loss = learner_stats.get("policy_loss", learner_stats.get("pi_loss", 0.0))
        vf_loss = learner_stats.get("vf_loss", learner_stats.get("value_function_loss", 0.0))
        entropy = learner_stats.get("entropy", 0.0)
        vf_explained_var = learner_stats.get("vf_explained_var", 0.0)  # Value Function Explained Variance
        total_loss = learner_stats.get("total_loss", abs(policy_loss) + abs(vf_loss))
        
        # Store histories
        total_loss_history.append(total_loss)
        policy_loss_history.append(policy_loss)
        q_loss_history.append(vf_loss)  # Reusing q_loss_history for value loss
        entropy_history.append(entropy)
        vf_explained_var_history.append(vf_explained_var)  # Track Critic accuracy
        reward_history.append(mean_rew)
        episode_length_history.append(ep_len)
        
        # Enhanced progress display with warm-up phase indicator
        phase_indicator = "[WARM-UP] 🧊" if i <= WARMUP_ITERATIONS else "[FINE-TUNE] 🔥"
        
        # During warm-up, show detailed loss breakdown to verify freeze is working
        if i <= WARMUP_ITERATIONS:
            print(f"Stage 2 {phase_indicator} - Iter {i}/{target_iters} | Reward: {mean_rew:.3f} | Total Loss: {total_loss:.4f}")
            print(f"       Policy Loss: {policy_loss:.4f} | Value Loss: {vf_loss:.4f} | Entropy: {entropy:.4f}")
            print(f"       VF Explained Var: {vf_explained_var:.4f} (Critic accuracy: 1.0=perfect, 0=guessing)")
            print(f"       (Policy should have minimal loss during warm-up due to frozen LR)")
        else:
            # After warm-up, show entropy to track exploration vs exploitation balance
            print(f"Stage 2 {phase_indicator} - Iter {i}/{target_iters} | Reward: {mean_rew:.3f} | Loss: {total_loss:.3f} | Entropy: {entropy:.3f} | VF_Var: {vf_explained_var:.3f}")
            if i == WARMUP_ITERATIONS + 2:
                print(f"       (Entropy should be higher now for exploration - PPO will gradually reduce it)")

        # --- Best Checkpoint Tracking ---
        if i > 10 and not np.isnan(mean_rew) and mean_rew > best_reward:
            best_reward = mean_rew
            best_reward_iteration = i
            best_checkpoint_dir = os.path.join(CHECKPOINT_DIR, f"best_iter_{i:05d}")
            # Use 'checkpoint=' arg if saving locally, or just path
            res = algo.save(best_checkpoint_dir)
            best_checkpoint_path = res.checkpoint.path if hasattr(res, 'checkpoint') else str(res)
            print(f"   ⭐ New best reward: {best_reward:.3f}")

        # --- Early Stopping Logic (Keep your code) ---
        # ...

        # --- EVALUATION INTERVAL (Your Request) ---
        # This is where your code was correct, just ensure EVALUATION_INTERVAL is defined
        if EVALUATION_INTERVAL and i % EVALUATION_INTERVAL == 0:
            print(f"\n🔄 EVALUATION at iteration {i}")
            # Save periodic checkpoint
            algo.save(CHECKPOINT_DIR)
            
            # Run custom evaluation function
            try:
                eval_metrics = run_fixed_eval(algo, n_episodes=10, n_agents=N_AGENTS)
                print(f"   [Eval] Avg Reward: {eval_metrics['avg_reward']:.3f}")
                print(f"   [Eval] Waypoint Rate: {eval_metrics['waypoint_rate']*100:.1f}%")
                print(f"   [Eval] Avg Intrusions: {eval_metrics['avg_intrusions']:.2f}")
                
                # Save evaluation metrics to CSV
                eval_csv_dir = os.path.join(METRICS_DIR, f"run_{RUN_ID}")
                _write_eval_row(eval_metrics, iteration=i, out_dir=eval_csv_dir)
                print(f"   [Eval] ✅ Saved to evaluation_progress.csv")
            except Exception as e:
                print(f"   [Eval] ❌ Error during evaluation at iter {i}: {e}")
                import traceback, datetime
                tb = traceback.format_exc()
                # Ensure metrics dir exists and write a traceback log for debugging
                try:
                    os.makedirs(eval_csv_dir, exist_ok=True)
                    log_path = os.path.join(eval_csv_dir, f"eval_error_iter_{i}.log")
                    with open(log_path, "w", encoding="utf-8") as fh:
                        fh.write(f"Timestamp: {datetime.datetime.utcnow().isoformat()}Z\n")
                        fh.write(tb)
                    print(f"   [Eval] Traceback written to: {log_path}")
                except Exception as _e_log:
                    print(f"   [Eval] Failed to write traceback log: {_e_log}")

                # As a fallback, write a placeholder CSV row so plotting tools see a file
                try:
                    placeholder = {
                        "avg_reward": float("nan"),
                        "avg_length": float("nan"),
                        "avg_intrusions": float("nan"),
                        "waypoint_rate": float("nan"),
                    }
                    _write_eval_row(placeholder, iteration=i, out_dir=eval_csv_dir)
                    print(f"   [Eval] ⚠️ Wrote placeholder evaluation row for iteration {i}")
                except Exception as _e_csv:
                    print(f"   [Eval] Failed to write placeholder CSV row: {_e_csv}")

    # ... [END OF LOOP] ...

    # ========================================================================
    # POST-TRAINING: Save all metrics to CSV
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"💾 SAVING TRAINING METRICS TO CSV")
    print(f"{'='*60}")
    
    # Save training metrics (entropy, loss, reward, etc.)
    training_metrics_path = os.path.join(METRICS_DIR, f"run_{RUN_ID}", "stage2_training_metrics.csv")
    os.makedirs(os.path.dirname(training_metrics_path), exist_ok=True)
    
    try:
        with open(training_metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "reward", "episode_length", "total_loss", "policy_loss", "value_loss", "entropy", "vf_explained_var"])
            for idx in range(len(reward_history)):
                writer.writerow([
                    idx + 1,
                    reward_history[idx],
                    episode_length_history[idx] if idx < len(episode_length_history) else 0,
                    total_loss_history[idx] if idx < len(total_loss_history) else 0,
                    policy_loss_history[idx] if idx < len(policy_loss_history) else 0,
                    q_loss_history[idx] if idx < len(q_loss_history) else 0,
                    entropy_history[idx] if idx < len(entropy_history) else 0,
                    vf_explained_var_history[idx] if idx < len(vf_explained_var_history) else 0
                ])
        print(f"✅ Training metrics saved to: {training_metrics_path}")
    except Exception as e:
        print(f"❌ Error saving training metrics: {e}")
    
    # ========================================================================
    # POST-TRAINING: Merge per-agent CSV files
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"📊 MERGING PER-AGENT CSV FILES")
    print(f"{'='*60}")
    
    metrics_run_dir = os.path.join(METRICS_DIR, f"run_{RUN_ID}")
    
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
    axes[1].set_title("Training Losses")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Plot Entropy
    axes[2].plot(entropy_history, label="Entropy", color="green")
    axes[2].axvline(x=WARMUP_ITERATIONS, color='red', linestyle='--', alpha=0.5, label=f'Warmup End (iter {WARMUP_ITERATIONS})')
    axes[2].set_title("Policy Entropy (Exploration)")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Entropy")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # Plot Ep Length
    axes[3].plot(episode_length_history, label="Episode Length", color="teal")
    axes[3].set_title("Episode Length")
    axes[3].set_xlabel("Iteration")
    axes[3].set_ylabel("Steps")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()
    
    # Plot VF Explained Variance (Critic Accuracy)
    axes[4].plot(vf_explained_var_history, label="VF Explained Var", color="darkorange", linewidth=2)
    axes[4].axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect (1.0)')
    axes[4].axhline(y=0.0, color='red', linestyle='--', alpha=0.5, label='Random (0.0)')
    axes[4].axvline(x=WARMUP_ITERATIONS, color='red', linestyle='--', alpha=0.5, label=f'Warmup End (iter {WARMUP_ITERATIONS})')
    axes[4].set_title("Value Function Explained Variance (Critic Accuracy)")
    axes[4].set_xlabel("Iteration")
    axes[4].set_ylabel("Explained Variance")
    axes[4].set_ylim(-0.1, 1.05)  # Cap Y-axis at 1.0 with small margin
    axes[4].grid(True, alpha=0.3)
    axes[4].legend(loc='best', fontsize=8)
    
    # Hide the 6th subplot (unused in 3x2 grid)
    axes[5].set_visible(False)
    
    fig.suptitle(f"Stage 2 Training Summary (RUN_ID={RUN_ID})", fontsize=14, fontweight='bold')
    plot_path = os.path.join(METRICS_DIR, f"run_{RUN_ID}", "stage2_training_summary.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"✅ Training plots saved to: {plot_path}")
    plt.close() # Close memory
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    training_duration = time.time() - training_start_time
    print(f"\n{'='*60}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Duration: {training_duration/60:.1f} minutes")
    print(f"Total iterations: {target_iters}")
    print(f"Best reward: {best_reward:.3f} at iteration {best_reward_iteration}")
    print(f"\nFiles saved:")
    print(f"  - Training metrics CSV: {training_metrics_path}")
    print(f"  - Merged agent data: {merged_path if all_dfs else 'N/A'}")
    print(f"  - Evaluation progress: {os.path.join(METRICS_DIR, f'run_{RUN_ID}', 'evaluation_progress.csv')}")
    print(f"  - Training plots: {plot_path}")
    print(f"  - Best checkpoint: {best_checkpoint_path if 'best_checkpoint_path' in locals() else 'N/A'}")
    print(f"\nTo analyze results, run:")
    print(f"  python analyze_csv.py")
    print(f"  python plot_eval_progress.py")

    ray.shutdown()