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

from bluesky_gym.envs.ma_env_two_stage_AM_PPO_NOISE_autoencoder import SectorEnv
from ray.tune.registry import register_env

import torch
import torch.nn.functional as F

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.sample_batch import SampleBatch


from run_config import RUN_ID

# --- Path to pretrained Autoencoder ---
# Points to the .pt file copied into this folder.
AE_MODEL_PATH = os.path.join(script_dir, "autoencoder_pretrained.pt")
if not os.path.isfile(AE_MODEL_PATH):
    print(f"[AE] Pretrained model not found at {AE_MODEL_PATH} — AE noise signal will be 0.")
    AE_MODEL_PATH = None

# Register your custom environment with Gymnasium
# Register your custom environment directly for RLlib
register_env("sector_env", lambda config: SectorEnv(**config))
ModelCatalog.register_custom_model("attention_sac", AttentionSACModel)

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
        Capture Stage 1 imitation loss and attention metrics for TensorBoard logging.
        This ensures the loss and attention stats are available in trainer.train() results.
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
            
            # Get attention model metrics if available
            if hasattr(policy, 'model') and hasattr(policy.model, 'metrics'):
                try:
                    attention_metrics = policy.model.metrics()
                    for key, value in attention_metrics.items():
                        result.setdefault("custom_metrics", {})[f"attention_{key}"] = value
                except Exception:
                    pass
                    
        except Exception:
            # Don't break training if logging fails
            pass

# --- Parameters ---
N_AGENTS = 20  # Number of agents for training

# --- STAGE CONTROL ---
RUN_STAGE_2 = True  # Set to True to run Stage 2 after Stage 1, False to only train Stage 1

# --- STAGE 1: IMITATION LEARNING (PPO with custom loss) ---
iterations_stage1 = 80  # Number of iterations for Stage 1 imitation learning

# --- WARM-UP PHASE SETTINGS ---
WARMUP_ITERATIONS = 10  # Number of iterations to warm up critic with frozen policy and attention
WARMUP_LR = 1e-4  # Critic needs higher LR to learn from scratch (was 3e-5, still too low)
FINETUNE_LR = 5e-5  # Learning rate after warm-up for joint optimization

# --- STAGE 2: RL FINE-TUNING (PPO with standard loss) ---
TOTAL_ITERS = WARMUP_ITERATIONS + 120  # Maximum total iterations for Stage 2

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
    Fixed Stage 1 loss: Uses logits directly to maintain gradient flow.
    """
    # 1. Differentiable forward pass
    # logits is a tensor with requires_grad=True
    logits, _ = model(train_batch)
    
    # 2. Extract only the predicted means (the actions)
    # Even if RLlib padded the output to 4 (mean + log_std), 
    # we only want the first action_dim columns.
    action_dim = policy.action_space.shape[0]
    predicted_actions = logits[:, :action_dim]
    
    # 3. Get teacher targets (ensure they are on the correct device)
    teacher_targets = train_batch["teacher_targets"]
    if not isinstance(teacher_targets, torch.Tensor):
        teacher_targets = torch.tensor(
            teacher_targets, 
            dtype=torch.float32,
            device=predicted_actions.device
        )
    else:
        teacher_targets = teacher_targets.to(predicted_actions.device)
    
    # # 4. Compute MSE loss
    # # Since predicted_actions has grad_fn, imitation_loss will too!
    # imitation_loss = F.mse_loss(predicted_actions, teacher_targets)
    
    # # 5. Store for logging
    # policy.loss_stats = {"imitation_loss": imitation_loss.item()}
    
    # return imitation_loss
    
    # NEW CODE FROM GEMINI to WEIGHT THE LOSS BASED ON TEACHER ACTIVITY
    # 3. Calculate raw squared error (element-wise)
    squared_error = (predicted_actions - teacher_targets) ** 2
    
    # --- FIX: Weighted Loss Strategy ---
    # Calculate magnitude of teacher's action
    # We want to punish errors MORE if the teacher is actually doing something.
    teacher_magnitude = torch.norm(teacher_targets, dim=1, keepdim=True)
    
    # Create a weight vector:
    # If teacher is acting (magnitude > 0.05), multiply weight by 10.0
    # Otherwise keep weight at 1.0
    weights = torch.ones_like(teacher_magnitude)
    weights = torch.where(teacher_magnitude > 0.05, weights * 10.0, weights)
    
    # Apply weights (broadcasts to action dimension)
    weighted_loss = (squared_error * weights).mean()
    # -----------------------------------
    
    policy.loss_stats = {"imitation_loss": weighted_loss.item()}
    return weighted_loss


def build_trainer(n_agents, stage=1, restore_path=None):
    """
    Builds the PPO algorithm for both stages with different configurations.
    """
    
    # 1. Define Policy Mapping
    def policy_map(agent_id, *_, **__):
        return "shared_policy"

    # 2. Determine Stage-Specific Settings
    if stage == 1:
        # --- STAGE 1: IMITATION LEARNING ---
        print("[Config] Stage 1: Using PPO with custom imitation loss + Attention Model")
        current_callbacks = MVPDataBridgeCallback
        
        training_config = {
            # Optimization Params
            "lr": 1e-4,
            "train_batch_size": 32000,
            "minibatch_size": 2000, 
            "num_sgd_iter": 10,
            "grad_clip": 1.0,
            "gamma": 0.99,
            
            # Dead PPO Params (Ignored by custom loss)
            "entropy_coeff": 0.0,
            "vf_loss_coeff": 0.01,
            
            # --- MODEL CONFIGURATION ---
            "model": {
                "custom_model": "attention_sac",  # Must match registration string
                "custom_model_config": {
                    "hidden_dims": [512, 512],
                    "is_critic": False,
                    "n_agents": n_agents,   # Pass this to help reshape inputs
                    "embed_dim": 128,
                },
                "free_log_std": True,      # Allow PPO to learn std_dev separately
                "vf_share_layers": False,  # Separate Value Branch
            },
        }
        
    else:
        # --- STAGE 2: RL FINE-TUNING ---
        print("[Config] Stage 2: Using standard PPO for RL with Attention Model")
        current_callbacks = None  # No teacher needed
        
        training_config = {
            "lr": WARMUP_LR,  # Start with higher LR for critic learning during warm-up
            "train_batch_size": 64000,
            "minibatch_size": 2000,
            "num_sgd_iter": 12,
            "clip_param": 0.2,
            "vf_loss_coeff": 2.0,  # Increase to 2.0 so critic gradients are stronger during warm-up
            "entropy_coeff": 0.001,  # Very small → minimal exploration during warm-up (combined with low log_std)
            "grad_clip": 0.5,
            "gamma": 0.99,
            "lambda_": 0.95,
            
            "model": {
                "custom_model": "attention_sac", # Reuse the same architecture
                "custom_model_config": {
                    "hidden_dims": [512, 512],
                    "is_critic": False,
                    "n_agents": n_agents,
                },
                "free_log_std": True,
                "vf_share_layers": False,
            }
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
                "autoencoder_path": AE_MODEL_PATH,
            },
            disable_env_checking=True,
        )
        .framework("torch")
        .env_runners(
            num_env_runners= os.cpu_count() - 1,
            num_envs_per_env_runner=1,
            sample_timeout_s=120.0,
        )
        .callbacks(current_callbacks) 
        .training(**training_config)
        .multi_agent(
            policies={"shared_policy": (None, None, None, {})},
            policy_mapping_fn=policy_map,
        )
        .resources(num_gpus=1)
    )
    
    # 4. Build the Algorithm Instance
    algo = cfg.build()
    
    # 5. Inject Custom Loss (Stage 1 Only)
    if stage == 1:
        policy = algo.get_policy("shared_policy")
        original_loss_fn = policy.loss
        
        def custom_loss_wrapper(model, dist_class, train_batch):
            return stage1_imitation_loss(policy, model, dist_class, train_batch)
        
        policy.loss = custom_loss_wrapper
        print("[Config] Custom imitation loss function injected into PPO policy")
    
    # 5.5. For Stage 2, set log_std BEFORE any restore (critical for warm-up)
    if stage == 2:
        policy = algo.get_policy("shared_policy")
        model = policy.model
        
        # Debug: Print all parameter names to find log_std
        print("\n[DEBUG] All model parameters:")
        for name, param in model.named_parameters():
            print(f"  {name}: shape={param.shape}")
        print()
        
        # Set log_std to -2.5 for low entropy during warm-up
        # For custom attention model, log_std is a direct parameter
        if hasattr(model, 'log_std'):
            with torch.no_grad():
                old_val = model.log_std.mean().item()
                model.log_std.fill_(-2.5)
                new_val = model.log_std.mean().item()
            expected_std = np.exp(new_val)
            expected_entropy = 0.5 * np.log(2 * np.pi * np.e * expected_std**2) * 2
            print(f"[Config] Set log_std for warm-up: {old_val:.3f} → {new_val:.3f} (std={expected_std:.4f}, entropy≈{expected_entropy:.2f})")
        else:
            print(f"[WARNING] Model does not have log_std parameter!")
            print(f"[WARNING] Entropy control will not work!")

    # 6. Restore Weights (if provided)
    if restore_path:
        print(f"Restoring weights from: {restore_path}")
        algo.restore(restore_path)
        
        # After restoring, re-apply log_std setting for Stage 2 (checkpoint may have overwritten it)
        if stage == 2:
            policy = algo.get_policy("shared_policy")
            model = policy.model
            
            # Reset log_std again after restore
            if hasattr(model, 'log_std'):
                with torch.no_grad():
                    old_val = model.log_std.mean().item()
                    model.log_std.fill_(-2.5)
                    new_val = model.log_std.mean().item()
                print(f"[Config] Reset log_std after restore: {old_val:.3f} → {new_val:.3f}")
            else:
                print(f"[WARNING] Could not reset log_std after restore")

    return algo



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

def run_fixed_eval(algo: Algorithm, n_episodes: int = 20, render: bool = False, n_agents: int = N_AGENTS, silent: bool = True, compare_teacher: bool = False):
    """Run a small deterministic evaluation (no exploration) and return metrics.

    Returns a dict with avg_reward, avg_length, avg_intrusions, waypoint_rate,
    and raw per-episode lists.
    
    Args:
        silent: If True, suppresses BlueSky simulation output during evaluation.
        n_agents: Number of agents to use in evaluation environment.
        compare_teacher: If True, print teacher (MVP) vs model actions for comparison.
    """
    # OLD API: Use get_policy instead of get_module
    policy = algo.get_policy("shared_policy")
    
    # Wrap the entire evaluation in output suppression if silent=True
    def _run_episodes():
        env = SectorEnv(
            render_mode="human" if render else None,
            n_agents=n_agents,
            run_id=RUN_ID,
            metrics_base_dir=METRICS_DIR,
            autoencoder_path=AE_MODEL_PATH,
        )
        rewards, lengths, intrusions, waypoints = [], [], [], []

        for ep_idx in range(n_episodes):
            obs, _ = env.reset()
            ep_rew = 0.0
            ep_len = 0
            step_count = 0
            while env.agents:
                # OLD API: Use policy.compute_actions
                agent_ids = list(obs.keys())
                obs_array = np.stack(list(obs.values()))
                
                # Compute deterministic actions (no exploration)
                actions_np = policy.compute_actions(obs_array, explore=False)[0]
                
                actions = {aid: act for aid, act in zip(agent_ids, actions_np)}
                
                # Compare with teacher actions if requested
                if compare_teacher and step_count % 20 == 0:  # Print every 20 steps to avoid spam
                    # Get teacher actions for comparison
                    print(f"\n[Eval Episode {ep_idx+1}, Step {step_count}] Teacher vs Model Actions:")
                    for i, agent_id in enumerate(agent_ids[:3]):  # Show first 3 agents only
                        teacher_action = env._calculate_mvp_action(agent_id)
                        model_action = actions_np[i]
                        print(f"  {agent_id}:")
                        print(f"    Teacher: [{teacher_action[0]:+.3f}, {teacher_action[1]:+.3f}]")
                        print(f"    Model:   [{model_action[0]:+.3f}, {model_action[1]:+.3f}]")
                        # Calculate action difference
                        diff = np.abs(teacher_action - model_action)
                        print(f"    Diff:    [{diff[0]:.3f}, {diff[1]:.3f}]")
                
                obs, rew, term, trunc, infos = env.step(actions)
                if rew:
                    ep_rew += sum(rew.values())
                ep_len += 1
                step_count += 1
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
    import numpy as np
    write_header = not os.path.exists(path)
    
    # Calculate IQR for intrusions and waypoint rate
    intrusions_data = metrics.get("per_episode_intrusions", [])
    waypoints_data = metrics.get("per_episode_waypoints", [])
    n_episodes = len(intrusions_data)
    n_agents = 20  # Adjust if different
    
    # Calculate percentiles for intrusions
    intr_q25 = np.percentile(intrusions_data, 25) if intrusions_data else 0
    intr_q75 = np.percentile(intrusions_data, 75) if intrusions_data else 0
    
    # Calculate percentiles for waypoint rate (per episode)
    if waypoints_data and n_episodes > 0:
        waypoint_rates = [w / n_agents * 100 for w in waypoints_data]  # Convert to %
        wp_q25 = np.percentile(waypoint_rates, 25)
        wp_q75 = np.percentile(waypoint_rates, 75)
    else:
        wp_q25 = 0
        wp_q75 = 0
    
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "iteration",
                "avg_reward",
                "avg_length",
                "avg_intrusions",
                "intrusions_q25",
                "intrusions_q75",
                "waypoint_rate",
                "waypoint_q25",
                "waypoint_q75",
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
                "intrusions_q25": round(intr_q25, 2),
                "intrusions_q75": round(intr_q75, 2),
                "waypoint_rate": round(metrics["waypoint_rate"], 4),
                "waypoint_q25": round(wp_q25, 2),
                "waypoint_q75": round(wp_q75, 2),
            }
        )

if __name__ == "__main__":
    
    # Start timing
    training_start_time = time.time()
    
    ray.shutdown()
    # Initialize Ray with runtime environment so workers can find attention_model_A
    ray.init(runtime_env={
        "working_dir": script_dir,
        "py_modules": [os.path.join(script_dir, "attention_model_A.py")],
    })

    print("-" * 30)

    # ==============================================================================
    # STAGE 1: TEACHER IMITATION (Supervised)
    # ==============================================================================
    # We only run this if we are NOT restoring from an existing Stage 2 checkpoint
    # and if we actually want to run stage 1.
    
    stage1_checkpoint = os.path.join(CHECKPOINT_DIR, "stage1_best_weights")
    run_stage1 = False if RUN_STAGE_2 else True  # Only run Stage 1 if not running Stage 2, otherwise we will restore from checkpoint
    restored_from = None  # Initialize to None - will be set if checkpoint found or Stage 1 runs
    
    # Check if we are trying to resume a Stage 2 run
    latest_checkpoint = _find_latest_checkpoint(CHECKPOINT_DIR)
    if latest_checkpoint:
        print(f"🔄 Found existing Stage 2 checkpoint: {latest_checkpoint}")
        print("⏭️  Skipping Stage 1 and resuming Stage 2 directly.")
        run_stage1 = False
        restored_from = latest_checkpoint
    
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
            except Exception:
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
    print(f"🧊 WARM-UP PHASE: Critic Learning for {WARMUP_ITERATIONS} iterations")
    print(f"   Learning Rate: {WARMUP_LR:.2e} (high enough for critic to learn from scratch)")
    print(f"   VF Loss Coeff: 2.0 (increased for stronger critic gradients)")
    print(f"   Entropy: ~-2.0 (low std=0.082, policy stays near teacher actions)")
    print(f"   Attention Temperature: 3.0 (learnable, will adapt focus sharpness)")
    print(f"   Critic will learn to evaluate pre-trained policy behavior")
    print(f"{'='*60}\n")
    
    # ... [YOUR METRIC TRACKING VARIABLES REMAIN THE SAME] ...
    total_loss_history = []
    policy_loss_history = []
    entropy_history = []
    alpha_history = []
    q_loss_history = []
    vf_explained_var_history = []
    temperature_history = []
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
        # Check if we need to increase LR and entropy after warm-up
        if i == WARMUP_ITERATIONS + 1 and not warmup_complete:
            warmup_complete = True
            print(f"\n{'='*60}")
            print(f"🔥 FINE-TUNING PHASE: Warm-up complete!")
            print(f"   Increasing learning rate: {WARMUP_LR:.2e} → {FINETUNE_LR:.2e}")
            print(f"   Increasing log_std: -2.5 → 0.0 (enabling exploration)")
            print(f"   Note: entropy_coeff stays at {algo.config.entropy_coeff} (can't change after init)")
            print(f"   Critic has learned to evaluate policy, now jointly optimizing")
            print(f"{'='*60}\n")
            
            # Update learning rate by accessing worker's policy optimizer
            policy = algo.get_policy("shared_policy")
            
            # Try multiple paths to find the optimizer
            optimizer_found = False
            
            # Try _optimizer attribute first (most common in old API)
            if hasattr(policy, '_optimizer') and policy._optimizer is not None:
                for param_group in policy._optimizer.param_groups:
                    param_group['lr'] = FINETUNE_LR
                optimizer_found = True
                print(f"✅ Learning rate updated via policy._optimizer")
            
            # Try _optimizers (sometimes it's a list)
            elif hasattr(policy, '_optimizers') and len(policy._optimizers) > 0:
                for opt in policy._optimizers:
                    for param_group in opt.param_groups:
                        param_group['lr'] = FINETUNE_LR
                optimizer_found = True
                print(f"✅ Learning rate updated via policy._optimizers")
            
            # Last resort: rebuild config with new LR (not ideal but works)
            if not optimizer_found:
                print(f"⚠️  WARNING: Could not find optimizer directly")
                print(f"   Learning rate will stay at {WARMUP_LR:.2e}")
                print(f"   (LR increase less critical than log_std for exploration)")
            
            # Increase log_std to enable exploration
            model = policy.model
            if hasattr(model, 'log_std'):
                with torch.no_grad():
                    old_val = model.log_std.mean().item()
                    model.log_std.fill_(0.0)  # std = 1.0
                    new_val = model.log_std.mean().item()
                expected_std = np.exp(new_val)
                expected_entropy = 0.5 * np.log(2 * np.pi * np.e * expected_std**2) * 2
                print(f"✅ Increased log_std: {old_val:.3f} → {new_val:.3f} (std={expected_std:.3f}, entropy≈{expected_entropy:.2f})")
            else:
                print(f"⚠️  WARNING: Could not find log_std to increase!")
            print()
        
        result = algo.train()

        # ... [YOUR METRIC EXTRACTION CODE REMAINS THE SAME] ...
        # (It was very good, keep it exactly as you wrote it)
        env_runners = result.get("env_runners", {})
        mean_rew = env_runners.get("episode_return_mean", float("nan"))
        ep_len = env_runners.get("episode_len_mean", float("nan"))
        
        # ... [Keep your scalar conversion and history appending logic] ...
        # For brevity, I am assuming you paste your logic here
        
        # (Re-creating necessary variables for the print statement below)
        timesteps_this_iter = result.get("num_env_steps_sampled_this_iter", 0)
        total_training_steps += int(timesteps_this_iter) if isinstance(timesteps_this_iter, (int, float)) else 0
        
        # Simplified extraction for context:
        info = result.get("info", {})
        learner_stats = info.get("learner", {}).get("shared_policy", {}).get("learner_stats", {})
        policy_loss = learner_stats.get("policy_loss", learner_stats.get("pi_loss", 0.0))
        vf_loss = learner_stats.get("vf_loss", learner_stats.get("value_function_loss", 0.0))
        entropy = learner_stats.get("entropy", 0.0)
        vf_explained_var = learner_stats.get("vf_explained_var", 0.0)
        total_loss = learner_stats.get("total_loss", abs(policy_loss) + abs(vf_loss))
        
        total_loss_history.append(total_loss)
        policy_loss_history.append(policy_loss)
        q_loss_history.append(vf_loss)
        entropy_history.append(entropy)
        vf_explained_var_history.append(vf_explained_var)
        reward_history.append(mean_rew)
        episode_length_history.append(ep_len)
        
        # Extract temperature from attention model
        policy = algo.get_policy("shared_policy")
        if hasattr(policy, 'model') and hasattr(policy.model, 'temperature'):
            current_temp = policy.model.temperature.item()
            temperature_history.append(current_temp)
        else:
            temperature_history.append(float('nan'))
        
        # Enhanced progress display with warm-up phase indicator
        phase_indicator = "[WARM-UP] 🧊" if i <= WARMUP_ITERATIONS else "[FINE-TUNE] 🔥"
        
        # During warm-up, show detailed loss breakdown to monitor critic learning
        if i <= WARMUP_ITERATIONS:
            print(f"Stage 2 {phase_indicator} - Iter {i}/{target_iters} | Reward: {mean_rew:.3f} | Total Loss: {total_loss:.4f}")
            print(f"       Policy Loss: {policy_loss:.4f} | Value Loss: {vf_loss:.4f} | Entropy: {entropy:.4f}")
            print(f"       VF Explained Var: {vf_explained_var:.4f} (1.0=perfect, 0.0=random)")
            print(f"       (LR={WARMUP_LR:.2e}, VF_Coeff=2.0 - critic learning to evaluate teacher policy)")
        else:
            # After warm-up, show entropy to track exploration vs exploitation balance
            print(f"Stage 2 {phase_indicator} - Iter {i}/{target_iters} | Reward: {mean_rew:.3f} | Loss: {total_loss:.3f} | Entropy: {entropy:.3f} | VF_Var: {vf_explained_var:.3f}")
            if i == WARMUP_ITERATIONS + 2:
                print(f"       (LR={FINETUNE_LR:.2e} - entropy increased for exploration)")

        # --- Best Checkpoint Tracking ---
        if i > 10 and not np.isnan(mean_rew) and mean_rew > best_reward:
            best_reward = mean_rew
            best_reward_iteration = i
            best_checkpoint_dir = os.path.join(CHECKPOINT_DIR, f"best_iter_{i:05d}")
            try:
                res = algo.save(best_checkpoint_dir)
                # Ray 2.x returns a Checkpoint object directly, not a Result with .checkpoint
                if hasattr(res, 'checkpoint') and hasattr(res.checkpoint, 'path'):
                    best_checkpoint_path = res.checkpoint.path
                elif hasattr(res, 'path'):
                    best_checkpoint_path = res.path
                else:
                    best_checkpoint_path = str(res)
            except Exception as _e:
                best_checkpoint_path = best_checkpoint_dir
            print(f"   ⭐ New best reward: {best_reward:.3f}")

        # --- Early Stopping Logic (Keep your code) ---
        # ...

        # --- EVALUATION INTERVAL (Your Request) ---
        # This is where your code was correct, just ensure EVALUATION_INTERVAL is defined
        if EVALUATION_INTERVAL and i % EVALUATION_INTERVAL == 0:
            print(f"\n🔄 EVALUATION at iteration {i}")
            # Save periodic checkpoint
            algo.save(CHECKPOINT_DIR)
            
            # Run custom evaluation function with teacher comparison
            if 'run_fixed_eval' in globals():
                try:
                    eval_metrics = run_fixed_eval(algo, n_episodes=10, n_agents=N_AGENTS, compare_teacher=True)
                    print(f"   [Eval] Avg Reward: {eval_metrics['avg_reward']:.3f}")
                    # Write evaluation metrics to CSV
                    eval_out_dir = os.path.join(METRICS_DIR, f"run_{RUN_ID}")
                    _write_eval_row(eval_metrics, i, eval_out_dir)
                except Exception as e:
                    print(f"   [Eval] Error: {e}")

    # ... [END OF LOOP] ...

    # --- Plotting (UPDATED) ---
    # Use savefig instead of show to prevent freezing
    fig, axes = plt.subplots(6, 1, figsize=(10, 24))
    
    # Plot Reward
    axes[0].plot(reward_history, label="Reward")
    axes[0].set_title("Training Reward")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Reward")
    axes[0].grid(True)
    axes[0].legend()
    
    # Plot Loss
    axes[1].plot(total_loss_history, label="Total Loss", color="orange")
    axes[1].set_title("Training Loss")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True)
    axes[1].legend()
    
    # Plot Entropy
    axes[2].plot(entropy_history, label="Entropy", color="purple")
    axes[2].set_title("Policy Entropy")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Entropy")
    axes[2].grid(True)
    axes[2].legend()
    
    # Plot Value Function Explained Variance (Critic Accuracy)
    axes[3].plot(vf_explained_var_history, label="VF Explained Variance", color="red")
    axes[3].set_title("Value Function Explained Variance (Critic Accuracy)")
    axes[3].set_xlabel("Iteration")
    axes[3].set_ylabel("Explained Variance")
    axes[3].axhline(y=1.0, color='green', linestyle='--', alpha=0.3, label='Perfect (1.0)')
    axes[3].axhline(y=0.0, color='gray', linestyle='--', alpha=0.3, label='Random (0.0)')
    axes[3].grid(True)
    axes[3].legend()
    
    # Plot Attention Temperature
    axes[4].plot(temperature_history, label="Attention Temperature", color="cyan")
    axes[4].set_title("Attention Temperature (Learnable Parameter)")
    axes[4].set_xlabel("Iteration")
    axes[4].set_ylabel("Temperature")
    axes[4].axhline(y=3.0, color='gray', linestyle='--', alpha=0.3, label='Initial (3.0)')
    axes[4].grid(True)
    axes[4].legend()
    
    # Plot Ep Length
    axes[5].plot(episode_length_history, label="Ep Length", color="green")
    axes[5].set_title("Episode Length")
    axes[5].set_xlabel("Iteration")
    axes[5].set_ylabel("Steps")
    axes[5].grid(True)
    axes[5].legend()
    
    plot_path = os.path.join(METRICS_DIR, f"training_summary_{RUN_ID}.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"\n📊 Training plots saved to: {plot_path}")
    plt.close() # Close memory
    
    # Save training metrics for later plotting
    metrics_dir = os.path.join(script_dir, "metrics", f"run_{RUN_ID}")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_file = os.path.join(metrics_dir, "training_metrics.pkl")
    import pickle
    with open(metrics_file, 'wb') as f:
        pickle.dump({
            'reward_history': reward_history,
            'total_loss_history': total_loss_history,
            'entropy_history': entropy_history,
            'vf_explained_var_history': vf_explained_var_history,
            'temperature_history': temperature_history,
            'episode_length_history': episode_length_history,
        }, f)
    print(f"📦 Training metrics saved to: {metrics_file}")

    ray.shutdown()