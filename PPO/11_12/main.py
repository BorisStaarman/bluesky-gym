# standard imports
import os
import sys
import shutil
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import io

# MARL ray imports
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig

# Make sure these imports point to your custom environment registration
from bluesky_gym import register_envs
from bluesky_gym.envs.ma_env_ppo import SectorEnv

from run_config import RUN_ID

# Register your custom environment with Gymnasium
register_envs()

# --- Parameters ---
# CURRICULUM LEARNING CONFIGURATION - SIMPLIFIED
# Train ONE stage at a time. After each stage completes, update CURRENT_STAGE and 
# LOAD_CHECKPOINT_FROM to continue with the next stage.

# Which stage to train (1, 2, or 3)
# CURRENT_STAGE = 1  # Change this to 1, 2, or 3

# Checkpoint to load weights from (set to None for stage 1, or paste path from previous stage)
# Example: r"c:\Users\boris\Documents\bsgym\bluesky-gym\PPO\11_12\models\sectorcr_ma_ppo\stage_1_easy\best_iter_00001"

# CHANGE THIS TO THE PATH OF PREVIOUS STAGE BEST RUN IF NOT STAGE 1
# checkpoint for stage 2
CURRENT_STAGE = 2
LOAD_CHECKPOINT_FROM = r"c:\Users\boris\Documents\bsgym\bluesky-gym\PPO\11_12\models\sectorcr_ma_ppo\stage_1_easy\best_iter_00080"
# checkpoint for stage 3
# LOAD_CHECKPOINT_FROM = r"c:\Users\boris\Documents\bsgym\bluesky-gym\PPO\11_12\models\sectorcr_ma_ppo\stage_2_medium\best_iter_00001"

# Stage configurations
STAGE_CONFIGS = {
    1: {"n_agents": 6, "iterations": 100, "name": "stage_1_easy"},      # TESTING: 1 iteration
    2: {"n_agents": 12, "iterations": 100, "name": "stage_2_medium"},   # TESTING: 1 iteration  
    3: {"n_agents": 20, "iterations": 100, "name": "stage_3_hard"},     # TESTING: 1 iteration
}
# For full training, change iterations to:
# 1: {"n_agents": 6, "iterations": 100, "name": "stage_1_easy"},
# 2: {"n_agents": 12, "iterations": 150, "name": "stage_2_medium"},
# 3: {"n_agents": 20, "iterations": 300, "name": "stage_3_hard"},

FORCE_RETRAIN = True        # TESTING: Set to True to start fresh
# Optional: Only useful if you want periodic checkpoints mid-training.
EVALUATION_INTERVAL = 10  # TESTING: Set to 1 to save every iteration

# --- Early Stopping Parameters ---
ENABLE_EARLY_STOPPING = False   # TESTING: Disabled to ensure all 3 stages run
EARLY_STOP_PATIENCE = 50        # Number of iterations without improvement before stopping (increased for 20 agents)
EARLY_STOP_MIN_DELTA = 0.5      # Minimum improvement in smoothed reward to count as progress
EARLY_STOP_USE_SMOOTHED = True  # Use moving average of last 5 rewards for stability

# --- Metrics Directory ---
# When copying to a new folder, update this to match your folder name!


script_dir = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.join(script_dir, "metrics")

# --- Path for model ---
CHECKPOINT_BASE_DIR = os.path.join(script_dir, "models/sectorcr_ma_ppo")

def _find_latest_checkpoint(base_dir: str) -> str | None:
    """Return the directory path containing algorithm_state.json with latest mtime.

    Scans base_dir recursively for files named 'algorithm_state.json'. If found,
    returns the parent directory of the newest one; else returns None.
    """
    if not os.path.exists(base_dir):
        return None
    
    latest_path = None
    latest_mtime = -1.0
    
    # First, check if base_dir itself is a checkpoint
    base_state = os.path.join(base_dir, "algorithm_state.json")
    if os.path.exists(base_state):
        try:
            mtime = os.path.getmtime(base_state)
            latest_path = base_dir
            latest_mtime = mtime
        except OSError:
            pass
    
    # Then scan subdirectories
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

def build_trainer(n_agents, total_iterations):
    """Builds and configures the PPO algorithm.
    
    Args:
        n_agents: Number of agents for the environment
        total_iterations: Total iterations for this stage (used for schedules)
    """
    def policy_map(agent_id, *_, **__):
        return "shared_policy"
    
    # Scale hyperparameters based on agent count
    # For 6 agents: baseline config
    # For 12 agents: moderate increase
    # For 20 agents: significant increase to handle complexity
    if n_agents <= 6:
        train_batch_size = 32000
        minibatch_size = 2000
        entropy_schedule = [[0, 0.02], [total_iterations//2, 0.01], [total_iterations, 0.003]]
    elif n_agents <= 12:
        train_batch_size = 48000
        minibatch_size = 3000
        entropy_schedule = [[0, 0.02], [total_iterations//2, 0.01], [total_iterations, 0.005]]
    else:  # 20+ agents
        train_batch_size = 64000
        minibatch_size = 4000
        # Slower entropy decay to maintain exploration longer with more agents
        entropy_schedule = [[0, 0.02], [150, 0.01], [300, 0.005]]
    
    # Learning rate schedule: decay to 1e-4 instead of near-zero for stability
    lr_schedule = [[0, 1.5e-4], [total_iterations, 1e-4]]

    cfg = (
        PPOConfig()
        .environment(
            "sector_env",
            env_config={"n_agents": n_agents,
                        "run_id": RUN_ID,
                        "metrics_base_dir": METRICS_DIR},
            disable_env_checking=False  # Enable checking to catch environment issues
        )
        .framework("torch")
        .env_runners(num_env_runners=os.cpu_count() - 1,
                     num_envs_per_env_runner=1,
                     sample_timeout_s=60.0)
        .training(
            # Scaled batch size for more agents
            train_batch_size=train_batch_size,
            # Network: moderate capacity to prevent overfitting
            model={"fcnet_hiddens": [512, 512]},  
            # Discount and GAE: standard values
            gamma=0.99,              # Standard discount factor
            lambda_=0.95,            # Standard GAE parameter
            # Learning rate: decay to 1e-4 for stability (not near-zero)
            lr=lr_schedule,
            # PPO clipping: standard conservative values
            clip_param=0.2,          # Standard PPO clip range
            vf_clip_param=10.0,      # Standard value function clip
            # Entropy: scaled schedule based on agent count
            entropy_coeff=entropy_schedule,
            # Gradient clipping: prevent exploding gradients
            grad_clip=0.5,           # Conservative gradient clipping
            # SGD iterations: balanced learning per batch
            num_sgd_iter=12,         # Standard number of epochs
            # Scaled minibatch size
            minibatch_size=minibatch_size,
            # KL divergence: soft constraint for policy stability
            use_kl_loss=True,
            kl_target=0.01,          # Target KL divergence
            kl_coeff=1.0,            # Standard coefficient
        )
        .multi_agent(
            policies={"shared_policy": (None, None, None, {})},
            policy_mapping_fn=policy_map,
        )
        .resources(num_gpus=1)
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

def run_fixed_eval(algo: Algorithm, n_episodes: int = 20, render: bool = False, n_agents: int = 6, silent: bool = True):
    """Run a small deterministic evaluation (no exploration) and return metrics.

    Returns a dict with avg_reward, avg_length, avg_intrusions, waypoint_rate,
    and raw per-episode lists.
    
    Args:
        silent: If True, suppresses BlueSky simulation output during evaluation.
        n_agents: Number of agents to use in evaluation environment.
    """
    module = algo.get_module("shared_policy")
    
    # Wrap the entire evaluation in output suppression if silent=True
    def _run_episodes():
        env = SectorEnv(
            render_mode="human" if render else None, 
            n_agents=n_agents, 
            metrics_base_dir=METRICS_DIR
        )
        rewards, lengths, intrusions, waypoints = [], [], [], []

        for _ in range(n_episodes):
            obs, _ = env.reset()
            ep_rew = 0.0
            ep_len = 0
            while env.agents:
                agent_ids = list(obs.keys())
                obs_list = list(obs.values())
                input_dict = {"obs": torch.from_numpy(np.stack(obs_list))}
                output_dict = module.forward_inference(input_dict)
                dist_class = module.get_inference_action_dist_cls()
                action_dist = dist_class.from_logits(output_dict["action_dist_inputs"])
                actions_np = action_dist.loc.cpu().numpy()  # deterministic mean action
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
    
    # Get current stage configuration
    if CURRENT_STAGE not in STAGE_CONFIGS:
        raise ValueError(f"Invalid CURRENT_STAGE={CURRENT_STAGE}. Must be 1, 2, or 3.")
    
    stage = STAGE_CONFIGS[CURRENT_STAGE]
    n_agents = stage["n_agents"]
    target_iters = stage["iterations"]
    stage_name = stage["name"]
    
    print("=" * 60)
    print(f"🎯 TRAINING STAGE {CURRENT_STAGE}/3: {stage_name}")
    print("=" * 60)
    print(f"Agents: {n_agents}")
    print(f"Iterations: {target_iters}")
    if LOAD_CHECKPOINT_FROM:
        print(f"Loading from: {LOAD_CHECKPOINT_FROM}")
    else:
        print(f"Starting with random weights (no checkpoint)")
    print("=" * 60)
    
    # Set up stage-specific checkpoint directory
    CHECKPOINT_DIR = os.path.join(CHECKPOINT_BASE_DIR, stage_name)
    
    # Clean up old checkpoints if force retraining
    if FORCE_RETRAIN:
        if os.path.exists(CHECKPOINT_DIR):
            print(f"\nFORCE_RETRAIN is True. Deleting old checkpoint directory:\n{CHECKPOINT_DIR}")
            try:
                shutil.rmtree(CHECKPOINT_DIR)
                print("✅ Old checkpoint directory removed.")
            except OSError as e:
                print(f"Error: {e.strerror} - {CHECKPOINT_DIR}")
        
        # Delete metrics directory for this stage
        stage_metrics_dir = os.path.join(METRICS_DIR, f"run_{RUN_ID}", stage_name)
        if os.path.exists(stage_metrics_dir):
            print(f"Deleting old metrics directory:\n{stage_metrics_dir}")
            try:
                shutil.rmtree(stage_metrics_dir)
                print("✅ Old metrics directory removed.")
            except OSError as e:
                print(f"Error: {e.strerror} - {stage_metrics_dir}")
    
    print("\n" + "-" * 30)
    
    # Build fresh trainer with new config
    print(f"\n🏗️  Building trainer for {n_agents} agents...")
    algo = build_trainer(n_agents, target_iters)
    
    # Load weights from checkpoint if specified
    if LOAD_CHECKPOINT_FROM:
        if os.path.exists(LOAD_CHECKPOINT_FROM):
            print(f"\n🔄 Loading weights from checkpoint...")
            print(f"   Path: {LOAD_CHECKPOINT_FROM}")
            try:
                # Use restore() to load weights into the new config
                algo.restore(LOAD_CHECKPOINT_FROM)
                print(f"✅ Successfully loaded weights!")
                print(f"   Policy network parameters transferred to {n_agents}-agent configuration")
            except Exception as e:
                print(f"❌ Failed to load checkpoint: {e}")
                import traceback
                traceback.print_exc()
                print(f"   Continuing with randomly initialized weights...")
        else:
            print(f"\n⚠️  WARNING: LOAD_CHECKPOINT_FROM specified but path doesn't exist:")
            print(f"   {LOAD_CHECKPOINT_FROM}")
            print(f"   Starting with fresh randomly initialized weights...")
    else:
        print(f"\n🆕 Starting with fresh randomly initialized weights")

        # Loss history for different components
        total_loss_history = []
        policy_loss_history = []
        value_loss_history = []
        entropy_loss_history = []
        kl_divergence_history = []
        reward_history = []
        
        # Early stopping tracking
        best_reward = float('-inf')  # Best single-iteration reward (for saving checkpoints)
        best_reward_iteration = 0
        best_checkpoint_path = None
        iterations_without_improvement = 0  # Based on smoothed reward (for stopping)
        early_stop_triggered = False
    
    # Ensure history and early-stopping variables are defined even when loading a checkpoint.
    # When restoring from a checkpoint the user may still want to track the current run's metrics.
    try:
        total_loss_history
    except NameError:
        total_loss_history = []
        policy_loss_history = []
        value_loss_history = []
        entropy_loss_history = []
        kl_divergence_history = []
        reward_history = []
        best_reward = float('-inf')
        best_reward_iteration = 0
        best_checkpoint_path = None
        iterations_without_improvement = 0
        early_stop_triggered = False
        
    # --- Main Training Loop ---
    print(f"\n🏋️  Starting training loop...\n")
    for i in range(1, target_iters + 1):
        result = algo.train()

        # Use RLlib new API fields
        envr = result.get("env_runners", {})
        mean_rew = envr.get("episode_return_mean", float("nan"))
        ep_len  = envr.get("episode_len_mean", float("nan"))
        learner_stats = result.get("learners", {}).get("shared_policy", {})
        total_loss = learner_stats.get("total_loss", float("nan"))
        
        # Extract loss components
        policy_loss = learner_stats.get("policy_loss", float("nan"))
        value_loss = learner_stats.get("vf_loss", float("nan"))
        entropy_loss = learner_stats.get("entropy", float("nan"))
        
        # Try different possible KL divergence keys
        kl_divergence = (
            learner_stats.get("kl", None) or 
            learner_stats.get("mean_kl", None) or 
            learner_stats.get("mean_kl_loss", None) or
            learner_stats.get("curr_kl_coeff", None) or
            float("nan")
        )
        
        # Debug: Print available keys on first iteration to help troubleshoot
        if i == 1:
            print(f"\n[DEBUG] Available learner_stats keys: {list(learner_stats.keys())}\n")

        # Append to history
        total_loss_history.append(total_loss)
        policy_loss_history.append(policy_loss)
        value_loss_history.append(value_loss)
        entropy_loss_history.append(entropy_loss)
        kl_divergence_history.append(kl_divergence)
        reward_history.append(mean_rew)

        # Also report reward per agent for comparability
        mean_rew_per_agent = mean_rew / max(1, n_agents)
        print(
            f"Stage {CURRENT_STAGE} | Iter {i}/{target_iters} | Mean Reward: {mean_rew:.3f}"
            f" (per-agent: {mean_rew_per_agent:.3f}) | Loss: {total_loss:.3f} | EpLenMean: {ep_len:.1f}"
            f" | KL: {kl_divergence:.6f}"
        )

        # --- Best Checkpoint Tracking (ALWAYS ACTIVE for curriculum learning) ---
        if not np.isnan(mean_rew) and mean_rew > best_reward:
            best_reward = mean_rew
            best_reward_iteration = i
            
            # Save checkpoint for new best reward
            best_checkpoint_dir = os.path.join(CHECKPOINT_DIR, f"best_iter_{i:05d}")
            checkpoint_result = algo.save(best_checkpoint_dir)
            # Extract path from checkpoint result
            if hasattr(checkpoint_result, 'checkpoint') and hasattr(checkpoint_result.checkpoint, 'path'):
                best_checkpoint_path = checkpoint_result.checkpoint.path
            else:
                best_checkpoint_path = best_checkpoint_dir
            
            print(f"   ⭐ New best reward: {best_reward:.3f} (saved to {os.path.basename(best_checkpoint_path)})")
        
        # --- Early Stopping Check (only if enabled) ---
        if ENABLE_EARLY_STOPPING and not np.isnan(mean_rew):
            # Use smoothed reward (moving average of last 5 iterations) for early stopping decision
            if EARLY_STOP_USE_SMOOTHED and len(reward_history) >= 5:
                smoothed_reward = np.mean(reward_history[-5:])
            else:
                smoothed_reward = mean_rew
            
            # Check for improvement in SMOOTHED reward (for early stopping patience)
            if smoothed_reward > np.mean(reward_history[-min(len(reward_history), EARLY_STOP_PATIENCE):]) + EARLY_STOP_MIN_DELTA:
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1
                if iterations_without_improvement >= EARLY_STOP_PATIENCE:
                    print(f"\n⏹️  Early stopping: No improvement in smoothed reward for {EARLY_STOP_PATIENCE} iterations")
                    print(f"   Best reward achieved: {best_reward:.3f} at iteration {best_reward_iteration}")
                    if best_checkpoint_path:
                        print(f"   Best checkpoint saved at: {best_checkpoint_path}")
                    early_stop_triggered = True
        
        # Break if early stopping triggered
        if early_stop_triggered:
            print(f"   Stopping at iteration {i}/{target_iters}")
            break

        # Optional periodic checkpointing
        if EVALUATION_INTERVAL and i % EVALUATION_INTERVAL == 0:
            checkpoint_result = algo.save(CHECKPOINT_DIR)
            # Extract just the path from the result to avoid printing massive object
            if hasattr(checkpoint_result, 'checkpoint') and hasattr(checkpoint_result.checkpoint, 'path'):
                path = checkpoint_result.checkpoint.path
            else:
                path = str(checkpoint_result)
            print(f"✅ Checkpoint saved to: {path}")

            # --- Fixed-seed mini evaluation ---
            try:
                eval_metrics = run_fixed_eval(
                    algo, 
                    n_episodes=30, 
                    render=False, 
                    n_agents=n_agents
                )
                print(
                    "[Eval] stage=%d iter=%d | avg_rew=%.3f | avg_len=%.1f | avg_intr=%.2f | wp_rate=%.1f%%"
                    % (
                        CURRENT_STAGE,
                        i,
                        eval_metrics["avg_reward"],
                        eval_metrics["avg_length"],
                        eval_metrics["avg_intrusions"],
                        eval_metrics["waypoint_rate"] * 100.0,
                    )
                )
                _write_eval_row(metrics=eval_metrics, iteration=i, out_dir=os.path.join(METRICS_DIR, f"run_{RUN_ID}", stage_name))
                
            except Exception as e:
                print(f"[Eval] skipped due to error: {e}")

    print(f"\n🚀 Stage {CURRENT_STAGE} training finished.")
    
    # Early stopping summary
    if early_stop_triggered:
        print(f"   ✋ Early stopping was triggered")
        print(f"   📊 Best reward achieved: {best_reward:.3f}")
    
    # Calculate and display stage training time
    stage_training_time = time.time() - training_start_time
    actual_iters = len(reward_history)
    print(f"⏱️  Training time: {stage_training_time/60:.2f} minutes ({stage_training_time/3600:.2f} hours) for {actual_iters} iters.")
    
    # Save final checkpoint (current state)
    final_checkpoint_result = algo.save(CHECKPOINT_DIR)
    if hasattr(final_checkpoint_result, 'checkpoint') and hasattr(final_checkpoint_result.checkpoint, 'path'):
        final_path = final_checkpoint_result.checkpoint.path
    else:
        final_path = str(final_checkpoint_result)
    print(f"✅ Final checkpoint saved to: {final_path}")
    
    # Summary and next steps
    print(f"\n{'='*60}")
    print(f"📁 CHECKPOINT SUMMARY - STAGE {CURRENT_STAGE}")
    print(f"{'='*60}")
    if best_checkpoint_path:
        print(f"Best checkpoint (reward {best_reward:.3f}):")
        print(f"  {best_checkpoint_path}")
    print(f"\nFinal checkpoint:")
    print(f"  {final_path}")
    
    # Show next steps
    if CURRENT_STAGE < 3:
        next_stage = CURRENT_STAGE + 1
        next_config = STAGE_CONFIGS[next_stage]
        print(f"\n{'='*60}")
        print(f"📋 NEXT STEPS - Continue to Stage {next_stage}")
        print(f"{'='*60}")
        print(f"\n1. Update the configuration at the top of main.py:")
        print(f"\n   CURRENT_STAGE = {next_stage}")
        if best_checkpoint_path:
            print(f"   LOAD_CHECKPOINT_FROM = r\"{best_checkpoint_path}\"")
        else:
            print(f"   LOAD_CHECKPOINT_FROM = r\"{final_path}\"")
        print(f"\n2. Run the script again to train Stage {next_stage}")
        print(f"   ({next_config['n_agents']} agents × {next_config['iterations']} iterations)")
        print(f"\n{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"🎉 ALL STAGES COMPLETE!")
        print(f"{'='*60}")
        
        # --- Plot the Loss and Reward in a Single Figure ---
    # Plot Loss Components
        # Ensure `axes` exists (created earlier only when all stages complete). Create if missing.
        try:
            axes
        except NameError:
            fig, axes = plt.subplots(3, 1, figsize=(12, 16))  # Create 3 subplots (3 rows, 1 column)

        # Plot Loss Components
    axes[0].plot(range(1, len(total_loss_history) + 1), total_loss_history, label="Total Loss", marker='o', linestyle='-')
    axes[0].plot(range(1, len(policy_loss_history) + 1), policy_loss_history, label="Policy Loss", marker='s', linestyle='--')
    axes[0].plot(range(1, len(value_loss_history) + 1), value_loss_history, label="Value Loss", marker='^', linestyle='-.')
    axes[0].plot(range(1, len(entropy_loss_history) + 1), entropy_loss_history, label="Entropy Loss", marker='d', linestyle=':')
    axes[0].set_title(f"Loss Components - Stage {CURRENT_STAGE} ({stage_name})")
    axes[0].set_xlabel("Training Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Plot Reward
    axes[1].plot(range(1, len(reward_history) + 1), reward_history, marker='o', linestyle='-', label='Mean Reward')
    
    # Mark the best checkpoint iteration if early stopping was used
    if best_checkpoint_path and best_reward_iteration > 0:
        axes[1].axvline(x=best_reward_iteration, color='green', linestyle='--', linewidth=2, 
                        label=f'Best Checkpoint (iter {best_reward_iteration})')
        axes[1].plot(best_reward_iteration, reward_history[best_reward_iteration-1], 
                    'g*', markersize=15, label=f'Best Reward: {best_reward:.2f}')
    
    axes[1].set_title(f"Mean Reward - Stage {CURRENT_STAGE} ({stage_name})")
    axes[1].set_xlabel("Training Iteration")
    axes[1].set_ylabel("Mean Reward")
    axes[1].legend()
    axes[1].grid(True)

    # Plot KL Divergence
    axes[2].plot(range(1, len(kl_divergence_history) + 1), kl_divergence_history, marker='o', linestyle='-', color='green')
    axes[2].axhline(y=0.01, color='red', linestyle='--', linewidth=1, alpha=0.7, label='KL Target (0.01)')
    axes[2].set_title(f"KL Divergence - Stage {CURRENT_STAGE} ({stage_name})")
    axes[2].set_xlabel("Training Iteration")
    axes[2].set_ylabel("KL Divergence")
    axes[2].legend()
    axes[2].grid(True)

    # Adjust layout and save the figure
    plt.tight_layout()
    
    # Save the figure to the stage directory
    plot_path = os.path.join(script_dir, f"training_metrics_{stage_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Training metrics plot saved to: {plot_path}")
    
    # Close the figure to avoid displaying it
    plt.close(fig)

    ray.shutdown()
