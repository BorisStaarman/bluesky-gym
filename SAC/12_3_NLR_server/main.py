# standard imports
import os
import sys
import shutil

# Set matplotlib backend to non-interactive (for headless servers)
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt

import numpy as np
import time
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import io

# MARL ray imports
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.sac import SACConfig

# Make sure these imports point to your custom environment registration
from bluesky_gym import register_envs
from bluesky_gym.envs.ma_env_SAC_new import SectorEnv

import torch
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from run_config import RUN_ID

# Register your custom environment with Gymnasium
register_envs()

# --- Parameters ---
N_AGENTS = 25  # Number of agents for training
TOTAL_ITERS = 10  # Maximum total iterations
EXTRA_ITERS = 50           # When resuming, run this many more iterations
FORCE_RETRAIN = True       # Start fresh with new hyperparameters
# Optional: Only useful if you want periodic checkpoints mid-training.
EVALUATION_INTERVAL = 1000  # e.g., set to 1 or 5 to save during training

# --- Early Stopping Parameters ---
ENABLE_EARLY_STOPPING = True    # Set to False to disable early stopping
EARLY_STOP_PATIENCE = 10000        # Number of iterations without improvement before stopping
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

class ForceAlphaCallback(DefaultCallbacks):
    def on_train_result(self, *, algorithm, result, **kwargs):
        # --- CONFIGURATION ---
        START_ALPHA = 0.25
        END_ALPHA = 0.01       # The "Floor" (Don't go lower than this!)
        DECAY_ITERS = 10000   # How long to take to get to the floor
        
        # 1. Get current iteration
        # RLlib reports this in the result dict
        current_iter = result["training_iteration"]
        
        # 2. Calculate Target Alpha (Linear Schedule)
        if current_iter >= DECAY_ITERS:
            # We are past the decay period, lock it at the floor
            target_alpha = END_ALPHA
        else:
            # Calculate the fraction of progress (0.0 to 1.0)
            progress = current_iter / DECAY_ITERS
            # Interpolate: Start - (Difference * Progress)
            target_alpha = START_ALPHA - ((START_ALPHA - END_ALPHA) * progress)

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
            actor_lr=1e-4, # LR for actor, which decides the actions, small means slower learning but better converging
            critic_lr=5e-4,          # evaluates quality of actions. hihger is better of exploration, 
            # ---- Option A: fixed alpha (stable baseline) ----
            target_entropy = -1.5,   # -1.0 for more exploration. larger negative value is more exploitation
            # alpha_lr = 1e-5,            # was 3e-5.   lr for updating entropy / alpha. lower means slower alpha updates
            alpha_lr=[
                [0,        0],   # from step 0 to 1M: 3e-4
                [TOTAL_ITERS/2, 1e-5],
                [TOTAL_ITERS, 1e-6],  # then slowly decay to 3e-5
            ],
            # alpha_lr=5e-5,            # was 3e-5.   lr for updating entropy / alpha. lower means slower alpha updates
            
            initial_alpha = 0.5, # initial alpha/entropy, higher means more exploration
            grad_clip=0.5,

            # Hyperparameters
            gamma=0.99, # discount factor future rewards
            tau=0.003, # soft update parameter for target    networks, smaller makes target network update more slowly
            
            twin_q=True, # use two networks, for more stable learning
            n_step=3, #  enables multi-step q-learning, agent will use rewards over multiple timestep

            # Replay/batching - REDUCED for more episode diversity
            replay_buffer_config={
                "type": "MultiAgentReplayBuffer",
                "capacity": 1_500_000,  # Reduced from 1M to encourage fresher samples
            },
            num_steps_sampled_before_learning_starts=10_000,  # Reduced from 5000
            train_batch_size=4096,  # Reduced from 2048 for more frequent updates
            
            # Force more environment interaction relative to training
            # training_intensity=10,  # Number of training updates per sampled item (lower = more sampling)
            # Models
            policy_model_config={"fcnet_hiddens": [512, 512]},
            q_model_config={"fcnet_hiddens": [1024, 1024]},
        )

        .multi_agent(
            policies={"shared_policy": (None, None, None, {})},
            policy_mapping_fn=policy_map,
        )
        .resources(num_gpus=0)
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

    if restored_from:
        print(f"Restoring from checkpoint: {restored_from}")
        algo = Algorithm.from_checkpoint(restored_from)
        # Run exactly EXTRA_ITERS more beyond the restored iteration
        target_iters = algo.iteration + max(1, int(EXTRA_ITERS))
    else:
        print("Building new trainer...")
        print(f"Starting training with {N_AGENTS} agents")
        algo = build_trainer(N_AGENTS)
        # Fresh training: run up to TOTAL_ITERS
        target_iters = int(TOTAL_ITERS)

    # Loss history for different components
    total_loss_history = []
    policy_loss_history = []
    entropy_history = []
    alpha_history = []
    q_loss_history = []
    reward_history = []
    episode_length_history = []
    
    # Training step tracking
    total_training_steps = 0  # Total environment steps used during training
    
    # Early stopping tracking
    best_reward = float('-inf')  # Best single-iteration reward (for saving checkpoints)
    best_reward_iteration = 0
    best_checkpoint_path = None
    best_smoothed_reward = float('-inf')  # Best smoothed reward (for early stopping)
    iterations_without_improvement = 0  # Based on smoothed reward (for stopping)
    early_stop_triggered = False
    
    # --- Main Training Loop ---
    for i in range(algo.iteration + 1, target_iters + 1):
        result = algo.train()

        # Extract metrics from env_runners (new location in hybrid API)
        env_runners = result.get("env_runners", {})
        mean_rew = env_runners.get("episode_return_mean", float("nan"))
        ep_len = env_runners.get("episode_len_mean", float("nan"))
        
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
        
        # SAC metrics from learner_stats
        policy_loss = learner_info.get("actor_loss", float("nan"))
        q_loss = learner_info.get("critic_loss", float("nan"))
        alpha_raw = learner_info.get("alpha_value", float("nan"))
        mean_q = learner_info.get("mean_q", float("nan"))
        
        # Convert arrays to scalars (SAC returns arrays sometimes)
        def to_scalar(val):
            if isinstance(val, (list, tuple, np.ndarray)):
                return float(np.mean(val)) if len(val) > 0 else float("nan")
            return float(val) if not isinstance(val, str) else float("nan")
        
        q_loss = to_scalar(q_loss)
        alpha = to_scalar(alpha_raw)
        policy_loss = to_scalar(policy_loss)
        entropy = to_scalar(mean_q)  # Use mean_q as proxy for entropy
        
        # Calculate total loss as sum of components
        if not np.isnan(policy_loss) and not np.isnan(q_loss):
            total_loss = abs(policy_loss) + abs(q_loss)
        else:
            total_loss = float("nan")

        # Append to history
        total_loss_history.append(total_loss)
        policy_loss_history.append(policy_loss)
        q_loss_history.append(q_loss)
        entropy_history.append(entropy)
        alpha_history.append(alpha)
        reward_history.append(mean_rew)
        episode_length_history.append(ep_len)

        # Calculate moving average for reward (5 iterations)
        if len(reward_history) >= 5:
            reward_ma5 = np.mean(reward_history[-5:])
        else:
            reward_ma5 = mean_rew
        
        # Enhanced training progress display
        print(
            f"Iter {i}/{target_iters} | "
            f"Reward: {mean_rew:.3f} (MA5: {reward_ma5:.3f}) | "
            f"EpLen: {ep_len:.1f} | "
            f"Loss: {total_loss:.3f} (Critic: {q_loss:.3f}, Actor: {policy_loss:.3f}) | "
            f"MeanQ: {entropy:.4f} | Alpha: {alpha:.4f}"
        )

        # --- Best Checkpoint Tracking (ALWAYS ACTIVE) ---
        # Skip saving best model for first 10 iterations (warm-up period)
        if i > 10 and not np.isnan(mean_rew) and mean_rew > best_reward:
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
            print(f"[Eval] Starting evaluation with {30} episodes...")
            try:
                eval_metrics = run_fixed_eval(
                    algo, 
                    n_episodes=30, 
                    render=False, 
                    n_agents=N_AGENTS
                )
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
    fig, axes = plt.subplots(4, 1, figsize=(14, 18))  # Create 4 subplots (4 rows, 1 column)

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

    # Adjust layout and save the figure
    plt.tight_layout()
    
    # Save figure to file instead of showing it (for headless servers)
    figure_path = os.path.join(script_dir, f"training_metrics_run_{RUN_ID}.png")
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Training metrics plot saved to: {figure_path}")
    plt.close()  # Close the figure to free memory
    
    ray.shutdown()
