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
from ray.rllib.algorithms.sac import SACConfig

# Make sure these imports point to your custom environment registration
from bluesky_gym import register_envs
from bluesky_gym.envs.ma_env_SAC import SectorEnv

from run_config import RUN_ID

# Register your custom environment with Gymnasium
register_envs()

# --- Parameters ---
N_AGENTS = 6  # Number of agents for training
TOTAL_ITERS = 10  # Maximum total iterations
TOTAL_ITERS_R = 50
EXTRA_ITERS = 50           # When resuming, run this many more iterations
FORCE_RETRAIN = True       # Start fresh with new hyperparameters
# Optional: Only useful if you want periodic checkpoints mid-training.
EVALUATION_INTERVAL = 5  # e.g., set to 1 or 5 to save during training

# --- Early Stopping Parameters ---
ENABLE_EARLY_STOPPING = True    # Set to False to disable early stopping
EARLY_STOP_PATIENCE = 20        # Number of iterations without improvement before stopping
EARLY_STOP_MIN_DELTA = 0.5      # Minimum improvement in smoothed reward to count as progress
EARLY_STOP_USE_SMOOTHED = True  # Use moving average of last 5 rewards for stability

# --- Metrics Directory ---
# When copying to a new folder, update this to match your folder name!

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
# METRICS_DIR = os.path.join(repo_root, "metrics_5_11")
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

def build_trainer(n_agents):
    """Builds and configures the PPO algorithm.
    
    Args:
        n_agents: Number of agents for the environment
    """
    def policy_map(agent_id, *_, **__):
        return "shared_policy"

    cfg = (
        SACConfig()
        .environment(
            "sector_env",
            env_config={"n_agents": n_agents,
                        "run_id": RUN_ID,
                        "metrics_base_dir": METRICS_DIR},
            disable_env_checking=True
        )
        .framework("torch")
        .env_runners(num_env_runners=os.cpu_count() - 1)
        # ✅ Turn OFF new API stack (important)
        .api_stack(enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False)
        .training( # exapmle
            actor_lr=0.001,
            critic_lr=0.002,
            alpha_lr=1e-5,
            gamma=0.99,
            initial_alpha=0.2,
            target_entropy="auto", # ✅ correct for 2-dim actions
            tau=0.005,
            twin_q=True,
            n_step=1,
            grad_clip=0.5,
            training_intensity=1.0,
            train_batch_size_per_learner=256,
            replay_buffer_config = {
                "type": "MultiAgentReplayBuffer",   # <– REQUIRED
                "capacity": 1000000,                 # or 1e6 for real training
                "chunk_size": 1,                    # store single transitions
            },
            policy_model_config={"fcnet_hiddens": [256, 256]},
            q_model_config={"fcnet_hiddens": [256, 256]},
            num_steps_sampled_before_learning_starts = 5000,
                  
            # twin_q = use two Q networks 
            # q_model_config = config of Q networks
            # policy_model_config = config of policy network
            # tau = update the target by tau*...
            # inital_alpha = Initial value to use for the entropy weight alpha.
            # target_entropy = Target entropy lower bound. has option auto
            # n_steps = N-environment step before target updates.
            # replay_buffer_config = configs of replay buffer
            # clip_actions= clip actions? should be false if actions are already normalized
            # clip_grad = If not None, clip gradients during optimization at this value.
            # optimization_config={
            #     "actor_learning_rate": 3e-4,
            #     "critic_learning_rate": 3e-4,
            #     "entropy_learning_rate": 3e-4,
            # }
            # actor_lr = 
            # critic_lr = 
            # alpha_lr = 
            # training_intensity =  controlling epochs per data in deep learning.
            # num_steps_sampled_before_learning_starts           
            # train_batch_size_per_learner
        )
       
        # tune.run(
        #     "SAC",
        #     config=config,
        #     stop={"timesteps_total": 1_000_000},   # <--- TRAIN LENGTH HERE
        # )
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
    value_loss_history = []
    entropy_loss_history = []
    kl_divergence_history = []
    # reward_history = []
    reward_history = []
    
    # Early stopping tracking
    best_reward = float('-inf')  # Best single-iteration reward (for saving checkpoints)
    best_reward_iteration = 0
    best_checkpoint_path = None
    iterations_without_improvement = 0  # Based on smoothed reward (for stopping)
    early_stop_triggered = False
    
    # --- Main Training Loop ---
    for i in range(algo.iteration + 1, target_iters + 1):
        result = algo.train()

        # Use RLlib new API fields
        envr = result.get("env_runners", {})
        mean_rew = envr.get("episode_return_mean", float("nan"))
        ep_len  = envr.get("episode_len_mean", float("nan"))
        learner_stats = result.get("learners", {}).get("shared_policy", {})
        total_loss = learner_stats.get("total_loss", float("nan"))
        
        # Extract loss components
        policy_loss = learner_stats.get("policy_loss", float("nan"))
        # value_loss = learner_stats.get("vf_loss", float("nan"))
        value_loss = learner_stats.get("q_loss", float("nan"))

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
        if i == algo.iteration + 1:
            print(f"\n[DEBUG] Available learner_stats keys: {list(learner_stats.keys())}\n")

        # Append to history
        total_loss_history.append(total_loss)
        policy_loss_history.append(policy_loss)
        value_loss_history.append(value_loss)
        entropy_loss_history.append(entropy_loss)
        kl_divergence_history.append(kl_divergence)
        reward_history.append(mean_rew)

        # Also report reward per agent for comparability
        mean_rew_per_agent = mean_rew / max(1, N_AGENTS)
        print(
            f"Iter {i}/{TOTAL_ITERS} | Mean Reward: {mean_rew:.3f}"
            f" (per-agent: {mean_rew_per_agent:.3f}) | Loss: {total_loss:.3f} | EpLenMean: {ep_len:.1f}"
            f" | KL: {kl_divergence:.6f}"
        )

        # --- Early Stopping Check ---
        if ENABLE_EARLY_STOPPING and not np.isnan(mean_rew):
            # Use smoothed reward (moving average of last 5 iterations) for early stopping decision
            if EARLY_STOP_USE_SMOOTHED and len(reward_history) >= 5:
                smoothed_reward = np.mean(reward_history[-5:])
            else:
                smoothed_reward = mean_rew
            
            # Check if current iteration has the best ACTUAL reward (for saving checkpoint)
            if mean_rew > best_reward:
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
                    n_agents=N_AGENTS
                )
                print(
                    "[Eval] iter=%d | avg_rew=%.3f | avg_len=%.1f | avg_intr=%.2f | wp_rate=%.1f%%"
                    % (
                        i,
                        eval_metrics["avg_reward"],
                        eval_metrics["avg_length"],
                        eval_metrics["avg_intrusions"],
                        eval_metrics["waypoint_rate"] * 100.0,
                    )
                )
                _write_eval_row(metrics=eval_metrics, iteration=i, out_dir=os.path.join(METRICS_DIR, f"run_{RUN_ID}"))
                
            except Exception as e:
                print(f"[Eval] skipped due to error: {e}")

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
    
    # Calculate and display total training time
    total_training_time = time.time() - training_start_time
    actual_iters = len(reward_history)
    print(f"⏱️  Total training time: {total_training_time/60:.2f} minutes ({total_training_time/3600:.2f} hours) for {actual_iters} iters.")
    
    # Save final checkpoint (current state)
    final_checkpoint_result = algo.save(CHECKPOINT_DIR)
    # Extract just the path from the result to avoid printing massive object
    if hasattr(final_checkpoint_result, 'checkpoint') and hasattr(final_checkpoint_result.checkpoint, 'path'):
        final_path = final_checkpoint_result.checkpoint.path
    else:
        final_path = str(final_checkpoint_result)
    print(f"✅ Final checkpoint (last iteration) saved to: {final_path}")
    
    # Summary of available checkpoints
    if best_checkpoint_path:
        print(f"\n📁 Checkpoint Summary:")
        print(f"   • Best model (iteration {best_reward_iteration}, reward {best_reward:.3f}): {best_checkpoint_path}")
        print(f"   • Final model (iteration {actual_iters}): {final_path}")
        print(f"\n   💡 Tip: Use the best checkpoint for evaluation to get optimal performance!")
    
    # --- Plot the Loss and Reward in a Single Figure ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 16))  # Create 3 subplots (3 rows, 1 column)

    # Plot Loss Components
    axes[0].plot(range(1, len(total_loss_history) + 1), total_loss_history, label="Total Loss", marker='o', linestyle='-')
    axes[0].plot(range(1, len(policy_loss_history) + 1), policy_loss_history, label="Policy Loss", marker='s', linestyle='--')
    axes[0].plot(range(1, len(value_loss_history) + 1), value_loss_history, label="Value Loss", marker='^', linestyle='-.')
    axes[0].plot(range(1, len(entropy_loss_history) + 1), entropy_loss_history, label="Entropy Loss", marker='d', linestyle=':')
    axes[0].set_title("Loss Components Over Training Iterations")
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
    
    axes[1].set_title("Mean Reward Over Training Iterations")
    axes[1].set_xlabel("Training Iteration")
    axes[1].set_ylabel("Mean Reward")
    axes[1].legend()
    axes[1].grid(True)

    # Plot KL Divergence
    axes[2].plot(range(1, len(kl_divergence_history) + 1), kl_divergence_history, marker='o', linestyle='-', color='green')
    axes[2].axhline(y=0.01, color='red', linestyle='--', linewidth=1, alpha=0.7, label='KL Target (0.01)')
    axes[2].set_title("KL Divergence Over Training Iterations")
    axes[2].set_xlabel("Training Iteration")
    axes[2].set_ylabel("KL Divergence")
    axes[2].legend()
    axes[2].grid(True)

    # Adjust layout and show the figure
    plt.tight_layout()
    plt.show()

    ray.shutdown()
