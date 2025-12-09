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
# Optional: If register_envs() fully registers "sector_env", the next line can be removed.
from bluesky_gym.envs.ma_env_ppo import SectorEnv  # optional
from datetime import datetime

from run_config import RUN_ID

# Register your custom environment with Gymnasium
register_envs()

# for logging the .csv files

# --- Parameters ---
N_AGENTS = 6  # Back to 6 agents
TOTAL_ITERS = 100          # Extended training for better convergence
EXTRA_ITERS = 50           # When resuming, run this many more iterations
FORCE_RETRAIN = True       # Start fresh with new hyperparameters
# Optional: Only useful if you want periodic checkpoints mid-training.
EVALUATION_INTERVAL = 10  # e.g., set to 1 or 5 to save during training

# --- Metrics Directory ---
# When copying to a new folder, update this to match your folder name!
# e.g., for boris_test_files/24_10, use "metrics_24_10"
# This points to the ROOT bluesky-gym/metrics_23_10/ folder
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
METRICS_DIR = os.path.join(repo_root, "metrics_23_10")

# --- Scheduling Parameters ---
def get_intrusion_penalty(iteration: int, total_iters: int) -> float:
    """Return intrusion penalty based on current iteration (stepped curriculum)."""
    # Stepped schedule: agents get time to adapt at each level
    third = total_iters // 3
    if iteration <= third:
        return -3.0   
    elif iteration <= 2 * third:
        return -4.5
    else:
        return -6.0

# --- Path for model ---
CHECKPOINT_DIR = os.path.join(script_dir, "models/sectorcr_ma_ppo")

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

def build_trainer():
    """Builds and configures the PPO algorithm."""
    def policy_map(agent_id, *_, **__):
        return "shared_policy"

    # Start with initial scheduled values
    initial_intrusion = get_intrusion_penalty(1, TOTAL_ITERS)


    cfg = (
        PPOConfig()
        .environment(
            "sector_env",
            env_config={"n_agents": N_AGENTS,
                        "run_id": RUN_ID,
                        "intrusion_penalty": initial_intrusion,
                        "metrics_base_dir": METRICS_DIR},
            disable_env_checking=True
        )
        .framework("torch")
        .env_runners(num_env_runners=os.cpu_count() - 1)
        .training(
            train_batch_size=8000,
            # Simpler network - large network may be overfitting or unstable
            model={"fcnet_hiddens": [256, 256]},
            gamma=0.99,
            lambda_=0.90,
            # Learning rate - balanced
            lr= 2e-4, 
            # More conservative PPO clipping for stability
            clip_param=0.2,
            vf_clip_param=5.0,
            # Entropy - low for more exploitation
            entropy_coeff = 0.01,
            # Gradient clipping
            grad_clip=1.0,
            # More thorough learning per batch
            num_sgd_iter=15,
            # Remove KL penalty - entropy already controls this
            use_kl_loss=False
            
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

def run_fixed_eval(algo: Algorithm, n_episodes: int = 20, render: bool = False, intrusion_penalty: float = -3.0, silent: bool = True):
    """Run a small deterministic evaluation (no exploration) and return metrics.

    Returns a dict with avg_reward, avg_length, avg_intrusions, waypoint_rate,
    and raw per-episode lists.
    
    Args:
        silent: If True, suppresses BlueSky simulation output during evaluation.
    """
    module = algo.get_module("shared_policy")
    
    # Wrap the entire evaluation in output suppression if silent=True
    def _run_episodes():
        env = SectorEnv(
            render_mode="human" if render else None, 
            n_agents=N_AGENTS, 
            intrusion_penalty=intrusion_penalty,
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
    waypoint_rate = (float(np.sum(waypoints)) / (n_episodes * N_AGENTS)) if waypoints else 0.0
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
        algo = build_trainer()
        # Fresh training: run up to TOTAL_ITERS
        target_iters = int(TOTAL_ITERS)

    # Loss history for different components
    total_loss_history = []
    policy_loss_history = []
    value_loss_history = []
    entropy_loss_history = []
    # reward_history = []
    reward_history = []
    
    # Track previous intrusion penalty to detect changes
    prev_intrusion_penalty = get_intrusion_penalty(algo.iteration if algo.iteration > 0 else 1, target_iters)
    last_printed_penalty = prev_intrusion_penalty
    
    # --- Main Training Loop ---
    for i in range(algo.iteration + 1, target_iters + 1):
        # Update intrusion penalty schedule (changes every iteration with linear schedule)
        curr_intrusion_penalty = get_intrusion_penalty(i, target_iters)
        
        # Always update environments if penalty changed
        if curr_intrusion_penalty != prev_intrusion_penalty:
            # Update in all worker environments
            def update_env_penalty(env):
                if hasattr(env, 'update_intrusion_penalty'):
                    env.update_intrusion_penalty(curr_intrusion_penalty)
            
            try:
                # Try different API methods based on RLlib version
                if hasattr(algo, 'workers') and hasattr(algo.workers, 'foreach_env'):
                    algo.workers.foreach_env(update_env_penalty)
                elif hasattr(algo, 'env_runner_group'):
                    # New API - iterate through runners and update their environments
                    for runner in algo.env_runner_group.remote_workers():
                        runner.foreach_env.remote(update_env_penalty)
                # If neither works, silently skip (penalty is set in initial config anyway)
            except Exception:
                # Silently skip - the initial penalty is set via env_config
                pass
            
            # Only print if change is significant (reduce console spam)
            if abs(curr_intrusion_penalty - last_printed_penalty) >= 0.5:
                print(f"[Schedule] Iter {i}: Intrusion penalty now {curr_intrusion_penalty:.2f} (was {last_printed_penalty:.2f})")
                last_printed_penalty = curr_intrusion_penalty
            
            prev_intrusion_penalty = curr_intrusion_penalty
        
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

        # Append to history
        total_loss_history.append(total_loss)
        policy_loss_history.append(policy_loss)
        value_loss_history.append(value_loss)
        entropy_loss_history.append(entropy_loss)
        reward_history.append(mean_rew)

        # Also report reward per agent for comparability across N_AGENTS
        mean_rew_per_agent = mean_rew / max(1, N_AGENTS)
        print(
            f"Iter {i}/{TOTAL_ITERS} | Mean Reward: {mean_rew:.3f}"
            f" (per-agent: {mean_rew_per_agent:.3f}) | Loss: {total_loss:.3f} | EpLenMean: {ep_len:.1f}"
        )

        # Optional periodic checkpointing
        if EVALUATION_INTERVAL and i % EVALUATION_INTERVAL == 0:
            path = algo.save(CHECKPOINT_DIR)
            print(f"✅ Checkpoint saved to: {path}")

            # --- Fixed-seed mini evaluation ---
            try:
                eval_metrics = run_fixed_eval(algo, n_episodes=20, render=False, intrusion_penalty=curr_intrusion_penalty)
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
    
    # Calculate and display total training time
    total_training_time = time.time() - training_start_time
    print(f"⏱️  Total training time: {total_training_time/60:.2f} minutes ({total_training_time/3600:.2f} hours) for {TOTAL_ITERS} iters.")
    
    final_path = algo.save(CHECKPOINT_DIR)
    print(f"✅ Final checkpoint saved to: {final_path}")
    
    # --- Plot the Loss and Reward in a Single Figure ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))  # Create 2 subplots (2 rows, 1 column)

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
    axes[1].plot(range(1, len(reward_history) + 1), reward_history, marker='o', linestyle='-')
    axes[1].set_title("Mean Reward Over Training Iterations")
    axes[1].set_xlabel("Training Iteration")
    axes[1].set_ylabel("Mean Reward")
    axes[1].grid(True)

    # Adjust layout and show the figure
    plt.tight_layout()
    plt.show()

    ray.shutdown()

# ---------------- Evaluation Helpers -----------------
def run_fixed_eval(algo: Algorithm, n_episodes: int = 20, render: bool = False):
    """Run a small deterministic evaluation (no exploration) and return metrics.

    Returns a dict with avg_reward, avg_length, avg_intrusions, waypoint_rate,
    and raw per-episode lists.
    """
    module = algo.get_module("shared_policy")
    env = SectorEnv(render_mode="human" if render else None, n_agents=N_AGENTS)
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
    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    avg_length = float(np.mean(lengths)) if lengths else 0.0
    avg_intrusions = float(np.mean(intrusions)) if intrusions else 0.0
    waypoint_rate = (float(np.sum(waypoints)) / (n_episodes * N_AGENTS)) if waypoints else 0.0
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
