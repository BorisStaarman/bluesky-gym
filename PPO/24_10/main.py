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
# Performance-based curriculum learning: advance when performance thresholds are met
CURRICULUM_STAGES = [4, 5, 6]  # Agent counts to progress through
CURRICULUM_THRESHOLDS = {
    'avg_intrusions': 1.0,   # Must have ≤ 1.0 intrusions per episode
    'waypoint_rate': 0.55,    # Must have ≥ 55% waypoint success
}
MIN_ITERS_PER_STAGE = 5  # Minimum iterations before allowing stage transition
MAX_ITERS_PER_STAGE = 20  # Maximum iterations at each stage before forced transition
TOTAL_ITERS = 60  # Maximum total iterations
N_AGENTS = 6  # Maximum agents (used for final evaluation)
EXTRA_ITERS = 50           # When resuming, run this many more iterations
FORCE_RETRAIN = True       # Start fresh with new hyperparameters
# Optional: Only useful if you want periodic checkpoints mid-training.
EVALUATION_INTERVAL = 5  # e.g., set to 1 or 5 to save during training

# --- Metrics Directory ---
# When copying to a new folder, update this to match your folder name!
# e.g., for boris_test_files/24_10, use "metrics_24_10"
# This points to the ROOT bluesky-gym/metrics_23_10/ folder
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
METRICS_DIR = os.path.join(repo_root, "metrics_24_10")

# --- Penalty Configuration ---
INTRUSION_PENALTY = -6.0  # Fixed intrusion penalty (episode terminates on violation)

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

def build_trainer(n_agents):
    """Builds and configures the PPO algorithm.
    
    Args:
        n_agents: Number of agents for the environment
    """
    def policy_map(agent_id, *_, **__):
        return "shared_policy"

    cfg = (
        PPOConfig()
        .environment(
            "sector_env",
            env_config={"n_agents": n_agents,
                        "run_id": RUN_ID,
                        "intrusion_penalty": INTRUSION_PENALTY,
                        "metrics_base_dir": METRICS_DIR},
            disable_env_checking=True
        )
        .framework("torch")
        .env_runners(num_env_runners=os.cpu_count() - 1)
        .training(
            #normalize_values=True,
            train_batch_size=24000,
            # Simpler network - large network may be overfitting or unstable
            model={"fcnet_hiddens": [256, 256]},
            gamma=0.98,
            lambda_=0.90,
            # Learning rate - balanced
            lr= 5e-5, 
            # More conservative PPO clipping for stability
            clip_param=0.1,
            vf_clip_param=30.0,
            # Entropy - smooth linear decay from exploration to exploitation
            # Format: [[timestep, value], [timestep, value]]
            # Start high (0.02) for exploration, end low (0.005) for exploitation
            entropy_coeff=[[0, 0.02], [20, 0.012], [TOTAL_ITERS, 0.008]],
            # Gradient clipping
            grad_clip=1.0,
            # More thorough learning per batch
            num_sgd_iter=10,
            # Remove KL penalty - entropy already controls this
            use_kl_loss=True,            # add a soft guardrail
            kl_target=0.01,
            kl_coeff=0.4,
            
        )
        # .rollouts(
        #     observation_filter="MeanStdFilter",    # add
        #     rollout_fragment_length=400,
        #     batch_mode="truncate_episodes",
        # )
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

def run_fixed_eval(algo: Algorithm, n_episodes: int = 20, render: bool = False, intrusion_penalty: float = -3.0, n_agents: int = 6, silent: bool = True):
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
    restored_from = None
    current_n_agents = None  # Track current agent count for curriculum
    curriculum_stage_idx = 0  # Which stage we're in (0=4 agents, 1=5 agents, 2=6 agents)
    iters_in_current_stage = 0  # How many iterations at current stage

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
        # TODO: When resuming, we'd need to track which stage we're in
        # For now, assume starting fresh with FORCE_RETRAIN=True
        current_n_agents = CURRICULUM_STAGES[0]
        curriculum_stage_idx = 0
    else:
        print("Building new trainer...")
        # Start with the first curriculum stage
        current_n_agents = CURRICULUM_STAGES[0]
        print(f"🎓 Curriculum Stage 1/{len(CURRICULUM_STAGES)}: Starting with {current_n_agents} agents")
        print(f"   Advancement criteria: intrusions ≤ {CURRICULUM_THRESHOLDS['avg_intrusions']}, "
              f"waypoint success ≥ {CURRICULUM_THRESHOLDS['waypoint_rate']*100:.0f}%")
        algo = build_trainer(current_n_agents)
        # Fresh training: run up to TOTAL_ITERS
        target_iters = int(TOTAL_ITERS)

    # Loss history for different components
    total_loss_history = []
    policy_loss_history = []
    value_loss_history = []
    entropy_loss_history = []
    # reward_history = []
    reward_history = []
    
    # --- Main Training Loop with Performance-Based Curriculum ---
    for i in range(algo.iteration + 1, target_iters + 1):
        iters_in_current_stage += 1
        
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

        # Also report reward per agent for comparability across different agent counts
        mean_rew_per_agent = mean_rew / max(1, current_n_agents)
        print(
            f"Iter {i}/{TOTAL_ITERS} [n_agents={current_n_agents}] | Mean Reward: {mean_rew:.3f}"
            f" (per-agent: {mean_rew_per_agent:.3f}) | Loss: {total_loss:.3f} | EpLenMean: {ep_len:.1f}"
        )

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
                    intrusion_penalty=INTRUSION_PENALTY,
                    n_agents=current_n_agents
                )
                print(
                    "[Eval] iter=%d | n_agents=%d | avg_rew=%.3f | avg_len=%.1f | avg_intr=%.2f | wp_rate=%.1f%%"
                    % (
                        i,
                        current_n_agents,
                        eval_metrics["avg_reward"],
                        eval_metrics["avg_length"],
                        eval_metrics["avg_intrusions"],
                        eval_metrics["waypoint_rate"] * 100.0,
                    )
                )
                _write_eval_row(metrics=eval_metrics, iteration=i, out_dir=os.path.join(METRICS_DIR, f"run_{RUN_ID}"))
                
                # --- Performance-Based Curriculum Advancement ---
                # Check if we should advance to next curriculum stage
                can_advance = (
                    curriculum_stage_idx < len(CURRICULUM_STAGES) - 1 and  # Not at final stage
                    iters_in_current_stage >= MIN_ITERS_PER_STAGE and  # Minimum iterations met
                    eval_metrics["avg_intrusions"] <= CURRICULUM_THRESHOLDS['avg_intrusions'] and
                    eval_metrics["waypoint_rate"] >= CURRICULUM_THRESHOLDS['waypoint_rate']
                )
                
                force_advance = iters_in_current_stage >= MAX_ITERS_PER_STAGE
                
                if can_advance or (force_advance and curriculum_stage_idx < len(CURRICULUM_STAGES) - 1):
                    next_stage_idx = curriculum_stage_idx + 1
                    next_n_agents = CURRICULUM_STAGES[next_stage_idx]
                    
                    print(f"\n{'='*70}")
                    if can_advance:
                        print(f"🎓 CURRICULUM ADVANCEMENT (Performance Criteria Met!)")
                        print(f"   ✅ Intrusions: {eval_metrics['avg_intrusions']:.2f} ≤ {CURRICULUM_THRESHOLDS['avg_intrusions']}")
                        print(f"   ✅ Waypoint Success: {eval_metrics['waypoint_rate']*100:.1f}% ≥ {CURRICULUM_THRESHOLDS['waypoint_rate']*100:.0f}%")
                    else:
                        print(f"⚠️  FORCED CURRICULUM ADVANCEMENT (Max iterations reached)")
                        print(f"   Current performance: intrusions={eval_metrics['avg_intrusions']:.2f}, wp_success={eval_metrics['waypoint_rate']*100:.1f}%")
                    
                    print(f"   Stage {curriculum_stage_idx + 1}/{len(CURRICULUM_STAGES)} → {next_stage_idx + 1}/{len(CURRICULUM_STAGES)}")
                    print(f"   Agents: {current_n_agents} → {next_n_agents}")
                    print(f"   Iterations in stage: {iters_in_current_stage}")
                    print(f"{'='*70}\n")
                    
                    # Save checkpoint before transition
                    print("💾 Saving checkpoint before curriculum transition...")
                    checkpoint_result = algo.save(checkpoint_dir=CHECKPOINT_DIR)
                    if hasattr(checkpoint_result, 'checkpoint') and hasattr(checkpoint_result.checkpoint, 'path'):
                        path = checkpoint_result.checkpoint.path
                    else:
                        path = str(checkpoint_result)
                    print(f"✅ Checkpoint saved to: {path}")
                    
                    # Stop the current algorithm
                    algo.stop()
                    
                    # Build new trainer with updated agent count
                    print(f"Building new trainer with {next_n_agents} agents...")
                    algo = build_trainer(next_n_agents)
                    
                    # Restore weights from the saved checkpoint
                    print(f"Restoring weights from checkpoint...")
                    algo.restore(path)
                    
                    # Update tracking variables
                    curriculum_stage_idx = next_stage_idx
                    current_n_agents = next_n_agents
                    iters_in_current_stage = 0
                    
                    print(f"✅ Curriculum transition complete. Continuing training...")
                    print(f"   New advancement criteria: intrusions ≤ {CURRICULUM_THRESHOLDS['avg_intrusions']}, "
                          f"waypoint success ≥ {CURRICULUM_THRESHOLDS['waypoint_rate']*100:.0f}%\n")
                
            except Exception as e:
                print(f"[Eval] skipped due to error: {e}")

    print("\n🚀 Training finished.")
    
    # Calculate and display total training time
    total_training_time = time.time() - training_start_time
    print(f"⏱️  Total training time: {total_training_time/60:.2f} minutes ({total_training_time/3600:.2f} hours) for {TOTAL_ITERS} iters.")
    
    final_checkpoint_result = algo.save(CHECKPOINT_DIR)
    # Extract just the path from the result to avoid printing massive object
    if hasattr(final_checkpoint_result, 'checkpoint') and hasattr(final_checkpoint_result.checkpoint, 'path'):
        final_path = final_checkpoint_result.checkpoint.path
    else:
        final_path = str(final_checkpoint_result)
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
