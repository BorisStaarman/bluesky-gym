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

from ray.rllib.models import ModelCatalog
from attention_model_A import AttentionSACModel

# Make sure these imports point to your custom environment registration
from bluesky_gym.envs.ma_env_ppo_AM import SectorEnv
from ray.tune.registry import register_env

from run_config import RUN_ID

# Register your custom environment directly for RLlib
register_env("sector_env", lambda config: SectorEnv(**config))
ModelCatalog.register_custom_model("d2mav_attention", AttentionSACModel)

# --- Parameters ---
N_AGENTS = 20  # Number of agents for training
TOTAL_ITERS = 5  # Maximum total iterations
TOTAL_ITERS_R = 3 # For scheduling purposes
EVALUATION_INTERVAL = 2  # e.g., set to 1 or 5 to save during training


# --- Metrics Directory ---
# When copying to a new folder, update this to match your folder name!
script_dir = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.join(script_dir, "metrics")
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
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
        .environment(
            "sector_env",
            env_config={"n_agents": n_agents,
                        "run_id": RUN_ID,
                        "metrics_base_dir": METRICS_DIR},
            disable_env_checking=True
        )
        .framework("torch")
        .env_runners(num_env_runners=os.cpu_count() - 1)
        .training(
            model={
                "custom_model": "d2mav_attention",
                "custom_model_config": {
                    "hidden_dims": [512, 512],
                    "is_critic": False,
                    "n_agents": n_agents,
                },
                "free_log_std": True,
                "vf_share_layers": False,
            },
            # Training configuration
            train_batch_size=64000,
            minibatch_size=2000,
            num_sgd_iter=12,
            
            # PPO clipping
            clip_param=0.2,
            vf_clip_param=10.0,
            
            # Loss coefficients
            vf_loss_coeff=2.0,  # Increase to 2.0 so critic gradients are stronger
            entropy_coeff=0.01,  # Start with good exploration (can be manually adjusted later if needed)
            
            # Gradient and optimization
            grad_clip=0.5,
            lr=1.5e-4,
            
            # Discount and GAE
            gamma=0.99,
            lambda_=0.95,

            # KL divergence: soft constraint for policy stability
            use_kl_loss=True,
            kl_target=0.01,
            kl_coeff=1.0,
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
    policy = algo.get_policy("shared_policy")
    
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
                # Use old API: compute_actions with explore=False for deterministic evaluation
                actions_np = policy.compute_actions(obs_list, explore=False)[0]
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

    
    print("-" * 30)

    target_iters = None
    restored_from = None

    base_state = os.path.join(CHECKPOINT_DIR, "algorithm_state.json")
    # if not FORCE_RETRAIN and os.path.exists(base_state):
    #     restored_from = CHECKPOINT_DIR
    # elif not FORCE_RETRAIN:
    #     # Try to find a checkpoint in subfolders (e.g., iter_00050, final_YYYYMMDD...)
    #     cand = _find_latest_checkpoint(CHECKPOINT_DIR)
    #     if cand:
    #         restored_from = cand

    
    print("Building new trainer...")
    print(f"Starting training with {N_AGENTS} agents")
    algo = build_trainer(N_AGENTS)
    # Fresh training: run up to TOTAL_ITERS
    target_iters = int(TOTAL_ITERS)

    # Loss history for different components
    total_loss_history = []
    policy_loss_history = []
    value_loss_history = []
    entropy_history = []
    kl_divergence_history = []
    vf_explained_var_history = []
    temperature_history = []
    reward_history = []
    episode_length_history = []
    
    # Early stopping tracking
    best_reward = float('-inf')  # Best single-iteration reward (for saving checkpoints)
    best_reward_iteration = 0
    best_checkpoint_path = None
    iterations_without_improvement = 0  # Based on smoothed reward (for stopping)
    early_stop_triggered = False
    
    # --- Main Training Loop ---
    for i in range(algo.iteration + 1, target_iters + 1):
        result = algo.train()

        # Extract metrics from result (matching working Stage 2 code)
        env_runners = result.get("env_runners", {})
        mean_rew = env_runners.get("episode_return_mean", float("nan"))
        ep_len = env_runners.get("episode_len_mean", float("nan"))
        
        # Extract learner stats (OLD API path)
        info = result.get("info", {})
        learner_stats = info.get("learner", {}).get("shared_policy", {}).get("learner_stats", {})
        
        # Extract loss components with fallback keys
        policy_loss = learner_stats.get("policy_loss", learner_stats.get("pi_loss", 0.0))
        vf_loss = learner_stats.get("vf_loss", learner_stats.get("value_function_loss", 0.0))
        entropy = learner_stats.get("entropy", 0.0)
        vf_explained_var = learner_stats.get("vf_explained_var", 0.0)
        total_loss = learner_stats.get("total_loss", abs(policy_loss) + abs(vf_loss))
        
        # Append to history
        total_loss_history.append(total_loss)
        policy_loss_history.append(policy_loss)
        value_loss_history.append(vf_loss)
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
        
        # Enhanced progress display with detailed metrics
        print(f"Iter {i}/{TOTAL_ITERS} | Reward: {mean_rew:.3f} | Total Loss: {total_loss:.4f}")
        print(f"       Policy Loss: {policy_loss:.4f} | Value Loss: {vf_loss:.4f} | Entropy: {entropy:.4f}")
        print(f"       VF Explained Var: {vf_explained_var:.4f} (1.0=perfect, 0.0=random)")

        # --- Early Stopping Check ---
        

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
    
    # --- Save Training Metrics for Later Plotting ---
    import pickle
    metrics_dir = os.path.join(METRICS_DIR, f"run_{RUN_ID}")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_file = os.path.join(metrics_dir, "training_metrics.pkl")
    
    training_metrics = {
        'reward_history': reward_history,
        'total_loss_history': total_loss_history,
        'policy_loss_history': policy_loss_history,
        'value_loss_history': value_loss_history,
        'entropy_history': entropy_history,
        'vf_explained_var_history': vf_explained_var_history,
        'temperature_history': temperature_history,
        'episode_length_history': episode_length_history,
        'best_reward': best_reward,
        'best_reward_iteration': best_reward_iteration,
    }
    
    with open(metrics_file, 'wb') as f:
        pickle.dump(training_metrics, f)
    print(f"💾 Training metrics saved to: {metrics_file}")
    
    # --- Plot the Loss and Reward in a Single Figure ---
    fig, axes = plt.subplots(6, 1, figsize=(10, 24))  # Create 6 subplots matching Stage 2 format
    
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
    
    # Plot Episode Length
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
    plt.close()  # Close to free memory

    ray.shutdown()
