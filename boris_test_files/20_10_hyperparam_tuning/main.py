# standard imports
import os
import shutil
import matplotlib.pyplot as plt

# MARL ray imports
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig

# Make sure these imports point to your custom environment registration
from bluesky_gym import register_envs
# Optional: If register_envs() fully registers "sector_env", the next line can be removed.
from bluesky_gym.envs.ma_env import SectorEnv  # optional
from datetime import datetime

from run_config import RUN_ID

# Register your custom environment with Gymnasium
register_envs()

# for logging the .csv files

# --- Parameters ---
N_AGENTS = 6
TOTAL_ITERS = 25
FORCE_RETRAIN = True  # Set to True to delete old model and start fresh
# Optional: Only useful if you want periodic checkpoints mid-training.
EVALUATION_INTERVAL = None  # e.g., set to 1 or 5 to save during training

# --- Path for model ---
script_dir = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(script_dir, "models/sectorcr_ma_ppo")


def build_trainer():
    """Builds and configures the PPO algorithm."""
    def policy_map(agent_id, *_, **__):
        return "shared_policy"

    cfg = (
        PPOConfig()
        .environment(
            "sector_env",
            env_config={"n_agents": N_AGENTS,
                        "run_id": RUN_ID},
            disable_env_checking=True
        )
        .framework("torch")
        .env_runners(num_env_runners=os.cpu_count() - 1)
        .training(
            train_batch_size=4000,
            model={"fcnet_hiddens": [256, 256]},
            gamma=0.98, # was 0.99
            lr= 5e-5, # was 3e-4
            vf_clip_param=25.0, # value function clipping, was 100
            entropy_coeff=0.02, # entropy coefficient ensures exploration, used to be the standard value idk what it is
        )
        .multi_agent(
            policies={"shared_policy": (None, None, None, {})},
            policy_mapping_fn=policy_map,
        )
        .resources(num_gpus=1)
    )
    return cfg.build()

if __name__ == "__main__":
    ray.shutdown()
    ray.init()

    if FORCE_RETRAIN and os.path.exists(CHECKPOINT_DIR):
        print(f"FORCE_RETRAIN is True. Deleting old checkpoint directory:\n{CHECKPOINT_DIR}")
        try:
            shutil.rmtree(CHECKPOINT_DIR)
            print("✅ Old checkpoint directory removed.")
        except OSError as e:
            print(f"Error: {e.strerror} - {CHECKPOINT_DIR}")
    print("-" * 30)

    if os.path.exists(os.path.join(CHECKPOINT_DIR, "algorithm_state.json")):
        print("Restoring from checkpoint...")
        algo = Algorithm.from_checkpoint(CHECKPOINT_DIR)
    else:
        print("Building new trainer...")
        algo = build_trainer()

    # Loss history for different components
    total_loss_history = []
    policy_loss_history = []
    value_loss_history = []
    entropy_loss_history = []
    # reward_history = []
    reward_history = []
    
    # --- Main Training Loop ---
    for i in range(algo.iteration + 1, TOTAL_ITERS + 1):
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

        print(f"Iter {i}/{TOTAL_ITERS} | Mean Reward: {mean_rew:.3f} | Loss: {total_loss:.3f} | EpLenMean: {ep_len:.1f}")

        # Optional periodic checkpointing
        if EVALUATION_INTERVAL and i % EVALUATION_INTERVAL == 0:
            path = algo.save(CHECKPOINT_DIR)
            print(f"✅ Checkpoint saved to: {path}")

    print("\n🚀 Training finished.")
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
