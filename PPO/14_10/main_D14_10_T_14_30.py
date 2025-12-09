# standard imports
import os
import numpy as np
import torch 
import matplotlib.pyplot as plt

# MARL ray imports
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks

# Make sure these imports point to your custom environment files
from bluesky_gym.envs.ma_env_ppo import SectorEnv
from bluesky_gym import register_envs

# Register your custom environment with Gymnasium
register_envs()

# --- Parameters ---
N_AGENTS = 6
TOTAL_ITERS = 10
EVALUATION_INTERVAL = 10  # Run an evaluation every 10 training iterations
EVALUATION_EPISODES = 5   # Number of episodes to run for each evaluation

script_dir = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(script_dir, "models/sectorcr_ma_ppo")

def build_trainer(): 
    """
    Builds and configures the PPO algorithm.
    """
    def policy_map(agent_id, *_, **__):
        return "shared_policy"

    cfg = (
        PPOConfig()
        .environment("sector_env", env_config={"n_agents": N_AGENTS}, disable_env_checking=True)
        .framework("torch")  
        .env_runners(num_env_runners=os.cpu_count() - 1)
        .training(
            train_batch_size=4000,
            model={"fcnet_hiddens": [256, 256]},
            gamma=0.99,
            lr=3e-4,
            vf_clip_param=100.0,
        )
        .multi_agent(
            policies={"shared_policy": (None, None, None, {})},
            policy_mapping_fn=policy_map,
        )
        .resources(num_gpus=0)
    )
    return cfg.build()

def run_periodic_evaluation(algo, n_agents, num_episodes):
    """
    Runs a detailed evaluation on the CURRENT state of the algorithm.
    This is your original 'evaluate_loop' refactored into a function.
    """
    print("\n" + "="*40)
    print(f"🎯 Running Evaluation for {num_episodes} episodes...")
    
    env = SectorEnv(render_mode="human" if RENDER else None, n_agents=n_agents)
    module = algo.get_module("shared_policy")

    episode_rewards = []
    episode_steps_list = []
    total_intrusions = 0
    total_waypoints_reached = 0

    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        
        while env.agents:
            agent_ids = list(obs.keys())
            obs_list = list(obs.values())
            input_dict = {"obs": torch.from_numpy(np.stack(obs_list))}
            output_dict = module.forward_inference(input_dict)
            
            dist_class = module.get_inference_action_dist_cls()
            action_dist = dist_class.from_logits(output_dict["action_dist_inputs"])
            actions_tensor = action_dist.sample()
            actions_np = actions_tensor.cpu().numpy()
            actions = {agent_id: action for agent_id, action in zip(agent_ids, actions_np)}
            
            obs, rewards, _, _, _ = env.step(actions)
            if rewards:
                episode_reward += sum(rewards.values())
            episode_steps += 1
        
        episode_rewards.append(episode_reward)
        episode_steps_list.append(episode_steps)
        total_intrusions += env.total_intrusions
        total_waypoints_reached += len(env.waypoint_reached_agents)

    env.close()

    avg_reward = np.mean(episode_rewards)
    avg_steps = np.mean(episode_steps_list)
    avg_intrusions = total_intrusions / num_episodes
    waypoint_rate = (total_waypoints_reached / (num_episodes * n_agents)) * 100

    print(f"✅ Evaluation Complete:")
    print(f"  - Average Reward: {avg_reward:.3f}")
    print(f"  - Average Episode Length: {avg_steps:.1f} steps")
    print(f"  - Average Intrusions: {avg_intrusions:.2f}")
    print(f"  - Waypoint Reached Rate: {waypoint_rate:.1f}%")
    print("="*40 + "\n")
    
    return avg_reward, avg_steps, avg_intrusions, waypoint_rate

if __name__ == "__main__":
    ray.shutdown()
    ray.init()

    # Set to False for long training runs to improve speed
    RENDER = False

    # Initialize or restore the algorithm from a checkpoint
    if os.path.exists(os.path.join(CHECKPOINT_DIR, "algorithm_state.json")):
        print("Restoring from checkpoint...")
        algo = Algorithm.from_checkpoint(CHECKPOINT_DIR)
    else:
        print("Building new trainer...")
        algo = build_trainer()

    # --- Lists to store training history for plotting ---
    training_rewards = []
    training_losses = []
    reward_progress_hist = []
    reward_drift_hist = []
    reward_intrusion_hist = []

    # --- Main Training and Evaluation Loop ---
    for i in range(algo.iteration + 1, TOTAL_ITERS + 1):
        result = algo.train()
        
        hist_stats = result.get("hist_stats", {})
         # RLlib automatically finds 'reward_drift' from the info dict,
        # calculates stats, and puts them in hist_stats.
        mean_rew = np.mean(hist_stats.get("episode_reward", [np.nan]))
        reward_progress = np.mean(hist_stats.get("reward_progress", [np.nan]))
        reward_drift = np.mean(hist_stats.get("reward_drift", [np.nan]))
        reward_intrusion = np.mean(hist_stats.get("reward_intrusion", [np.nan]))

        learner_stats = result.get("learners", {}).get("shared_policy", {})
        total_loss = learner_stats.get("total_loss", np.nan)
        
        training_rewards.append(mean_rew)
        training_losses.append(total_loss)
        reward_progress_hist.append(reward_progress)
        reward_drift_hist.append(reward_drift)
        reward_intrusion_hist.append(reward_intrusion)

        print(f"Iter {i}/{TOTAL_ITERS} | Mean Reward: {mean_rew:.3f} | Loss: {total_loss:.3f}")

        if i % EVALUATION_INTERVAL == 0 and i > 0:
            path = algo.save(CHECKPOINT_DIR)
            print(f"✅ Checkpoint saved to: {path}")
            # run_periodic_evaluation(algo, N_AGENTS, EVALUATION_EPISODES)
            
    print("\n🚀 Training finished.")
    final_path = algo.save(CHECKPOINT_DIR)
    print(f"✅ Final checkpoint saved to: {final_path}")

    # --- FINAL PLOTTING SECTION ---
    fig, axs = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    fig.suptitle('Training Metrics Over Iterations', fontsize=16)
    iterations = range(1, len(training_rewards) + 1)

    # Plot 1: Overall Mean Reward
    axs[0].plot(iterations, np.nan_to_num(training_rewards), marker='o', linestyle='-')
    axs[0].set_title('Mean Episode Reward')
    axs[0].set_ylabel('Reward')
    axs[0].grid(True)

    # Plot 2: Total Loss
    axs[1].plot(iterations, np.nan_to_num(training_losses), marker='o', linestyle='-', color='r')
    axs[1].set_title('Total Loss')
    axs[1].set_ylabel('Loss')
    axs[1].grid(True)

    # Plot 3: Reward Components
    axs[2].plot(iterations, np.nan_to_num(reward_progress_hist), marker='.', linestyle='-', label='Progress Reward (mean)')
    axs[2].plot(iterations, np.nan_to_num(reward_drift_hist), marker='.', linestyle='-', label='Drift Penalty (mean)')
    axs[2].plot(iterations, np.nan_to_num(reward_intrusion_hist), marker='.', linestyle='-', label='Intrusion Penalty (mean)')
    axs[2].set_title('Mean of Step-wise Reward Components')
    axs[2].set_ylabel('Reward Value')
    axs[2].set_xlabel('Training Iterations')
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

    ray.shutdown()
