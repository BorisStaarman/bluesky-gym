# standard imports
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import time

# MARL ray imports
import ray
from ray.rllib.algorithms.algorithm import Algorithm

# Make sure these imports point to your custom environment files
from bluesky_gym.envs.ma_env import SectorEnv
from bluesky_gym import register_envs

# Register your custom environment with Gymnasium
register_envs()

# --- Parameters for Evaluation ---
N_AGENTS = 6  # The number of agents the model was trained with
NUM_EVAL_EPISODES = 10  # How many episodes to run for evaluation
RENDER = True  # Set to True to watch the agent play

# This path MUST match the checkpoint directory from your main.py training script
script_dir = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(script_dir, "models/sectorcr_ma_ppo")


if __name__ == "__main__":
    # Initialize Ray
    ray.shutdown()
    ray.init(include_dashboard=False)

    # --- Check if a checkpoint exists ---
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"❌ Checkpoint directory not found at: {CHECKPOINT_DIR}")
        print("Please run the `main.py` script first to train and save a model.")
        ray.shutdown()
        exit()

    print(f"\n🎯 Evaluating policy from checkpoint: {CHECKPOINT_DIR}\n")

    # --- Load the trained algorithm and policy ---
    algo = Algorithm.from_checkpoint(CHECKPOINT_DIR)
    module = algo.get_module("shared_policy")
    env = SectorEnv(render_mode="human" if RENDER else None, n_agents=N_AGENTS)

    # --- Lists to store metrics from the evaluation run ---
    episode_rewards = []
    episode_steps_list = []
    episode_intrusions = []
    total_waypoints_reached = 0

    # --- Main Evaluation Loop ---
    for episode in range(1, NUM_EVAL_EPISODES + 1):
        print(f"\n--- Starting Evaluation Episode {episode}/{NUM_EVAL_EPISODES} ---")

        obs, info = env.reset()
        episode_reward = 0.0
        episode_steps = 0

        # Run the episode until it's done
        while env.agents:
            # The logic to get an action from the policy is the same as in your example
            agent_ids = list(obs.keys())
            obs_list = list(obs.values())
            input_dict = {"obs": torch.from_numpy(np.stack(obs_list))}

            # Get action from the trained module
            output_dict = module.forward_inference(input_dict)
            dist_class = module.get_inference_action_dist_cls()
            action_dist = dist_class.from_logits(output_dict["action_dist_inputs"])
            actions_tensor = action_dist.sample()
            actions_np = actions_tensor.cpu().numpy()
            actions = {agent_id: action for agent_id, action in zip(agent_ids, actions_np)}

            # Step the environment
            obs, rewards, terminateds, truncateds, infos = env.step(actions)
            if rewards:
                episode_reward += sum(rewards.values())
            episode_steps += 1
            
            # Slow down rendering to make it watchable
            if RENDER:
                time.sleep(0.1)

        # After the episode is finished, collect and store the final stats
        print(f"-> Episode finished in {episode_steps} steps.")
        print(f"   - Total Reward: {episode_reward:.3f}")
        print(f"   - Intrusions: {env.total_intrusions}")
        print(f"   - Waypoints Reached: {len(env.waypoint_reached_agents)}/{N_AGENTS}")

        episode_rewards.append(episode_reward)
        episode_steps_list.append(episode_steps)
        episode_intrusions.append(env.total_intrusions)
        total_waypoints_reached += len(env.waypoint_reached_agents)

    # --- Print Final Summary Statistics ---
    print("\n" + "="*50)
    print("✅ EVALUATION COMPLETE")
    print(f"Ran {NUM_EVAL_EPISODES} episodes.")
    print(f"  - Average Reward: {np.mean(episode_rewards):.3f}")
    print(f"  - Average Episode Length: {np.mean(episode_steps_list):.1f} steps")
    print(f"  - Average Intrusions per Episode: {np.mean(episode_intrusions):.2f}")
    
    waypoint_rate = (total_waypoints_reached / (NUM_EVAL_EPISODES * N_AGENTS)) * 100
    print(f"  - Overall Waypoint Reached Rate: {waypoint_rate:.1f}%")
    print("="*50 + "\n")

    # --- Plot the results ---
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, NUM_EVAL_EPISODES + 1), episode_rewards, marker='o', linestyle='-')
    plt.title("Total Reward per Evaluation Episode")
    plt.xlabel("Episode Number")
    plt.ylabel("Total Reward")
    plt.xticks(range(1, NUM_EVAL_EPISODES + 1))
    plt.grid(True)
    plt.show()

    # --- Clean up ---
    env.close()
    ray.shutdown()
