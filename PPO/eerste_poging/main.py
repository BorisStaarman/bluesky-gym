# This code has the option to only evaluate or only train
# Use toggle switch for that 

# standard imports
import os
import gymnasium as gym
import numpy as np
import torch 
import matplotlib.pyplot as plt
# MARL ray imports
import ray
from ray.rllib.algorithms.algorithm import Algorithm # Use the generic Algorithm class
from ray.rllib.algorithms.ppo import PPOConfig
from bluesky_gym.envs.ma_env_ppo import SectorEnv
from bluesky_gym import register_envs
register_envs()

# Switches 
TRAIN = False             # Set to False for evaluation
RENDER = True             # Set to True for visualization
# set parameters
N_AGENTS = 6 # agents in environment present
TOTAL_ITERS = 100 # number of training iterations
# path for model
script_dir = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(script_dir, "models/sectorcr_ma_ppo")


def build_trainer(): 
    # Function that sets the policy and parameters
    def policy_map(agent_id, *_, **__):
        return "shared_policy"

    cfg = (
        PPOConfig()
        .environment("sector_env", env_config={"n_agents": N_AGENTS})
        .framework("torch")
        .env_runners(num_env_runners=os.cpu_count() - 1) # Using parallel workers
        .training(
            train_batch_size=4000, # number of steps to take per iteration before updating the model
            model={"fcnet_hiddens": [256, 256]},  # number of hidden layers and nodes
            gamma=0.99, # discount factor, determines impact of future rewards
            lr=3e-4, # learning rate, determines how much weights are updated 
            vf_clip_param=100.0, # value function clipping, clips the value function (critic)
            # other paramaters to include
            # num_epochs= standard is 30 (number of iterations that weights are updated per iteration)
            # min_batch_size, number of steps in a minibatch, default is 128
            
        )
        .multi_agent(
            policies={"shared_policy": (None, None, None, {})},
            policy_mapping_fn=policy_map,
        )
        .resources(num_gpus=0)
    )
    return cfg.build()

def train_loop():
    print(f"\n🚀 Training PPO for {TOTAL_ITERS} iters | agents={N_AGENTS}\n")
    loss = []
    reward = []
    mean_intrusions = []
    algo = build_trainer()
    for i in range(1, TOTAL_ITERS + 1):
        result = algo.train() # start collect the data
        # environment provides an observation, the NN predicts an action
        # action is send to environments step method, which determines a 
        # a new observation and a reward
        
        # gather data to review
        mean_rew = result.get("episode_reward_mean", float("nan"))
        # append loss
        learner_stats = result.get("learners", {}).get("shared_policy", {})
        loss.append(learner_stats.get("total_loss", float("nan")))
        # append reward
        reward.append(mean_rew)
        # number of loss of seperations 
        custom_metrics = result.get("env_runners", {}).get("custom_metrics", {})
        mean_intrusions.append(custom_metrics.get("total_intrusions_mean", float("nan")))
        # print iteration, mean reward, total loss and intrusions
        print(f"Iter {i}/{TOTAL_ITERS} | Mean reward: {mean_rew:.3f}| loss: {learner_stats.get('total_loss', float('nan')):.3f}| Intrusions: {custom_metrics.get('total_intrusions_mean', float('nan')):.3f}" )
        
        # save all losses seperately
        # if i % 10 == 0:
            
    path = algo.save(CHECKPOINT_DIR)
    print(f"\n✅ Saved checkpoint to:\n{path}\n")
    return loss, reward, mean_intrusions



def evaluate_loop():
    print("\n🎯 Evaluating policy...\n")
    
    algo = Algorithm.from_checkpoint(CHECKPOINT_DIR)
    module = algo.get_module("shared_policy")
    env = SectorEnv(render_mode="human", n_agents=N_AGENTS)
    
    num_episodes_to_run = 5
    # It must be INSIDE the function to reset the list for each evaluation run.
    episode_rewards = []

    for episode in range(1, num_episodes_to_run + 1):
        print(f"\n--- Starting Evaluation Episode {episode}/{num_episodes_to_run} ---")
        
        obs, infos = env.reset(seed=42 + episode)
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
            
            obs, rewards, terminateds, truncateds, infos = env.step(actions)
            if rewards:
                episode_reward += sum(rewards.values())
            episode_steps += 1
        
        # do something after each episode happens here

        print(f"Episode finished in {episode_steps} steps. Total reward: {episode_reward:.3f}")
        episode_rewards.append(episode_reward)

    print(f"\n✅ Finished running {num_episodes_to_run} episodes. Average reward: {np.mean(episode_rewards):.3f}\n")

    env.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_episodes_to_run + 1), episode_rewards, marker='o', linestyle='-')
    plt.title("Total Reward per Evaluation Episode")
    plt.xlabel("Episode Number")
    plt.ylabel("Total Reward")
    plt.xticks(range(1, num_episodes_to_run + 1))
    plt.grid(True)
    plt.show()
    
     
if __name__ == "__main__":
    ray.shutdown()
    ray.init(include_dashboard=False)

    if TRAIN:
        loss, reward, mean_intrusions = train_loop()
        
        # create graph of training data
        fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        fig.suptitle('Training Metrics Over Iterations', fontsize=16)
        # --- Plot 1: Loss ---
        axs[0].plot(loss, marker='o', linestyle='-')
        axs[0].set_title('Total Loss')
        axs[0].set_ylabel('Loss')
        axs[0].grid(True)
        
        # --- Plot 2: Reward ---
        axs[1].plot(reward, marker='o', linestyle='-', color='g')
        axs[1].set_title('Mean Episode Reward')
        axs[1].set_ylabel('Mean Reward')
        axs[1].grid(True)
        
        # --- Plot 3: Intrusions ---
        axs[2].plot(mean_intrusions, marker='o', linestyle='-', color='r')
        axs[2].set_title('Mean Intrusions per Episode')
        axs[2].set_ylabel('Mean Intrusions')
        axs[2].set_xlabel('Training Iterations') # Only need x-axis label on the bottom plot
        axs[2].grid(True)
        # Adjust layout to prevent titles from overlapping
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        # Show the single window with all three plots
        plt.show()
        
        #
    else:
        evaluate_loop()
       
    ray.shutdown()