# standard imports
import os
import numpy as np
import torch 
import matplotlib.pyplot as plt

# MARL ray imports
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
# from ray.rllib.algorithms.callbacks import DefaultCallbacks
# from ray.rllib.policy.sample_batch import SampleBatch



# Make sure these imports point to your custom environment files
from bluesky_gym.envs.ma_env import SectorEnv
from bluesky_gym import register_envs

# Register your custom environment with Gymnasium
register_envs()

# --- Parameters ---
N_AGENTS = 6
TOTAL_ITERS = 10
EVALUATION_INTERVAL = 10  # Run an evaluation every 10 training iterations
EVALUATION_EPISODES = 1   # Number of episodes to run for each evaluation

script_dir = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(script_dir, "models/sectorcr_ma_ppo")



# ✅ Works on Ray/RLlib 2.49.x (new API stack)
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.sample_batch import SampleBatch

def _eid(ep):
    # Robust episode-id getter across variants
    for name in ("episode_id", "id_", "id"):
        if hasattr(ep, name):
            return getattr(ep, name)
    return id(ep)

class RewardBreakdownCallbacks(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        # per-episode accumulators (don’t touch episode.user_data/custom_metrics)
        self._acc = {}  # {ep_id: {"prog": float, "drift": float, "intr": float}}

    def on_episode_start(self, *args, **kwargs):
        episode = kwargs.get("episode") or (len(args) > 3 and args[3])
        if episode is None:
            return
        self._acc[_eid(episode)] = {"prog": 0.0, "drift": 0.0, "intr": 0.0}

    def on_postprocess_trajectory(self, *args, **kwargs):
        # Preferred place to read step-wise infos on the new stack
        episode = kwargs.get("episode") or (len(args) > 1 and args[1])
        batch = kwargs.get("postprocessed_batch") or (len(args) > 5 and args[5])
        if episode is None or batch is None:
            return

        eid = _eid(episode)
        acc = self._acc.setdefault(eid, {"prog": 0.0, "drift": 0.0, "intr": 0.0})

        infos = batch.get(SampleBatch.INFOS) or batch.get("infos")
        if not infos:
            return

        for info in infos:
            if not info:
                continue
            acc["prog"]  += float(info.get("reward_progress", 0.0))
            acc["drift"] += float(info.get("reward_drift", 0.0))
            acc["intr"]  += float(info.get("reward_intrusion", 0.0))

    def on_episode_end(self, *args, **kwargs):
        episode = kwargs.get("episode") or (len(args) > 3 and args[3])
        if episode is None:
            return
        eid = _eid(episode)
        acc = self._acc.pop(eid, {"prog": 0.0, "drift": 0.0, "intr": 0.0})

        # ✅ New-API way to report metrics:
        metrics_logger = kwargs.get("metrics_logger")
        if metrics_logger is not None:
            metrics_logger.log_value("reward_progress",  float(acc["prog"]))
            metrics_logger.log_value("reward_drift",     float(acc["drift"]))
            metrics_logger.log_value("reward_intrusion", float(acc["intr"]))
        else:
            # Fallback for older stacks (won’t be hit on 2.49.2)
            if hasattr(episode, "custom_metrics"):
                episode.custom_metrics["reward_progress"]  = float(acc["prog"])
                episode.custom_metrics["reward_drift"]     = float(acc["drift"])
                episode.custom_metrics["reward_intrusion"] = float(acc["intr"])


        
        
def build_trainer(): 
    """
    Builds and configures the PPO algorithm.
    """
    # def policy_map(agent_id, *_, **__):
    #     return "shared_policy"

    cfg = (
        PPOConfig()
        .environment("sector_env", env_config={"n_agents": N_AGENTS}, disable_env_checking=True)
        .framework("torch")  
        #.env_runners(num_env_runners=os.cpu_count() - 1)
        .env_runners(num_env_runners=1, sample_timeout_s=120) # for during debugging
        .training(
            train_batch_size=4000,
            model={"fcnet_hiddens": [256, 256]},
            gamma=0.99,
            lr=3e-4,
            vf_clip_param=100.0,
        )
        .multi_agent(
            policies={"shared_policy": (None, None, None, {})},
            policy_mapping_fn=lambda aid, *a, **k: "shared_policy",
        )
        .callbacks(RewardBreakdownCallbacks)
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
        
        if result.get("num_env_steps_sampled", 0) == 0 and result.get("episodes_total", 0) == 0:
            print("⚠️ No samples this iteration; skipping.")
            continue
        
        # after result = algo.train()
        # hist_stats = result.get("hist_stats", {})
        cm = result.get("custom_metrics", {})
        progress_mean   = cm.get("reward_progress_mean",  np.nan)
        drift_mean      = cm.get("reward_drift_mean",     np.nan)
        intrusion_mean  = cm.get("reward_intrusion_mean", np.nan)

        # Learner stats can appear directly or under "learner_stats"
        learners = result.get("learners", {}) or {}
        policy_key = "shared_policy" if "shared_policy" in learners else (next(iter(learners), None))
        total_loss = np.nan
        if policy_key:
            l = learners[policy_key]
            total_loss = (
                l.get("total_loss", np.nan)
                if isinstance(l, dict) else np.nan
            )
            # Some builds nest under "learner_stats"
            if np.isnan(total_loss):
                total_loss = l.get("learner_stats", {}).get("total_loss", np.nan)

        #training_rewards.append(total_ep_return_mean)
        training_losses.append(total_loss)
        reward_progress_hist.append(progress_mean)
        reward_drift_hist.append(drift_mean)
        reward_intrusion_hist.append(intrusion_mean)

        print(
            f"Iter {i}/{TOTAL_ITERS} | "
            #f"Mean Reward: {total_ep_return_mean:.3f} | "
            f"Loss: {np.nan if np.isnan(total_loss) else round(float(total_loss),3)} | "
            f"Prog: {progress_mean:.3f} Drift: {drift_mean:.3f} Intr: {intrusion_mean:.3f}"
        )
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
    iters = range(1, len(training_rewards) + 1)

    axs[0].plot(iters, np.nan_to_num(training_rewards, nan=np.nan), marker='o')
    axs[0].set_title('Mean Episode Reward'); axs[0].set_ylabel('Reward'); axs[0].grid(True)

    axs[1].plot(iters, np.nan_to_num(training_losses, nan=np.nan), marker='o')
    axs[1].set_title('Total Loss (from learner)'); axs[1].set_ylabel('Loss'); axs[1].grid(True)

    axs[2].plot(iters, np.nan_to_num(reward_progress_hist,  nan=np.nan), marker='.', label='Progress (episode total mean)')
    axs[2].plot(iters, np.nan_to_num(reward_drift_hist,     nan=np.nan), marker='.', label='Drift (episode total mean)')
    axs[2].plot(iters, np.nan_to_num(reward_intrusion_hist, nan=np.nan), marker='.', label='Intrusion (episode total mean)')
    axs[2].set_title('Reward Components (per-episode totals averaged per iteration)')
    axs[2].set_ylabel('Value'); axs[2].set_xlabel('Training Iterations'); axs[2].grid(True); axs[2].legend()


    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

    ray.shutdown()
