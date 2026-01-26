# standard imports
import os
import sys
import shutil
import matplotlib.pyplot as plt
import numpy as np
import time
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import io

import bluesky as bs
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch
from mvp_2d import MVP_2D
import bluesky_gym.envs.common.functions as fn

# MARL ray imports
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.sac import SACConfig

from ray.rllib.models import ModelCatalog
# from attention_model_M import AttentionSACModel # Multiplicative method
from attention_model_A2 import AttentionSACModel # additive method

# Make sure these imports point to your custom environment registration
from bluesky_gym.envs.ma_env_SAC_AM import SectorEnv
from ray.rllib.policy.sample_batch import SampleBatch

from ray.tune.registry import register_env

import torch
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from run_config import RUN_ID

# Register your custom environment directly for RLlib
register_env("sector_env", lambda config: SectorEnv(**config))
ModelCatalog.register_custom_model("attention_sac", AttentionSACModel)

# --- Parameters ---
N_AGENTS = 20  # Number of agents for training
TOTAL_ITERS = 2000   # 18000 Maximum total iterations
EXTRA_ITERS = 50           # When resuming, run this many more iterations
FORCE_RETRAIN = True       # Start fresh with new hyperparameters
# Optional: Only useful if you want periodic checkpoints mid-training.
EVALUATION_INTERVAL = 400  # e.g., set to 1 or 5 to save during training

START_ALPHA = 0.15  # Initial alpha value for SAC entropy term

PRETRAIN_EPISODES = 150

# --- Burn-in Phase Parameters (Offline Learning from Expert Buffer) ---
BURN_IN_ITERATIONS = 2000     # Number of gradient steps on expert data only
BURN_IN_BATCH_SIZE = 4096     # Batch size for burn-in sampling
ENABLE_BURN_IN = True         # Set to False to skip burn-in phase

# --- Early Stopping Parameters ---
ENABLE_EARLY_STOPPING = True    # Set to False to disable early stopping
EARLY_STOP_PATIENCE = 18001        # Number of iterations without improvement before stopping
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
        # --- NIEUWE AGRESSIEVE CONFIGURATIE ---
        START_ALPHA = 0.15
        MIDDLE_ALPHA = 0.08  # Lager tussenpunt voor snellere focus
        END_ALPHA = 0.04     # Lager eindpunt voor maximale precisie
        
        # We verkorten de fases aanzienlijk
        FREEZE_UNTIL = TOTAL_ITERS * 0.10  # Slechts 10% (150 iters) op 0.15
        DECAY_PERIOD = TOTAL_ITERS * 0.40  # 40% (600 iters) voor de daling
        
        current_iter = result["training_iteration"]
        
        if current_iter < FREEZE_UNTIL:
            target_alpha = START_ALPHA
        elif current_iter < (FREEZE_UNTIL + DECAY_PERIOD):
            # Lineaire daling naar MIDDLE_ALPHA
            progress = (current_iter - FREEZE_UNTIL) / DECAY_PERIOD
            target_alpha = START_ALPHA - ((START_ALPHA - MIDDLE_ALPHA) * progress)
        else:
            # Langzame verfijning naar de scherpe END_ALPHA
            remaining_iters = TOTAL_ITERS - (FREEZE_UNTIL + DECAY_PERIOD)
            if remaining_iters > 0:
                progress = (current_iter - (FREEZE_UNTIL + DECAY_PERIOD)) / remaining_iters
                target_alpha = MIDDLE_ALPHA - ((MIDDLE_ALPHA - END_ALPHA) * progress)
            else:
                target_alpha = END_ALPHA 
            

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
        
        # 5. FIX FOR NaN ENTROPY: Clamp log_std to prevent numerical instability
        # This prevents log_std from becoming too negative, which causes entropy calculation to fail
        try:
            # Access the actor model (policy network)
            actor_model = None
            if hasattr(policy, 'model'):
                # For SAC, the policy model might wrap the action model
                if hasattr(policy.model, 'action_model'):
                    actor_model = policy.model.action_model
                elif not hasattr(policy.model, 'is_critic') or not policy.model.is_critic:
                    actor_model = policy.model
            
            # If the actor model has log_std parameters, clamp them
            if actor_model is not None:
                # Check for log_std as a parameter (common in SAC)
                if hasattr(actor_model, 'log_std'):
                    with torch.no_grad():
                        # Clamp to reasonable range:
                        # log_std ∈ [-5, 2] → std ∈ [0.0067, 7.39]
                        # This prevents both: 
                        # - Too deterministic (log_std < -5 causes entropy NaN)
                        # - Too random (log_std > 2 is excessive exploration)
                        actor_model.log_std.clamp_(-5.0, 2.0)
                        
                        # Log current std for monitoring (only occasionally to avoid spam)
                        if current_iter % 100 == 1:
                            current_log_std = actor_model.log_std.detach().cpu().numpy()
                            current_std = np.exp(current_log_std)
                            result["custom_metrics"]["policy_log_std_mean"] = float(np.mean(current_log_std))
                            result["custom_metrics"]["policy_std_mean"] = float(np.mean(current_std))
        except Exception as e:
            # Silently continue if clamping fails
            pass



class SACExpert:
    def __init__(self, safe_dist_m=100.0, lookahead_s=15.0):
        # We gebruiken de MVP logica die ook in Stage 1 PPO werkte
        self.mvp = MVP_2D(safe_distance=safe_dist_m, lookahead_time=lookahead_s)

    def get_action(self, env, agent_id):
        try:
            ac_idx = bs.traf.id2idx(agent_id)
            # 1. Huidige staat (meters t.o.v. center)
            pos = env.compute_relative_position(env.center, bs.traf.lat[ac_idx], bs.traf.lon[ac_idx])
            vel = np.array([
                np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * bs.traf.gs[ac_idx],
                np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * bs.traf.gs[ac_idx]
            ])

            # 2. Buren verzamelen
            neighbors = []
            for other in env.agents:
                if other == agent_id: continue
                o_idx = bs.traf.id2idx(other)
                neighbors.append({
                    'pos': env.compute_relative_position(env.center, bs.traf.lat[o_idx], bs.traf.lon[o_idx]),
                    'vel': np.array([np.cos(np.deg2rad(bs.traf.hdg[o_idx])) * bs.traf.gs[o_idx],
                                   np.sin(np.deg2rad(bs.traf.hdg[o_idx])) * bs.traf.gs[o_idx]])
                })

            # 3. MVP Berekening
            target_vel = self.mvp.calculate_avoidance_velocity(pos, vel, neighbors)
            
            # Indien geen conflict: direct naar waypoint (zoals in PPO Stage 1)
            if np.array_equal(target_vel, vel):
                wpt_lat, wpt_lon = env.agent_waypoints[agent_id]
                wpt_qdr, _ = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], wpt_lat, wpt_lon)
                target_hdg, target_spd = wpt_qdr, 35.0 
            else:
                target_hdg = np.rad2deg(np.arctan2(target_vel[1], target_vel[0]))
                target_spd = np.linalg.norm(target_vel) * 1.94384 # m/s -> kts
            
            # 4. Schalen naar de [-1, 1] actie-ruimte van SectorEnv
            dh = fn.bound_angle_positive_negative_180(target_hdg - bs.traf.hdg[ac_idx])
            dv = target_spd - (bs.traf.tas[ac_idx] * 1.94384)
            
            # Schaling: D_HEADING=45, D_VELOCITY=3.33 (10/3)
            return np.array([np.clip(dh / 45.0, -1, 1), np.clip(dv / (10/3), -1, 1)], dtype=np.float32)
        except Exception:
            return np.array([0.0, 0.0], dtype=np.float32)

def prefill_sac_buffer(algo, n_episodes=30):
    print(f"\n🚀 Start Buffer Pre-fill met {n_episodes} expert episodes...")
    expert = SACExpert()
    # Maak lokale omgeving aan om data te genereren
    env = SectorEnv(n_agents=20, run_id="prefill") 
    
    total_samples, waypoints_hit = 0, 0
    episode_id = 0  # Track episode IDs
    for ep in range(n_episodes):
        episode_id += 1
        timestep = 0  # Track timestep within episode
        obs, _ = env.reset()
        while env.agents:
            timestep += 1
            # Only get actions for currently active agents
            active_agents = set(obs.keys())
            agent_actions = {aid: expert.get_action(env, aid) for aid in active_agents}
            current_obs_snapshot = {aid: o.copy() for aid, o in obs.items()}
            
            obs, rewards, terms, truncs, infos = env.step(agent_actions)
            
            for aid, action in agent_actions.items():
                # Skip if agent observation is missing (edge case handling)
                if aid not in current_obs_snapshot:
                    print(f"[WARNING] Agent {aid} missing from obs snapshot")
                    continue
                
                # Get next observation (zeros if terminated)
                next_obs = obs.get(aid, np.zeros_like(current_obs_snapshot[aid]))
                
                # For terminated agents, ensure next_obs is zeros
                if terms.get(aid, False) and not np.allclose(next_obs, 0.0):
                    next_obs = np.zeros_like(current_obs_snapshot[aid])
                
                batch = SampleBatch({
                    SampleBatch.OBS: [current_obs_snapshot[aid]],
                    SampleBatch.ACTIONS: [action],
                    SampleBatch.REWARDS: [rewards.get(aid, 0.0)], 
                    SampleBatch.NEXT_OBS: [next_obs],
                    SampleBatch.TERMINATEDS: [terms.get(aid, False)],
                    SampleBatch.TRUNCATEDS: [truncs.get(aid, False)],
                    SampleBatch.INFOS: [infos.get(aid, {})],  # Info dict
                    "weights": [1.0],  # For prioritized experience replay (uniform weight)
                    SampleBatch.EPS_ID: [episode_id],  # Episode ID
                    SampleBatch.AGENT_INDEX: [aid],  # Agent index
                    SampleBatch.UNROLL_ID: [episode_id],  # Unroll ID (same as episode for replay buffer)
                    SampleBatch.T: [timestep],  # Timestep within episode
                })
                
                # Debug: Print batch keys on first sample to verify completeness
                if total_samples == 0:
                    print(f"[DEBUG] SampleBatch keys: {list(batch.keys())}")
                
                # Add to the algorithm's local replay buffer
                # Multi-agent replay buffer expects MultiAgentBatch with policy_id mapping
                ma_batch = MultiAgentBatch({"shared_policy": batch}, env_steps=1)
                algo.local_replay_buffer.add(ma_batch)
                total_samples += 1
        
        waypoints_hit += len(env.waypoint_reached_agents)
        if (ep + 1) % 10 == 0:
            print(f"   Episode {ep+1}/{n_episodes} | Samples: {total_samples} | Avg WP: {waypoints_hit/(ep+1):.2f}")
            
    print(f"✅ Buffer gevuld met {total_samples} samples.")
    print(f"   Expert WP Success Rate: {(waypoints_hit/(n_episodes*20))*100:.1f}%")
    try:
        buffer_size = algo.local_replay_buffer._num_added
        print(f"   Buffer size after pre-fill: {buffer_size}")
    except AttributeError:
        print(f"   Buffer content verified: {total_samples} samples added")
    print()
    env.close()

    
    
    
def burn_in_on_expert_buffer(algo, n_iterations=2000, batch_size=4096):
    print(f"\n🔥 Starting Burn-in Phase: {n_iterations} iterations on expert buffer...")
    print(f"   Goal: Calibrate Critic to recognize expert-level Q-values")
    
    policy = algo.get_policy("shared_policy")
    
    # 1. Zoek de juiste SAC Alpha parameter (NIET de model temperature)
    alpha_param = None
    if hasattr(policy, "model") and hasattr(policy.model, "log_alpha"):
        alpha_param = policy.model.log_alpha
    elif hasattr(policy, "log_alpha"):
        alpha_param = policy.log_alpha
        
    # Sla de originele log_alpha op om te kunnen herstellen
    original_log_alpha = alpha_param.detach().item() if alpha_param is not None else np.log(START_ALPHA)

    mean_q_history = []
    critic_loss_history = []
    actor_loss_history = []
    actual_alpha_history = []
    
    for i in range(1, n_iterations + 1):
        batch = algo.local_replay_buffer.sample(batch_size)
        
        if isinstance(batch, MultiAgentBatch):
            train_batch = batch.policy_batches.get("shared_policy")
        else:
            train_batch = batch

        if train_batch is None or len(train_batch) == 0:
            continue

        # --- 2. SAC ALPHA ONDERDRUKKING (Voor pure imitatie) ---
        if alpha_param is not None:
            with torch.no_grad():
                # We zetten SAC Alpha op 0.01 (log(0.01) ≈ -4.6)
                alpha_param.fill_(np.log(0.01))
        
        # 3. Voer de update uit
        train_results = policy.learn_on_batch(train_batch)
        learner_stats = train_results.get('learner_stats', {})
        
        # --- 4. HERSTEL ORIGINELE ALPHA ---
        if alpha_param is not None:
            with torch.no_grad():
                alpha_param.fill_(original_log_alpha)
        
        # Convert metrics to scalars
        def to_scalar(val):
            if isinstance(val, (list, tuple, np.ndarray)):
                if len(val) > 0:
                    clean_val = np.array(val)
                    clean_val = clean_val[~np.isnan(clean_val)]
                    if len(clean_val) > 0:
                        return float(np.mean(clean_val))
                return float('nan')
            return float(val) if not isinstance(val, str) else float('nan')
        
        mean_q = to_scalar(learner_stats.get('mean_q', float('nan')))
        critic_loss = to_scalar(learner_stats.get('critic_loss', float('nan')))
        actor_loss = to_scalar(learner_stats.get('actor_loss', float('nan')))
        alpha_value = to_scalar(learner_stats.get('alpha_value', float('nan')))
        # Record the actual used alpha for this update (scalar)
        actual_used_alpha = alpha_value
        actual_alpha_history.append(actual_used_alpha)
        
        # Track history
        mean_q_history.append(mean_q)
        critic_loss_history.append(critic_loss)
        actor_loss_history.append(actor_loss)
        
        # Print progress every 50 iterations
        if i % 50 == 0 or i == 1:
            # Calculate moving averages for stability
            recent_q = np.nanmean(mean_q_history[-10:]) if len(mean_q_history) >= 10 else mean_q
            recent_critic_loss = np.nanmean(critic_loss_history[-10:]) if len(critic_loss_history) >= 10 else critic_loss
            recent_actor_loss = np.nanmean(actor_loss_history[-10:]) if len(actor_loss_history) >= 10 else actor_loss
            
            print(
                f"   Burn-in {i}/{n_iterations} | "
                f"MeanQ: {recent_q:.4f} | "
                f"Critic Loss: {recent_critic_loss:.4f} | "
                f"Actor Loss: {recent_actor_loss:.4f} | "
                f"UsedAlpha: {actual_used_alpha:.4f} | "
                f"TargetAlpha: {alpha_value:.4f}"
            )
    
    # Final summary
    final_mean_q = np.nanmean(mean_q_history[-50:]) if len(mean_q_history) >= 50 else np.nanmean(mean_q_history)
    initial_mean_q = np.nanmean(mean_q_history[:10]) if len(mean_q_history) >= 10 else mean_q_history[0]
    
    print(f"\n✅ Burn-in Phase Complete!")
    print(f"   Initial MeanQ: {initial_mean_q:.4f}")
    print(f"   Final MeanQ: {final_mean_q:.4f}")
    print(f"   Change: {final_mean_q - initial_mean_q:+.4f}")
    print(f"   Critic is now calibrated to expert demonstrations.")
    print(f"   Ready to start main training with environment interaction.\n")
    
    return {
        'mean_q_history': mean_q_history,
        'critic_loss_history': critic_loss_history,
        'actor_loss_history': actor_loss_history,
        'actual_alpha_history': actual_alpha_history,
        'final_mean_q': final_mean_q,
        'initial_mean_q': initial_mean_q
    }
    
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

def verify_buffer_rewards(algo):
    # Haal een grote batch samples uit de buffer
    batch = algo.local_replay_buffer.sample(5000)
    
    if isinstance(batch, MultiAgentBatch):
        rewards = batch.policy_batches.get("shared_policy")["rewards"]
    else:
        rewards = batch["rewards"]
    
    avg_reward = np.mean(rewards)
    max_reward = np.max(rewards)
    
    print("\n🔍 BUFFER REWARD VERIFICATION")
    print("-" * 30)
    print(f"Gemiddelde beloning in buffer: {avg_reward:.4f}")
    print(f"Hoogste beloning in buffer:    {max_reward:.4f}")
    
    if max_reward < 1.0:
        print("⚠️ WAARSCHUWING: Je buffer bevat nog de OUDE beloningen (< 1.0).")
        print("De Critic zal nooit naar MeanQ 1.125 stijgen met deze data.")
    else:
        print("✅ SUCCESS: Buffer bevat de nieuwe, hogere beloningen.")
    print("-" * 30 + "\n")

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
            actor_lr=5e-5  , # was 1e-4 LR for actor, which decides the actions, small means slower learning but better converging
            critic_lr=1e-4, # was 5e-4          # evaluates quality of actions. hihger is better of exploration, 
            train_batch_size=4096,
            # ---- Option A: fixed alpha (stable baseline) ----
            target_entropy = -2.0,   #was -2.0. -1.0 for more exploration. larger negative value is more exploitation
            # alpha_lr = 1e-5,            # was 3e-5.   lr for updating entropy / alpha. lower means slower alpha updates
            # alpha_lr=[
            #     [0,        0],   # from step 0 to 1M: 3e-4
            #     [TOTAL_ITERS/2, 1e-5],
            #     [TOTAL_ITERS, 1e-6],  # then slowly decay to 3e-5
            # ],
            # alpha_lr=[ # DIT DOET DUS HELEMAAL NIKS DOOR DIE CALLBACK BOVENIN DE CODE
            #     [0, 0],                            # Tot 70% van de tijd: maximale exploratie
            #     [TOTAL_ITERS * 0.5, 0],         # Daarna: heel voorzichtig de chaos afbouwen
            #     [TOTAL_ITERS * 0.7, 5e-6],         # Daarna: heel voorzichtig de chaos afbouwen
            #     [TOTAL_ITERS, 1e-6],               # Eindigen met subtiele verfijning
            # ], 
            # alpha_lr=5e-5,            # was 3e-5.   lr for updating entropy / alpha. lower means slower alpha updates
            
            initial_alpha = START_ALPHA, #  DIT DOET NIKS DOOR DIE CALLBACK. was 0.5. initial alpha/entropy, higher means more exploration
            grad_clip=0.5,
            
            # Add numerical stability for entropy calculation
            # This prevents log(0) in TanhGaussian entropy computation
            clip_actions=True,  # Ensure actions stay in valid range

            # Hyperparameters
            gamma=0.99, # discount factor future rewards
            tau=0.001, # soft update parameter for target    networks, smaller makes target network update more slowly
            
            twin_q=True, # use two networks, for more stable learning
            n_step=3, #  enables multi-step q-learning, agent will use rewards over multiple timestep

            # Replay/batching - REDUCED for more episode diversity
            replay_buffer_config={
                "type": "MultiAgentReplayBuffer",
                "capacity": 1_000_000,  # Reduced from 1M to encourage fresher samples
                "prioritized_replay": False,
            },
            num_steps_sampled_before_learning_starts=0,  # kan ook 1000 zijn ofzo, of gewoon weinig Reduced from 5000
            # train_batch_size=16_384,  # Reduced from 2048 for more frequent updates
            
            # Actor (Policy) Configuration 
            policy_model_config={
                "custom_model": "attention_sac",
                "custom_model_config": {
                    "hidden_dims": [512, 512], 
                    "is_critic": False
                },
                "fcnet_hiddens": [], # Ensure default MLP is disabled
            },

            # --- CRITIC (Q-Network) Configuration ---
            q_model_config={
                "custom_model": "attention_sac",
                "custom_model_config": {
                    "hidden_dims": [512, 512],
                    "is_critic": True
                },
                "fcnet_hiddens": [], # Ensure default MLP is disabled
            },
        )

        .multi_agent(
            policies={"shared_policy": (None, None, None, {})},
            policy_mapping_fn=policy_map,
        )
        .resources(num_gpus=1)
        # .learners(num_gpus_per_learner=1)
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
            
    # NEW LOGIC: Pre-fill buffer before training starts
    if restored_from:
        algo = Algorithm.from_checkpoint(restored_from)
        target_iters = algo.iteration + max(1, int(EXTRA_ITERS))
    else:
        algo = build_trainer(N_AGENTS)
        target_iters = int(TOTAL_ITERS)
        
        # --- NIEUW: VUL DE BUFFER VOORDAT DE TRAINING START ---
        if not restored_from: # Alleen bij een verse run
            prefill_sac_buffer(algo, n_episodes=PRETRAIN_EPISODES)

            # --- Pre-Burn-in Evaluation (baseline before offline updates) ---
            try:
                print("\n🌟 Pre-Burn-in evaluation (before offline burn-in)...")
                pre_eval_metrics = run_fixed_eval(
                    algo,
                    n_episodes=30,
                    render=False,
                    n_agents=N_AGENTS,
                    silent=True,
                )
                pre_wp_pct = pre_eval_metrics.get("waypoint_rate", float("nan")) * 100.0 if pre_eval_metrics.get("waypoint_rate") is not None else float("nan")
                print(
                    f"[Pre-Eval] avg_rew={pre_eval_metrics['avg_reward']:.3f} | "
                    f"avg_len={pre_eval_metrics['avg_length']:.1f} | avg_intr={pre_eval_metrics['avg_intrusions']:.2f} | "
                    f"wp_rate={pre_wp_pct:.1f}%"
                )
            except Exception as e:
                print(f"[Pre-Eval] FAILED: {e}")

            # --- BURN-IN: Offline learning from expert buffer only ---
            if ENABLE_BURN_IN:
                burn_in_results = burn_in_on_expert_buffer(
                    algo,
                    n_iterations=BURN_IN_ITERATIONS,
                    batch_size=BURN_IN_BATCH_SIZE,
                )
                # Summarize burn-in improvements
                try:
                    im_init = burn_in_results.get('initial_mean_q', float('nan'))
                    im_final = burn_in_results.get('final_mean_q', float('nan'))
                    print(f"\n🔥 Burn-in summary: initial_mean_q={im_init:.4f} -> final_mean_q={im_final:.4f} (Δ={im_final-im_init:+.4f})")
                except Exception:
                    pass
    # print
    results_buffer_verification = verify_buffer_rewards(algo)
    print("-" * 30)
    print(f" ✅ BUFFER VERIFICATION: {results_buffer_verification}")
    

    # EVALUATE INITIAL PERFORMANCE
    print(f"\n🌟 Evaluating initial performance before training...")
    try:
        checkpoint_result = algo.save(CHECKPOINT_DIR)
        # Extract just the path from the result to avoid printing massive object
        if hasattr(checkpoint_result, 'checkpoint') and hasattr(checkpoint_result.checkpoint, 'path'):
            path = checkpoint_result.checkpoint.path
        else:
            path = str(checkpoint_result)
        print(f"✅ Checkpoint saved to: {path}")

        # --- Fixed-seed mini evaluation ---is 
        print(f"[Eval] Starting evaluation with {30} episodes...")
        try:
            eval_metrics = run_fixed_eval(
                algo, 
                n_episodes=30, 
                render=False, 
                n_agents=N_AGENTS
            )
            iter_label = f"INIT({getattr(algo, 'iteration', 0)})"
            wp_pct = eval_metrics.get("waypoint_rate", float("nan")) * 100.0 if eval_metrics.get("waypoint_rate") is not None else float("nan")
            print(
                f"[Eval] ✅ {iter_label} | avg_rew={eval_metrics['avg_reward']:.3f} | "
                f"avg_len={eval_metrics['avg_length']:.1f} | avg_intr={eval_metrics['avg_intrusions']:.2f} | "
                f"wp_rate={wp_pct:.1f}%"
            )
        except Exception as e:
            print(f"[Eval] ❌ Evaluation failed: {e}")
    except Exception as e:
        print(f"❌ Checkpoint saving failed: {e}")
                            
    # Loss history for different components
    total_loss_history = []
    policy_loss_history = []
    entropy_history = []
    alpha_history = []
    q_loss_history = []
    reward_history = []
    episode_length_history = []
    reward_entropy_ratio_history = []  # Track reward-to-entropy balance
    waypoint_rate_history = []  # Track waypoint success rate
    avg_intrusions_history = []  # Track mean intrusions per iteration
    
    # Attention metrics tracking
    attention_sharpness_history = []
    wq_weight_norm_history = []
    wq_grad_norm_history = []
    wk_weight_norm_history = []
    wk_grad_norm_history = []
    wv_weight_norm_history = []
    wv_grad_norm_history = []
    
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
        
        # Extract custom metrics for waypoint reach rate
        custom_metrics = result.get("custom_metrics", {})
        waypoint_rate = custom_metrics.get("waypoint_rate_mean", float("nan"))
        avg_intrusions = custom_metrics.get("intrusions_mean", float("nan"))
        
        # Convert waypoint_rate to scalar if needed
        if isinstance(waypoint_rate, (list, np.ndarray)):
            waypoint_rate = float(np.mean(waypoint_rate)) if len(waypoint_rate) > 0 else float("nan")
        # Convert avg_intrusions to scalar if needed
        if isinstance(avg_intrusions, (list, np.ndarray)):
            avg_intrusions = float(np.mean(avg_intrusions)) if len(avg_intrusions) > 0 else float("nan")
        
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
        
        # Check if we're in pre-learning phase
        total_timesteps = result.get('timesteps_total', 0)
        in_prelearning = total_timesteps < 10_000
        
        
        # SAC metrics from learner_stats
        policy_loss = learner_info.get("actor_loss", float("nan"))
        q_loss = learner_info.get("critic_loss", float("nan"))
        alpha_raw = learner_info.get("alpha_value", float("nan"))
        mean_q = learner_info.get("mean_q", float("nan"))
        real_entropy = learner_info.get("policy_entropy", float("nan"))
        
        # Convert arrays to scalars (SAC returns arrays sometimes)
        def to_scalar(val):
            if isinstance(val, (list, tuple, np.ndarray)):
                # Check for NaN values in arrays
                if len(val) > 0:
                    clean_val = np.array(val)
                    if np.all(np.isnan(clean_val)):
                        return float("nan")
                    # Filter out NaN values before taking mean
                    clean_val = clean_val[~np.isnan(clean_val)]
                    if len(clean_val) > 0:
                        return float(np.mean(clean_val))
                return float("nan")
            return float(val) if not isinstance(val, str) else float("nan")
        
        q_loss = to_scalar(q_loss)
        alpha = to_scalar(alpha_raw)
        policy_loss = to_scalar(policy_loss)
        current_mean_q = to_scalar(mean_q)  # Use mean_q as proxy for entropy
        entropy = to_scalar(real_entropy) # THIS IS NOW TRUE ENTROPY
        
        # If entropy is NaN, try to get log_std statistics as a diagnostic
        manual_entropy = None  # Will try to compute if RLlib's is NaN
        
        if np.isnan(entropy):
            
            # Try to compute entropy manually from the policy's distribution
            try:
                policy = algo.get_policy("shared_policy")
                
                # Deep inspection on iteration 42 to understand model structure
                if i == 42:
                    print(f"\n   [Deep Inspection] SAC Model Structure:")
                    if hasattr(policy, 'model'):
                        model = policy.model
                        print(f"     - policy.model type: {type(model).__name__}")
                        print(f"     - model attributes: {[a for a in dir(model) if not a.startswith('_')][:20]}")
                        
                        if hasattr(model, 'action_model'):
                            am = model.action_model
                            print(f"     - action_model type: {type(am).__name__}")
                            print(f"     - action_model attributes: {[a for a in dir(am) if not a.startswith('_')][:20]}")
                            
                            # Check for log_std
                            if hasattr(am, 'log_std'):
                                print(f"     - ✓ action_model.log_std exists!")
                                print(f"     - log_std type: {type(am.log_std).__name__}")
                            else:
                                print(f"     - ✗ action_model.log_std NOT found")
                                print(f"     - Checking for other std-related attributes...")
                                std_attrs = [a for a in dir(am) if 'std' in a.lower() or 'log' in a.lower()]
                                print(f"     - std-related attrs: {std_attrs}")
                    print(f"")
                
                # Try multiple methods to get entropy/log_std
                computed_entropy = False
                
                # Method 1: Direct log_std from action_model
                if hasattr(policy, 'model') and hasattr(policy.model, 'action_model'):
                    am = policy.model.action_model
                    if hasattr(am, 'log_std'):
                        log_std = am.log_std.detach().cpu().numpy()
                        action_dim = len(log_std)
                        manual_entropy = 0.5 * action_dim * (1 + np.log(2 * np.pi)) + np.sum(log_std)
                        computed_entropy = True
                        
                        if i % 50 == 0 or i == 42:
                            std = np.exp(log_std)
                            print(f"   [Manual Entropy] Method 1: Gaussian H = {manual_entropy:.4f}")
                            print(f"     log_std: {log_std}, std: {std}")
                            print(f"     Note: True SAC entropy ≈ {manual_entropy - 0.7:.4f} (after tanh correction)")
                
                # Method 2: Check policy.model directly
                if not computed_entropy and hasattr(policy, 'model'):
                    if hasattr(policy.model, 'log_std'):
                        log_std = policy.model.log_std.detach().cpu().numpy()
                        action_dim = len(log_std)
                        manual_entropy = 0.5 * action_dim * (1 + np.log(2 * np.pi)) + np.sum(log_std)
                        computed_entropy = True
                        
                        if i % 50 == 0 or i == 42:
                            std = np.exp(log_std)
                            print(f"   [Manual Entropy] Method 2: Gaussian H = {manual_entropy:.4f}")
                            print(f"     log_std: {log_std}, std: {std}")
                
                # Method 3: Sample actions and compute std empirically
                if not computed_entropy and i % 100 == 42:
                    print(f"   [Manual Entropy] Method 3: SAC uses state-dependent std")
                    print(f"     Cannot compute entropy without forward pass")
                    print(f"     Entropy will remain NaN - use Alpha as proxy")
                    
            except Exception as e:
                if i == 42 or (i % 100 == 0 and i <= 200):
                    print(f"   [Manual Entropy] Exception: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Use manual entropy if available, otherwise keep NaN
            if manual_entropy is not None:
                entropy = manual_entropy
            else:
                # Keep NaN but stop spamming messages after iteration 100
                entropy = float("nan")
        
        # Calculate total loss as sum of components
        if not np.isnan(policy_loss) and not np.isnan(q_loss):
            total_loss = abs(policy_loss) + abs(q_loss)
        else:
            total_loss = float("nan")

        # Extract attention metrics from actor model
        policy = algo.get_policy("shared_policy")
        attention_metrics = {}
        
        # Try different ways to access the actor model in SAC
        actor_model = None
        try:
            # Debug info on first iteration
            if i == 1:
                print(f"\n   [Debug] Inspecting policy structure:")
                print(f"   - hasattr(policy, 'model'): {hasattr(policy, 'model')}")
                if hasattr(policy, 'model'):
                    print(f"   - type(policy.model): {type(policy.model).__name__}")
                    print(f"   - hasattr(policy.model, 'action_model'): {hasattr(policy.model, 'action_model')}")
                    print(f"   - hasattr(policy.model, 'is_critic'): {hasattr(policy.model, 'is_critic')}")
                    if hasattr(policy.model, 'is_critic'):
                        print(f"   - policy.model.is_critic: {policy.model.is_critic}")
                print(f"   - hasattr(policy, 'pi_model'): {hasattr(policy, 'pi_model')}")
            
            # New API: Try to get the actor model from the policy
            if hasattr(policy, 'model'):
                # For SAC, check if this is the action_model (actor)
                if hasattr(policy.model, 'action_model'):
                    actor_model = policy.model.action_model
                elif not hasattr(policy.model, 'is_critic') or not policy.model.is_critic:
                    # If not marked as critic, assume it's the actor
                    actor_model = policy.model
            
            # Alternative: Try to access via pi_model or action_model directly on policy
            if actor_model is None and hasattr(policy, 'pi_model'):
                actor_model = policy.pi_model
                
            # --- SUCCESS MONITOR (Expert vs Learner) ---
            if i % 10 == 0:
                # 1. Haal de belangrijkste metrics op
                current_mean_q = result.get("info", {}).get("learner", {}).get("shared_policy", {}).get("learner_stats", {}).get("mean_q", 0.0)
                explained_var = result.get("info", {}).get("learner", {}).get("shared_policy", {}).get("learner_stats", {}).get("vf_explained_var", 0.0)
                
                # 2. Probeer de waypoint rate uit custom_metrics te halen (indien beschikbaar)
                custom_metrics = result.get("custom_metrics", {})
                wp_rate = custom_metrics.get("waypoint_rate_mean", 0.0) * 100 # Als percentage
                avg_intr = custom_metrics.get("intrusions_mean", float("nan"))
                if isinstance(avg_intr, (list, np.ndarray)):
                    avg_intr = float(np.mean(avg_intr)) if len(avg_intr) > 0 else float("nan")
                
                # 3. Print de vergelijking
                print(f"Waypoint Rate   | {wp_rate:>6.1f}%")
                print(f"Mean Q-Value    | {current_mean_q:>6.3f}")
                print(f"Avg Intrusions  | {avg_intr:>6.1f}")
                print(f"Critic accuracy | {explained_var:>6.3f}")
                
               
            
            if i == 1:
                print(f"   - Found actor_model: {actor_model is not None}")
                if actor_model is not None:
                    print(f"   - Actor model type: {type(actor_model).__name__}")
                    print(f"   - hasattr(actor_model, 'metrics'): {hasattr(actor_model, 'metrics')}")
            
            # If we found the actor model, get metrics
            if actor_model is not None and hasattr(actor_model, 'metrics'):
                attention_metrics = actor_model.metrics()
                if i == 1:
                    print(f"   - Metrics extracted: {list(attention_metrics.keys())}")
        except Exception as e:
            # Debug: print why metrics extraction failed
            if i <= 3:  # Only print for first few iterations
                print(f"   [Debug] Could not extract attention metrics: {e}")
                import traceback
                traceback.print_exc()
            pass
        
        # Append to history
        total_loss_history.append(total_loss)
        policy_loss_history.append(policy_loss)
        q_loss_history.append(q_loss)
        entropy_history.append(entropy)
        alpha_history.append(alpha)
        reward_history.append(mean_rew)
        episode_length_history.append(ep_len)
        waypoint_rate_history.append(waypoint_rate * 100.0 if not np.isnan(waypoint_rate) else 0.0)  # Convert to percentage
        avg_intrusions_history.append(avg_intrusions if not np.isnan(avg_intrusions) else 0.0)
        
        # Append attention metrics (use NaN if not available)
        attention_sharpness_history.append(attention_metrics.get('attention_sharpness', float('nan')))
        wq_weight_norm_history.append(attention_metrics.get('wq_weight_norm', float('nan')))
        wq_grad_norm_history.append(attention_metrics.get('wq_grad_norm', float('nan')))
        wk_weight_norm_history.append(attention_metrics.get('wk_weight_norm', float('nan')))
        wk_grad_norm_history.append(attention_metrics.get('wk_grad_norm', float('nan')))
        wv_weight_norm_history.append(attention_metrics.get('wv_weight_norm', float('nan')))
        wv_grad_norm_history.append(attention_metrics.get('wv_grad_norm', float('nan')))

        # Calculate moving average for reward (5 iterations)
        if len(reward_history) >= 5:
            reward_ma5 = np.mean(reward_history[-5:])
        else:
            reward_ma5 = mean_rew
        
        # Calculate Reward-to-Entropy (R/E) Ratio
        # This measures if agent prioritizes reward over exploration
        # 
        # Formula: R/E = |mean_reward| / (α × H + ε)
        # where:
        #   - α (alpha) = temperature/scaling factor for entropy
        #   - H (entropy) = actual policy entropy (randomness of actions)
        #   - ε = 1e-6 (small constant to prevent division by zero)
        #
        # Interpretation:
        #   - High R/E ratio: Agent focuses on reward (good if learning mission)
        #   - Low R/E ratio: Agent focuses on exploration (may ignore mission)
        #   - If entropy is NaN: R/E ratio is INVALID and should be ignored
        
        if not np.isnan(alpha) and not np.isnan(entropy) and not np.isnan(mean_rew):
            # Weighted Entropy = α × H (the actual exploration signal)
            weighted_entropy = alpha * entropy  # Note: entropy is typically negative
            
            # R/E Ratio = |reward| / |α × H + ε|
            # We use abs() because both reward and entropy can be negative
            reward_entropy_ratio = abs(mean_rew) / (abs(weighted_entropy) + 1e-6)
            
            # Warn if weighted entropy is suspiciously small (policy collapsed)
            if abs(weighted_entropy) < 0.001 and i % 50 == 0:
                print(f"   ⚠️  Weighted Entropy very low: {weighted_entropy:.6f} (α={alpha:.4f}, H={entropy:.4f})")
                print(f"      Policy may have collapsed - agent not exploring!")
        else:
            # Cannot compute valid R/E ratio - set to NaN for visibility
            weighted_entropy = float('nan')
            reward_entropy_ratio = float('nan')
            
            # Warn on first occurrence and periodically
            if i == 71 or (i % 100 == 0 and i <= 500):
                print(f"   ⚠️  R/E Ratio is INVALID (entropy is NaN)")
                print(f"      Cannot assess reward vs exploration balance")
                print(f"      Track Alpha (α={alpha:.4f}) as exploration proxy instead")
        
        reward_entropy_ratio_history.append(reward_entropy_ratio)
        
        # Enhanced training progress display with entropy ratio
        prelearning_note = " [PRE-LEARNING]" if in_prelearning else ""
        
        # Format R/E ratio display - show "INVALID" if NaN due to entropy
        if np.isnan(reward_entropy_ratio) and not np.isnan(mean_rew):
            re_ratio_str = "INVALID (H=nan)"
        else:
            re_ratio_str = f"{reward_entropy_ratio:.2f}"
        
        # Print only every 10th iteration (or in pre-learning phase)
        if i % 10 == 0 or in_prelearning:
            # Format waypoint rate as percentage
            wp_rate_str = f"{waypoint_rate*100:.1f}%" if not np.isnan(waypoint_rate) else "nan"
            
            print(
                f"Iter {i}/{target_iters}{prelearning_note} | "
                f"Reward: {mean_rew:.3f} (MA5: {reward_ma5:.3f}) | "
                f"EpLen: {ep_len:.1f} | WP: {wp_rate_str} | "
                f"Avg Intrusions   | {avg_intrusions:>6.1f} | "
                f"Loss: {total_loss:.3f} (Critic: {q_loss:.3f}, Actor: {policy_loss:.3f}) | "
                f"MeanQ: {current_mean_q:.4f} | Entropy: {entropy:.4f} | Alpha: {alpha:.4f} | "
                f"R/E Ratio: {re_ratio_str}"
            )

        # --- Best Checkpoint Tracking (ALWAYS ACTIVE) ---
        # Skip saving best model for first 10 iterations (warm-up period)
        if i > 50 and not np.isnan(mean_rew) and mean_rew > best_reward:
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
                wp_pct = eval_metrics.get("waypoint_rate", float("nan")) * 100.0 if eval_metrics.get("waypoint_rate") is not None else float("nan")
                print(
                    f"[Eval] ✅ iter={i} | avg_rew={eval_metrics['avg_reward']:.3f} | "
                    f"avg_len={eval_metrics['avg_length']:.1f} | avg_intr={eval_metrics['avg_intrusions']:.2f} | "
                    f"wp_rate={wp_pct:.1f}%"
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
    # Create 6 subplots (6 rows, 1 column) so the waypoint subplot (axes[5]) exists
    fig, axes = plt.subplots(6, 1, figsize=(14, 24))

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

    # Plot 5: Reward-to-Entropy Ratio
    axes[4].plot(range(1, len(reward_entropy_ratio_history) + 1), reward_entropy_ratio_history, 
                marker='o', linestyle='-', color='teal', linewidth=2)
    # Add reference lines for healthy ranges
    axes[4].axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Ratio = 1.0 (Entropy Dominant)')
    axes[4].axhline(y=10.0, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Ratio = 10.0 (Reward Dominant)')
    axes[4].set_title("Reward-to-Entropy Ratio Over Training Iterations\n(Low < 1.0: Entropy dominant | High > 10.0: Too greedy)")
    axes[4].set_xlabel("Training Iteration")
    axes[4].set_ylabel("Reward / (Alpha × Entropy)")
    axes[4].legend(loc='best')
    axes[4].grid(True)
    axes[4].set_yscale('log')  # Log scale for better visualization of wide ratio ranges
    
    # Plot 6: Waypoint Success Rate (left y-axis) and Mean Intrusions (right y-axis)
    axes[5].set_title("Waypoint Success Rate and Mean Intrusions Over Training")
    axes[5].set_xlabel("Training Iteration")
    # Plot waypoint rate as percentage on left axis
    if len(waypoint_rate_history) > 0:
        axes[5].plot(range(1, len(waypoint_rate_history) + 1), waypoint_rate_history, 
                     marker='o', linestyle='-', color='green', label='Waypoint Rate (%)')
    axes[5].set_ylabel("Waypoint Rate (%)", color='green')
    axes[5].tick_params(axis='y', labelcolor='green')
    axes[5].grid(True)

    # Plot mean intrusions on twin axis (right)
    if len(avg_intrusions_history) > 0:
        ax5_twin = axes[5].twinx()
        ax5_twin.plot(range(1, len(avg_intrusions_history) + 1), avg_intrusions_history, 
                      marker='s', linestyle='--', color='red', label='Mean Intrusions')
        ax5_twin.set_ylabel('Mean Intrusions', color='red')
        ax5_twin.tick_params(axis='y', labelcolor='red')
        # Combine legends if both series exist
        try:
            lines, labels = axes[5].get_legend_handles_labels()
            lines2, labels2 = ax5_twin.get_legend_handles_labels()
            axes[5].legend(lines + lines2, labels + labels2, loc='upper left')
        except Exception:
            pass

    # Adjust layout and show the figure
    plt.tight_layout()
    
    # Save figure to file instead of showing it (for headless servers)
    figure_path = os.path.join(script_dir, f"training_metrics_run_{RUN_ID}.png")
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Training metrics plot saved to: {figure_path}")
    plt.close()  # Close the figure to free memory
    
    # --- Save Attention Metrics to CSV ---
    import csv
    attention_csv_path = os.path.join(script_dir, f"attention_metrics_run_{RUN_ID}.csv")
    with open(attention_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'iteration', 'attention_sharpness', 
            'wq_weight_norm', 'wq_grad_norm',
            'wk_weight_norm', 'wk_grad_norm',
            'wv_weight_norm', 'wv_grad_norm'
        ])
        for idx in range(len(attention_sharpness_history)):
            writer.writerow([
                idx + 1,
                attention_sharpness_history[idx],
                wq_weight_norm_history[idx],
                wq_grad_norm_history[idx],
                wk_weight_norm_history[idx],
                wk_grad_norm_history[idx],
                wv_weight_norm_history[idx],
                wv_grad_norm_history[idx]
            ])
    print(f"📊 Attention metrics saved to: {attention_csv_path}")
    
    

    ray.shutdown()
