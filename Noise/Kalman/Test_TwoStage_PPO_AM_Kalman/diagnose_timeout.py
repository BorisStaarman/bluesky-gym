"""
Quick diagnostic: measures how long one episode takes and estimates whether
workers would hit the sample_timeout_s before finishing their rollout fragment.

Run from the repo root with your normal python environment:
    python Noise/Kalman/Test_TwoStage_PPO_AM_Kalman/diagnose_timeout.py
"""

import os, sys, time
import numpy as np

# ---- path setup so bluesky_gym is importable ----
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from bluesky_gym.envs.ma_env_two_stage_AM_PPO_NOISE_kalman import SectorEnv

# ---- config (must match main.py Stage 2) ----
N_AGENTS             = 20
TRAIN_BATCH_SIZE     = 64_000
NUM_ENV_RUNNERS      = max(1, os.cpu_count() - 1)
# RLlib default: rollout_fragment_length = ceil(train_batch_size / num_env_runners)
rollout_fragment_len = int(np.ceil(TRAIN_BATCH_SIZE / NUM_ENV_RUNNERS))

OLD_TIMEOUT_S        = 120.0   # what you had before
NEW_TIMEOUT_S        = 600.0   # what you set now

N_TIMING_EPISODES    = 3       # how many episodes to time (keeps test fast)
USE_KALMAN           = True    # must match your training config

# ----------------------------------------------------------------
print(f"CPU count: {os.cpu_count()}, num_env_runners: {NUM_ENV_RUNNERS}")
print(f"train_batch_size: {TRAIN_BATCH_SIZE}, rollout_fragment per worker: {rollout_fragment_len} steps")
print(f"\nCreating SectorEnv (use_kalman_filter={USE_KALMAN}) ...")

env = SectorEnv(n_agents=N_AGENTS, use_kalman_filter=USE_KALMAN)

step_times = []
ep_step_counts = []

for ep in range(N_TIMING_EPISODES):
    obs, _ = env.reset()
    ep_steps = 0
    ep_start = time.perf_counter()

    while env.agents:
        agent_ids = list(obs.keys())
        # random actions — we only care about speed, not quality
        actions = {aid: env.action_space[aid].sample() for aid in agent_ids}

        t0 = time.perf_counter()
        obs, rew, term, trunc, infos = env.step(actions)
        step_times.append(time.perf_counter() - t0)
        ep_steps += 1

    ep_elapsed = time.perf_counter() - ep_start
    ep_step_counts.append(ep_steps)
    print(f"  Episode {ep+1}: {ep_steps} steps in {ep_elapsed:.2f}s  ({ep_elapsed/ep_steps*1000:.1f} ms/step)")

env.close()

# ---- summary ----
avg_step_s  = np.mean(step_times)
p95_step_s  = np.percentile(step_times, 95)
avg_ep_steps = np.mean(ep_step_counts)

est_fragment_time_avg = avg_step_s  * rollout_fragment_len
est_fragment_time_p95 = p95_step_s  * rollout_fragment_len
est_episode_time_avg  = avg_step_s  * avg_ep_steps

print(f"\n--- Timing summary ---")
print(f"  Avg step time:          {avg_step_s*1000:.2f} ms")
print(f"  P95 step time:          {p95_step_s*1000:.2f} ms")
print(f"  Avg episode length:     {avg_ep_steps:.0f} steps  ({est_episode_time_avg:.1f}s per episode)")
print(f"  Rollout fragment:       {rollout_fragment_len} steps")
print(f"  Est. fragment time avg: {est_fragment_time_avg:.1f}s")
print(f"  Est. fragment time p95: {est_fragment_time_p95:.1f}s")

print(f"\n--- Timeout verdict ---")
for label, est in [("avg", est_fragment_time_avg), ("p95", est_fragment_time_p95)]:
    old_ok = "✅ OK" if est < OLD_TIMEOUT_S else f"❌ TIMEOUT (exceeds {OLD_TIMEOUT_S:.0f}s)"
    new_ok = "✅ OK" if est < NEW_TIMEOUT_S else f"❌ TIMEOUT (exceeds {NEW_TIMEOUT_S:.0f}s)"
    print(f"  [{label}]  old timeout ({OLD_TIMEOUT_S:.0f}s): {old_ok}")
    print(f"  [{label}]  new timeout ({NEW_TIMEOUT_S:.0f}s): {new_ok}")

print(f"\nConclusion:")
if est_fragment_time_avg > OLD_TIMEOUT_S:
    print(f"  Workers would CONSISTENTLY timeout with sample_timeout_s={OLD_TIMEOUT_S:.0f}s.")
    print(f"  This explains Loss=0 / Entropy=0 / VF_Var=0 — no updates were running.")
elif est_fragment_time_p95 > OLD_TIMEOUT_S:
    print(f"  Workers would OCCASIONALLY timeout with sample_timeout_s={OLD_TIMEOUT_S:.0f}s (slow steps).")
else:
    print(f"  Timeout is NOT the problem — something else is causing bad rewards.")
    print(f"  Consider checking: observation scaling, reward function, or checkpoint loading.")
