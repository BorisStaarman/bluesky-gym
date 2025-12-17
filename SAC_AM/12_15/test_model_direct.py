"""
Direct model testing script to check if the attention model produces sensible actions
"""
import os
import sys
import numpy as np
import torch
from ray.rllib.algorithms.algorithm import Algorithm

# Set working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Import and register environment BEFORE loading checkpoint
sys.path.insert(0, script_dir)
from bluesky_gym import register_envs
from ray.rllib.models import ModelCatalog
from attention_model import AttentionSACModel

register_envs()
ModelCatalog.register_custom_model("attention_sac", AttentionSACModel)

# Find best checkpoint
BASE_CHECKPOINT_DIR = os.path.join(script_dir, "models/sectorcr_ma_sac")
best_checkpoints = [
    d for d in os.listdir(BASE_CHECKPOINT_DIR) 
    if os.path.isdir(os.path.join(BASE_CHECKPOINT_DIR, d)) and d.startswith("best_iter_")
]
best_checkpoints.sort(key=lambda x: int(x.split("_")[-1]), reverse=True)
CHECKPOINT_DIR = os.path.join(BASE_CHECKPOINT_DIR, best_checkpoints[0])

print(f"Loading checkpoint: {CHECKPOINT_DIR}")

# Load algorithm
import ray
ray.init(local_mode=True, ignore_reinit_error=True)
algo = Algorithm.from_checkpoint(CHECKPOINT_DIR)
policy = algo.get_policy("shared_policy")

print(f"\nPolicy model: {policy.model.__class__.__name__}")
print(f"Model in training mode: {policy.model.training}")

# Set model to eval mode
policy.model.eval()
print(f"Model in training mode after .eval(): {policy.model.training}")

# Create dummy observations (25 agents, 171-dim each)
n_agents = 20
obs_dim = 3 + 7 * 24  # 171
dummy_obs = np.random.randn(n_agents, obs_dim).astype(np.float32)

print(f"\nTest with dummy observations:")
print(f"  Observation shape: {dummy_obs.shape}")

# Test action computation
actions = policy.compute_actions(dummy_obs, explore=False)[0]

print(f"\nActions produced:")
print(f"  Shape: {actions.shape}")
print(f"  Range: [{actions.min():.4f}, {actions.max():.4f}]")
print(f"  Mean: {actions.mean():.4f}")
print(f"  Std: {actions.std():.4f}")
print(f"\nFirst 5 actions:")
for i in range(min(5, len(actions))):
    print(f"  Agent {i}: heading={actions[i,0]:+.4f}, speed={actions[i,1]:+.4f}")

# Test with different observations
print(f"\n--- Testing with varied observations ---")
for test_name, obs_modifier in [
    ("All zeros", lambda x: np.zeros_like(x)),
    ("All ones", lambda x: np.ones_like(x)),
    ("Large values", lambda x: np.full_like(x, 10.0)),
]:
    test_obs = obs_modifier(dummy_obs)
    test_actions = policy.compute_actions(test_obs, explore=False)[0]
    print(f"\n{test_name}:")
    print(f"  Action range: [{test_actions.min():.4f}, {test_actions.max():.4f}]")
    print(f"  Action mean: {test_actions.mean():.4f}")
    print(f"  Sample: heading={test_actions[0,0]:+.4f}, speed={test_actions[0,1]:+.4f}")

ray.shutdown()
