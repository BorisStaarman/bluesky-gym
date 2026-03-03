"""
Runs exactly 2 training iterations and prints the full nested key structure
of the result dict so you can fix the learner_stats extraction path.

Run from repo root:
    python Noise/Kalman/Test_TwoStage_PPO_AM_Kalman/diagnose_result_keys.py
"""
import os, sys, json
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from attention_model_A import AttentionSACModel
from bluesky_gym.envs.ma_env_two_stage_AM_PPO_NOISE_kalman import SectorEnv
from run_config import RUN_ID

register_env("sector_env", lambda config: SectorEnv(**config))
ModelCatalog.register_custom_model("attention_sac", AttentionSACModel)

ray.shutdown()  # Clean up any leftover Ray instance
ray.init(ignore_reinit_error=True, runtime_env={
    "working_dir": script_dir,
    "excludes": ["models/", "metrics/", "__pycache__/", "*.zip", "*.pt", "*.pth", "*.ckpt"],
})

cfg = (
    PPOConfig()
    .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    .environment("sector_env", env_config={"n_agents": 4, "run_id": "diag", "use_kalman_filter": True}, disable_env_checking=True)
    .framework("torch")
    .env_runners(num_env_runners=1, num_envs_per_env_runner=1, sample_timeout_s=120.0)
    .training(
        lr=1e-4, train_batch_size=400, minibatch_size=100,
        model={"custom_model": "attention_sac", "custom_model_config": {"hidden_dims": [64], "is_critic": False, "n_agents": 4}, "free_log_std": True, "vf_share_layers": False},
    )
    .multi_agent(policies={"shared_policy": (None, None, None, {})}, policy_mapping_fn=lambda agent_id, *_, **__: "shared_policy")
    .resources(num_gpus=0)
)

algo = cfg.build()
result = algo.train()

def _print_keys(d, prefix="", max_depth=4, depth=0):
    if depth > max_depth or not isinstance(d, dict):
        return
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{prefix}{k}/")
            _print_keys(v, prefix + "  ", max_depth, depth + 1)
        elif isinstance(v, (int, float)):
            print(f"{prefix}{k} = {v:.4f}")
        else:
            print(f"{prefix}{k} = {type(v).__name__}")

print("\n" + "="*60)
print("RESULT DICT KEY STRUCTURE")
print("="*60)
_print_keys(result)

# Print specific paths that main.py is trying to use
print("\n" + "="*60)
print("CHECKING MAIN.PY EXTRACTION PATHS")
print("="*60)

# Path 1: env_runners
print(f"\nresult['env_runners']['episode_return_mean'] = "
      f"{result.get('env_runners', {}).get('episode_return_mean', 'NOT FOUND')}")

# Path 2: learner_stats (what main.py uses)
learner = result.get('info', {}).get('learner', {})
sp = learner.get('shared_policy', {})
ls = sp.get('learner_stats', {})
print(f"\nresult['info']['learner']['shared_policy']['learner_stats'] keys: {list(ls.keys()) if ls else 'PATH NOT FOUND'}")
print(f"  entropy       = {ls.get('entropy', 'NOT FOUND')}")
print(f"  policy_loss   = {ls.get('policy_loss', 'NOT FOUND')}")
print(f"  vf_loss       = {ls.get('vf_loss', 'NOT FOUND')}")
print(f"  vf_explained_var = {ls.get('vf_explained_var', 'NOT FOUND')}")
print(f"  total_loss    = {ls.get('total_loss', 'NOT FOUND')}")

# Print the full 'info' -> 'learner' subtree if it exists
info_learner = result.get('info', {}).get('learner', {})
if info_learner:
    print(f"\nFull info.learner structure:")
    _print_keys(info_learner, prefix="  ")
else:
    print(f"\n'info'.'learner' key NOT FOUND in result")
    print(f"Available keys in result['info']: {list(result.get('info', {}).keys())}")

algo.stop()
ray.shutdown()
print("\nDone.")
