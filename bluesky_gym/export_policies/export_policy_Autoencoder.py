from pathlib import Path
from stable_baselines3 import PPO
import torch
import ray
import sys
import numpy as np

from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.algorithms.algorithm import Algorithm

# Import environment registration
from bluesky_gym import register_envs

# met deze code kan je stage 1 en stage 2 exporteren van sac _am
# alleen ff de file names aanpassen naar de juiste checkpoint


def export_policy_torch_old_api(checkpoint_dir, policy_id, export_file, env_creator=None, runtime_env=None):
    """Export policy weights for OLD API checkpoints (PPO with enable_rl_module_and_learner=False)"""
    
    # 1. Define the exclusions
    files_to_exclude = [
        "*.pkl",                # Exclude policy state files
        "*.csv",                # Exclude large metric files
        "*/metrics/*",          # Ignore everything inside metrics folders
        "*/models/*",           # Ignore everything inside models folders
        "*/Two_stage_AM/*"      # Ignore the specific heavy training folder
    ]
    
    # 2. Initialize Ray (CORRECTED LOGIC)
    if not ray.is_initialized():
        # If runtime_env wasn't provided, create a blank dictionary
        if runtime_env is None:
            runtime_env = {}
            
        # Ensure the 'excludes' key exists in the dictionary
        if "excludes" not in runtime_env:
            runtime_env["excludes"] = []
            
        # Add your excluded files to the configuration
        runtime_env["excludes"].extend(files_to_exclude)
        
        print(f"🚀 Starting Ray with excludes: {runtime_env['excludes']}")
        ray.init(ignore_reinit_error=True, runtime_env=runtime_env)
    
    # Register the custom environment
    if env_creator is None:
        register_envs()
        print("✅ Registered custom environments")
    else:
        # Use provided env_creator for custom environments
        from ray.tune.registry import register_env
        register_env("sector_env", env_creator)
        print("✅ Registered custom environment with provided creator")
    
    # Load the full algorithm from checkpoint
    print(f"📂 Loading checkpoint from: {checkpoint_dir}")
    algo = Algorithm.from_checkpoint(checkpoint_dir)
    
    # Get the policy
    policy = algo.get_policy(policy_id)
    
    # Extract the model's state dict
    state_dict = policy.model.state_dict()
    
    # Save to file
    torch.save(state_dict, export_file)
    print(f"✅ Saved Torch weights (OLD API): {export_file}")
    print(f"   Model has {len(state_dict)} parameter tensors")
    
    # Print model structure for verification
    print(f"\n📊 Model structure:")
    for name, tensor in state_dict.items():
        print(f"   {name:50s} {tuple(tensor.shape)}")
    
    # Cleanup
    algo.stop()
    print("\n✅ Export complete!")


# ================================ CODE FOR TWO_STAGE_AM MODELS ==================================
# Export Stage 1 (Behavior Cloning) model from Two_stage_AM training
# IMPORTANT: This requires attention_model_A.py to be importable!

# Shutdown Ray first to ensure clean start with new runtime_env
if ray.is_initialized():
    ray.shutdown()

# Point to the Two_stage_AM directory containing attention_model_A.py
two_stage_am_dir = r"C:\Users\boris\Documents\bsgym\bluesky-gym\Noise\Kalman\PPO_Symmetric_Autoencoder"
if two_stage_am_dir not in sys.path:
    sys.path.insert(0, two_stage_am_dir)

# Now import and register the attention model AND environment
from ray.rllib.models import ModelCatalog
from attention_model_A import AttentionSACModel  # 3-head additive attention
from bluesky_gym.envs.ma_env_two_stage_AM_PPO_NOISE_autoencoder import SectorEnv

ModelCatalog.register_custom_model("attention_sac", AttentionSACModel)

# Create environment creator function for Two_stage_AM
def sector_env_creator(config):
    return SectorEnv(**config)

# Create runtime environment so Ray workers can find attention_model_A
runtime_env = {
    "env_vars": {"PYTHONPATH": two_stage_am_dir},
    "py_modules": [two_stage_am_dir],  # This makes the directory available to all workers
}

# ================================ EXPORT  ppo symmetric  autoencoder==================================

#UNCOMMENT THE FOLLOWING BIT TO EXPORT STAGE 2 MODEL PPO 
print("\n" + "="*70)
print("🚀 EXPORTING STAGE 2 (RL FINE-TUNED) MODEL")
print("="*70)

# Export the best Stage 2 checkpoint from 1_9_PPO
export_policy_torch_old_api(
    r"C:\Users\boris\Documents\bsgym\bluesky-gym\Noise\Kalman\PPO_Symmetric_Autoencoder\models\sectorcr_ma_sac\best_iter_00130",
    "shared_policy",
    r"C:\Users\boris\BS_setup\bluesky-master\plugins\models_boris\PPO_Symmetric_Autoencoder.pt",
    env_creator=sector_env_creator,
    runtime_env=runtime_env
)

print("\n" + "="*70)
print("✅ STAGE 2 EXPORT COMPLETE")
print("   Model saved to: models_boris/Two_stage_AM_Stage2_iter112.pt")
print("   This is the RL-optimized model from iteration 112 (maximizes reward)")
print("="*70)


