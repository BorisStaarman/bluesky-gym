from pathlib import Path
import torch
import ray
import sys
import numpy as np

from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.algorithms.algorithm import Algorithm

# Import environment registration
from bluesky_gym import register_envs

def export_policy_torch_new_api(checkpoint_dir, policy_id, export_file):
    """Export policy weights for NEW API (RLModule-based) checkpoints"""
    ckpt = Path(checkpoint_dir)
    module_path = ckpt / "learner_group" / "learner" / "rl_module" / policy_id
    module = RLModule.from_checkpoint(module_path).eval()
    torch.save(module.state_dict(), export_file)
    print(f"✅ Saved Torch weights (NEW API): {export_file}")

def export_policy_torch_old_api(checkpoint_dir, policy_id, export_file, env_creator=None, runtime_env=None):
    """Export policy weights for OLD API checkpoints (like SAC with enable_rl_module_and_learner=False)"""
    
    # 1. Define the exclusions
    files_to_exclude = [
        "*.pkl",                # Exclude policy state files
        "*.csv",                # Exclude large metric files
        "*/metrics/*",          # Ignore everything inside metrics folders
        "*/models/*",           # Ignore everything inside models folders
        "*/SAC_AM/*"            # Ignore the specific heavy training folder
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
    
    # --- The rest of your function remains the same ---
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


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

# code for model 1, PPO (NEW API)
# export_policy_torch_new_api(
#     r"C:\Users\boris\Documents\bsgym\bluesky-gym\boris_test_files\29_10\models\sectorcr_ma_ppo\best_iter_00098",
#     "shared_policy",
#     r"C:\Users\boris\bluesky\models_boris\PPO_model1\ppo.pt",
# )

# code for model 2, SAC/11_13 (OLD API) - FINAL MODEL
# export_policy_torch_old_api(
#     r"C:\Users\boris\Documents\bsgym\bluesky-gym\SAC\11_13\models\sectorcr_ma_sac\final_model",
#     "shared_policy",
#     r"C:\Users\boris\AppData\Local\Programs\Python\Python312\Lib\site-packages\bluesky\plugins\models_boris\SAC_2.pt",
# )

# code for model 3, SAC/11_17 (OLD API) - BEST ITERATION
# export_policy_torch_old_api(
#     r"C:\Users\boris\Documents\bsgym\bluesky-gym\SAC\11_17\models\sectorcr_ma_sac\best_iter_05780",
#     "shared_policy",
#     r"C:\Users\boris\AppData\Local\Programs\Python\Python312\Lib\site-packages\bluesky\plugins\models_boris\SAC_3.pt",
# )


# # code for model 4, SAC/11_27_2 (OLD API) - BEST ITERATION
# export_policy_torch_old_api(
#     r"C:\Users\boris\Documents\bsgym\bluesky-gym\SAC\11_27_2\models\sectorcr_ma_sac\best_iter_09590",
#     "shared_policy",
#     r"C:\Users\boris\AppData\Local\Programs\Python\Python312\Lib\site-packages\bluesky\plugins\models_boris\SAC_4.pt",
# )


# code for model 5 SAC/12_2/
# export_policy_torch_old_api(
#     r"C:\Users\boris\Documents\bsgym\bluesky-gym\SAC\12_2\models\sectorcr_ma_sac\best_iter_13727",
#     "shared_policy",
#     r"C:\Users\boris\BS_setup\bluesky-master\plugins\models_boris\SAC_5.pt",
# )


# code for model 6 sac 12_3_nlr_server , deze model is van de nlr server getrokken
# export_policy_torch_old_api(
#     r"C:\Users\boris\Documents\bsgym\bluesky-gym\SAC\12_3_NLR_server\models\best_iter_18640",
#     "shared_policy",
#     r"C:\Users\boris\BS_setup\bluesky-master\plugins\models_boris\SAC_6.pt",
# )


# code for model 7 - Stage 1 Two-Stage (PPO OLD API - Imitation Learning)
# export_policy_torch_old_api(
#     r"C:\Users\boris\Documents\bsgym\bluesky-gym\Two_stage\12_11\models\sectorcr_ma_sac\stage1_weights",
#     "shared_policy",
#     r"C:\Users\boris\BS_setup\bluesky-master\plugins\models_boris\Stage1_Imitation.pt",
# )

# second time exporting stage 1 weights. now from 12_16_2 folder
export_policy_torch_old_api(
    r"C:\Users\boris\Documents\bsgym\bluesky-gym\Two_stage\12_16_2\models\sectorcr_ma_sac\stage1_best_weights",
    "shared_policy",
    r"C:\Users\boris\BS_setup\bluesky-master\plugins\models_boris\Stage1_Imitation_vs2.pt",
)