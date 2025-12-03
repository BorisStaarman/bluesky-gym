from pathlib import Path
import torch
import ray
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

def export_policy_torch_old_api(checkpoint_dir, policy_id, export_file):
    """Export policy weights for OLD API checkpoints (like SAC with enable_rl_module_and_learner=False)"""
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # Register the custom environment
    register_envs()
    print("✅ Registered custom environments")
    
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


# code for model 4, SAC/11_27_2 (OLD API) - BEST ITERATION
export_policy_torch_old_api(
    r"C:\Users\boris\Documents\bsgym\bluesky-gym\SAC\11_27_2\models\sectorcr_ma_sac\best_iter_09590",
    "shared_policy",
    r"C:\Users\boris\AppData\Local\Programs\Python\Python312\Lib\site-packages\bluesky\plugins\models_boris\SAC_4.pt",
)