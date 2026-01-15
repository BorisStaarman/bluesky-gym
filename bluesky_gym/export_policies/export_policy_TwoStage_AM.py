from pathlib import Path
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

"""
================================================================================
EXPORTED MODEL ARCHITECTURE EXPLANATION
================================================================================

The exported .pt file contains a PyTorch state_dict with the complete trained
attention-based policy network. This model uses a 3-head additive attention
mechanism to process multi-agent observations and output actions.

ARCHITECTURE OVERVIEW:
----------------------

1. **3-HEAD ADDITIVE ATTENTION MECHANISM** (Groot et al. 2025)
   - Processes ownship state (7 features) and intruder states (N x 5 features)
   - Each head independently learns to focus on different aspects of the scene
   - Outputs a 15-dimensional context vector (3 heads × 5 features)

2. **POLICY NETWORK (ACTOR)**
   - Input: Concatenated [ownship (7) + attention_context (15)] = 22 features
   - Hidden layers: 512 → 512 (LeakyReLU activation)
   - Output: Action means (2D: heading_change, speed_change)
   - Log_std: Separate parameter for action exploration (2D)

3. **VALUE NETWORK (CRITIC)** [Optional, for PPO training only]
   - Same architecture as policy (22 → 512 → 512)
   - Output: Single scalar value estimate
   - Not needed for inference in simulation

STATE_DICT PARAMETERS:
----------------------
The .pt file contains these parameter groups:

**Attention Mechanism (3 heads):**
- W_q_heads.{0,1,2}.weight: (5, 7)   - Query projection from ownship
- W_q_heads.{0,1,2}.bias: (5,)
- W_k_heads.{0,1,2}.weight: (5, 5)   - Key projection from intruders  
- W_k_heads.{0,1,2}.bias: (5,)
- W_v_heads.{0,1,2}.weight: (5, 5)   - Value projection from intruders
- W_v_heads.{0,1,2}.bias: (5,)
- v_att_heads.{0,1,2}: (5, 1)        - Scoring vectors for attention weights

**Policy Network:**
- hidden_layers.0.weight: (512, 22)  - First hidden layer
- hidden_layers.0.bias: (512,)
- hidden_layers.1.weight: (512, 512) - Second hidden layer
- hidden_layers.1.bias: (512,)
- final_layer.weight: (2, 512)       - Output layer (action means)
- final_layer.bias: (2,)
- log_std: (2,)                      - Action standard deviations (log scale)

**Value Network (optional):**
- value_branch.0.weight: (512, 22)
- value_branch.0.bias: (512,)
- value_branch.2.weight: (512, 512)
- value_branch.2.bias: (512,)
- value_branch.4.weight: (1, 512)
- value_branch.4.bias: (1,)

INPUT FORMAT:
-------------
The model expects a flat observation vector with this structure:

[ownship_state (7 features) | intruder_states (N × 5 features)]

**Ownship state (7 features):**
- cos_drift: cos(heading_error)
- sin_drift: sin(heading_error)
- speed: normalized speed
- x: x position
- y: y position  
- vx: x velocity
- vy: y velocity

**Intruder states (5 features per intruder, relative to ownship):**
- rel_x: relative x position
- rel_y: relative y position
- rel_vx: relative x velocity
- rel_vy: relative y velocity
- dist: distance to intruder

Note: If fewer than N intruders exist, pad with zeros.

OUTPUT FORMAT:
--------------
The model outputs:

**For deterministic actions (inference mode):**
- Action means only: [heading_change, speed_change] (2D vector)

**For stochastic actions (exploration mode):**
- Full output: [mean_0, mean_1, log_std_0, log_std_1] (4D vector)
- Sample action: action = mean + std * noise, where std = exp(log_std)

INFERENCE PROCEDURE:
--------------------
To use this model in your simulation software:

1. Load the state_dict:
   ```python
   state_dict = torch.load("Two_stage_AM_Stage1.pt")
   ```

2. Create the AttentionSACModel instance (you need attention_model_A.py):
   ```python
   from attention_model_A import AttentionSACModel
   model = AttentionSACModel(obs_space, action_space, num_outputs=2, 
                             model_config={'custom_model_config': {
                                 'hidden_dims': [512, 512],
                                 'is_critic': False,
                                 'n_agents': 20
                             }}, name="policy")
   model.load_state_dict(state_dict)
   model.eval()  # Set to evaluation mode
   ```

3. Prepare observation (batch format):
   ```python
   obs = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)  # (1, obs_dim)
   ```

4. Get action (deterministic):
   ```python
   with torch.no_grad():
       logits, _ = model.forward({'obs': obs}, state=[], seq_lens=None)
       action_means = logits[0, :2]  # First 2 outputs are action means
       action = action_means.numpy()
   ```

5. Apply action to agent in simulation:
   ```python
   heading_change = action[0]
   speed_change = action[1]
   ```

IMPORTANT NOTES FOR SIMULATION INTEGRATION:
-------------------------------------------
- The model expects observations in the EXACT format used during training
- Attention mechanism automatically handles variable numbers of intruders (via masking)
- For deterministic behavior (no exploration), only use the first 2 outputs (action means)
- The model was trained with N=19 other agents (20 total), but can handle different numbers
- All inputs should be normalized the same way as during training
- The model outputs continuous actions that need to be scaled/clipped as appropriate

DEPENDENCIES:
-------------
- PyTorch (torch)
- NumPy
- attention_model_A.py (contains the AttentionSACModel class definition)

================================================================================
"""

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
two_stage_am_dir = r"C:\Users\boris\Documents\bsgym\bluesky-gym\Two_stage_AM\1_13_PPO"
if two_stage_am_dir not in sys.path:
    sys.path.insert(0, two_stage_am_dir)

# Now import and register the attention model AND environment
from ray.rllib.models import ModelCatalog
from attention_model_A import AttentionSACModel  # 3-head additive attention
from bluesky_gym.envs.ma_env_two_stage_AM import SectorEnv

ModelCatalog.register_custom_model("attention_sac", AttentionSACModel)

# Create environment creator function for Two_stage_AM
def sector_env_creator(config):
    return SectorEnv(**config)

# Create runtime environment so Ray workers can find attention_model_A
runtime_env = {
    "env_vars": {"PYTHONPATH": two_stage_am_dir},
    "py_modules": [two_stage_am_dir],  # This makes the directory available to all workers
}

# ================================ EXPORT STAGE 1 MODEL ==================================
# Stage 1 is the behavior cloning (imitation learning) phase
# This model has learned to mimic the MVP teacher actions
print("\n" + "="*70)
print("🎯 EXPORTING STAGE 1 (BEHAVIOR CLONING) MODEL")
print("="*70)

export_policy_torch_old_api(
    r"C:\Users\boris\Documents\bsgym\bluesky-gym\Two_stage_AM\1_13_PPO\models\sectorcr_ma_sac\stage1_best_weights",
    "shared_policy",
    r"C:\Users\boris\BS_setup\bluesky-master\plugins\models_boris\Two_stage_AM_Stage1_vs2.pt",
    env_creator=sector_env_creator,
    runtime_env=runtime_env
)

print("\n" + "="*70)
print("✅ STAGE 1 EXPORT COMPLETE")
print("   Model saved to: models_boris/Two_stage_AM_Stage1.pt")
print("   This is the behavior cloning model (teacher imitation)")
# print("="*70)

# ================================ EXPORT STAGE 2 MODEL ==================================
# Stage 2 is the RL fine-tuning phase
# This exports the best checkpoint from 1_9_PPO training

# print("\n" + "="*70)
# print("🚀 EXPORTING STAGE 2 (RL FINE-TUNED) MODEL")
# print("="*70)

# # Export the best Stage 2 checkpoint from 1_9_PPO
# export_policy_torch_old_api(
#     r"C:\Users\boris\Documents\bsgym\bluesky-gym\Two_stage_AM\1_9_PPO\models\sectorcr_ma_sac\best_iter_00112",
#     "shared_policy",
#     r"C:\Users\boris\BS_setup\bluesky-master\plugins\models_boris\Two_stage_AM_PPO_Stage2.pt",
#     env_creator=sector_env_creator,
#     runtime_env=runtime_env
# )

# print("\n" + "="*70)
# print("✅ STAGE 2 EXPORT COMPLETE")
# print("   Model saved to: models_boris/Two_stage_AM_Stage2_iter112.pt")
# print("   This is the RL-optimized model from iteration 112 (maximizes reward)")
# print("="*70)
