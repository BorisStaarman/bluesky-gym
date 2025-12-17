import sys
import pickle
import os
import types
import numpy as np
import torch
import ray
from ray.rllib.policy.policy import Policy
from bluesky_gym import register_envs

# ==============================================================================
# ☢️ NUCLEAR FIX: Monkey-Patch Pickle for NumPy 2.0 -> 1.x
# ==============================================================================
# 1. Apply standard sys.modules patches
sys.modules["numpy._core"] = np.core
if hasattr(np.core, "numeric"):
    sys.modules["numpy._core.numeric"] = np.core.numeric
if hasattr(np.core, "multiarray"):
    sys.modules["numpy._core.multiarray"] = np.core.multiarray
if not hasattr(np, "_core"):
    np._core = np.core

# 2. Apply Custom Unpickler Patch
class NumpyCompatibilityUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "numpy._core.numeric":
            module = "numpy.core.numeric"
        elif module == "numpy._core.multiarray":
            module = "numpy.core.multiarray"
        elif module == "numpy._core.umath":
            module = "numpy.core.umath"
        elif module == "numpy._core":
            module = "numpy.core"
        return super().find_class(module, name)

# Overwrite pickle.load
def patched_pickle_load(file, *args, **kwargs):
    return NumpyCompatibilityUnpickler(file, *args, **kwargs).load()

pickle.load = patched_pickle_load
print("✅ Pickle.load patched for NumPy 2.0 compatibility.")
# ==============================================================================

def export_policy_only(checkpoint_dir, policy_id, export_file):
    """
    Loads ONLY the Policy (neural network) from the checkpoint.
    This avoids creating environments or workers, preventing EnvError.
    """
    
    # Initialize Ray (lightweight mode)
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, local_mode=True)

    # Construct the path to the specific policy inside the checkpoint
    # Structure is usually: checkpoint_dir / "policies" / policy_id
    policy_dir = os.path.join(checkpoint_dir, "policies", policy_id)
    
    if not os.path.exists(policy_dir):
        print(f"⚠️ Could not find policy folder at: {policy_dir}")
        print("   Attempting to load from root checkpoint dir (fallback)...")
        policy_dir = checkpoint_dir

    print(f"📂 Loading Policy from: {policy_dir}")
    
    try:
        # Load ONLY the policy, not the whole Algorithm
        policy = Policy.from_checkpoint(policy_dir)
        
        # Extract weights
        state_dict = policy.model.state_dict()
        
        # Save to file
        torch.save(state_dict, export_file)
        print(f"✅ SUCCESS: Saved Torch weights to: {export_file}")
        print(f"   Model has {len(state_dict)} tensors")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

# ==============================================================================
# EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # Settings for Model 6 (NLR Server)
    CHECKPOINT_PATH = r"C:\Users\boris\Documents\bsgym\bluesky-gym\SAC\12_3_NLR_server\models\best_iter_18640"
    OUTPUT_PATH = r"C:\Users\boris\BS_setup\bluesky-master\plugins\models_boris\SAC_6.pt"
    POLICY_ID = "shared_policy"

    export_policy_only(CHECKPOINT_PATH, POLICY_ID, OUTPUT_PATH)