from pathlib import Path

export_file = Path(r"C:\Users\boris\bluesky\models_boris\PPO_model1\shared_policy.pt")
print(f"Exists: {export_file.exists()}")
print(f"File size: {export_file.stat().st_size / 1024:.1f} KB")


import torch

# path = r"C:\Users\boris\bluesky\models_boris\PPO_model1\shared_policy.pt"
path = r"C:\Users\boris\BS_setup\bluesky-master\plugins\models_boris\SAC_4.pt"
state_dict = torch.load(path)

print(f"Number of tensors: {len(state_dict)}")
for name, tensor in state_dict.items():
    print(f"{name:50s} {tuple(tensor.shape)}")


from pathlib import Path
import torch
from ray.rllib.core.rl_module.rl_module import RLModule

# paths
ckpt = Path(r"C:\Users\boris\Documents\bsgym\bluesky-gym\boris_test_files\29_10\models\sectorcr_ma_ppo\best_iter_00098")
pt_path = r"C:\Users\boris\bluesky\models_boris\PPO_model1\shared_policy.pt"

# 1) restore architecture from checkpoint
module = RLModule.from_checkpoint(ckpt / "learner_group" / "learner" / "rl_module" / "shared_policy").eval()

# 2) load your saved weights
state = torch.load(pt_path, map_location="cpu")
module.load_state_dict(state, strict=True)  # will raise if shapes don’t match

# 3) quick forward pass
dummy_obs = torch.zeros((1, 27), dtype=torch.float32)
with torch.no_grad():
    out = module.forward_inference({"obs": dummy_obs})
    # usually: out["action_dist_inputs"] with shape (1, 4)
    k = "action_dist_inputs" if "action_dist_inputs" in out else "actions"
    print("Output key:", k, "Shape:", tuple(out[k].shape))
    # deterministisch: pak de eerste 2 (means)
    means = out[k][:, :2]
    print("Deterministic action (means) shape:", tuple(means.shape))
