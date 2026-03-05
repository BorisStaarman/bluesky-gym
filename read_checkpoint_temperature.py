"""
Quick utility: read the attention temperature (and other scalars) from any
Stage 1 or Stage 2 RLlib checkpoint without starting a simulation.

Usage:
    python read_checkpoint_temperature.py <path_to_checkpoint_dir>

Example:
    python read_checkpoint_temperature.py C:\...\stage1_best_weights
    python read_checkpoint_temperature.py C:\...\best_iter_00042
"""

# do in terminal: python read_checkpoint_temperature.py "C:\Users\boris\Documents\bsgym\bluesky-gym\Noise\Kalman\Test_TwoStage_PPO_AM\models\sectorcr_ma_sac\stage1_best_weights"

import sys
import os
import pickle
import numpy as np

SCALAR_KEYS = ["temperature", "log_std"]   # extend if you want more


def read_weights(checkpoint_dir: str):
    pkl = os.path.join(checkpoint_dir, "policies", "shared_policy", "policy_state.pkl")
    if not os.path.isfile(pkl):
        # Try the checkpoint dir itself (some versions save it there)
        pkl = os.path.join(checkpoint_dir, "policy_state.pkl")
    if not os.path.isfile(pkl):
        print(f"[ERROR] policy_state.pkl not found under:\n  {checkpoint_dir}")
        sys.exit(1)

    with open(pkl, "rb") as f:
        state = pickle.load(f)

    weights = state.get("weights", {})

    print(f"\nCheckpoint: {checkpoint_dir}")
    print("-" * 60)

    found_any = False
    for key, val in weights.items():
        arr = np.array(val).ravel()
        if arr.size <= 8:          # print any small tensors (scalars / log_std)
            found_any = True
            if arr.size == 1:
                print(f"  {key:30s}  =  {arr[0]:.6f}")
            else:
                vals_str = ", ".join(f"{v:.4f}" for v in arr)
                print(f"  {key:30s}  =  [{vals_str}]")

    if not found_any:
        print("  (no scalar/small parameters found)")

    # Always print temperature explicitly
    temp = weights.get("temperature")
    if temp is not None:
        t = float(np.array(temp).ravel()[0])
        print(f"\n>>> Attention temperature: {t:.4f}")
    else:
        print("\n>>> 'temperature' key not found in weights")

    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    read_weights(sys.argv[1])
