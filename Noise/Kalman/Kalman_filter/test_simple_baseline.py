"""
Test if a simple moving average outperforms the LSTM
======================================================
"""
import numpy as np
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from train_denoiser import (
    collect_mvp_trajectories, make_dataset_from_trajectories,
    X_NORM, Y_NORM, V_NORM, POS_NOISE_STD_M, VEL_NOISE_STD_MS
)

print("Collecting test trajectories...")
trajectories = collect_mvp_trajectories(n_episodes=50, n_agents=20, verbose=True)
X_test, Y_test = make_dataset_from_trajectories(trajectories, seq_len=10, seed=99999)

print(f"\nTest set size: {X_test.shape}")

# Baseline 1: Use last noisy observation (no denoising)
noisy_last = X_test[:, -1, :]

# Baseline 2: Moving average over window
moving_avg = X_test.mean(axis=1)  # Average over seq_len dimension

# Baseline 3: Weighted average (more weight on recent)
weights = np.exp(np.linspace(-2, 0, 10))  # Exponential weights
weights /= weights.sum()
weighted_avg = np.sum(X_test * weights[np.newaxis, :, np.newaxis], axis=1)

# Ground truth
clean = Y_test

# Compute RMSEs
unnorm = np.array([X_NORM, Y_NORM, V_NORM, V_NORM])
feature_names = ["x", "y", "vx", "vy"]

print(f"\n{'='*100}")
print(f"{'Feature':<8} | {'Noisy RMSE':>15} | {'MovAvg RMSE':>15} | {'WeightedAvg':>15} | {'Best Method':<20}")
print(f"{'-'*100}")

for i, name in enumerate(feature_names):
    noisy_rmse = np.sqrt(np.mean((noisy_last[:, i] - clean[:, i]) ** 2)) * unnorm[i]
    mavg_rmse = np.sqrt(np.mean((moving_avg[:, i] - clean[:, i]) ** 2)) * unnorm[i]
    wavg_rmse = np.sqrt(np.mean((weighted_avg[:, i] - clean[:, i]) ** 2)) * unnorm[i]
    
    best_val = min(noisy_rmse, mavg_rmse, wavg_rmse)
    if best_val == noisy_rmse:
        best = "Noisy (no denoise)"
    elif best_val == mavg_rmse:
        best = f"MovAvg ({(1 - mavg_rmse/noisy_rmse)*100:.1f}% better)"
    else:
        best = f"WeightedAvg ({(1 - wavg_rmse/noisy_rmse)*100:.1f}% better)"
    
    unit = "m" if i < 2 else "m/s"
    print(f"{name:<8} | {noisy_rmse:>12.4f} {unit:<2} | {mavg_rmse:>12.4f} {unit:<2} | "
          f"{wavg_rmse:>12.4f} {unit:<2} | {best:<20}")

print(f"{'='*100}")

# Overall
overall_noisy = np.sqrt(np.mean((noisy_last - clean) ** 2))
overall_mavg = np.sqrt(np.mean((moving_avg - clean) ** 2))
overall_wavg = np.sqrt(np.mean((weighted_avg - clean) ** 2))

print(f"\nOVERALL (normalized space):")
print(f"  Noisy:        {overall_noisy:.8f}")
print(f"  MovAvg:       {overall_mavg:.8f}  ({(1 - overall_mavg/overall_noisy)*100:+.1f}%)")
print(f"  WeightedAvg:  {overall_wavg:.8f}  ({(1 - overall_wavg/overall_noisy)*100:+.1f}%)")

print(f"\n💡 If simple averaging beats LSTM, the LSTM architecture needs rethinking!")
