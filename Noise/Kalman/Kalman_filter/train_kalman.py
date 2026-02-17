"""
Kalman Filter Evaluation
=========================
The Kalman filter doesn't require training (it's mathematically optimal
for linear Gaussian systems). This script:
1. Collects MVP trajectories
2. Evaluates Kalman filter performance
3. Saves the filter configuration

Usage:
    python train_kalman.py --n_episodes 100
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import io

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from kalman_denoiser import KalmanDenoiser, KalmanDenoiserBatch
from train_denoiser import (
    collect_mvp_trajectories, make_dataset_from_trajectories,
    X_NORM, Y_NORM, V_NORM, POS_NOISE_STD_M, VEL_NOISE_STD_MS
)


def evaluate_kalman(args):
    """Evaluate Kalman filter on MVP trajectories."""
    
    print(f"\n{'='*70}")
    print(f"  KALMAN FILTER DENOISER EVALUATION")
    print(f"{'='*70}")
    print(f"  No training required - mathematically optimal solution!")
    print(f"  Position noise: {POS_NOISE_STD_M} m")
    print(f"  Velocity noise: {VEL_NOISE_STD_MS} m/s")
    
    # Collect test trajectories
    print(f"\n[kalman] Collecting test trajectories...")
    trajectories = collect_mvp_trajectories(
        n_episodes=args.n_episodes,
        n_agents=args.n_agents,
        verbose=True
    )
    
    # Build dataset with seq_len window
    print(f"\n[kalman] Building dataset (seq_len={args.seq_len})...")
    X, Y = make_dataset_from_trajectories(trajectories, args.seq_len, seed=42)
    print(f"[kalman] Dataset: X={X.shape}, Y={Y.shape}")
    
    # Create Kalman filter with tunable process noise
    print(f"\n[kalman] Creating Kalman filter...")
    kalman = KalmanDenoiser(
        dt=1.0,
        pos_noise_std=POS_NOISE_STD_M,
        vel_noise_std=VEL_NOISE_STD_MS,
        process_noise_std=args.process_noise,  # Tunable parameter
        x_norm=X_NORM,
        y_norm=Y_NORM,
        v_norm=V_NORM,
    )
    
    # Run Kalman filter on all windows
    print(f"\n[kalman] Running Kalman filter on {len(X)} windows...")
    kalman_batch = KalmanDenoiserBatch(
        dt=1.0,
        pos_noise_std=POS_NOISE_STD_M,
        vel_noise_std=VEL_NOISE_STD_MS,
        process_noise_std=args.process_noise,
        x_norm=X_NORM,
        y_norm=Y_NORM,
        v_norm=V_NORM,
    )
    
    predictions = kalman_batch.denoise_batch(X)
    
    # Evaluate
    print(f"\n[kalman] Evaluating performance...")
    evaluate_performance(X, Y, predictions)
    
    # Save configuration
    save_dir = os.path.join(script_dir, "denoiser_models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "kalman_denoiser.npz")
    kalman.save(save_path)
    
    # Visualize some examples
    visualize_examples(X, Y, predictions, save_dir)
    
    print(f"\n✅ Kalman filter evaluation complete!")


def evaluate_performance(X, Y, predictions):
    """Evaluate and print performance metrics."""
    
    # Baseline: noisy last observation
    noisy_last = X[:, -1, :]
    clean = Y
    
    feature_names = ["x", "y", "vx", "vy"]
    unnorm = np.array([X_NORM, Y_NORM, V_NORM, V_NORM])
    
    print(f"\n{'='*65}")
    print(f"{'Feature':<8} | {'Noisy RMSE (phys)':<20} | {'Kalman RMSE (phys)':<20} | {'Improvement':<12}")
    print(f"{'-'*65}")
    
    improvements = []
    for i, name in enumerate(feature_names):
        noisy_rmse_norm = np.sqrt(np.mean((noisy_last[:, i] - clean[:, i]) ** 2))
        kalman_rmse_norm = np.sqrt(np.mean((predictions[:, i] - clean[:, i]) ** 2))
        
        noisy_rmse_phys = noisy_rmse_norm * unnorm[i]
        kalman_rmse_phys = kalman_rmse_norm * unnorm[i]
        
        improvement = (1 - kalman_rmse_phys / noisy_rmse_phys) * 100 if noisy_rmse_phys > 0 else 0
        improvements.append(improvement)
        
        unit = "m" if i < 2 else "m/s"
        print(f"  {name:<6} | {noisy_rmse_phys:>12.4f} {unit:<6} | "
              f"{kalman_rmse_phys:>12.4f} {unit:<6} | {improvement:>8.1f}%")
    
    # Overall
    noisy_overall = np.sqrt(np.mean((noisy_last - clean) ** 2))
    kalman_overall = np.sqrt(np.mean((predictions - clean) ** 2))
    overall_improvement = (1 - kalman_overall / noisy_overall) * 100
    
    print(f"{'-'*65}")
    print(f"  {'Total':<6} | {noisy_overall:>12.8f} (norm)  | "
          f"{kalman_overall:>12.8f} (norm)  | {overall_improvement:>8.1f}%")
    print(f"{'='*65}")
    
    avg_improvement = np.mean(improvements)
    print(f"\n📈 Average improvement: {avg_improvement:.1f}%")
    
    # Theoretical best (if noise was perfectly removed)
    theoretical_rmse = 0.0
    print(f"💡 Theoretical best RMSE: {theoretical_rmse:.8f} (perfect denoising)")
    print(f"   Kalman achieves: {(1 - kalman_overall / noisy_overall) * 100:.1f}% of optimal")


def visualize_examples(X, Y, predictions, save_dir, n_examples=5):
    """Visualize some denoising examples."""
    
    indices = np.random.choice(len(X), size=min(n_examples, len(X)), replace=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Kalman Filter Denoising Examples", fontsize=14, fontweight='bold')
    
    feature_names = ["X Position", "Y Position", "X Velocity", "Y Velocity"]
    unnorm = [X_NORM, Y_NORM, V_NORM, V_NORM]
    units = ["m", "m", "m/s", "m/s"]
    
    for feat_idx, (ax, name, scale, unit) in enumerate(zip(axes.flat, feature_names, unnorm, units)):
        for i in indices:
            window = X[i]
            clean_val = Y[i, feat_idx] * scale
            kalman_val = predictions[i, feat_idx] * scale
            noisy_vals = window[:, feat_idx] * scale
            
            timesteps = np.arange(len(window))
            ax.plot(timesteps, noisy_vals, 'o-', alpha=0.3, color='gray', markersize=4)
            ax.axhline(clean_val, color='green', linestyle='--', linewidth=2, label='Clean' if i == indices[0] else '')
            ax.axhline(kalman_val, color='blue', linestyle='-', linewidth=2, label='Kalman' if i == indices[0] else '')
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel(f'{name} ({unit})')
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        if feat_idx == 0:
            ax.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "kalman_examples.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[kalman] Visualization saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Kalman Filter Denoiser")
    parser.add_argument("--n_episodes", type=int, default=100,
                       help="Number of episodes to collect (default: 100)")
    parser.add_argument("--n_agents", type=int, default=20,
                       help="Number of agents per episode (default: 20)")
    parser.add_argument("--seq_len", type=int, default=3,
                       help="Sequence length for windows (default: 3)")
    parser.add_argument("--process_noise", type=float, default=0.5,
                       help="Process noise std for velocity changes in m/s^2 (default: 0.5 - higher for maneuvering drones!)")
    
    args = parser.parse_args()
    evaluate_kalman(args)


if __name__ == "__main__":
    main()
