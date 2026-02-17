"""
Tune Kalman Filter Parameters
===============================
Grid search over process noise to find optimal denoising performance.

Usage:
    python tune_kalman.py --n_episodes 50
"""

import os
import sys
import numpy as np
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from kalman_denoiser import KalmanDenoiser, KalmanDenoiserBatch
from train_denoiser import (
    collect_mvp_trajectories, make_dataset_from_trajectories,
    X_NORM, Y_NORM, V_NORM, POS_NOISE_STD_M, VEL_NOISE_STD_MS
)


def evaluate_kalman_with_params(X, Y, process_noise):
    """Evaluate Kalman filter with specific process noise."""
    kalman_batch = KalmanDenoiserBatch(
        dt=1.0,
        pos_noise_std=POS_NOISE_STD_M,
        vel_noise_std=VEL_NOISE_STD_MS,
        process_noise_std=process_noise,
        x_norm=X_NORM,
        y_norm=Y_NORM,
        v_norm=V_NORM,
    )
    
    predictions = kalman_batch.denoise_batch(X)
    
    # Compute metrics
    noisy_last = X[:, -1, :]
    clean = Y
    
    # Position RMSE (in physical units)
    pos_noisy = np.sqrt(np.mean((noisy_last[:, :2] - clean[:, :2])**2)) * (X_NORM + Y_NORM) / 2
    pos_kalman = np.sqrt(np.mean((predictions[:, :2] - clean[:, :2])**2)) * (X_NORM + Y_NORM) / 2
    pos_improvement = (1 - pos_kalman / pos_noisy) * 100
    
    # Velocity RMSE (in physical units)
    vel_noisy = np.sqrt(np.mean((noisy_last[:, 2:] - clean[:, 2:])**2)) * V_NORM
    vel_kalman = np.sqrt(np.mean((predictions[:, 2:] - clean[:, 2:])**2)) * V_NORM
    vel_improvement = (1 - vel_kalman / vel_noisy) * 100
    
    # Overall
    overall_noisy = np.sqrt(np.mean((noisy_last - clean)**2))
    overall_kalman = np.sqrt(np.mean((predictions - clean)**2))
    overall_improvement = (1 - overall_kalman / overall_noisy) * 100
    
    return {
        'process_noise': process_noise,
        'pos_rmse': pos_kalman,
        'pos_improvement': pos_improvement,
        'vel_rmse': vel_kalman,
        'vel_improvement': vel_improvement,
        'overall_improvement': overall_improvement,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--n_agents", type=int, default=20)
    parser.add_argument("--seq_len", type=int, default=3)
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"  KALMAN FILTER PARAMETER TUNING")
    print(f"{'='*70}")
    
    # Collect test data
    print(f"\nCollecting trajectories...")
    trajectories = collect_mvp_trajectories(args.n_episodes, args.n_agents, verbose=False)
    X, Y = make_dataset_from_trajectories(trajectories, args.seq_len, seed=42)
    print(f"Dataset: {X.shape[0]} windows")
    
    # Grid search over process noise
    process_noise_values = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    
    print(f"\n{'='*90}")
    print(f"{'Process Noise':<15} | {'Pos Impr %':<12} | {'Vel Impr %':<12} | {'Overall %':<12} | {'Rating':<10}")
    print(f"{'-'*90}")
    
    results = []
    for pn in process_noise_values:
        print(f"Testing process_noise={pn}...", end=" ", flush=True)
        result = evaluate_kalman_with_params(X, Y, pn)
        results.append(result)
        
        # Rating based on balanced improvement
        rating = "★★★★★" if result['overall_improvement'] > 20 else \
                 "★★★★☆" if result['overall_improvement'] > 15 else \
                 "★★★☆☆" if result['overall_improvement'] > 10 else \
                 "★★☆☆☆" if result['overall_improvement'] > 5 else "★☆☆☆☆"
        
        print(f"{pn:<15.2f} | {result['pos_improvement']:>10.1f}% | "
              f"{result['vel_improvement']:>10.1f}% | {result['overall_improvement']:>10.1f}% | {rating}")
    
    print(f"{'='*90}")
    
    # Find best
    best = max(results, key=lambda r: r['overall_improvement'])
    print(f"\n🏆 Best process_noise: {best['process_noise']:.2f}")
    print(f"   Position improvement: {best['pos_improvement']:.1f}%")
    print(f"   Velocity improvement: {best['vel_improvement']:.1f}%")
    print(f"   Overall improvement: {best['overall_improvement']:.1f}%")
    
    # Save best config
    kalman = KalmanDenoiser(
        dt=1.0,
        pos_noise_std=POS_NOISE_STD_M,
        vel_noise_std=VEL_NOISE_STD_MS,
        process_noise_std=best['process_noise'],
        x_norm=X_NORM,
        y_norm=Y_NORM,
        v_norm=V_NORM,
    )
    
    save_dir = os.path.join(script_dir, "denoiser_models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "kalman_denoiser_tuned.npz")
    kalman.save(save_path)
    print(f"\n💾 Saved tuned Kalman filter to: {save_path}")
    print(f"\nTest with: python evaluate_lstm_mvp.py --denoiser_path \"{save_path}\"")


if __name__ == "__main__":
    main()
