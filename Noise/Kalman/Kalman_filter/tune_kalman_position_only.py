"""
Tune Position-Only Kalman Filter
==================================
Compare standard Kalman vs position-only Kalman filter.
Position-only filter ignores noisy velocity measurements.

Usage:
    python tune_kalman_position_only.py --n_episodes 50
"""

import os
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from kalman_denoiser import KalmanDenoiser, KalmanDenoiserBatch
from kalman_denoiser_position_only import KalmanDenoiserPositionOnly, KalmanDenoiserPositionOnlyBatch
from train_denoiser import (
    collect_mvp_trajectories, make_dataset_from_trajectories,
    X_NORM, Y_NORM, V_NORM, POS_NOISE_STD_M, VEL_NOISE_STD_MS
)


def evaluate_filter(X, Y, kalman_batch, filter_name="Kalman"):
    """Evaluate a filter and return detailed metrics."""
    predictions = kalman_batch.denoise_batch(X)
    
    noisy_last = X[:, -1, :]
    clean = Y
    
    # Per-feature metrics
    feature_names = ["x", "y", "vx", "vy"]
    unnorm = np.array([X_NORM, Y_NORM, V_NORM, V_NORM])
    
    feature_results = {}
    for i, name in enumerate(feature_names):
        noisy_rmse_phys = np.sqrt(np.mean((noisy_last[:, i] - clean[:, i])**2)) * unnorm[i]
        kalman_rmse_phys = np.sqrt(np.mean((predictions[:, i] - clean[:, i])**2)) * unnorm[i]
        improvement = (1 - kalman_rmse_phys / noisy_rmse_phys) * 100
        
        feature_results[name] = {
            'noisy_rmse': noisy_rmse_phys,
            'kalman_rmse': kalman_rmse_phys,
            'improvement': improvement
        }
    
    # Aggregate metrics
    pos_noisy = np.sqrt(np.mean((noisy_last[:, :2] - clean[:, :2])**2)) * (X_NORM + Y_NORM) / 2
    pos_kalman = np.sqrt(np.mean((predictions[:, :2] - clean[:, :2])**2)) * (X_NORM + Y_NORM) / 2
    pos_improvement = (1 - pos_kalman / pos_noisy) * 100
    
    vel_noisy = np.sqrt(np.mean((noisy_last[:, 2:] - clean[:, 2:])**2)) * V_NORM
    vel_kalman = np.sqrt(np.mean((predictions[:, 2:] - clean[:, 2:])**2)) * V_NORM
    vel_improvement = (1 - vel_kalman / vel_noisy) * 100
    
    return {
        'name': filter_name,
        'pos_improvement': pos_improvement,
        'vel_improvement': vel_improvement,
        'pos_rmse': pos_kalman,
        'vel_rmse': vel_kalman,
        'features': feature_results,
    }


def tune_position_only_filter(X, Y, process_noise_vel_values=None):
    """Tune position-only filter over process noise for velocity."""
    if process_noise_vel_values is None:
        process_noise_vel_values = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    
    print(f"\n{'='*100}")
    print(f"  POSITION-ONLY KALMAN FILTER TUNING")
    print(f"{'='*100}")
    print(f"Note: We only use POSITION measurements [x,y], velocity [vx,vy] is ESTIMATED")
    
    print(f"\n{'Proc Noise Vel':<15} | {'Pos Impr':<10} | {'Vel Impr':<10} | {'Rating'}")
    print(f"{'-'*60}")
    
    results = []
    for pn_vel in process_noise_vel_values:
        kalman_batch = KalmanDenoiserPositionOnlyBatch(
            dt=1.0,
            pos_noise_std=POS_NOISE_STD_M,
            process_noise_pos=0.1,  # Small position process noise
            process_noise_vel=pn_vel,  # Tune this
            x_norm=X_NORM,
            y_norm=Y_NORM,
            v_norm=V_NORM,
        )
        
        result = evaluate_filter(X, Y, kalman_batch, f"PosOnly-{pn_vel:.1f}")
        result['process_noise_vel'] = pn_vel
        results.append(result)
        
        rating = "★★★★★" if result['pos_improvement'] > 40 else \
                 "★★★★☆" if result['pos_improvement'] > 38 else \
                 "★★★☆☆" if result['pos_improvement'] > 35 else \
                 "★★☆☆☆" if result['pos_improvement'] > 30 else "★☆☆☆☆"
        
        print(f"{pn_vel:<15.2f} | {result['pos_improvement']:>8.1f}% | "
              f"{result['vel_improvement']:>8.1f}% | {rating}")
    
    print(f"{'='*100}")
    
    best = max(results, key=lambda r: r['pos_improvement'])
    return results, best


def compare_filters(X, Y, process_noise_standard=1.0, process_noise_posonly=1.0):
    """Compare standard Kalman vs position-only."""
    print(f"\n{'='*100}")
    print(f"  COMPARISON: Standard Kalman vs Position-Only Kalman")
    print(f"{'='*100}")
    
    # Standard Kalman (measures position + velocity)
    standard_batch = KalmanDenoiserBatch(
        dt=1.0,
        pos_noise_std=POS_NOISE_STD_M,
        vel_noise_std=VEL_NOISE_STD_MS,
        process_noise_std=process_noise_standard,
        x_norm=X_NORM,
        y_norm=Y_NORM,
        v_norm=V_NORM,
    )
    
    # Position-only Kalman (measures position only)
    posonly_batch = KalmanDenoiserPositionOnlyBatch(
        dt=1.0,
        pos_noise_std=POS_NOISE_STD_M,
        process_noise_pos=0.1,
        process_noise_vel=process_noise_posonly,
        x_norm=X_NORM,
        y_norm=Y_NORM,
        v_norm=V_NORM,
    )
    
    standard_result = evaluate_filter(X, Y, standard_batch, "Standard Kalman")
    posonly_result = evaluate_filter(X, Y, posonly_batch, "Position-Only")
    
    print(f"\n{'Filter Type':<25} | {'Pos Impr':<12} | {'Vel Impr':<12} | {'Winner'}")
    print(f"{'-'*80}")
    
    print(f"{'Standard (measures x,y,vx,vy)':<25} | {standard_result['pos_improvement']:>10.1f}% | "
          f"{standard_result['vel_improvement']:>10.1f}% | "
          f"{'🏆' if standard_result['pos_improvement'] > posonly_result['pos_improvement'] else ''}")
    
    print(f"{'Position-Only (measures x,y)':<25} | {posonly_result['pos_improvement']:>10.1f}% | "
          f"{posonly_result['vel_improvement']:>10.1f}% | "
          f"{'🏆' if posonly_result['pos_improvement'] > standard_result['pos_improvement'] else ''}")
    
    print(f"{'Difference':<25} | {posonly_result['pos_improvement'] - standard_result['pos_improvement']:>+10.1f}% | "
          f"{posonly_result['vel_improvement'] - standard_result['vel_improvement']:>+10.1f}%")
    
    print(f"{'='*100}")
    
    # Detailed breakdown
    print(f"\n{'='*100}")
    print(f"  DETAILED FEATURE COMPARISON")
    print(f"{'='*100}")
    print(f"{'Feature':<10} | {'Standard RMSE':<18} | {'PosOnly RMSE':<18} | {'Difference'}")
    print(f"{'-'*100}")
    
    for feat in ['x', 'y', 'vx', 'vy']:
        std_rmse = standard_result['features'][feat]['kalman_rmse']
        pos_rmse = posonly_result['features'][feat]['kalman_rmse']
        diff_pct = (1 - pos_rmse / std_rmse) * 100 if std_rmse > 0 else 0
        
        unit = "m" if feat in ['x', 'y'] else "m/s"
        winner = "🏆" if diff_pct > 0 else ""
        
        print(f"{feat:<10} | {std_rmse:>12.4f} {unit:<4} | {pos_rmse:>12.4f} {unit:<4} | "
              f"{diff_pct:>+7.1f}% {winner}")
    
    print(f"{'='*100}")
    
    return standard_result, posonly_result


def visualize_comparison(standard_result, posonly_result, save_path):
    """Visualize comparison between filter types."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Standard Kalman vs Position-Only Kalman", fontsize=14, fontweight='bold')
    
    # Position and velocity improvements
    ax = axes[0]
    filters = ['Standard\n(measures x,y,vx,vy)', 'Position-Only\n(measures x,y)']
    pos_imprs = [standard_result['pos_improvement'], posonly_result['pos_improvement']]
    vel_imprs = [standard_result['vel_improvement'], posonly_result['vel_improvement']]
    
    x_pos = np.arange(len(filters))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, pos_imprs, width, label='Position', color='blue', alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, vel_imprs, width, label='Velocity', color='green', alpha=0.8)
    
    ax.set_ylabel('Improvement (%)', fontsize=12)
    ax.set_title('Overall Improvement by Filter Type')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(filters)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    
    # Per-feature RMSE
    ax = axes[1]
    features = ['x', 'y', 'vx', 'vy']
    std_rmses = [standard_result['features'][f]['kalman_rmse'] for f in features]
    pos_rmses = [posonly_result['features'][f]['kalman_rmse'] for f in features]
    
    x_pos = np.arange(len(features))
    bars1 = ax.bar(x_pos - width/2, std_rmses, width, label='Standard', color='orange', alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, pos_rmses, width, label='Position-Only', color='purple', alpha=0.8)
    
    ax.set_ylabel('RMSE (physical units)', fontsize=12)
    ax.set_title('Per-Feature RMSE (Lower is Better)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(features)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n📊 Comparison visualization saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--n_agents", type=int, default=20)
    parser.add_argument("--seq_len", type=int, default=3)
    parser.add_argument("--standard_process_noise", type=float, default=1.0,
                       help="Process noise for standard Kalman filter")
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"  POSITION-ONLY KALMAN FILTER EVALUATION")
    print(f"{'='*70}")
    
    # Collect test data
    print(f"\nCollecting trajectories...")
    trajectories = collect_mvp_trajectories(args.n_episodes, args.n_agents, verbose=False)
    X, Y = make_dataset_from_trajectories(trajectories, args.seq_len, seed=42)
    print(f"Dataset: {X.shape[0]} windows from {args.n_episodes} episodes")
    
    # Tune position-only filter
    posonly_results, best_posonly = tune_position_only_filter(X, Y)
    
    print(f"\n🏆 Best Position-Only Configuration:")
    print(f"   Process noise (velocity): {best_posonly['process_noise_vel']:.2f} m/s²")
    print(f"   Position improvement: {best_posonly['pos_improvement']:.1f}%")
    print(f"   Velocity improvement: {best_posonly['vel_improvement']:.1f}%")
    
    # Compare with standard filter
    standard_result, posonly_result = compare_filters(
        X, Y, 
        process_noise_standard=args.standard_process_noise,
        process_noise_posonly=best_posonly['process_noise_vel']
    )
    
    # Visualize
    save_dir = os.path.join(script_dir, "denoiser_models")
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, "kalman_comparison.png")
    visualize_comparison(standard_result, posonly_result, plot_path)
    
    # Save best position-only filter
    if posonly_result['pos_improvement'] > standard_result['pos_improvement']:
        print(f"\n✅ Position-Only filter is BETTER for position estimates!")
        print(f"   Position improvement gain: +{posonly_result['pos_improvement'] - standard_result['pos_improvement']:.1f}%")
        
        best_filter = KalmanDenoiserPositionOnly(
            dt=1.0,
            pos_noise_std=POS_NOISE_STD_M,
            process_noise_pos=0.1,
            process_noise_vel=best_posonly['process_noise_vel'],
            x_norm=X_NORM,
            y_norm=Y_NORM,
            v_norm=V_NORM,
        )
        save_path = os.path.join(save_dir, "kalman_denoiser_position_only.npz")
        best_filter.save(save_path)
        print(f"💾 Saved position-only filter to: {save_path}")
    else:
        print(f"\n⚠️ Standard filter is still better for position estimates")
        print(f"   Position improvement loss: {posonly_result['pos_improvement'] - standard_result['pos_improvement']:.1f}%")
        print(f"   Stick with standard Kalman filter")
    
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"Standard Kalman:")
    print(f"  • Uses noisy measurements: [x, y, vx, vy]")
    print(f"  • Position improvement: {standard_result['pos_improvement']:.1f}%")
    print(f"  • Velocity improvement: {standard_result['vel_improvement']:.1f}%")
    print(f"\nPosition-Only Kalman:")
    print(f"  • Uses only measurements: [x, y]")
    print(f"  • Estimates velocity from position changes")
    print(f"  • Position improvement: {posonly_result['pos_improvement']:.1f}%")
    print(f"  • Velocity improvement: {posonly_result['vel_improvement']:.1f}%")
    print(f"\nRecommendation:")
    if posonly_result['pos_improvement'] > standard_result['pos_improvement']:
        print(f"  🎯 Use Position-Only Kalman for best position estimates!")
    else:
        print(f"  🎯 Use Standard Kalman (process_noise={args.standard_process_noise:.1f})")


if __name__ == "__main__":
    main()
