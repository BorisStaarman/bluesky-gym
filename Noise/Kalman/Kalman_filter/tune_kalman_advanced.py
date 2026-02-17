"""
Advanced Kalman Filter Parameter Tuning
========================================
Grid search over multiple parameters with detailed diagnostics.

Usage:
    python tune_kalman_advanced.py --n_episodes 50 --fine_tune
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
from train_denoiser import (
    collect_mvp_trajectories, make_dataset_from_trajectories,
    X_NORM, Y_NORM, V_NORM, POS_NOISE_STD_M, VEL_NOISE_STD_MS
)


def evaluate_kalman_detailed(X, Y, process_noise, initial_vel_cov=0.5, 
                             pos_noise_mult=1.0, vel_noise_mult=1.0):
    """Evaluate Kalman filter with detailed metrics."""
    
    # Create custom Kalman filter
    kalman_batch = KalmanDenoiserBatch(
        dt=1.0,
        pos_noise_std=POS_NOISE_STD_M * pos_noise_mult,
        vel_noise_std=VEL_NOISE_STD_MS * vel_noise_mult,
        process_noise_std=process_noise,
        x_norm=X_NORM,
        y_norm=Y_NORM,
        v_norm=V_NORM,
    )
    
    predictions = kalman_batch.denoise_batch(X)
    
    # Compute metrics
    noisy_last = X[:, -1, :]
    clean = Y
    
    # Individual feature RMSEs
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
    
    # Overall (normalized space - equally weighted)
    overall_noisy = np.sqrt(np.mean((noisy_last - clean)**2))
    overall_kalman = np.sqrt(np.mean((predictions - clean)**2))
    overall_improvement = (1 - overall_kalman / overall_noisy) * 100
    
    # Balanced score (position and velocity both matter)
    # Penalize if either gets worse
    if pos_improvement > 0 and vel_improvement > 0:
        balanced_score = np.sqrt(pos_improvement * vel_improvement)  # Geometric mean
    else:
        balanced_score = min(pos_improvement, vel_improvement)  # Penalty for negative
    
    return {
        'process_noise': process_noise,
        'pos_rmse': pos_kalman,
        'pos_improvement': pos_improvement,
        'vel_rmse': vel_kalman,
        'vel_improvement': vel_improvement,
        'overall_improvement': overall_improvement,
        'balanced_score': balanced_score,
        'features': feature_results,
        'initial_vel_cov': initial_vel_cov,
        'pos_noise_mult': pos_noise_mult,
        'vel_noise_mult': vel_noise_mult,
    }


def coarse_search(X, Y):
    """Coarse grid search over wide range of process noise."""
    print(f"\n{'='*100}")
    print(f"  COARSE SEARCH - Finding the right ballpark")
    print(f"{'='*100}")
    
    # Test wide range including MUCH higher values
    process_noise_values = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
    
    print(f"\n{'Process Noise':<15} | {'Pos Impr':<10} | {'Vel Impr':<10} | {'Overall':<10} | {'Balanced':<10} | {'Rating'}")
    print(f"{'-'*100}")
    
    results = []
    for pn in process_noise_values:
        result = evaluate_kalman_detailed(X, Y, pn)
        results.append(result)
        
        # Rating based on balanced score
        score = result['balanced_score']
        rating = "★★★★★" if score > 30 else \
                 "★★★★☆" if score > 25 else \
                 "★★★☆☆" if score > 20 else \
                 "★★☆☆☆" if score > 15 else \
                 "★☆☆☆☆" if score > 5 else "☆☆☆☆☆"
        
        print(f"{pn:<15.2f} | {result['pos_improvement']:>8.1f}% | "
              f"{result['vel_improvement']:>8.1f}% | {result['overall_improvement']:>8.1f}% | "
              f"{score:>8.1f}  | {rating}")
    
    print(f"{'='*100}")
    
    return results


def fine_search(X, Y, center, width=2.0, n_points=15):
    """Fine grid search around best coarse value."""
    print(f"\n{'='*100}")
    print(f"  FINE SEARCH - Optimizing around process_noise={center:.2f}")
    print(f"{'='*100}")
    
    # Fine grid around best value
    process_noise_values = np.linspace(max(0.1, center - width), center + width, n_points)
    
    print(f"\n{'Process Noise':<15} | {'Pos Impr':<10} | {'Vel Impr':<10} | {'Overall':<10} | {'Balanced':<10}")
    print(f"{'-'*100}")
    
    results = []
    for pn in process_noise_values:
        result = evaluate_kalman_detailed(X, Y, pn)
        results.append(result)
        
        print(f"{pn:<15.3f} | {result['pos_improvement']:>8.1f}% | "
              f"{result['vel_improvement']:>8.1f}% | {result['overall_improvement']:>8.1f}% | "
              f"{result['balanced_score']:>8.1f}")
    
    print(f"{'='*100}")
    
    return results


def visualize_results(results, save_path):
    """Create visualization of tuning results."""
    process_noise = [r['process_noise'] for r in results]
    pos_impr = [r['pos_improvement'] for r in results]
    vel_impr = [r['vel_improvement'] for r in results]
    balanced = [r['balanced_score'] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Kalman Filter Parameter Tuning Results", fontsize=14, fontweight='bold')
    
    # Position improvement
    ax = axes[0, 0]
    ax.plot(process_noise, pos_impr, 'o-', linewidth=2, markersize=6, color='blue')
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Process Noise (m/s²)')
    ax.set_ylabel('Position Improvement (%)')
    ax.set_title('Position Denoising Performance')
    ax.grid(True, alpha=0.3)
    
    # Velocity improvement
    ax = axes[0, 1]
    ax.plot(process_noise, vel_impr, 'o-', linewidth=2, markersize=6, color='green')
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Process Noise (m/s²)')
    ax.set_ylabel('Velocity Improvement (%)')
    ax.set_title('Velocity Denoising Performance')
    ax.grid(True, alpha=0.3)
    
    # Both together
    ax = axes[1, 0]
    ax.plot(process_noise, pos_impr, 'o-', linewidth=2, markersize=6, color='blue', label='Position')
    ax.plot(process_noise, vel_impr, 's-', linewidth=2, markersize=6, color='green', label='Velocity')
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Process Noise (m/s²)')
    ax.set_ylabel('Improvement (%)')
    ax.set_title('Position vs Velocity Trade-off')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Balanced score
    ax = axes[1, 1]
    ax.plot(process_noise, balanced, 'o-', linewidth=2, markersize=6, color='purple')
    best_idx = np.argmax(balanced)
    ax.plot(process_noise[best_idx], balanced[best_idx], '*', markersize=20, 
            color='gold', markeredgecolor='black', markeredgewidth=2,
            label=f'Best: {process_noise[best_idx]:.2f}')
    ax.set_xlabel('Process Noise (m/s²)')
    ax.set_ylabel('Balanced Score')
    ax.set_title('Overall Performance (Geometric Mean)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n📊 Visualization saved to: {save_path}")


def print_feature_breakdown(result):
    """Print detailed per-feature results."""
    print(f"\n{'='*80}")
    print(f"  DETAILED BREAKDOWN - Process Noise = {result['process_noise']:.3f}")
    print(f"{'='*80}")
    print(f"{'Feature':<10} | {'Noisy RMSE':<15} | {'Kalman RMSE':<15} | {'Improvement':<12}")
    print(f"{'-'*80}")
    
    for name, metrics in result['features'].items():
        unit = "m" if name in ['x', 'y'] else "m/s"
        print(f"{name:<10} | {metrics['noisy_rmse']:>10.4f} {unit:<3} | "
              f"{metrics['kalman_rmse']:>10.4f} {unit:<3} | "
              f"{metrics['improvement']:>8.1f}%")
    
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=50,
                       help="Number of episodes for evaluation")
    parser.add_argument("--n_agents", type=int, default=20,
                       help="Agents per episode")
    parser.add_argument("--seq_len", type=int, default=3,
                       help="Sequence length")
    parser.add_argument("--fine_tune", action='store_true',
                       help="Run fine-tuning around best coarse value")
    parser.add_argument("--center", type=float, default=None,
                       help="Center for fine search (auto-detected if not provided)")
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"  ADVANCED KALMAN FILTER PARAMETER TUNING")
    print(f"{'='*70}")
    
    # Collect test data
    print(f"\nCollecting trajectories...")
    trajectories = collect_mvp_trajectories(args.n_episodes, args.n_agents, verbose=False)
    X, Y = make_dataset_from_trajectories(trajectories, args.seq_len, seed=42)
    print(f"Dataset: {X.shape[0]} windows from {args.n_episodes} episodes")
    
    # Coarse search
    coarse_results = coarse_search(X, Y)
    
    # Find best by balanced score
    best_coarse = max(coarse_results, key=lambda r: r['balanced_score'])
    print(f"\n🏆 Best coarse result:")
    print(f"   Process noise: {best_coarse['process_noise']:.2f} m/s²")
    print(f"   Position improvement: {best_coarse['pos_improvement']:.1f}%")
    print(f"   Velocity improvement: {best_coarse['vel_improvement']:.1f}%")
    print(f"   Balanced score: {best_coarse['balanced_score']:.1f}")
    
    print_feature_breakdown(best_coarse)
    
    # Visualize coarse results
    save_dir = os.path.join(script_dir, "denoiser_models")
    os.makedirs(save_dir, exist_ok=True)
    coarse_plot = os.path.join(save_dir, "kalman_tuning_coarse.png")
    visualize_results(coarse_results, coarse_plot)
    
    all_results = coarse_results
    best_overall = best_coarse
    
    # Fine search if requested
    if args.fine_tune:
        center = args.center if args.center is not None else best_coarse['process_noise']
        fine_results = fine_search(X, Y, center)
        best_fine = max(fine_results, key=lambda r: r['balanced_score'])
        
        print(f"\n🎯 Best fine-tuned result:")
        print(f"   Process noise: {best_fine['process_noise']:.3f} m/s²")
        print(f"   Position improvement: {best_fine['pos_improvement']:.1f}%")
        print(f"   Velocity improvement: {best_fine['vel_improvement']:.1f}%")
        print(f"   Balanced score: {best_fine['balanced_score']:.1f}")
        
        print_feature_breakdown(best_fine)
        
        fine_plot = os.path.join(save_dir, "kalman_tuning_fine.png")
        visualize_results(fine_results, fine_plot)
        
        all_results = fine_results
        best_overall = best_fine
    
    # Save best configuration
    kalman = KalmanDenoiser(
        dt=1.0,
        pos_noise_std=POS_NOISE_STD_M * best_overall['pos_noise_mult'],
        vel_noise_std=VEL_NOISE_STD_MS * best_overall['vel_noise_mult'],
        process_noise_std=best_overall['process_noise'],
        x_norm=X_NORM,
        y_norm=Y_NORM,
        v_norm=V_NORM,
    )
    
    save_path = os.path.join(save_dir, "kalman_denoiser_optimized.npz")
    kalman.save(save_path)
    print(f"\n💾 Saved optimized Kalman filter to: {save_path}")
    
    # Save results to CSV
    results_csv = os.path.join(save_dir, "kalman_tuning_results.csv")
    import csv
    with open(results_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['process_noise', 'pos_improvement', 
                                                'vel_improvement', 'overall_improvement', 
                                                'balanced_score'])
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r[k] for k in ['process_noise', 'pos_improvement', 
                                                'vel_improvement', 'overall_improvement',
                                                'balanced_score']})
    print(f"📊 Results saved to: {results_csv}")
    
    print(f"\n{'='*70}")
    print(f"  RECOMMENDATIONS")
    print(f"{'='*70}")
    print(f"Based on your drone collision avoidance scenario:")
    print(f"• Your drones make frequent maneuvers (velocity changes)")
    print(f"• Process noise needs to be HIGH to track these changes")
    print(f"• Optimal value: {best_overall['process_noise']:.2f} m/s²")
    print(f"• This balances position smoothing vs velocity tracking")
    print(f"\nTest with: python evaluate_lstm_mvp.py --denoiser_path \"{save_path}\"")


if __name__ == "__main__":
    main()
