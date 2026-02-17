"""
LSTM Denoiser Diagnostic Tool
===============================
Analyzes the trained LSTM denoiser to understand:
1. Actual denoising performance in physical units
2. Whether the model has reached optimal performance
3. Suggestions for improvement

Usage:
    python diagnose_lstm.py
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from lstm_denoiser import LSTMDenoiser
from train_denoiser import (
    collect_mvp_trajectories, make_dataset_from_trajectories,
    X_NORM, Y_NORM, V_NORM, POS_NOISE_STD_M, VEL_NOISE_STD_MS
)

def analyze_denoising_performance():
    """Detailed analysis of LSTM denoising quality."""
    
    print("\n" + "="*70)
    print("  LSTM DENOISER DIAGNOSTIC")
    print("="*70)
    
    # Load model
    model_path = os.path.join(script_dir, "denoiser_models", "lstm_denoiser_best.pt")
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print("   Run train_denoiser.py first!")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMDenoiser.load(model_path, device=device)
    print(f"✅ Loaded model from: {model_path}")
    print(f"   Device: {device}")
    
    # Generate test data from MVP (different from training)
    print(f"\n📊 Collecting test trajectories from MVP controller...")
    test_trajectories = collect_mvp_trajectories(
        n_episodes=50,  # Smaller for faster testing
        n_agents=20,
        verbose=True
    )
    
    print(f"\n📊 Building test dataset...")
    X_test, Y_test = make_dataset_from_trajectories(
        test_trajectories,
        seq_len=10,
        seed=99999  # Different seed from training
    )
    print(f"   Test size: {X_test.shape}")
    
    # Run inference
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        Y_pred = model(X_tensor).cpu().numpy()
    
    # Compare: Noisy baseline vs LSTM vs Ground Truth
    noisy_last = X_test[:, -1, :]  # Just use last noisy observation
    clean = Y_test
    
    feature_names = ["x", "y", "vx", "vy"]
    unnorm = np.array([X_NORM, Y_NORM, V_NORM, V_NORM])
    units = ["m", "m", "m/s", "m/s"]
    
    print(f"\n" + "="*90)
    print(f"{'Feature':<6} | {'Noise σ':<10} | {'Noisy RMSE':<12} | {'LSTM RMSE':<12} | {'Improvement':<12} | {'% of σ':<10}")
    print("-"*90)
    
    improvements = []
    for i, name in enumerate(feature_names):
        # RMSE in normalized space
        noisy_rmse_norm = np.sqrt(np.mean((noisy_last[:, i] - clean[:, i]) ** 2))
        lstm_rmse_norm = np.sqrt(np.mean((Y_pred[:, i] - clean[:, i]) ** 2))
        
        # Convert to physical units
        noisy_rmse_phys = noisy_rmse_norm * unnorm[i]
        lstm_rmse_phys = lstm_rmse_norm * unnorm[i]
        
        # Original noise std
        if i < 2:
            noise_std = POS_NOISE_STD_M
        else:
            noise_std = VEL_NOISE_STD_MS
        
        # Improvement percentage
        improvement = (1 - lstm_rmse_phys / noisy_rmse_phys) * 100 if noisy_rmse_phys > 0 else 0
        improvements.append(improvement)
        
        # LSTM error as percentage of original noise
        pct_of_noise = (lstm_rmse_phys / noise_std) * 100 if noise_std > 0 else 0
        
        unit = units[i]
        print(f"{name:<6} | {noise_std:>6.2f} {unit:<2} | "
              f"{noisy_rmse_phys:>9.4f} {unit:<2} | "
              f"{lstm_rmse_phys:>9.4f} {unit:<2} | "
              f"{improvement:>9.1f}% | "
              f"{pct_of_noise:>7.1f}%")
    
    print("="*90)
    avg_improvement = np.mean(improvements)
    print(f"\n📈 Average improvement across all features: {avg_improvement:.1f}%")
    
    # Analysis in normalized space (what the RL agent sees)
    print(f"\n" + "="*70)
    print("  NORMALIZED SPACE ANALYSIS (RL Agent's View)")
    print("="*70)
    
    overall_noisy = np.sqrt(np.mean((noisy_last - clean) ** 2))
    overall_lstm = np.sqrt(np.mean((Y_pred - clean) ** 2))
    overall_improvement = (1 - overall_lstm / overall_noisy) * 100
    
    print(f"Overall RMSE (normalized):")
    print(f"  Noisy baseline: {overall_noisy:.6f}")
    print(f"  LSTM denoised:  {overall_lstm:.6f}")
    print(f"  Improvement:    {overall_improvement:.1f}%")
    
    # Check per-feature distribution
    print(f"\n" + "="*70)
    print("  ERROR DISTRIBUTION ANALYSIS")
    print("="*70)
    
    for i, name in enumerate(feature_names):
        noisy_errors = np.abs(noisy_last[:, i] - clean[:, i]) * unnorm[i]
        lstm_errors = np.abs(Y_pred[:, i] - clean[:, i]) * unnorm[i]
        
        print(f"\n{name.upper()} ({units[i]}):")
        print(f"  Noisy: Mean={np.mean(noisy_errors):.4f}, Median={np.median(noisy_errors):.4f}, "
              f"P95={np.percentile(noisy_errors, 95):.4f}")
        print(f"  LSTM:  Mean={np.mean(lstm_errors):.4f}, Median={np.median(lstm_errors):.4f}, "
              f"P95={np.percentile(lstm_errors, 95):.4f}")
    
    # Recommendations
    print(f"\n" + "="*70)
    print("  RECOMMENDATIONS")
    print("="*70)
    
    if avg_improvement > 40:
        print("✅ EXCELLENT: LSTM is providing strong denoising (>40% improvement)")
        print("   Your current model is working very well!")
    elif avg_improvement > 25:
        print("✅ GOOD: LSTM is effectively reducing noise (25-40% improvement)")
        print("   Consider these improvements:")
        print("   • Increase model capacity (hidden_dim=256, num_layers=3)")
        print("   • Use longer sequences (seq_len=15 or 20)")
    else:
        print("⚠️  MODERATE: LSTM shows some improvement but has potential issues")
        print("   Suggested actions:")
        print("   • Check if training data matches deployment scenarios")
        print("   • Increase model capacity")
        print("   • Train longer with lower learning rate")
    
    if avg_improvement < 50:
        print("\n💡 Potential Improvements:")
        print("   1. Train with more diverse trajectories")
        print("   2. Add trajectory variations (accelerations, sharp turns)")
        print("   3. Increase LSTM hidden dimension to 256")
        print("   4. Try GRU instead of LSTM")
        print("   5. Add batch normalization or layer normalization")
    
    # Check if early stopping is working
    print(f"\n📊 TRAINING TIPS:")
    print(f"   • Your training plateaus quickly (~10 epochs)")
    print(f"   • This is OK if final performance is good")
    print(f"   • Consider reducing epochs to 30 with early stopping")
    print(f"   • Use ReduceLROnPlateau (already implemented ✓)")
    
    # Visualize some examples
    print(f"\n📊 Generating visualization...")
    plot_examples(X_test[:5], Y_test[:5], Y_pred[:5], noisy_last[:5])
    
    print(f"\n✅ Diagnostic complete!")


def plot_examples(X, Y, Y_pred, noisy_last, n_examples=5):
    """Plot example trajectories showing denoising effect."""
    
    feature_names = ["X Position", "Y Position", "X Velocity", "Y Velocity"]
    unnorm = np.array([X_NORM, Y_NORM, V_NORM, V_NORM])
    units = ["m", "m", "m/s", "m/s"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for feat_idx in range(4):
        ax = axes[feat_idx]
        
        # Plot multiple examples
        for i in range(min(n_examples, len(X))):
            # Convert to physical units
            noisy_seq = X[i, :, feat_idx] * unnorm[feat_idx]
            clean_val = Y[i, feat_idx] * unnorm[feat_idx]
            lstm_val = Y_pred[i, feat_idx] * unnorm[feat_idx]
            noisy_val = noisy_last[i, feat_idx] * unnorm[feat_idx]
            
            # Plot trajectory
            timesteps = np.arange(len(noisy_seq))
            ax.plot(timesteps, noisy_seq, 'o-', alpha=0.3, color='gray', 
                   linewidth=1, markersize=3)
            
            # Mark last timestep
            ax.plot([len(noisy_seq)-1], [noisy_val], 'ro', 
                   label='Noisy (last)' if i == 0 else '', markersize=8, alpha=0.6)
            ax.plot([len(noisy_seq)-1], [lstm_val], 'go', 
                   label='LSTM' if i == 0 else '', markersize=8, alpha=0.8)
            ax.axhline(clean_val, color='blue', linestyle='--', alpha=0.5,
                      label='Ground Truth' if i == 0 else '', linewidth=2)
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel(f'{feature_names[feat_idx]} ({units[feat_idx]})')
        ax.set_title(f'{feature_names[feat_idx]}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(script_dir, "denoiser_models", "denoising_examples.png")
    plt.savefig(save_path, dpi=150)
    print(f"   Saved visualization to: {save_path}")
    plt.close()


if __name__ == "__main__":
    analyze_denoising_performance()
