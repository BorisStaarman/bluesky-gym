"""
Simple Example: Using Kalman Filter to Denoise Data
====================================================
Run this to see the Kalman filter in action!

Usage:
    python example_use_kalman.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add current directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from kalman_denoiser import KalmanDenoiser


def generate_fake_trajectory(n_timesteps=50):
    """Generate a fake drone trajectory with noise."""
    
    # Normalization constants
    X_NORM = 8500.0
    Y_NORM = 8000.0
    V_NORM = 36.0
    
    # Generate clean trajectory (normalized)
    t = np.linspace(0, 10, n_timesteps)
    x_clean = 0.3 + 0.1 * np.sin(0.5 * t)  # Smooth x motion
    y_clean = 0.4 + 0.1 * np.cos(0.5 * t)  # Smooth y motion
    vx_clean = np.gradient(x_clean, t[1] - t[0]) * (1.0 * X_NORM / V_NORM)  # vx from position
    vy_clean = np.gradient(y_clean, t[1] - t[0]) * (1.0 * Y_NORM / V_NORM)  # vy from position
    
    clean_trajectory = np.column_stack([x_clean, y_clean, vx_clean, vy_clean])
    
    # Add noise (in normalized space)
    pos_noise = 3.5 / X_NORM  # 3.5m in normalized units
    vel_noise = 0.1 / V_NORM  # 0.1 m/s in normalized units
    
    noise = np.random.randn(n_timesteps, 4) * np.array([pos_noise, pos_noise, vel_noise, vel_noise])
    noisy_trajectory = clean_trajectory + noise
    
    return clean_trajectory, noisy_trajectory


def example_1_denoise_full_sequence():
    """Example 1: Denoise an entire trajectory."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Denoise Full Trajectory")
    print("="*70)
    
    # Generate fake data
    clean, noisy = generate_fake_trajectory(n_timesteps=50)
    print(f"Generated trajectory with {len(noisy)} timesteps")
    
    # Create Kalman filter with optimal parameters
    kalman = KalmanDenoiser(
        dt=1.0,
        pos_noise_std=3.5,
        vel_noise_std=0.1,
        process_noise_std=1.0,  # Optimal tuned value
    )
    print("Created Kalman filter with optimal parameters")
    
    # Denoise entire sequence
    filtered = kalman.denoise_sequence(noisy)
    print(f"Filtered trajectory shape: {filtered.shape}")
    
    # Calculate improvements
    noisy_error = np.sqrt(np.mean((noisy - clean)**2, axis=0))
    filtered_error = np.sqrt(np.mean((filtered - clean)**2, axis=0))
    improvement = (1 - filtered_error / noisy_error) * 100
    
    print(f"\nResults:")
    print(f"  X improvement:  {improvement[0]:.1f}%")
    print(f"  Y improvement:  {improvement[1]:.1f}%")
    print(f"  Vx improvement: {improvement[2]:.1f}%")
    print(f"  Vy improvement: {improvement[3]:.1f}%")
    
    return clean, noisy, filtered


def example_2_denoise_window():
    """Example 2: Get current estimate from recent window."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Get Current Estimate from Recent Observations")
    print("="*70)
    
    # Generate fake data
    clean, noisy = generate_fake_trajectory(n_timesteps=50)
    
    # Take last 5 observations as "recent window"
    recent_window = noisy[-5:]
    true_current = clean[-1]
    
    print(f"Using last 5 observations to estimate current state")
    print(f"True current state: {true_current}")
    print(f"Noisy observation:  {noisy[-1]}")
    
    # Create Kalman filter
    kalman = KalmanDenoiser(process_noise_std=1.0)
    
    # Get filtered estimate of CURRENT state only
    current_estimate = kalman.denoise(recent_window)
    
    print(f"Kalman estimate:    {current_estimate}")
    print(f"\nNoisy error:   {np.linalg.norm(noisy[-1] - true_current):.6f}")
    print(f"Kalman error:  {np.linalg.norm(current_estimate - true_current):.6f}")
    
    improvement = (1 - np.linalg.norm(current_estimate - true_current) / 
                   np.linalg.norm(noisy[-1] - true_current)) * 100
    print(f"Improvement:   {improvement:.1f}%")


def example_3_batch_processing():
    """Example 3: Process multiple trajectories at once."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Batch Processing Multiple Trajectories")
    print("="*70)
    
    from kalman_denoiser import KalmanDenoiserBatch
    
    # Generate multiple trajectories
    n_trajectories = 100
    window_size = 5
    
    all_windows = []
    for i in range(n_trajectories):
        _, noisy = generate_fake_trajectory(n_timesteps=window_size)
        all_windows.append(noisy)
    
    batch = np.array(all_windows)
    print(f"Processing {n_trajectories} trajectories simultaneously")
    print(f"Batch shape: {batch.shape}")
    
    # Create batch processor
    kalman_batch = KalmanDenoiserBatch(process_noise_std=1.0)
    
    # Process all at once
    final_estimates = kalman_batch.denoise_batch(batch)
    print(f"Output shape: {final_estimates.shape}")
    print(f"\nFirst 5 estimates:")
    for i in range(5):
        print(f"  Trajectory {i}: {final_estimates[i]}")


def example_4_realtime_processing():
    """Example 4: Process measurements as they arrive (streaming)."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Real-Time / Streaming Processing")
    print("="*70)
    
    # Generate fake data
    clean, noisy = generate_fake_trajectory(n_timesteps=10)
    
    # Create Kalman filter
    kalman = KalmanDenoiser(process_noise_std=1.0)
    
    print("Processing measurements one at a time (like real-time):\n")
    
    # Initialize with first measurement
    kalman.reset(noisy[0])
    print(f"t=0: Initialized with {noisy[0]}")
    
    # Process subsequent measurements
    for t in range(1, len(noisy)):
        # Predict next state
        kalman.predict()
        
        # Update with new measurement
        kalman.update(noisy[t])
        
        # Get current estimate
        current_estimate = kalman.x.copy()
        
        error = np.linalg.norm(current_estimate - clean[t])
        print(f"t={t}: Estimate error = {error:.6f}")
    
    print(f"\nFinal filtered state: {kalman.x}")


def visualize_results(clean, noisy, filtered):
    """Create visualization of filtering results."""
    print("\n" + "="*70)
    print("Creating Visualization...")
    print("="*70)
    
    # Denormalize for plotting (convert to physical units)
    X_NORM = 8500.0
    Y_NORM = 8000.0
    V_NORM = 36.0
    
    clean_phys = clean * np.array([X_NORM, Y_NORM, V_NORM, V_NORM])
    noisy_phys = noisy * np.array([X_NORM, Y_NORM, V_NORM, V_NORM])
    filtered_phys = filtered * np.array([X_NORM, Y_NORM, V_NORM, V_NORM])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Kalman Filter Denoising Results', fontsize=14, fontweight='bold')
    
    timesteps = np.arange(len(clean))
    
    # X position
    ax = axes[0, 0]
    ax.plot(timesteps, clean_phys[:, 0], 'g-', linewidth=2, label='True (clean)')
    ax.plot(timesteps, noisy_phys[:, 0], 'r.', alpha=0.3, markersize=4, label='Noisy')
    ax.plot(timesteps, filtered_phys[:, 0], 'b-', linewidth=1.5, label='Kalman filtered')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('X Position (m)')
    ax.set_title('X Position')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Y position
    ax = axes[0, 1]
    ax.plot(timesteps, clean_phys[:, 1], 'g-', linewidth=2, label='True (clean)')
    ax.plot(timesteps, noisy_phys[:, 1], 'r.', alpha=0.3, markersize=4, label='Noisy')
    ax.plot(timesteps, filtered_phys[:, 1], 'b-', linewidth=1.5, label='Kalman filtered')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Y Position')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # X velocity
    ax = axes[1, 0]
    ax.plot(timesteps, clean_phys[:, 2], 'g-', linewidth=2, label='True (clean)')
    ax.plot(timesteps, noisy_phys[:, 2], 'r.', alpha=0.3, markersize=4, label='Noisy')
    ax.plot(timesteps, filtered_phys[:, 2], 'b-', linewidth=1.5, label='Kalman filtered')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('X Velocity (m/s)')
    ax.set_title('X Velocity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Y velocity
    ax = axes[1, 1]
    ax.plot(timesteps, clean_phys[:, 3], 'g-', linewidth=2, label='True (clean)')
    ax.plot(timesteps, noisy_phys[:, 3], 'r.', alpha=0.3, markersize=4, label='Noisy')
    ax.plot(timesteps, filtered_phys[:, 3], 'b-', linewidth=1.5, label='Kalman filtered')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Y Velocity (m/s)')
    ax.set_title('Y Velocity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(script_dir, "kalman_example_results.png")
    plt.savefig(save_path, dpi=150)
    print(f"Saved visualization to: {save_path}")
    plt.close()


def main():
    print("\n" + "="*70)
    print("KALMAN FILTER USAGE EXAMPLES")
    print("="*70)
    print("This script demonstrates how to use the Kalman filter in your code.")
    
    # Run all examples
    clean, noisy, filtered = example_1_denoise_full_sequence()
    example_2_denoise_window()
    example_3_batch_processing()
    example_4_realtime_processing()
    
    # Visualize
    visualize_results(clean, noisy, filtered)
    
    print("\n" + "="*70)
    print("✅ All examples completed successfully!")
    print("="*70)
    print("\nSummary:")
    print("  • Example 1: Full trajectory denoising")
    print("  • Example 2: Current estimate from window")
    print("  • Example 3: Batch processing")
    print("  • Example 4: Real-time streaming")
    print(f"\nVisualization saved to: kalman_example_results.png")
    print("\nNow you know how to use the Kalman filter in your own code!")


if __name__ == "__main__":
    main()
