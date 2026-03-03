"""
Kalman Filter Process Noise Tuning Script
==========================================

This script helps find the optimal process_noise_std value for your Kalman filter.

The process noise models how much drone velocity changes between timesteps due to:
- Collision avoidance maneuvers
- Acceleration/deceleration
- Control actions

TRADE-OFF:
----------
- Too LOW (e.g., 0.01):  
  ✓ Very smooth tracking (filters noise well)
  ✗ Slow response to real maneuvers (lags behind true motion)
  
- Too HIGH (e.g., 1.0):  
  ✓ Fast response to maneuvers  
  ✗ Doesn't filter noise well (trusts noisy measurements too much)
  
CURRENT VALUE: 0.1
- Expects velocity changes of ~3.6 m/s per timestep (0.1 × 36.0 normalization)
- With measurement noise of 0.1 m/s, this provides good balance
- For your AC_SPD=9 m/s drones, D_VELOCITY=10/3 kts = 1.7 m/s changes are typical

TUNING PROCEDURE:
----------------
1. Run training with different values: [0.05, 0.1, 0.2, 0.5]
2. Compare:
   - Waypoint success rate (higher is better)  
   - Position error metrics (lower is better after convergence)
   - Episode reward (higher is better)
3. Choose value that maximizes waypoint success

VALUES TO TRY:
-------------
- 0.05: Very smooth, conservative (good for stable flight)
- 0.1:  Balanced (current default) ← START HERE
- 0.2:  Responsive (good for aggressive maneuvering)
- 0.5:  Very responsive (only if drones make sudden changes)

MODIFICATION:
------------
Edit in: bluesky_gym/envs/ma_env_two_stage_AM_PPO_NOISE_kalman_ASSYMETRIC.py

Search for: KalmanDenoiser(process_noise_std=0.1)
Change to:  KalmanDenoiser(process_noise_std=YOUR_VALUE)

(3 locations to change - all instances should match)
"""

import numpy as np
import matplotlib.pyplot as plt
from bluesky_gym.kalman_filter import KalmanDenoiser

def simulate_maneuver(process_noise_values=[0.05, 0.1, 0.2, 0.5]):
    """
    Simulate a drone maneuver with different process noise values.
    Shows how each setting tracks a sudden velocity change.
    """
    # Simulation parameters
    T = 50  # timesteps
    dt = 1.0
    
    # True trajectory: straight flight, then sudden turn at t=20
    true_x = np.zeros(T)
    true_y = np.zeros(T)
    true_vx = np.ones(T) * 9.0  # 9 m/s forward
    true_vy = np.zeros(T)
    
    # Sudden maneuver at t=20 (velocity change = 5 m/s)
    true_vx[20:] = 6.0
    true_vy[20:] = 4.0
    
    # Integrate to get positions
    for t in range(1, T):
        true_x[t] = true_x[t-1] + true_vx[t-1] * dt
        true_y[t] = true_y[t-1] + true_vy[t-1] * dt
    
    # Add measurement noise (realistic values from your environment)
    pos_noise = 3.5  # meters
    vel_noise = 0.1  # m/s
    
    noisy_x = true_x + np.random.normal(0, pos_noise, T)
    noisy_y = true_y + np.random.normal(0, vel_noise, T)
    noisy_vx = true_vx + np.random.normal(0, vel_noise, T)
    noisy_vy = true_vy + np.random.normal(0, vel_noise, T)
    
    # Test each process noise value
    results = {}
    
    for pn in process_noise_values:
        kf = KalmanDenoiser(
            process_noise_std=pn,
            pos_noise_std=pos_noise,
            vel_noise_std=vel_noise
        )
        
        filtered_x = []
        filtered_y = []
        filtered_vx = []
        filtered_vy = []
        
        for t in range(T):
            # Normalize
            obs_norm = np.array([
                noisy_x[t] / 8500.0,
                noisy_y[t] / 8000.0,
                noisy_vx[t] / 36.0,
                noisy_vy[t] / 36.0
            ], dtype=np.float32)
            
            if t == 0:
                kf.reset(obs_norm)
                # Apply predict-update immediately (like your fix)
                kf.predict()
                kf.update(obs_norm)
            else:
                kf.predict()
                kf.update(obs_norm)
            
            # Denormalize
            filtered_x.append(kf.x[0] * 8500.0)
            filtered_y.append(kf.x[1] * 8000.0)
            filtered_vx.append(kf.x[2] * 36.0)
            filtered_vy.append(kf.x[3] * 36.0)
        
        results[pn] = {
            'x': np.array(filtered_x),
            'y': np.array(filtered_y),
            'vx': np.array(filtered_vx),
            'vy': np.array(filtered_vy)
        }
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Kalman Filter Performance: Different Process Noise Values', fontsize=14)
    
    t_range = np.arange(T)
    
    # Velocity X comparison
    ax = axes[0, 0]
    ax.plot(t_range, true_vx, 'k-', linewidth=2, label='True', alpha=0.8)
    ax.plot(t_range, noisy_vx, 'gray', alpha=0.3, label='Noisy')
    for pn in process_noise_values:
        ax.plot(t_range, results[pn]['vx'], label=f'PN={pn}', linewidth=1.5)
    ax.axvline(20, color='red', linestyle='--', alpha=0.5, label='Maneuver')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Velocity X (m/s)')
    ax.set_title('Velocity Tracking (X component)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Velocity Y comparison  
    ax = axes[0, 1]
    ax.plot(t_range, true_vy, 'k-', linewidth=2, label='True', alpha=0.8)
    ax.plot(t_range, noisy_vy, 'gray', alpha=0.3, label='Noisy')
    for pn in process_noise_values:
        ax.plot(t_range, results[pn]['vy'], label=f'PN={pn}', linewidth=1.5)
    ax.axvline(20, color='red', linestyle='--', alpha=0.5, label='Maneuver')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Velocity Y (m/s)')
    ax.set_title('Velocity Tracking (Y component)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Position error over time
    ax = axes[1, 0]
    for pn in process_noise_values:
        pos_error = np.sqrt((results[pn]['x'] - true_x)**2 + (results[pn]['y'] - true_y)**2)
        ax.plot(t_range, pos_error, label=f'PN={pn}', linewidth=1.5)
    ax.axvline(20, color='red', linestyle='--', alpha=0.5, label='Maneuver')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Position Error (m)')
    ax.set_title('Position Tracking Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Velocity error over time
    ax = axes[1, 1]
    for pn in process_noise_values:
        vel_error = np.sqrt((results[pn]['vx'] - true_vx)**2 + (results[pn]['vy'] - true_vy)**2)
        ax.plot(t_range, vel_error, label=f'PN={pn}', linewidth=1.5)
    ax.axvline(20, color='red', linestyle='--', alpha=0.5, label='Maneuver')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Velocity Error (m/s)')
    ax.set_title('Velocity Tracking Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kalman_process_noise_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Plot saved: kalman_process_noise_comparison.png")
    plt.show()
    
    # Print statistics
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON (After Maneuver, t=21-50)")
    print("="*80)
    
    for pn in process_noise_values:
        # Compute errors after maneuver
        post_maneuver = slice(21, T)
        
        pos_err = np.sqrt((results[pn]['x'][post_maneuver] - true_x[post_maneuver])**2 + 
                          (results[pn]['y'][post_maneuver] - true_y[post_maneuver])**2)
        vel_err = np.sqrt((results[pn]['vx'][post_maneuver] - true_vx[post_maneuver])**2 + 
                          (results[pn]['vy'][post_maneuver] - true_vy[post_maneuver])**2)
        
        # Convergence time (timesteps to get within 10% of final velocity)
        target_vx = true_vx[21]
        target_vy = true_vy[21]
        threshold = 0.1 * np.sqrt(target_vx**2 + target_vy**2)
        
        convergence_idx = None
        for t in range(21, T):
            current_vel = np.sqrt(results[pn]['vx'][t]**2 + results[pn]['vy'][t]**2)
            target_vel = np.sqrt(target_vx**2 + target_vy**2)
            if abs(current_vel - target_vel) < threshold:
                convergence_idx = t - 20
                break
        
        convergence_time = convergence_idx if convergence_idx else "Did not converge"
        
        print(f"\nProcess Noise = {pn}:")
        print(f"  Position Error: {pos_err.mean():.2f} ± {pos_err.std():.2f} m")
        print(f"  Velocity Error: {vel_err.mean():.3f} ± {vel_err.std():.3f} m/s")
        print(f"  Convergence Time: {convergence_time} timesteps")
    
    print("\n" + "="*80)
    print("RECOMMENDATION:")
    print("="*80)
    print("Choose process noise that:")
    print("  1. Minimizes velocity error after convergence")
    print("  2. Has fast convergence time (< 5 timesteps)")
    print("  3. Balances smoothness vs responsiveness for YOUR task")
    print("\nFor collision avoidance with AC_SPD=9 m/s and D_VELOCITY=1.7 m/s:")
    print("  → process_noise_std = 0.1 is a good starting point")
    print("  → Try 0.05 if you want smoother tracking")
    print("  → Try 0.2 if drones make aggressive maneuvers")
    print("="*80 + "\n")

if __name__ == "__main__":
    print("Kalman Filter Process Noise Tuning")
    print("===================================\n")
    print("Simulating drone maneuver with different process noise values...")
    print("This will show how each setting tracks a sudden velocity change.\n")
    
    simulate_maneuver(process_noise_values=[0.05, 0.1, 0.2, 0.5])
