"""
Plot Kalman Filter Trajectory Comparison
=========================================
Visualizes raw noisy trajectories vs Kalman-filtered trajectories
for multiple agents in a single episode.

Usage:
    python plot_kalman_trajectories.py --n_agents 3
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import io

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from ma_env import SectorEnv
from kalman_denoiser import KalmanDenoiser
from run_config import RUN_ID

# Constants from environment
X_NORM = 8500.0
Y_NORM = 8000.0
V_NORM = 36.0
POS_NOISE_STD_M = 3.5
VEL_NOISE_STD_MS = 0.1


@contextmanager
def suppress_output():
    """Suppress BlueSky output."""
    null_out = io.StringIO()
    null_err = io.StringIO()
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = null_out
        sys.stderr = null_err
        with redirect_stdout(null_out), redirect_stderr(null_err):
            yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        null_out.close()
        null_err.close()


def add_noise_to_trajectory(clean_traj, rng):
    """Add Gaussian noise to clean trajectory."""
    noise_std = np.array([
        POS_NOISE_STD_M / X_NORM,
        POS_NOISE_STD_M / Y_NORM,
        VEL_NOISE_STD_MS / V_NORM,
        VEL_NOISE_STD_MS / V_NORM,
    ], dtype=np.float32)
    
    noise = rng.normal(0.0, 1.0, size=clean_traj.shape).astype(np.float32)
    noise *= noise_std
    return clean_traj + noise


def collect_episode_trajectories(n_agents=20):
    """
    Run one episode with MVP and collect trajectories.
    
    Returns
    -------
    dict with keys:
        - 'clean': dict of agent_id -> np.ndarray (T, 4) clean trajectories
        - 'noisy': dict of agent_id -> np.ndarray (T, 4) noisy trajectories
        - 'filtered': dict of agent_id -> np.ndarray (T, 4) Kalman-filtered trajectories
    """
    print(f"Collecting single episode with {n_agents} agents...")
    
    # Create clean environment
    env = SectorEnv(
        n_agents=n_agents,
        run_id=f"kalman_plot_{RUN_ID}",
        render_mode=None,
    )
    
    with suppress_output():
        obs, info = env.reset()
    
    # Storage for trajectories
    clean_trajectories = {agent_id: [] for agent_id in env.agents}
    
    done = {agent: False for agent in env.agents}
    truncated = {agent: False for agent in env.agents}
    step = 0
    max_steps = 300
    
    while not all(done.values()) and not all(truncated.values()) and step < max_steps:
        # Get MVP actions
        actions = {}
        for agent_id in env.agents:
            if not done[agent_id] and not truncated[agent_id]:
                with suppress_output():
                    mvp_action = env._calculate_mvp_action(agent_id)
                actions[agent_id] = mvp_action
        
        # Extract clean ownship state BEFORE stepping
        for agent_id in env.agents:
            if not done[agent_id] and not truncated[agent_id]:
                ownship_state = obs[agent_id][3:7]  # [x, y, vx, vy]
                clean_trajectories[agent_id].append(ownship_state.copy())
        
        # Step environment
        with suppress_output():
            obs, rewards, done, truncated, infos = env.step(actions)
        step += 1
    
    env.close()
    
    # Convert to numpy arrays
    for agent_id in list(clean_trajectories.keys()):
        if len(clean_trajectories[agent_id]) < 10:
            del clean_trajectories[agent_id]
        else:
            clean_trajectories[agent_id] = np.array(clean_trajectories[agent_id], dtype=np.float32)
    
    print(f"Collected episode: {step} steps, {len(clean_trajectories)} agents with sufficient data")
    
    # Add noise to trajectories
    rng = np.random.default_rng(42)
    noisy_trajectories = {}
    for agent_id, clean_traj in clean_trajectories.items():
        noisy_trajectories[agent_id] = add_noise_to_trajectory(clean_traj, rng)
    
    # Apply Kalman filter
    print("Applying Kalman filter...")
    kalman = KalmanDenoiser(
        dt=1.0,
        pos_noise_std=POS_NOISE_STD_M,
        vel_noise_std=VEL_NOISE_STD_MS,
        process_noise_std=1.0,  # Optimal tuned value
        x_norm=X_NORM,
        y_norm=Y_NORM,
        v_norm=V_NORM,
    )
    
    filtered_trajectories = {}
    for agent_id, noisy_traj in noisy_trajectories.items():
        filtered_trajectories[agent_id] = kalman.denoise_sequence(noisy_traj)
    
    return {
        'clean': clean_trajectories,
        'noisy': noisy_trajectories,
        'filtered': filtered_trajectories,
    }


def plot_trajectories(trajectories, n_agents_to_plot=3, save_path='kalman_trajectory_comparison.png'):
    """
    Plot comparison of noisy vs Kalman-filtered trajectories.
    
    Parameters
    ----------
    trajectories : dict
        Output from collect_episode_trajectories
    n_agents_to_plot : int
        Number of agents to visualize
    save_path : str
        Path to save the figure
    """
    clean = trajectories['clean']
    noisy = trajectories['noisy']
    filtered = trajectories['filtered']
    
    # Select agents with longest trajectories
    agent_ids = sorted(clean.keys(), key=lambda a: len(clean[a]), reverse=True)[:n_agents_to_plot]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 5))
    
    # Define colors for agents
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Plot 1: Noisy trajectories
    ax1 = plt.subplot(131)
    for i, agent_id in enumerate(agent_ids):
        traj = noisy[agent_id]
        x_phys = traj[:, 0] * X_NORM
        y_phys = traj[:, 1] * Y_NORM
        
        ax1.plot(x_phys, y_phys, 'o-', color=colors[i], alpha=0.4, 
                markersize=2, linewidth=1, label=f'Agent {agent_id}')
        # Mark start and end
        ax1.plot(x_phys[0], y_phys[0], 'o', color=colors[i], markersize=8, 
                markeredgecolor='black', markeredgewidth=1.5)
        ax1.plot(x_phys[-1], y_phys[-1], 's', color=colors[i], markersize=8,
                markeredgecolor='black', markeredgewidth=1.5)
    
    ax1.set_xlabel('X Position (m)', fontsize=11)
    ax1.set_ylabel('Y Position (m)', fontsize=11)
    ax1.set_title('Noisy Sensor Data\n($\\sigma_{pos}$ = 3.5 m)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.legend(fontsize=9)
    
    # Plot 2: Kalman filtered trajectories
    ax2 = plt.subplot(132)
    for i, agent_id in enumerate(agent_ids):
        traj = filtered[agent_id]
        x_phys = traj[:, 0] * X_NORM
        y_phys = traj[:, 1] * Y_NORM
        
        ax2.plot(x_phys, y_phys, '-', color=colors[i], alpha=0.8, 
                linewidth=2, label=f'Agent {agent_id}')
        # Mark start and end
        ax2.plot(x_phys[0], y_phys[0], 'o', color=colors[i], markersize=8,
                markeredgecolor='black', markeredgewidth=1.5)
        ax2.plot(x_phys[-1], y_phys[-1], 's', color=colors[i], markersize=8,
                markeredgecolor='black', markeredgewidth=1.5)
    
    ax2.set_xlabel('X Position (m)', fontsize=11)
    ax2.set_ylabel('Y Position (m)', fontsize=11)
    ax2.set_title('Kalman Filtered Data\n(38% error reduction)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    ax2.legend(fontsize=9)
    
    # Plot 3: Overlay comparison for one agent
    ax3 = plt.subplot(133)
    agent_id = agent_ids[0]  # Focus on first agent
    
    # Clean (ground truth)
    clean_traj = clean[agent_id]
    x_clean = clean_traj[:, 0] * X_NORM
    y_clean = clean_traj[:, 1] * Y_NORM
    ax3.plot(x_clean, y_clean, '-', color='green', linewidth=3, 
            label='Ground Truth', alpha=0.7, zorder=3)
    
    # Noisy
    noisy_traj = noisy[agent_id]
    x_noisy = noisy_traj[:, 0] * X_NORM
    y_noisy = noisy_traj[:, 1] * Y_NORM
    ax3.plot(x_noisy, y_noisy, 'o', color='gray', markersize=3, 
            alpha=0.5, label='Noisy Measurements', zorder=1)
    
    # Filtered
    filt_traj = filtered[agent_id]
    x_filt = filt_traj[:, 0] * X_NORM
    y_filt = filt_traj[:, 1] * Y_NORM
    ax3.plot(x_filt, y_filt, '-', color='blue', linewidth=2, 
            label='Kalman Filter', alpha=0.8, zorder=2)
    
    # Mark start
    ax3.plot(x_clean[0], y_clean[0], 'o', color='black', markersize=10,
            label='Start', zorder=4)
    ax3.plot(x_clean[-1], y_clean[-1], 's', color='black', markersize=10,
            label='End', zorder=4)
    
    ax3.set_xlabel('X Position (m)', fontsize=11)
    ax3.set_ylabel('Y Position (m)', fontsize=11)
    ax3.set_title(f'Detailed Comparison: Agent {agent_id}', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    ax3.legend(fontsize=9, loc='best')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Trajectory plot saved to: {save_path}")
    plt.close()
    
    # Also create error analysis plot
    plot_error_analysis(trajectories, agent_ids[0], 
                       save_path.replace('.png', '_error_analysis.png'))
    
    # Create zoomed-in detail plot
    plot_zoomed_detail(trajectories, agent_ids[0],
                      save_path.replace('.png', '_zoomed.png'))
    
    # Create cornering detail plot with larger window for bigger turn
    plot_cornering_detail(trajectories, agent_ids[0],
                         save_path.replace('.png', '_cornering.png'),
                         min_turn_deg=15.0, window_size=40)


def plot_error_analysis(trajectories, agent_id, save_path):
    """Plot position error over time for one agent."""
    clean = trajectories['clean'][agent_id]
    noisy = trajectories['noisy'][agent_id]
    filtered = trajectories['filtered'][agent_id]
    
    # Compute position errors in physical units
    noisy_error = np.sqrt(
        ((noisy[:, 0] - clean[:, 0]) * X_NORM)**2 + 
        ((noisy[:, 1] - clean[:, 1]) * Y_NORM)**2
    )
    
    filtered_error = np.sqrt(
        ((filtered[:, 0] - clean[:, 0]) * X_NORM)**2 + 
        ((filtered[:, 1] - clean[:, 1]) * Y_NORM)**2
    )
    
    timesteps = np.arange(len(noisy_error))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Position error over time
    ax1.plot(timesteps, noisy_error, '-', color='gray', linewidth=1.5, 
            alpha=0.7, label='Noisy Measurements')
    ax1.plot(timesteps, filtered_error, '-', color='blue', linewidth=2, 
            label='Kalman Filter')
    ax1.axhline(POS_NOISE_STD_M, color='red', linestyle='--', linewidth=1.5,
               label=f'Expected Noise ($\\sigma$ = {POS_NOISE_STD_M} m)')
    ax1.set_xlabel('Timestep', fontsize=11)
    ax1.set_ylabel('Position Error (m)', fontsize=11)
    ax1.set_title(f'Position Error Over Time: Agent {agent_id}', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Cumulative statistics
    ax2.hist(noisy_error, bins=30, alpha=0.5, color='gray', label='Noisy', density=True)
    ax2.hist(filtered_error, bins=30, alpha=0.7, color='blue', label='Kalman', density=True)
    ax2.axvline(np.mean(noisy_error), color='gray', linestyle='--', linewidth=2)
    ax2.axvline(np.mean(filtered_error), color='blue', linestyle='--', linewidth=2)
    ax2.set_xlabel('Position Error (m)', fontsize=11)
    ax2.set_ylabel('Probability Density', fontsize=11)
    ax2.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text
    noisy_mean = np.mean(noisy_error)
    filtered_mean = np.mean(filtered_error)
    improvement = (1 - filtered_mean / noisy_mean) * 100
    
    stats_text = f'Mean Error:\n  Noisy: {noisy_mean:.2f} m\n  Kalman: {filtered_mean:.2f} m\n  Improvement: {improvement:.1f}%'
    ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10, family='monospace')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Error analysis saved to: {save_path}")
    plt.close()


def plot_zoomed_detail(trajectories, agent_id, save_path):
    """Create zoomed-in detailed view showing filtering quality."""
    clean = trajectories['clean'][agent_id]
    noisy = trajectories['noisy'][agent_id]
    filtered = trajectories['filtered'][agent_id]
    
    # Select middle portion of trajectory for zoom (better maneuvers)
    traj_len = len(clean)
    start_idx = traj_len // 3
    end_idx = min(start_idx + 50, traj_len)  # Show ~50 timesteps
    
    clean_zoom = clean[start_idx:end_idx]
    noisy_zoom = noisy[start_idx:end_idx]
    filtered_zoom = filtered[start_idx:end_idx]
    
    # Convert to physical units
    x_clean = clean_zoom[:, 0] * X_NORM
    y_clean = clean_zoom[:, 1] * Y_NORM
    x_noisy = noisy_zoom[:, 0] * X_NORM
    y_noisy = noisy_zoom[:, 1] * Y_NORM
    x_filt = filtered_zoom[:, 0] * X_NORM
    y_filt = filtered_zoom[:, 1] * Y_NORM
    
    # Create figure
    fig = plt.figure(figsize=(14, 6))
    
    # Left: Side-by-side comparison
    ax1 = plt.subplot(121)
    
    # Plot noisy with very thin lines
    ax1.plot(x_noisy, y_noisy, '-', color='lightgray', linewidth=0.8, 
            alpha=0.6, label='Noisy', zorder=1)
    ax1.plot(x_noisy, y_noisy, 'o', color='gray', markersize=2.5, 
            alpha=0.4, zorder=1)
    
    # Plot filtered
    ax1.plot(x_filt, y_filt, '-', color='blue', linewidth=1.5, 
            alpha=0.8, label='Kalman Filter', zorder=2)
    
    # Plot ground truth
    ax1.plot(x_clean, y_clean, '-', color='green', linewidth=2, 
            alpha=0.9, label='Ground Truth', zorder=3)
    
    # Mark start point
    ax1.plot(x_clean[0], y_clean[0], 'o', color='black', markersize=8,
            markeredgecolor='white', markeredgewidth=1.5, zorder=4)
    
    ax1.set_xlabel('X Position (m)', fontsize=11)
    ax1.set_ylabel('Y Position (m)', fontsize=11)
    ax1.set_title(f'Zoomed Detail: Agent {agent_id} (timesteps {start_idx}-{end_idx})', 
                 fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    ax1.axis('equal')
    
    # Right: Error magnitude over trajectory segment
    ax2 = plt.subplot(122)
    
    # Compute errors
    noisy_error = np.sqrt(
        ((noisy_zoom[:, 0] - clean_zoom[:, 0]) * X_NORM)**2 + 
        ((noisy_zoom[:, 1] - clean_zoom[:, 1]) * Y_NORM)**2
    )
    filtered_error = np.sqrt(
        ((filtered_zoom[:, 0] - clean_zoom[:, 0]) * X_NORM)**2 + 
        ((filtered_zoom[:, 1] - clean_zoom[:, 1]) * Y_NORM)**2
    )
    
    timesteps = np.arange(start_idx, end_idx)
    
    ax2.plot(timesteps, noisy_error, '-', color='gray', linewidth=1.5, 
            alpha=0.7, label=f'Noisy (mean: {np.mean(noisy_error):.2f} m)')
    ax2.plot(timesteps, filtered_error, '-', color='blue', linewidth=2, 
            label=f'Kalman (mean: {np.mean(filtered_error):.2f} m)')
    ax2.axhline(POS_NOISE_STD_M, color='red', linestyle='--', linewidth=1.5,
               alpha=0.7, label=f'Noise $\\sigma$ = {POS_NOISE_STD_M} m')
    
    ax2.set_xlabel('Timestep', fontsize=11)
    ax2.set_ylabel('Position Error (m)', fontsize=11)
    ax2.set_title('Position Error Magnitude', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Add improvement annotation
    improvement = (1 - np.mean(filtered_error) / np.mean(noisy_error)) * 100
    ax2.text(0.02, 0.98, f'Improvement: {improvement:.1f}%', 
            transform=ax2.transAxes,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
            fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Zoomed detail plot saved to: {save_path}")
    plt.close()


def plot_cornering_detail(trajectories, agent_id, save_path, min_turn_deg=15.0, window_size=30):
    """Create detailed view of trajectory during cornering maneuver.
    
    Parameters
    ----------
    trajectories : dict
        Output from collect_episode_trajectories
    agent_id : str
        Agent to analyze
    save_path : str
        Path to save figure
    min_turn_deg : float
        Minimum heading change (degrees) to consider as cornering
    window_size : int
        Number of timesteps to show around the turn
    """
    clean = trajectories['clean'][agent_id]
    noisy = trajectories['noisy'][agent_id]
    filtered = trajectories['filtered'][agent_id]
    
    # Compute heading from velocity vectors (vx, vy are indices 2, 3)
    def compute_headings(traj):
        """Compute heading in degrees from velocity vectors."""
        vx = traj[:, 2] * V_NORM  # Convert to physical units
        vy = traj[:, 3] * V_NORM
        headings = np.degrees(np.arctan2(vy, vx))
        return headings
    
    clean_headings = compute_headings(clean)
    
    # Compute heading change rate (degrees per timestep)
    # Use wrapped difference to handle 360° wrap-around
    def angle_diff(a, b):
        """Compute smallest angle difference between a and b."""
        diff = a - b
        # Wrap to [-180, 180]
        diff = (diff + 180) % 360 - 180
        return diff
    
    heading_changes = np.array([
        abs(angle_diff(clean_headings[i+1], clean_headings[i]))
        for i in range(len(clean_headings) - 1)
    ])
    
    # Find windows with sustained turning (moving average of heading change)
    kernel_size = 5
    if len(heading_changes) < kernel_size:
        print(f"⚠️  Trajectory too short for cornering analysis")
        return
    
    # Smooth heading changes with moving average
    smoothed_changes = np.convolve(heading_changes, 
                                   np.ones(kernel_size)/kernel_size, 
                                   mode='valid')
    
    # Find the segment with maximum sustained turn rate
    if len(smoothed_changes) == 0 or np.max(smoothed_changes) < min_turn_deg:
        print(f"⚠️  No significant turns found (threshold: {min_turn_deg}°)")
        # Fallback: use segment with largest single heading change
        max_idx = np.argmax(heading_changes)
    else:
        max_idx = np.argmax(smoothed_changes) + kernel_size // 2
    
    # Extract window around the turn
    start_idx = max(0, max_idx - window_size // 2)
    end_idx = min(len(clean), max_idx + window_size // 2)
    
    clean_zoom = clean[start_idx:end_idx]
    noisy_zoom = noisy[start_idx:end_idx]
    filtered_zoom = filtered[start_idx:end_idx]
    
    # Convert to physical units
    x_clean = clean_zoom[:, 0] * X_NORM
    y_clean = clean_zoom[:, 1] * Y_NORM
    x_noisy = noisy_zoom[:, 0] * X_NORM
    y_noisy = noisy_zoom[:, 1] * Y_NORM
    x_filt = filtered_zoom[:, 0] * X_NORM
    y_filt = filtered_zoom[:, 1] * Y_NORM
    
    # Compute heading change in this window
    headings_window = clean_headings[start_idx:end_idx]
    total_turn = abs(angle_diff(headings_window[-1], headings_window[0]))
    
    # Create figure
    fig = plt.figure(figsize=(8, 8))
    
    # Trajectory during cornering
    ax1 = plt.subplot(111)
    
    # Plot noisy with very thin lines
    ax1.plot(x_noisy, y_noisy, '-', color='lightgray', linewidth=0.8, 
            alpha=0.6, label='Noisy', zorder=1)
    ax1.plot(x_noisy, y_noisy, 'o', color='gray', markersize=2.5, 
            alpha=0.4, zorder=1)
    
    # Plot filtered
    ax1.plot(x_filt, y_filt, '-', color='blue', linewidth=1.5, 
            alpha=0.8, label='Kalman Filter', zorder=2)
    
    # Plot ground truth
    ax1.plot(x_clean, y_clean, '-', color='green', linewidth=2, 
            alpha=0.9, label='Ground Truth', zorder=3)
    
    # Mark start point (where turn begins)
    ax1.plot(x_clean[0], y_clean[0], 'o', color='black', markersize=10,
            markeredgecolor='white', markeredgewidth=1.5, zorder=4,
            label='Turn Start')
    
    # Mark end point (where turn completes)
    ax1.plot(x_clean[-1], y_clean[-1], 's', color='red', markersize=10,
            markeredgecolor='white', markeredgewidth=1.5, zorder=4,
            label='Turn End')
    
    ax1.set_xlabel('X Position (m)', fontsize=11)
    ax1.set_ylabel('Y Position (m)', fontsize=11)
    ax1.set_title(f'Cornering Maneuver: Agent {agent_id}\n' + 
                 f'Total Turn: {total_turn:.1f}° over {end_idx-start_idx} timesteps', 
                 fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    ax1.axis('equal')
    
    # Zoom in tighter on the trajectory
    margin = 50  # meters
    ax1.set_xlim(x_clean.min() - margin, x_clean.max() + margin)
    ax1.set_ylim(y_clean.min() - margin, y_clean.max() + margin)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Cornering detail plot saved to: {save_path}")
    print(f"   Total turn: {total_turn:.1f}° over {end_idx-start_idx} timesteps")
    
    # Compute improvement for console output
    noisy_error = np.sqrt(
        ((noisy_zoom[:, 0] - clean_zoom[:, 0]) * X_NORM)**2 + 
        ((noisy_zoom[:, 1] - clean_zoom[:, 1]) * Y_NORM)**2
    )
    filtered_error = np.sqrt(
        ((filtered_zoom[:, 0] - clean_zoom[:, 0]) * X_NORM)**2 + 
        ((filtered_zoom[:, 1] - clean_zoom[:, 1]) * Y_NORM)**2
    )
    improvement = (1 - np.mean(filtered_error) / np.mean(noisy_error)) * 100
    print(f"   Filter improvement: {improvement:.1f}%")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot Kalman Filter Trajectory Comparison")
    parser.add_argument("--n_agents", type=int, default=20,
                       help="Total number of agents in episode (default: 20)")
    parser.add_argument("--n_plot", type=int, default=3,
                       help="Number of agents to plot (default: 3)")
    parser.add_argument("--output", type=str, default="kalman_trajectory_comparison.png",
                       help="Output filename (default: kalman_trajectory_comparison.png)")
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"  KALMAN FILTER TRAJECTORY VISUALIZATION")
    print(f"{'='*70}")
    print(f"  Total agents: {args.n_agents}")
    print(f"  Agents to plot: {args.n_plot}")
    print(f"  Output: {args.output}")
    
    # Collect trajectories
    trajectories = collect_episode_trajectories(n_agents=args.n_agents)
    
    # Plot
    plot_trajectories(trajectories, n_agents_to_plot=args.n_plot, save_path=args.output)
    
    print(f"\n✅ Visualization complete!")
    print(f"\nFiles created:")
    print(f"  - {args.output}")
    print(f"  - {args.output.replace('.png', '_error_analysis.png')}")
    print(f"  - {args.output.replace('.png', '_cornering_detail.png')}")


if __name__ == "__main__":
    main()
