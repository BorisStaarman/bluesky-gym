"""
Evaluate LSTM Denoiser with MVP Controller
===========================================
Tests the effectiveness of LSTM denoising by running the MVP (Minimum Vector
Projection) controller with two configurations:
    1. MVP with Noisy Data (no LSTM)
    2. MVP with LSTM-Denoised Data

Metrics tracked:
    - Episode Length
    - Total Intrusions
    - Waypoints Reached
    - Collisions
    - Average Distance to Waypoint

This demonstrates the LSTM's ability to improve decision quality by providing
cleaner state estimates to the controller.

Usage:
    python evaluate_lstm_mvp.py --episodes 50 --render False
"""

import os
import sys
import argparse
import numpy as np
import csv
import time
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import io

# Add script directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from ma_env_lstm import SectorEnvLSTM
from run_config import RUN_ID


@contextmanager
def suppress_output():
    """Context manager to suppress BlueSky verbose output."""
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


def evaluate_mvp_with_config(
    n_episodes: int,
    n_agents: int,
    use_lstm: bool,
    render: bool = False,
    verbose: bool = True,
    denoiser_path: str = None,
):
    """
    Run episodes using MVP controller with specified configuration.
    
    Parameters
    ----------
    n_episodes : int
        Number of episodes to evaluate
    n_agents : int
        Number of agents in environment
    use_lstm : bool
        If True, use LSTM denoising; if False, use raw noisy observations
    render : bool
        Whether to render the environment
    verbose : bool
        Whether to print per-episode results
    
    Returns
    -------
    dict
        Dictionary with aggregated metrics
    """
    config_name = "MVP + LSTM Denoiser" if use_lstm else "MVP + Noisy Data"
    if use_lstm and denoiser_path and 'kalman' in denoiser_path.lower():
        config_name = "MVP + Kalman Filter"
    print(f"\n{'='*70}")
    print(f"  Evaluating: {config_name}")
    print(f"{'='*70}")
    
    # Create environment
    env = SectorEnvLSTM(
        n_agents=n_agents,
        run_id=f"lstm_eval_{RUN_ID}",
        noise_enabled=True,              # Always add noise
        use_denoiser=use_lstm,           # Toggle LSTM denoising
        add_intruder_noise=True,         # Noise on intruder observations too
        denoiser_path=denoiser_path,     # Custom path (e.g., for Kalman filter)
        render_mode="human" if render else None,
    )
    
    # Metrics storage
    episode_lengths = []
    episode_intrusions = []
    episode_waypoints = []
    episode_collisions = []
    episode_avg_distance = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = {agent: False for agent in env.agents}
        truncated = {agent: False for agent in env.agents}
        
        ep_length = 0
        ep_distance_sum = 0.0
        step_count = 0
        
        while not all(done.values()) and not all(truncated.values()):
            # Get MVP actions for all agents
            actions = {}
            for agent_id in env.agents:
                if not done[agent_id] and not truncated[agent_id]:
                    # Use the environment's built-in MVP calculator
                    # This uses the observations that have already been processed
                    # through the noise + LSTM pipeline
                    mvp_action = env._calculate_mvp_action(agent_id)
                    actions[agent_id] = mvp_action
            
            # Step environment
            obs, rewards, done, truncated, infos = env.step(actions)
            ep_length += 1
            step_count += 1
            
            # Track average distance to waypoint (for progress metric)
            for agent_id in env.agents:
                if agent_id in infos and "dist_to_waypoint" in infos[agent_id]:
                    ep_distance_sum += infos[agent_id]["dist_to_waypoint"]
            
            if render:
                time.sleep(0.05)
        
        # Collect episode metrics
        episode_lengths.append(ep_length)
        episode_intrusions.append(env.total_intrusions)
        episode_waypoints.append(len(env.waypoint_reached_agents))
        episode_collisions.append(getattr(env, 'total_collisions', 0))
        
        # Average distance per step per agent
        if step_count > 0:
            avg_dist = ep_distance_sum / (step_count * n_agents)
            episode_avg_distance.append(avg_dist)
        else:
            episode_avg_distance.append(0.0)
        
        if verbose and (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{n_episodes} | "
                  f"Length: {ep_length:3d} | "
                  f"Intrusions: {env.total_intrusions:2d} | "
                  f"Waypoints: {len(env.waypoint_reached_agents):2d}/{n_agents}")
    
    env.close()
    
    # Compute statistics
    results = {
        "config": config_name,
        "use_lstm": use_lstm,
        "n_episodes": n_episodes,
        "avg_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
        "avg_intrusions": float(np.mean(episode_intrusions)),
        "std_intrusions": float(np.std(episode_intrusions)),
        "q25_intrusions": float(np.percentile(episode_intrusions, 25)),
        "q75_intrusions": float(np.percentile(episode_intrusions, 75)),
        "avg_waypoints": float(np.mean(episode_waypoints)),
        "std_waypoints": float(np.std(episode_waypoints)),
        "waypoint_rate": float(np.sum(episode_waypoints)) / (n_episodes * n_agents),
        "avg_collisions": float(np.mean(episode_collisions)),
        "avg_distance_to_waypoint": float(np.mean(episode_avg_distance)),
        # Raw data
        "per_episode_lengths": episode_lengths,
        "per_episode_intrusions": episode_intrusions,
        "per_episode_waypoints": episode_waypoints,
        "per_episode_collisions": episode_collisions,
    }
    
    return results


def print_comparison(results_noisy, results_lstm):
    """Print comparison table between noisy and LSTM results."""
    print(f"\n{'='*70}")
    print(f"  COMPARISON: MVP Performance with Noisy vs LSTM-Denoised Data")
    print(f"{'='*70}")
    print(f"{'Metric':<30} | {'Noisy':>15} | {'LSTM':>15} | {'Improvement':>12}")
    print(f"{'-'*70}")
    
    # Episode Length
    len_noisy = results_noisy["avg_length"]
    len_lstm = results_lstm["avg_length"]
    len_change = ((len_lstm - len_noisy) / len_noisy * 100) if len_noisy > 0 else 0
    print(f"{'Avg Episode Length':<30} | {len_noisy:>15.2f} | {len_lstm:>15.2f} | {len_change:>10.1f}%")
    
    # Intrusions (lower is better)
    intr_noisy = results_noisy["avg_intrusions"]
    intr_lstm = results_lstm["avg_intrusions"]
    intr_change = ((intr_noisy - intr_lstm) / intr_noisy * 100) if intr_noisy > 0 else 0
    print(f"{'Avg Intrusions (↓ better)':<30} | {intr_noisy:>15.2f} | {intr_lstm:>15.2f} | {intr_change:>10.1f}%")
    
    # IQR for intrusions
    q25_noisy = results_noisy["q25_intrusions"]
    q75_noisy = results_noisy["q75_intrusions"]
    q25_lstm = results_lstm["q25_intrusions"]
    q75_lstm = results_lstm["q75_intrusions"]
    print(f"{'Intrusions Q25-Q75':<30} | {q25_noisy:>6.1f}-{q75_noisy:<7.1f} | {q25_lstm:>6.1f}-{q75_lstm:<7.1f} |")
    
    # Waypoint Rate
    wp_noisy = results_noisy["waypoint_rate"] * 100
    wp_lstm = results_lstm["waypoint_rate"] * 100
    wp_change = ((wp_lstm - wp_noisy) / wp_noisy * 100) if wp_noisy > 0 else 0
    print(f"{'Waypoint Success Rate %':<30} | {wp_noisy:>15.2f} | {wp_lstm:>15.2f} | {wp_change:>10.1f}%")
    
    # Collisions
    col_noisy = results_noisy["avg_collisions"]
    col_lstm = results_lstm["avg_collisions"]
    col_change = ((col_noisy - col_lstm) / col_noisy * 100) if col_noisy > 0 else 0
    print(f"{'Avg Collisions (↓ better)':<30} | {col_noisy:>15.2f} | {col_lstm:>15.2f} | {col_change:>10.1f}%")
    
    # Distance to Waypoint
    dist_noisy = results_noisy["avg_distance_to_waypoint"]
    dist_lstm = results_lstm["avg_distance_to_waypoint"]
    dist_change = ((dist_noisy - dist_lstm) / dist_noisy * 100) if dist_noisy > 0 else 0
    print(f"{'Avg Dist to Waypoint (↓)':<30} | {dist_noisy:>15.4f} | {dist_lstm:>15.4f} | {dist_change:>10.1f}%")
    
    print(f"{'='*70}")
    
    # Overall assessment
    improvements = []
    if intr_change > 0:
        improvements.append(f"Intrusions reduced by {intr_change:.1f}%")
    if wp_change > 0:
        improvements.append(f"Waypoints improved by {wp_change:.1f}%")
    if col_change > 0:
        improvements.append(f"Collisions reduced by {col_change:.1f}%")
    
    if improvements:
        print(f"\n✅ LSTM Denoising Benefits:")
        for imp in improvements:
            print(f"   • {imp}")
    else:
        print(f"\n⚠️  No significant improvement observed from LSTM denoising")
    print()


def save_results_to_csv(results_noisy, results_lstm, output_path):
    """Save detailed results to CSV file."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write summary statistics
        writer.writerow(["SUMMARY STATISTICS"])
        writer.writerow(["Metric", "Noisy MVP", "LSTM MVP", "Improvement %"])
        writer.writerow(["Avg Episode Length", 
                        f"{results_noisy['avg_length']:.2f}",
                        f"{results_lstm['avg_length']:.2f}",
                        f"{((results_lstm['avg_length'] - results_noisy['avg_length']) / results_noisy['avg_length'] * 100):.1f}"])
        writer.writerow(["Avg Intrusions", 
                        f"{results_noisy['avg_intrusions']:.2f}",
                        f"{results_lstm['avg_intrusions']:.2f}",
                        f"{((results_noisy['avg_intrusions'] - results_lstm['avg_intrusions']) / results_noisy['avg_intrusions'] * 100):.1f}"])
        writer.writerow(["Waypoint Rate %", 
                        f"{results_noisy['waypoint_rate']*100:.2f}",
                        f"{results_lstm['waypoint_rate']*100:.2f}",
                        f"{((results_lstm['waypoint_rate'] - results_noisy['waypoint_rate']) / results_noisy['waypoint_rate'] * 100):.1f}"])
        writer.writerow(["Avg Collisions", 
                        f"{results_noisy['avg_collisions']:.2f}",
                        f"{results_lstm['avg_collisions']:.2f}",
                        f"{((results_noisy['avg_collisions'] - results_lstm['avg_collisions']) / results_noisy['avg_collisions'] * 100):.1f}"])
        writer.writerow([])
        
        # Write per-episode data
        writer.writerow(["PER-EPISODE DATA"])
        writer.writerow(["Episode", "Config", "Length", "Intrusions", "Waypoints", "Collisions"])
        
        for i in range(results_noisy["n_episodes"]):
            writer.writerow([i+1, "Noisy", 
                           results_noisy["per_episode_lengths"][i],
                           results_noisy["per_episode_intrusions"][i],
                           results_noisy["per_episode_waypoints"][i],
                           results_noisy["per_episode_collisions"][i]])
        
        for i in range(results_lstm["n_episodes"]):
            writer.writerow([i+1, "LSTM", 
                           results_lstm["per_episode_lengths"][i],
                           results_lstm["per_episode_intrusions"][i],
                           results_lstm["per_episode_waypoints"][i],
                           results_lstm["per_episode_collisions"][i]])
    
    print(f"📊 Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate LSTM Denoiser with MVP Controller")
    parser.add_argument("--episodes", type=int, default=50,
                       help="Number of episodes to evaluate (default: 50)")
    parser.add_argument("--n_agents", type=int, default=20,
                       help="Number of agents (default: 20)")
    parser.add_argument("--render", action="store_true",
                       help="Enable rendering (slower)")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Print per-episode progress")
    parser.add_argument("--output", type=str, default="lstm_mvp_evaluation.csv",
                       help="Output CSV file path (default: lstm_mvp_evaluation.csv)")
    parser.add_argument("--silent", action="store_true",
                       help="Suppress BlueSky output completely")
    parser.add_argument("--denoiser_path", type=str, default=None,
                       help="Path to denoiser model (.pt for LSTM, .npz for Kalman)")
    
    args = parser.parse_args()
    
    print(f"\n🎯 LSTM Denoiser Evaluation with MVP Controller")
    print(f"   Episodes: {args.episodes}")
    print(f"   Agents: {args.n_agents}")
    print(f"   Render: {args.render}")
    print(f"   Output: {args.output}")
    
    # Run both configurations
    def run_evaluations():
        # 1. MVP with Noisy Data (no LSTM)
        results_noisy = evaluate_mvp_with_config(
            n_episodes=args.episodes,
            n_agents=args.n_agents,
            use_lstm=False,
            render=args.render,
            verbose=args.verbose,
        )
        
        # 2. MVP with LSTM-Denoised Data
        results_lstm = evaluate_mvp_with_config(
            n_episodes=args.episodes,
            n_agents=args.n_agents,
            use_lstm=True,
            render=args.render,
            denoiser_path=args.denoiser_path,
            verbose=args.verbose,
        )
        
        return results_noisy, results_lstm
    
    # Run with or without output suppression
    if args.silent:
        with suppress_output():
            results_noisy, results_lstm = run_evaluations()
    else:
        results_noisy, results_lstm = run_evaluations()
    
    # Print comparison
    print_comparison(results_noisy, results_lstm)
    
    # Save results
    save_results_to_csv(results_noisy, results_lstm, args.output)
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
