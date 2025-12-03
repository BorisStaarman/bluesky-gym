"""
Quick script to check if observation normalization is good.
Runs a few episodes and collects statistics about observation values.
"""
import os
import sys
import numpy as np

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, repo_root)

from bluesky_gym import register_envs
from bluesky_gym.envs.ma_env_ppo import SectorEnv
from run_config import RUN_ID

# Register environments
register_envs()

def check_normalization(n_agents=6, n_episodes=5):
    """
    Run a few episodes and check observation value ranges.
    
    Args:
        n_agents: Number of agents
        n_episodes: Number of episodes to run for statistics
    """
    print(f"\n{'='*80}")
    print(f"Checking Observation Normalization")
    print(f"Running {n_episodes} episodes with {n_agents} agents...")
    print(f"{'='*80}\n")
    
    # Create environment with stats collection enabled
    metrics_base = os.path.join(repo_root, "metrics_28_10")
    env = SectorEnv(
        n_agents=n_agents,
        run_id=f"{RUN_ID}_norm_check",
        collect_obs_stats=True,  # Enable statistics collection
        print_obs_stats_per_episode=False,  # Don't print each episode
        metrics_base_dir=metrics_base
    )
    
    # Run episodes with random actions
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        step_count = 0
        
        while not done:
            # Random actions for all agents
            actions = {
                agent: env.action_space[agent].sample() 
                for agent in env.agents
            }
            
            obs, rewards, terminateds, truncateds, infos = env.step(actions)
            step_count += 1
            
            # Check if all agents are done
            done = all(terminateds.values()) or all(truncateds.values())
            
            if step_count > 500:  # Safety limit
                break
        
        print(f"  Episode {ep+1}/{n_episodes} completed ({step_count} steps)")
    
    print("\n" + "="*80)
    print("Observation Statistics Summary")
    print("="*80 + "\n")
    
    # Get statistics
    stats = env._obs_stats
    if stats.get("count", 0) == 0:
        print("No statistics collected!")
        return
    
    n = stats["count"]
    mean = stats["mean"]
    variance = stats["M2"] / n
    std = np.sqrt(variance)
    min_vals = stats["min"]
    max_vals = stats["max"]
    
    # Define feature names (must match your observation vector structure)
    feature_names = [
        "own_x", "own_y", "own_dist_to_goal",
        # Other aircraft features (repeat for NUM_AC_STATE agents)
    ]
    
    # Auto-generate names if we have the right size
    obs_size = len(mean)
    if obs_size == 3 + 8 * 3:  # 3 own + 8 features * 3 nearest aircraft
        feature_names = ["own_x", "own_y", "own_dist_to_goal"]
        ac_features = ["ac_x", "ac_y", "ac_vx", "ac_vy", "ac_dist", "cpa_time", "cpa_dist", "collision_risk"]
        for i in range(3):  # 3 nearest aircraft
            for feat in ac_features:
                feature_names.append(f"{feat}_{i+1}")
    
    # Print table
    print(f"{'Feature':<20} {'Index':>6} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12} {'Status':>15}")
    print(f"{'-'*20} {'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*15}")
    
    issues = []
    for i in range(obs_size):
        feat_name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
        
        # Check if values are well-normalized (roughly in [-1, 1] or [0, 1])
        status = "✓ Good"
        if max_vals[i] > 2.0 or min_vals[i] < -2.0:
            status = "⚠️ Too large"
            issues.append((feat_name, i, min_vals[i], max_vals[i]))
        elif abs(mean[i]) > 0.8:
            status = "⚠️ Biased"
        elif max_vals[i] - min_vals[i] < 0.1:
            status = "⚠️ Too small"
        
        print(f"{feat_name:<20} {i:>6} {min_vals[i]:>12.4f} {max_vals[i]:>12.4f} "
              f"{mean[i]:>12.4f} {std[i]:>12.4f} {status:>15}")
    
    print("\n" + "="*80)
    
    # Summary
    if issues:
        print(f"\n⚠️  Found {len(issues)} features with normalization issues:\n")
        for feat_name, idx, min_val, max_val in issues:
            print(f"  • {feat_name} (index {idx}): range [{min_val:.4f}, {max_val:.4f}]")
        print(f"\nConsider adjusting normalization constants in ma_env.py")
    else:
        print("\n✓ All features appear to be well normalized!")
    
    print("\n" + "="*80 + "\n")
    
    # Clean up
    env.close()
    
    return stats


if __name__ == "__main__":
    check_normalization(n_agents=6, n_episodes=50)
