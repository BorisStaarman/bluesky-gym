# Script to check observation normalization by collecting statistics from a few episodes

import os
import sys
import numpy as np

# Add current directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from bluesky_gym.envs.ma_env_SAC_AM import SectorEnv
from run_config import RUN_ID

# Parameters
N_AGENTS = 20
NUM_EPISODES = 10  # Run a few episodes to gather statistics
METRICS_DIR = os.path.join(script_dir, "metrics")

if __name__ == "__main__":
    env = SectorEnv(
        render_mode=None,  # No rendering for faster collection
        n_agents=N_AGENTS,
        run_id=RUN_ID,
        metrics_base_dir=METRICS_DIR
    )
    
    # Observation dimension: 7 (ownship) + 7*19 (intruders) = 140
    num_features = 7  # Features per agent/intruder
    num_intruders = N_AGENTS - 1  # 19
    
    # Track min/max for each of the 7 feature types across all agents/intruders
    feature_names = ["cos_drift", "sin_drift", "airspeed", "x", "y", "vx", "vy"]
    
    # Initialize tracking arrays: [num_features]
    ownship_mins = np.full(num_features, np.inf)
    ownship_maxs = np.full(num_features, -np.inf)
    intruder_mins = np.full(num_features, np.inf)
    intruder_maxs = np.full(num_features, -np.inf)
    
    total_steps = 0
    
    print(f"Collecting normalization statistics over {NUM_EPISODES} episodes...")
    
    for episode in range(1, NUM_EPISODES + 1):
        obs, info = env.reset()
        done = False
        episode_steps = 0
        
        while env.agents:
            # Random actions (we just need to collect observations)
            actions = {agent_id: env.action_space.sample() for agent_id in obs.keys()}
            obs, rewards, terminateds, truncateds, infos = env.step(actions)
            
            # Process observations
            for agent_id, agent_obs in obs.items():
                # Split into ownship (first 7) and intruders (rest)
                ownship_feats = agent_obs[:7]
                intruder_feats = agent_obs[7:].reshape(num_intruders, num_features)  # Shape: (19, 7)
                
                # Update ownship min/max for each feature
                ownship_mins = np.minimum(ownship_mins, ownship_feats)
                ownship_maxs = np.maximum(ownship_maxs, ownship_feats)
                
                # Update intruder min/max for each feature (across all intruders)
                # Only consider non-zero intruders (non-padding)
                non_padding_mask = intruder_feats.sum(axis=1) != 0
                if non_padding_mask.any():
                    valid_intruders = intruder_feats[non_padding_mask]
                    intruder_mins = np.minimum(intruder_mins, valid_intruders.min(axis=0))
                    intruder_maxs = np.maximum(intruder_maxs, valid_intruders.max(axis=0))
            
            episode_steps += 1
        
        total_steps += episode_steps
        print(f"  Episode {episode}: {episode_steps} steps")
    
    print(f"\nTotal steps collected: {total_steps}")
    print("\n" + "="*80)
    print("NORMALIZATION ANALYSIS")
    print("="*80)
    
    print("\nOWNSHIP FEATURES:")
    print(f"{'Feature':<15} {'Min':>12} {'Max':>12} {'Range':>12}")
    print("-" * 55)
    for i, name in enumerate(feature_names):
        min_val = ownship_mins[i]
        max_val = ownship_maxs[i]
        range_val = max_val - min_val
        print(f"{name:<15} {min_val:>12.4f} {max_val:>12.4f} {range_val:>12.4f}")
    
    print("\nINTRUDER FEATURES:")
    print(f"{'Feature':<15} {'Min':>12} {'Max':>12} {'Range':>12}")
    print("-" * 55)
    for i, name in enumerate(feature_names):
        min_val = intruder_mins[i]
        max_val = intruder_maxs[i]
        range_val = max_val - min_val
        print(f"{name:<15} {min_val:>12.4f} {max_val:>12.4f} {range_val:>12.4f}")
    
    print("\n" + "="*80)
    print("NORMALIZATION RECOMMENDATIONS:")
    print("="*80)
    
    # Check if features are well-normalized (roughly in [-1, 1] or [0, 1] range)
    print("\nFeatures that may need better normalization (range > 10):")
    for i, name in enumerate(feature_names):
        own_range = ownship_maxs[i] - ownship_mins[i]
        int_range = intruder_maxs[i] - intruder_mins[i]
        
        if own_range > 10 or int_range > 10:
            print(f"  {name}:")
            if own_range > 10:
                print(f"    Ownship range: {own_range:.2f} (min={ownship_mins[i]:.2f}, max={ownship_maxs[i]:.2f})")
            if int_range > 10:
                print(f"    Intruder range: {int_range:.2f} (min={intruder_mins[i]:.2f}, max={intruder_maxs[i]:.2f})")
    
    print("\nFeatures with good normalization (range < 2):")
    for i, name in enumerate(feature_names):
        own_range = ownship_maxs[i] - ownship_mins[i]
        int_range = intruder_maxs[i] - intruder_mins[i]
        
        if own_range < 2 and int_range < 2:
            print(f"  ✓ {name}: ownship={own_range:.2f}, intruder={int_range:.2f}")
    
    env.close()
    print("\nDone!")
