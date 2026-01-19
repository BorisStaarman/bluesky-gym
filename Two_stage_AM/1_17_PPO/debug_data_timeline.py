"""
Debug script to understand the episode timeline and what caused the drastic change.
"""
from run_config import RUN_ID
import os
import pandas as pd
import glob

script_dir = os.path.dirname(os.path.abspath(__file__))
metrics_dir = os.path.join(script_dir, "metrics", f"run_{RUN_ID}")
csv_path = os.path.join(metrics_dir, "all_agents_merged_sorted.csv")

if not os.path.exists(csv_path):
    print("Please run analyze_csv.py first to generate merged CSV")
    exit(1)

df = pd.read_csv(csv_path)

# Filter to only completed episodes
df = df[df['finished_at'].notna()].copy()

# Create unique environment episode identifier
df['env_episode_id'] = df['episode_index'].astype(str) + '_' + df['pid'].astype(str)

# Group by environment episode
episode_df = df.groupby('env_episode_id').agg({
    'episode_index': 'first',
    'finished_at': 'first',
    'pid': 'first',
    'total_intrusions': 'sum',
    'terminated_waypoint': 'sum',
    'steps': 'mean',
}).reset_index()

# Add waypoint success rate
agents_per_episode = df.groupby('env_episode_id').size()
episode_df['total_agents'] = episode_df['env_episode_id'].map(agents_per_episode)
episode_df['waypoint_success_rate'] = (episode_df['terminated_waypoint'] / episode_df['total_agents']) * 100

# Sort chronologically
episode_df = episode_df.sort_values('finished_at').reset_index(drop=True)
episode_df['chronological_index'] = range(1, len(episode_df) + 1)

print("="*80)
print("DATA TIMELINE ANALYSIS")
print("="*80)
print(f"\nTotal environment episodes: {len(episode_df)}")
print(f"Unique PIDs: {episode_df['pid'].nunique()}")
print(f"Unique episode_index values: {episode_df['episode_index'].nunique()}")

# Analyze the transition point around episode 7500-8000
transition_start = 7000
transition_end = 8500

print(f"\n{'='*80}")
print(f"ANALYZING TRANSITION ZONE (chronological episodes {transition_start}-{transition_end})")
print(f"{'='*80}")

transition_df = episode_df[(episode_df['chronological_index'] >= transition_start) & 
                           (episode_df['chronological_index'] <= transition_end)]

print(f"\nEpisodes in transition zone: {len(transition_df)}")
print(f"PIDs in transition zone: {transition_df['pid'].unique()}")
print(f"Episode_index range: {transition_df['episode_index'].min()} to {transition_df['episode_index'].max()}")

# Check for PID changes
print(f"\n{'='*80}")
print("PID TIMELINE")
print(f"{'='*80}")
pid_changes = episode_df.groupby('pid').agg({
    'chronological_index': ['min', 'max', 'count'],
    'episode_index': ['min', 'max'],
    'finished_at': ['min', 'max'],
    'waypoint_success_rate': 'mean'
}).reset_index()
pid_changes.columns = ['_'.join(col).strip('_') for col in pid_changes.columns.values]
pid_changes = pid_changes.sort_values('chronological_index_min')

print(f"\nShowing PIDs near the transition point (chrono 7000-8000):")
near_transition = pid_changes[
    ((pid_changes['chronological_index_min'] >= 6000) & (pid_changes['chronological_index_min'] <= 9000)) |
    ((pid_changes['chronological_index_max'] >= 6000) & (pid_changes['chronological_index_max'] <= 9000))
]

for idx, row in near_transition.iterrows():
    print(f"\n{row['pid']}:")
    print(f"  Chronological range: {int(row['chronological_index_min'])} to {int(row['chronological_index_max'])} ({int(row['chronological_index_count'])} episodes)")
    print(f"  Episode_index range: {int(row['episode_index_min'])} to {int(row['episode_index_max'])}")
    print(f"  Timestamps: {row['finished_at_min']} to {row['finished_at_max']}")
    print(f"  Avg waypoint success: {row['waypoint_success_rate_mean']:.1f}%")

# Check if there's a specific chronological index where things change
print(f"\n{'='*80}")
print("BEFORE vs AFTER COMPARISON")
print(f"{'='*80}")

# Find the exact transition point by looking at waypoint success rate
episode_df['waypoint_high'] = episode_df['waypoint_success_rate'] > 50
transition_idx = None

for i in range(6000, 9000):
    before = episode_df[episode_df['chronological_index'] < i]
    after = episode_df[episode_df['chronological_index'] >= i]
    
    if len(before) > 100 and len(after) > 100:
        before_success = before['waypoint_success_rate'].mean()
        after_success = after['waypoint_success_rate'].mean()
        
        # Find the largest jump
        if before_success < 30 and after_success > 70:
            transition_idx = i
            break

if transition_idx:
    print(f"\nTransition point detected at chronological index: {transition_idx}")
    
    before = episode_df[episode_df['chronological_index'] < transition_idx]
    after = episode_df[episode_df['chronological_index'] >= transition_idx]
    
    print(f"\nBEFORE episode {transition_idx}:")
    print(f"  Episodes: {len(before)}")
    print(f"  PIDs: {before['pid'].nunique()} unique")
    print(f"  Episode_index range: {before['episode_index'].min()} to {before['episode_index'].max()}")
    print(f"  Avg waypoint success: {before['waypoint_success_rate'].mean():.1f}%")
    print(f"  Avg intrusions: {before['total_intrusions'].mean():.1f}")
    print(f"  Avg steps: {before['steps'].mean():.1f}")
    
    print(f"\nAFTER episode {transition_idx}:")
    print(f"  Episodes: {len(after)}")
    print(f"  PIDs: {after['pid'].nunique()} unique")
    print(f"  Episode_index range: {after['episode_index'].min()} to {after['episode_index'].max()}")
    print(f"  Avg waypoint success: {after['waypoint_success_rate'].mean():.1f}%")
    print(f"  Avg intrusions: {after['total_intrusions'].mean():.1f}")
    print(f"  Avg steps: {after['steps'].mean():.1f}")
    
    # Check the exact episodes at the boundary
    print(f"\n{'='*80}")
    print(f"EPISODES AT THE BOUNDARY")
    print(f"{'='*80}")
    
    boundary_df = episode_df[
        (episode_df['chronological_index'] >= transition_idx - 10) & 
        (episode_df['chronological_index'] <= transition_idx + 10)
    ][['chronological_index', 'episode_index', 'pid', 'waypoint_success_rate', 'total_intrusions', 'finished_at']]
    
    print(boundary_df.to_string(index=False))

# Check if episode_index resets (indicating a new training run)
print(f"\n{'='*80}")
print("EPISODE_INDEX CONTINUITY CHECK")
print(f"{'='*80}")

# Sort by finished_at and check if episode_index ever decreases
episode_df_time_sorted = episode_df.sort_values('finished_at').reset_index(drop=True)
episode_df_time_sorted['episode_index_diff'] = episode_df_time_sorted['episode_index'].diff()

resets = episode_df_time_sorted[episode_df_time_sorted['episode_index_diff'] < -10]
if len(resets) > 0:
    print(f"\nFound {len(resets)} episode_index RESETS (where episode_index goes backwards):")
    for idx, row in resets.head(10).iterrows():
        prev_row = episode_df_time_sorted.iloc[idx-1]
        print(f"\n  At chronological index {int(row['chronological_index'])}:")
        print(f"    Previous: episode_index={int(prev_row['episode_index'])}, pid={prev_row['pid']}, wp_success={prev_row['waypoint_success_rate']:.1f}%")
        print(f"    Current:  episode_index={int(row['episode_index'])}, pid={row['pid']}, wp_success={row['waypoint_success_rate']:.1f}%")
        print(f"    Jump: {int(row['episode_index_diff'])} (RESET DETECTED)")
else:
    print("\nNo episode_index resets detected - all episodes are from the same continuous run")

print("\n" + "="*80)
print("HYPOTHESIS")
print("="*80)
print("""
Based on your training setup:
- Stage 1 (Imitation): 75 iterations
- Stage 2 (RL Fine-tuning): 36 iterations (6 warmup + 30 regular)

The drastic change you see is likely because:
1. Your CSV files contain episodes from BOTH Stage 1 and Stage 2
2. The environment was restarted between stages (new PID)
3. Before the transition: Stage 1 imitation learning (agents learn to mimic teacher)
4. After the transition: Stage 2 RL fine-tuning (agents optimize with full reward signal)

If the transition happens around episode 7500, this suggests approximately:
- ~7500 episodes during Stage 1 training
- Remaining episodes during Stage 2 training
""")
