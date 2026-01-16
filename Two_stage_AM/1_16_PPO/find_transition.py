"""
Find the exact transition point where waypoint success rate jumps from low to high.
"""
from run_config import RUN_ID
import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
metrics_dir = os.path.join(script_dir, "metrics", f"run_{RUN_ID}")
csv_path = os.path.join(metrics_dir, "all_agents_merged_sorted.csv")

df = pd.read_csv(csv_path)
df = df[df['finished_at'].notna()].copy()
df['env_episode_id'] = df['episode_index'].astype(str) + '_' + df['pid'].astype(str)

episode_df = df.groupby('env_episode_id').agg({
    'episode_index': 'first',
    'finished_at': 'first',
    'pid': 'first',
    'total_intrusions': 'sum',
    'terminated_waypoint': 'sum',
    'steps': 'mean',
}).reset_index()

agents_per_episode = df.groupby('env_episode_id').size()
episode_df['total_agents'] = episode_df['env_episode_id'].map(agents_per_episode)
episode_df['waypoint_success_rate'] = (episode_df['terminated_waypoint'] / episode_df['total_agents']) * 100
episode_df = episode_df.sort_values('finished_at').reset_index(drop=True)
episode_df['chronological_index'] = range(1, len(episode_df) + 1)

# Find where waypoint success rate changes dramatically
window = 100
episode_df['wp_rolling_avg'] = episode_df['waypoint_success_rate'].rolling(window=window, center=True).mean()

print("Scanning for largest jump in waypoint success rate...")
max_jump = 0
max_jump_idx = 0

for i in range(500, len(episode_df) - 500, 50):
    before_avg = episode_df.iloc[i-200:i]['waypoint_success_rate'].mean()
    after_avg = episode_df.iloc[i:i+200]['waypoint_success_rate'].mean()
    jump = after_avg - before_avg
    
    if jump > max_jump:
        max_jump = jump
        max_jump_idx = i

print(f"\nLargest jump found at chronological index {max_jump_idx}")
print(f"Jump size: {max_jump:.1f}%")

# Analyze around this point
window_size = 500
start_idx = max(0, max_jump_idx - window_size)
end_idx = min(len(episode_df), max_jump_idx + window_size)

print(f"\n{'='*80}")
print(f"DETAILED VIEW AROUND THE TRANSITION (chrono {start_idx} to {end_idx})")
print(f"{'='*80}")

before_df = episode_df.iloc[start_idx:max_jump_idx]
after_df = episode_df.iloc[max_jump_idx:end_idx]

print(f"\nBEFORE (chrono {start_idx} to {max_jump_idx}):")
print(f"  Episodes: {len(before_df)}")
print(f"  Unique PIDs: {before_df['pid'].nunique()}")
print(f"  PID list: {before_df['pid'].unique()[:5]}...")
print(f"  Episode_index range: {before_df['episode_index'].min()} to {before_df['episode_index'].max()}")
print(f"  Avg waypoint success: {before_df['waypoint_success_rate'].mean():.1f}%")
print(f"  Avg intrusions: {before_df['total_intrusions'].mean():.2f}")
print(f"  Avg steps: {before_df['steps'].mean():.1f}")
print(f"  Timestamp range: {before_df['finished_at'].min()} to {before_df['finished_at'].max()}")

print(f"\nAFTER (chrono {max_jump_idx} to {end_idx}):")
print(f"  Episodes: {len(after_df)}")
print(f"  Unique PIDs: {after_df['pid'].nunique()}")
print(f"  PID list: {after_df['pid'].unique()[:5]}...")
print(f"  Episode_index range: {after_df['episode_index'].min()} to {after_df['episode_index'].max()}")
print(f"  Avg waypoint success: {after_df['waypoint_success_rate'].mean():.1f}%")
print(f"  Avg intrusions: {after_df['total_intrusions'].mean():.2f}")
print(f"  Avg steps: {after_df['steps'].mean():.1f}")
print(f"  Timestamp range: {after_df['finished_at'].min()} to {after_df['finished_at'].max()}")

print(f"\n{'='*80}")
print("20 EPISODES BEFORE TRANSITION:")
print(f"{'='*80}")
view = episode_df.iloc[max_jump_idx-20:max_jump_idx][['chronological_index', 'episode_index', 'pid', 'waypoint_success_rate', 'total_intrusions', 'steps']]
print(view.to_string(index=False))

print(f"\n{'='*80}")
print("20 EPISODES AFTER TRANSITION:")
print(f"{'='*80}")
view = episode_df.iloc[max_jump_idx:max_jump_idx+20][['chronological_index', 'episode_index', 'pid', 'waypoint_success_rate', 'total_intrusions', 'steps']]
print(view.to_string(index=False))

# Check if there's a checkpoint reload by examining the timestamps
print(f"\n{'='*80}")
print("TIME GAP ANALYSIS (looking for training restarts):")
print(f"{'='*80}")

episode_df['timestamp'] = pd.to_datetime(episode_df['finished_at'])
episode_df['time_diff'] = episode_df['timestamp'].diff().dt.total_seconds()

# Find large time gaps (> 1 hour = 3600 seconds)
large_gaps = episode_df[episode_df['time_diff'] > 3600].copy()
large_gaps = large_gaps[['chronological_index', 'episode_index', 'pid', 'time_diff', 'waypoint_success_rate', 'finished_at']]

if len(large_gaps) > 0:
    print(f"\nFound {len(large_gaps)} large time gaps (>1 hour):")
    for idx, row in large_gaps.iterrows():
        if abs(row['chronological_index'] - max_jump_idx) < 100:
            print(f"\n⚠️  NEAR TRANSITION at chrono {int(row['chronological_index'])}:")
            print(f"    Time gap: {row['time_diff']/3600:.1f} hours")
            print(f"    Episode_index: {int(row['episode_index'])}")
            print(f"    PID: {row['pid']}")
            print(f"    Waypoint success after gap: {row['waypoint_success_rate']:.1f}%")
            print(f"    Timestamp: {row['finished_at']}")
else:
    print("No large time gaps found - continuous training")
