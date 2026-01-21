"""
Find where waypoint success rate goes from LOW (<20%) to HIGH (>70%)
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

print("="*80)
print("SCANNING FOR LOW→HIGH TRANSITION IN WAYPOINT SUCCESS")
print("="*80)

# Find where rolling average crosses from <30% to >60%
window = 50
episode_df['wp_rolling'] = episode_df['waypoint_success_rate'].rolling(window=window, center=False).mean()

transition_found = False
for i in range(window, len(episode_df) - window):
    before_avg = episode_df.iloc[i-window:i]['waypoint_success_rate'].mean()
    after_avg = episode_df.iloc[i:i+window]['waypoint_success_rate'].mean()
    
    if before_avg < 30 and after_avg > 60:
        print(f"\n✅ FOUND LOW→HIGH TRANSITION at chronological index {i}")
        print(f"   Before (episodes {i-window} to {i}): {before_avg:.1f}% waypoint success")
        print(f"   After (episodes {i} to {i+window}): {after_avg:.1f}% waypoint success")
        transition_found = True
        transition_idx = i
        break

if not transition_found:
    print("\n❌ No clear LOW→HIGH transition found")
    print("   Checking if training started with high performance...")
    
    first_1000 = episode_df.head(1000)['waypoint_success_rate'].mean()
    print(f"\n   First 1000 episodes avg: {first_1000:.1f}%")
    
    if first_1000 > 50:
        print(f"   ⚠️  Training started with HIGH waypoint success ({first_1000:.1f}%)")
        print(f"   This suggests the model was loaded from a pre-trained checkpoint!")
        transition_idx = 0
    else:
        # Find the steepest upward slope
        print("\n   Looking for steepest improvement...")
        max_slope = 0
        max_slope_idx = 0
        window_small = 50
        
        for i in range(window_small, len(episode_df) - window_small, 10):
            before = episode_df.iloc[i-window_small:i]['waypoint_success_rate'].mean()
            after = episode_df.iloc[i:i+window_small]['waypoint_success_rate'].mean()
            slope = after - before
            if slope > max_slope:
                max_slope = slope
                max_slope_idx = i
        
        print(f"\n   Steepest improvement at chronological index {max_slope_idx}")
        print(f"   Improvement: +{max_slope:.1f}%")
        transition_idx = max_slope_idx

# Detailed analysis around transition
print(f"\n{'='*80}")
print(f"DETAILED ANALYSIS AROUND CHRONOLOGICAL INDEX {transition_idx}")
print(f"{'='*80}")

window_size = 200
before_start = max(0, transition_idx - window_size)
before_end = transition_idx
after_start = transition_idx
after_end = min(len(episode_df), transition_idx + window_size)

before_df = episode_df.iloc[before_start:before_end]
after_df = episode_df.iloc[after_start:after_end]

print(f"\nBEFORE (chrono {before_start} to {before_end}):")
print(f"  Episodes: {len(before_df)}")
print(f"  Unique PIDs: {before_df['pid'].nunique()}")
print(f"  First 5 PIDs: {list(before_df['pid'].unique()[:5])}")
print(f"  Episode_index range: {before_df['episode_index'].min()} to {before_df['episode_index'].max()}")
print(f"  Waypoint success: {before_df['waypoint_success_rate'].mean():.1f}%")
print(f"  Intrusions: {before_df['total_intrusions'].mean():.1f}")
print(f"  Steps: {before_df['steps'].mean():.1f}")

print(f"\nAFTER (chrono {after_start} to {after_end}):")
print(f"  Episodes: {len(after_df)}")
print(f"  Unique PIDs: {after_df['pid'].nunique()}")
print(f"  First 5 PIDs: {list(after_df['pid'].unique()[:5])}")
print(f"  Episode_index range: {after_df['episode_index'].min()} to {after_df['episode_index'].max()}")
print(f"  Waypoint success: {after_df['waypoint_success_rate'].mean():.1f}%")
print(f"  Intrusions: {after_df['total_intrusions'].mean():.1f}")
print(f"  Steps: {after_df['steps'].mean():.1f}")

# Check for PID change
if transition_idx > 0:
    before_pids = set(before_df['pid'].unique())
    after_pids = set(after_df['pid'].unique())
    common_pids = before_pids & after_pids
    new_pids = after_pids - before_pids
    old_pids = before_pids - after_pids
    
    print(f"\nPID ANALYSIS:")
    print(f"  Common PIDs: {len(common_pids)} {list(common_pids)[:3]}...")
    print(f"  New PIDs after transition: {len(new_pids)} {list(new_pids)[:3]}...")
    print(f"  Old PIDs before transition: {len(old_pids)} {list(old_pids)[:3]}...")

print(f"\n{'='*80}")
print(f"EPISODES AT BOUNDARY (±10 from chrono {transition_idx}):")
print(f"{'='*80}")

boundary = episode_df.iloc[max(0, transition_idx-10):min(len(episode_df), transition_idx+10)]
view = boundary[['chronological_index', 'episode_index', 'pid', 'waypoint_success_rate', 'total_intrusions', 'steps', 'finished_at']]
print(view.to_string(index=False))

# Check timestamps for gaps
boundary_df = episode_df.iloc[max(0, transition_idx-50):min(len(episode_df), transition_idx+50)].copy()
boundary_df['timestamp'] = pd.to_datetime(boundary_df['finished_at'])
boundary_df['time_diff_sec'] = boundary_df['timestamp'].diff().dt.total_seconds()

large_gaps_in_boundary = boundary_df[boundary_df['time_diff_sec'] > 300]  # >5 minutes
if len(large_gaps_in_boundary) > 0:
    print(f"\n{'='*80}")
    print(f"TIME GAPS NEAR TRANSITION (>5 minutes):")
    print(f"{'='*80}")
    for idx, row in large_gaps_in_boundary.iterrows():
        print(f"\nAt chronological index {int(row['chronological_index'])}:")
        print(f"  Gap: {row['time_diff_sec']/60:.1f} minutes")
        print(f"  PID: {row['pid']}")
        print(f"  Episode_index: {int(row['episode_index'])}")
        print(f"  Waypoint success: {row['waypoint_success_rate']:.1f}%")
