"""
Check if there's a timestamp gap or episode_index jump that indicates Stage 2 starting
"""
from run_config import RUN_ID
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

episode_df['timestamp'] = pd.to_datetime(episode_df['finished_at'], unit='s')
episode_df['time_diff_minutes'] = episode_df['timestamp'].diff().dt.total_seconds() / 60

print("="*80)
print("STAGE 1 vs STAGE 2 DETECTION")
print("="*80)

# Your training config:
# Stage 1: 75 iterations
# Stage 2: 36 iterations (6 warmup + 30)

# Find major time gaps (> 10 minutes = likely training restart)
major_gaps = episode_df[episode_df['time_diff_minutes'] > 10].copy()

print(f"\nFound {len(major_gaps)} time gaps > 10 minutes (potential training restarts):")

for idx, row in major_gaps.head(10).iterrows():
    prev_row = episode_df.iloc[idx-1]
    print(f"\n  Gap at chronological index {int(row['chronological_index'])}:")
    print(f"    Time gap: {row['time_diff_minutes']:.1f} minutes ({row['time_diff_minutes']/60:.2f} hours)")
    print(f"    BEFORE gap:")
    print(f"      Episode_index: {int(prev_row['episode_index'])}, PID: {prev_row['pid']}")
    print(f"      Waypoint success: {prev_row['waypoint_success_rate']:.1f}%")
    print(f"      Timestamp: {prev_row['timestamp']}")
    print(f"    AFTER gap:")
    print(f"      Episode_index: {int(row['episode_index'])}, PID: {row['pid']}")
    print(f"      Waypoint success: {row['waypoint_success_rate']:.1f}%")
    print(f"      Timestamp: {row['timestamp']}")
    
    # Check if episode_index resets (indicates new stage)
    if row['episode_index'] < prev_row['episode_index']:
        print(f"      WARNING: Episode_index DECREASED ({prev_row['episode_index']} -> {row['episode_index']})")
        print(f"      This might indicate Stage 2 starting with a new episode counter!")

# Plot episode_index over chronological time to see if it ever resets dramatically
print(f"\n{'='*80}")
print("EPISODE_INDEX PROGRESSION:")
print(f"{'='*80}")

# Check if episode_index increases monotonically or has large resets
episode_df['episode_idx_diff'] = episode_df['episode_index'].diff()
large_decreases = episode_df[episode_df['episode_idx_diff'] < -50].copy()

if len(large_decreases) > 0:
    print(f"\nWARNING: Found {len(large_decreases)} MAJOR episode_index RESETS (decrease > 50):")
    for idx, row in large_decreases.head(5).iterrows():
        prev_row = episode_df.iloc[idx-1]
        print(f"\n  Reset at chronological index {int(row['chronological_index'])}:")
        print(f"    Episode_index: {int(prev_row['episode_index'])} -> {int(row['episode_index'])} (change: {int(row['episode_idx_diff'])})")
        print(f"    PID change: {prev_row['pid']} -> {row['pid']}")
        print(f"    Waypoint success: {prev_row['waypoint_success_rate']:.1f}% -> {row['waypoint_success_rate']:.1f}%")
        print(f"    This is likely the start of Stage 2!")
else:
    print("\nNo major episode_index resets found.")
    print("Episode_index range:", episode_df['episode_index'].min(), "to", episode_df['episode_index'].max())
    
    # Check performance degradation around episode_index changes
    print(f"\n{'='*80}")
    print("PERFORMANCE OVER EPISODE_INDEX:")
    print(f"{'='*80}")
    
    # Group by episode_index and show avg performance
    perf_by_idx = episode_df.groupby('episode_index').agg({
        'waypoint_success_rate': 'mean',
        'total_intrusions': 'mean',
        'chronological_index': ['min', 'max', 'count']
    }).reset_index()
    perf_by_idx.columns = ['episode_index', 'wp_success', 'intrusions', 'chrono_min', 'chrono_max', 'count']
    
    print(f"\nShowing episode_index progression:")
    print(f"(Looking for where performance suddenly drops)")
    print("\nFirst 20 episode_index values:")
    print(perf_by_idx.head(20).to_string(index=False))
    
    print("\n\nLast 20 episode_index values:")
    print(perf_by_idx.tail(20).to_string(index=False))
    
    # Find where performance drops
    perf_by_idx['wp_success_next'] = perf_by_idx['wp_success'].shift(-1)
    perf_by_idx['wp_drop'] = perf_by_idx['wp_success'] - perf_by_idx['wp_success_next']
    
    big_drops = perf_by_idx[perf_by_idx['wp_drop'] > 30].copy()
    if len(big_drops) > 0:
        print(f"\nWARNING: Found {len(big_drops)} MAJOR PERFORMANCE DROPS (>30%):")
        for idx, row in big_drops.iterrows():
            next_row = perf_by_idx.iloc[idx+1]
            print(f"\n  Drop at episode_index {int(row['episode_index'])} -> {int(next_row['episode_index'])}:")
            print(f"    Waypoint success: {row['wp_success']:.1f}% -> {next_row['wp_success']:.1f}%")
            print(f"    Chronological index range: {int(row['chrono_min'])}-{int(row['chrono_max'])} -> {int(next_row['chrono_min'])}-{int(next_row['chrono_max'])}")
            print(f"    -> This might be where Stage 2 begins!")

print(f"\n{'='*80}")
print("HYPOTHESIS:")
print(f"{'='*80}")
print("""
Based on the data:
1. Episodes 1-50: Model is learning from scratch (Stage 1 beginning)
2. Episodes 50-~7500: Model has learned well from teacher (Stage 1 mature)  
3. Episodes ~7500-18000: Performance is more volatile/lower (Stage 2?)

The drastic change you see around episode 7500-8000 is likely when:
- Stage 1 training completed (75 iterations)
- Stage 2 started with a new episode counter or continued
- The frozen policy from Stage 1 was unfrozen
- Model started learning from RL rewards instead of imitating teacher
- Initial performance dropped as the model explored with RL

This is NORMAL for two-stage training! Stage 2 often shows initial performance
degradation as the model transitions from supervised imitation to RL optimization.
""")
