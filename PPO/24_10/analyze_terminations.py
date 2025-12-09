"""
Analyze episode termination reasons from CSV files.
Shows how many episodes ended due to:
- Intrusion (early termination)
- Waypoint reached (success)
- Truncation (out of bounds or time limit)
"""
import pandas as pd
import os
from run_config import RUN_ID

# Get path to merged CSV
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
csv_path = os.path.join(repo_root, "metrics_24_10", f"run_{RUN_ID}", "all_agents_merged_sorted.csv")

if not os.path.exists(csv_path):
    print(f"Merged CSV not found at: {csv_path}")
    print("Run merge_runs.py first!")
    exit(1)

# Load data
df = pd.read_csv(csv_path)

# Filter to completed episodes only
if 'finished_at' in df.columns:
    df = df[df['finished_at'].notna()]
    
print(f"\n{'='*60}")
print(f"EPISODE TERMINATION ANALYSIS (RUN_ID={RUN_ID})")
print(f"{'='*60}\n")

# Get unique episodes
total_episodes = df['episode_index'].nunique()
print(f"Total episodes analyzed: {total_episodes}\n")

# Group by episode to get termination reason (one row per episode)
episode_groups = df.groupby('episode_index').agg({
    'terminated_intrusion': 'max',  # True if ANY agent had intrusion termination
    'terminated_waypoint': 'sum',   # Count how many agents reached waypoint
    'truncated': 'max',             # True if ANY agent was truncated
    'steps': 'mean',
    'total_intrusions': 'sum'
}).reset_index()

# Count termination types
intrusion_episodes = episode_groups['terminated_intrusion'].sum()
truncated_episodes = episode_groups['truncated'].sum()
# Success = at least some agents reached waypoint and not intrusion/truncated
success_episodes = len(episode_groups[
    (episode_groups['terminated_waypoint'] > 0) & 
    (~episode_groups['terminated_intrusion']) & 
    (~episode_groups['truncated'])
])

print("Termination Reasons:")
print(f"  Intrusion (early termination):  {intrusion_episodes:4d} ({intrusion_episodes/total_episodes*100:.1f}%)")
print(f"  Success (waypoint reached):     {success_episodes:4d} ({success_episodes/total_episodes*100:.1f}%)")
print(f"  Truncated (bounds/time):        {truncated_episodes:4d} ({truncated_episodes/total_episodes*100:.1f}%)")

# Stats for intrusion-terminated episodes
intrusion_eps = episode_groups[episode_groups['terminated_intrusion']]
if len(intrusion_eps) > 0:
    print(f"\nIntrusion-terminated episodes:")
    print(f"  Average episode length:  {intrusion_eps['steps'].mean():.1f} steps")
    print(f"  Average intrusions:      {intrusion_eps['total_intrusions'].mean():.1f}")
    print(f"  Min episode length:      {intrusion_eps['steps'].min():.0f} steps")
    print(f"  Max episode length:      {intrusion_eps['steps'].max():.0f} steps")

# Stats for successful episodes
success_eps = episode_groups[
    (episode_groups['terminated_waypoint'] > 0) & 
    (~episode_groups['terminated_intrusion']) & 
    (~episode_groups['truncated'])
]
if len(success_eps) > 0:
    print(f"\nSuccessful episodes:")
    print(f"  Average episode length:  {success_eps['steps'].mean():.1f} steps")
    print(f"  Average intrusions:      {success_eps['total_intrusions'].mean():.1f}")
    print(f"  Agents reaching waypoint (avg): {success_eps['terminated_waypoint'].mean():.1f} / 6")

# Show trend over time (first half vs second half)
midpoint = total_episodes // 2
first_half = episode_groups[episode_groups['episode_index'] <= midpoint]
second_half = episode_groups[episode_groups['episode_index'] > midpoint]

print(f"\n{'='*60}")
print("Learning Progress (First Half vs Second Half):")
print(f"{'='*60}")
print(f"                           First {midpoint} eps    Last {total_episodes-midpoint} eps")
print(f"  Intrusion terminations:  {first_half['terminated_intrusion'].sum():4d}            {second_half['terminated_intrusion'].sum():4d}")
print(f"  Successful episodes:     {len(first_half[(first_half['terminated_waypoint'] > 0) & (~first_half['terminated_intrusion']) & (~first_half['truncated'])]):4d}            {len(second_half[(second_half['terminated_waypoint'] > 0) & (~second_half['terminated_intrusion']) & (~second_half['truncated'])]):4d}")
print(f"  Avg intrusions/episode:  {first_half['total_intrusions'].mean():5.1f}          {second_half['total_intrusions'].mean():5.1f}")
print(f"  Avg episode length:      {first_half['steps'].mean():5.1f}          {second_half['steps'].mean():5.1f}")
print()
