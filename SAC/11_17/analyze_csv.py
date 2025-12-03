from run_config import RUN_ID
import os
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import math

# get the path to the .csv file
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
root = os.path.join(script_dir, "metrics", f"run_{RUN_ID}", "all_agents_merged_sorted.csv")
df = pd.read_csv(root)

# --- Data Summary ---
print(f"Total rows in CSV: {len(df)}")
print(f"Total unique episodes (episode_index): {df['episode_index'].nunique()}")
print(f"Episode index range: {df['episode_index'].min()} to {df['episode_index'].max()}")

# --- Finished-at summary ---
# Count the number of unique agent-episodes (each episode ended once per agent)
if {'agent', 'episode_index'}.issubset(df.columns):
    episode_keys = df[['agent', 'episode_index']].dropna().drop_duplicates()
    total_agent_episodes = len(episode_keys)
    print(f"Total agent-episode records: {total_agent_episodes}")
else:
    # Fallback: unique episode_index if agent column is missing
    if 'episode_index' in df.columns:
        total_agent_episodes = df['episode_index'].dropna().nunique()
        print(f"Total episodes (by episode_index): {total_agent_episodes}")
    else:
        print("No episode_index column found; cannot summarize episode stops.")

# Filter to only completed episodes
if 'finished_at' in df.columns:
    episodes_before = df['episode_index'].nunique()
    rows_before = len(df)
    episodes_with_finished = df[df['finished_at'].notna()]['episode_index'].nunique()
    episodes_without_finished = df[df['finished_at'].isna()]['episode_index'].nunique()
    
    print(f"\nFinished_at Analysis:")
    print(f"  Episodes WITH finished_at timestamp: {episodes_with_finished}")
    print(f"  Episodes WITHOUT finished_at timestamp: {episodes_without_finished}")
    print(f"  Total rows before filtering: {rows_before}")
    
    df = df[df['finished_at'].notna()]
    episodes_after = df['episode_index'].nunique()
    rows_after = len(df)
    
    print(f"  Total rows after filtering: {rows_after}")
    print(f"  Episodes after filtering: {episodes_after}")
    print(f"  --> Removed {episodes_before - episodes_after} incomplete episodes")
    
    # Show completion percentage
    completion_rate = (episodes_with_finished / episodes_before) * 100 if episodes_before > 0 else 0
    print(f"  Completion rate: {completion_rate:.1f}%")
else:
    print("\nNo 'finished_at' column found - all episodes included")

# Create unique environment episode identifier
# Combine episode_index + pid to track each unique environment run
df['env_episode_id'] = df['episode_index'].astype(str) + '_' + df['pid'].astype(str)

# Count unique environment episodes
unique_env_episodes = df['env_episode_id'].nunique()
agents_per_env = df.groupby('env_episode_id').size()
print(f"\nEnvironment episode analysis:")
print(f"  Unique environment episodes: {unique_env_episodes}")
print(f"  Agents per environment (mean): {agents_per_env.mean():.2f}")
print(f"  Episodes with all 6 agents: {(agents_per_env == 6).sum()}")

# Group by environment episode to get per-environment statistics
episode_df = df.groupby('env_episode_id').agg({
    'episode_index': 'first',
    'finished_at': 'first',
    'steps': 'mean',
    'mean_reward_drift': 'mean',
    'mean_reward_progress': 'mean',
    'mean_reward_intrusion': 'mean',
    'mean_reward_proximity': 'mean',
    'sum_reward_drift': 'mean',
    'sum_reward_progress': 'mean',
    'sum_reward_intrusion': 'mean',
    'sum_reward_proximity': 'mean',
    'total_intrusions': 'sum',
    'terminated_waypoint': 'sum'  # Sum of waypoints reached per episode
}).reset_index()

# Calculate waypoint success rate per episode (percentage of agents that reached waypoint)
# Count agents per episode
agents_per_episode = df.groupby('env_episode_id').size()
episode_df['total_agents'] = episode_df['env_episode_id'].map(agents_per_episode)
episode_df['waypoint_success_rate'] = (episode_df['terminated_waypoint'] / episode_df['total_agents']) * 100

# Sort by finished_at to get chronological order
episode_df = episode_df.sort_values('finished_at').reset_index(drop=True)
episode_df['chronological_index'] = range(1, len(episode_df) + 1)

print(f"\nAnalyzing {len(episode_df)} actual environment episodes (chronologically ordered)")

# Define reward components to plot (organized by type)
# Left column: Total rewards (sum_reward_*)
# Right column: Per-step rewards (mean_reward_*)
reward_components = [
    ('sum_reward_drift', 'Total Drift Penalty (per episode)', 'blue'),
    ('mean_reward_drift', 'Drift Penalty (per step)', 'blue'),
    ('sum_reward_progress', 'Total Progress Reward (per episode)', 'green'),
    ('mean_reward_progress', 'Progress Reward (per step)', 'green'),
    ('sum_reward_intrusion', 'Total Intrusion Penalty (per episode)', 'red'),
    ('mean_reward_intrusion', 'Intrusion Penalty (per step)', 'red'),
    ('sum_reward_proximity', 'Total Proximity Penalty (per episode)', 'orange'),
    ('mean_reward_proximity', 'Proximity Penalty (per step)', 'orange'),
    ('total_intrusions', 'Total Intrusions (per episode)', 'darkred'),
    ('terminated_waypoint', 'Waypoints Reached (per episode)', 'cyan'),
    ('waypoint_success_rate', 'Waypoint Success Rate (%)', 'teal'),
    ('steps', 'Episode Length (steps)', 'purple'),
]

# Filter to only include columns that exist
reward_components = [(col, label, color) for col, label, color in reward_components if col in episode_df.columns]

# Moving average function
def moving_average(x, window_size):
    if len(x) < window_size:
        window_size = max(1, len(x) // 2)
    return np.convolve(x, np.ones(window_size), mode='valid') / window_size

window_size = min(50, len(episode_df) // 3)  # Adaptive window size

# --- Create subplot grid ---
num_plots = len(reward_components)
num_cols = 3  # 2 columns for cleaner layout
num_rows = (num_plots + num_cols - 1) // num_cols
fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 4 * num_rows))
axes = axes.flatten() if num_plots > 1 else [axes]

# --- Plot each reward component ---
for i, (col, label, color) in enumerate(reward_components):
    ax = axes[i]
    data = episode_df[col].values
    x_values = episode_df['chronological_index'].values
    
    # Calculate and plot moving average only
    if len(data) >= window_size:
        smoothed_data = moving_average(data, window_size)
        smoothed_x = x_values[window_size - 1:]
        ax.plot(smoothed_x, smoothed_data, label=f'Rolling avg (window={window_size})', 
                color=color, linewidth=2.5)
    else:
        # If not enough data for smoothing, plot raw data
        ax.plot(x_values, data, color=color, linewidth=2.5, label='Data')
    
    ax.set_title(label, fontsize=11, fontweight='bold')
    ax.set_ylabel("Value")
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at 0 for reward components
    if 'reward' in col:
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

# --- Clean up unused subplots ---
for j in range(num_plots, len(axes)):
    axes[j].set_visible(False)

fig.suptitle(f'All {len(episode_df)} Environment Episodes - Chronological Order (RUN_ID={RUN_ID})', 
             fontsize=14, fontweight='bold', y=0.995)
fig.subplots_adjust(hspace=0.35, wspace=0.25, top=0.97)

plt.show()

# Print summary statistics
print("\n" + "="*60)
print("REWARD COMPONENT SUMMARY")
print("="*60)
for col, label, _ in reward_components:
    if col in episode_df.columns:
        data = episode_df[col]
        print(f"\n{label}:")
        print(f"  Mean: {data.mean():.3f}")
        print(f"  Std:  {data.std():.3f}")
        print(f"  Min:  {data.min():.3f}")
        print(f"  Max:  {data.max():.3f}")
        print(f"  First 10 avg: {data.head(10).mean():.3f}")
        print(f"  Last 10 avg:  {data.tail(10).mean():.3f}")

# Print waypoint-specific summary
print("\n" + "="*60)
print("WAYPOINT SUCCESS SUMMARY")
print("="*60)
if 'terminated_waypoint' in episode_df.columns:
    total_waypoints = episode_df['terminated_waypoint'].sum()
    total_agents = episode_df['total_agents'].sum()
    overall_success_rate = (total_waypoints / total_agents * 100) if total_agents > 0 else 0
    
    print(f"\nOverall Statistics:")
    print(f"  Total waypoints reached: {int(total_waypoints)}")
    print(f"  Total agents: {int(total_agents)}")
    print(f"  Overall success rate: {overall_success_rate:.2f}%")
    
    print(f"\nTrend Analysis:")
    first_50 = episode_df.head(50)['waypoint_success_rate'].mean() if len(episode_df) >= 50 else episode_df.head(len(episode_df))['waypoint_success_rate'].mean()
    last_50 = episode_df.tail(50)['waypoint_success_rate'].mean() if len(episode_df) >= 50 else episode_df.tail(len(episode_df))['waypoint_success_rate'].mean()
    print(f"  First 50 episodes avg: {first_50:.2f}%")
    print(f"  Last 50 episodes avg: {last_50:.2f}%")
    improvement = last_50 - first_50
    if improvement > 0:
        print(f"  ✅ IMPROVEMENT: +{improvement:.2f}% waypoint success")
    else:
        print(f"  ❌ REGRESSION: {improvement:.2f}% waypoint success")

 