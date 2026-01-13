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
agg_dict = {
    'episode_index': 'first',
    'finished_at': 'first',
    'steps': 'mean',
    'total_intrusions': 'sum',
    'terminated_waypoint': 'sum',  # Sum of waypoints reached per episode
    'truncated': 'sum',  # How many agents were truncated (out of bounds / time limit)
}

# Add reward components dynamically (they may not all exist depending on environment version)
reward_cols = [
    'mean_reward_drift', 'sum_reward_drift',
    'mean_reward_progress', 'sum_reward_progress',
    'mean_reward_intrusion', 'sum_reward_intrusion',
    'mean_reward_path_efficiency', 'sum_reward_path_efficiency',
    'mean_reward_proximity', 'sum_reward_proximity',  # Old name, keep for backwards compat
    'mean_reward_boundary', 'sum_reward_boundary',
    'mean_reward_step', 'sum_reward_step',
]

for col in reward_cols:
    if col in df.columns:
        agg_dict[col] = 'mean'

episode_df = df.groupby('env_episode_id').agg(agg_dict).reset_index()

# Calculate waypoint success rate per episode (percentage of agents that reached waypoint)
# Count agents per episode
agents_per_episode = df.groupby('env_episode_id').size()
episode_df['total_agents'] = episode_df['env_episode_id'].map(agents_per_episode)
episode_df['waypoint_success_rate'] = (episode_df['terminated_waypoint'] / episode_df['total_agents']) * 100

# Sort by finished_at to get chronological order
episode_df = episode_df.sort_values('finished_at').reset_index(drop=True)
episode_df['chronological_index'] = range(1, len(episode_df) + 1)

print(f"\nAnalyzing {len(episode_df)} actual environment episodes (chronologically ordered)")

# Define reward components to plot
reward_components = [
    # Reward components - all use sum (total per episode)
    ('sum_reward_progress', 'Progress Reward (total per episode)', 'green'),
    ('sum_reward_drift', 'Drift Penalty (total per episode)', 'blue'),
    ('sum_reward_intrusion', 'Intrusion Penalty (total per episode)', 'red'),
    ('sum_reward_boundary', 'Boundary Penalty (total per episode)', 'darkred'),
    ('sum_reward_step', 'Step Penalty (total per episode)', 'purple'),
    
    # Performance metrics
    ('total_intrusions', 'Total Intrusions (per episode)', 'crimson'),
    ('terminated_waypoint', 'Waypoints Reached (per episode)', 'cyan'),
    ('waypoint_success_rate', 'Waypoint Success Rate (%)', 'teal'),
    ('truncated', 'Agents Truncated (out of bounds / time limit)', 'brown'),
    ('steps', 'Episode Length (steps)', 'navy'),
]

# Filter to only include columns that exist
reward_components = [(col, label, color) for col, label, color in reward_components if col in episode_df.columns]

# Exclude last 5% of data from plotting
cutoff_index = int(len(episode_df) * 0.95)
episode_df_plot = episode_df.iloc[:cutoff_index].copy()
print(f"Plotting first {cutoff_index} episodes (95% of data), excluding last {len(episode_df) - cutoff_index} episodes")

# Moving average function
def moving_average(x, window_size):
    if len(x) < window_size:
        window_size = max(1, len(x) // 2)
    return np.convolve(x, np.ones(window_size), mode='valid') / window_size

window_size = min(50, len(episode_df_plot) // 3)  # Adaptive window size

# --- Create subplot grid ---
num_plots = len(reward_components)
num_cols = 3  # 3 columns for cleaner layout
num_rows = (num_plots + num_cols - 1) // num_cols
fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 4 * num_rows))
axes = axes.flatten() if num_plots > 1 else [axes]

# --- Plot each reward component ---
for i, (col, label, color) in enumerate(reward_components):
    ax = axes[i]
    data = episode_df_plot[col].values
    x_values = episode_df_plot['chronological_index'].values
    
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

fig.suptitle(f'First {cutoff_index} Episodes (95% of data) - Chronological Order (RUN_ID={RUN_ID})', 
             fontsize=14, fontweight='bold', y=0.995)
fig.subplots_adjust(hspace=0.35, wspace=0.25, top=0.97)

plt.show()

# Print summary statistics
# Calculate total reward per episode (sum of all reward components)
sum_reward_cols = [col for col, _, _ in reward_components if col.startswith('sum_reward_') and col in episode_df.columns]
if sum_reward_cols:
    episode_df['total_reward'] = episode_df[sum_reward_cols].sum(axis=1)
    print(f"\n{'='*60}")
    print(f"TOTAL REWARD (sum of all components)")
    print(f"{'='*60}")
    print(f"  Mean: {episode_df['total_reward'].mean():.3f}")
    print(f"  Std:  {episode_df['total_reward'].std():.3f}")
    print(f"  Min:  {episode_df['total_reward'].min():.3f}")
    print(f"  Max:  {episode_df['total_reward'].max():.3f}")
    print(f"  First 50 avg: {episode_df['total_reward'].head(50).mean():.3f}")
    print(f"  Last 50 avg:  {episode_df['total_reward'].tail(50).mean():.3f}")
    improvement = episode_df['total_reward'].tail(50).mean() - episode_df['total_reward'].head(50).mean()
    print(f"  Improvement: {improvement:+.3f} ({improvement/abs(episode_df['total_reward'].head(50).mean())*100:+.1f}%)")

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
        first_n = min(50, len(data) // 4)
        last_n = min(50, len(data) // 4)
        print(f"  First {first_n} avg: {data.head(first_n).mean():.3f}")
        print(f"  Last {last_n} avg:  {data.tail(last_n).mean():.3f}")
        if not col.startswith('steps') and not col == 'waypoint_success_rate':
            improvement = data.tail(last_n).mean() - data.head(first_n).mean()
            print(f"  Change: {improvement:+.3f}")

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

 