from run_config import RUN_ID
import os
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import math

# get the path to the .csv file
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
root = os.path.join(repo_root, "metrics", f"run_{RUN_ID}", "all_agents_merged_sorted.csv")
df = pd.read_csv(root)

# --- Finished-at summary ---
# Count the number of unique agent-episodes (each episode ended once per agent)
if {'agent', 'episode_index'}.issubset(df.columns):
    episode_keys = df[['agent', 'episode_index']].dropna().drop_duplicates()
    total_agent_episodes = len(episode_keys)
    print(f"Total episodes (agent-episodes) logged: {total_agent_episodes}")
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
    df = df[df['finished_at'].notna()]
    episodes_after = df['episode_index'].nunique()
    print(f"Filtered to completed episodes: {episodes_before} -> {episodes_after}")

# Group by episode to get per-episode statistics
episode_df = df.groupby('episode_index').agg({
    'steps': 'mean',
    'mean_reward_drift': 'mean',
    'mean_reward_progress': 'mean',
    'mean_reward_intrusion': 'mean',
    # 'mean_reward_proximity': 'mean',
    'sum_reward_drift': 'mean',
    'sum_reward_progress': 'mean',
    'sum_reward_intrusion': 'mean',
    # 'sum_reward_proximity': 'mean',
    # 'total_intrusions': 'sum'
}).reset_index()

print(f"\nAnalyzing {len(episode_df)} completed episodes")

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
    ('steps', 'Episode Length (steps)', 'purple'),
]

# Filter to only include columns that exist
reward_components = [(col, label, color) for col, label, color in reward_components if col in episode_df.columns]

# Moving average function
def moving_average(x, window_size):
    if len(x) < window_size:
        window_size = max(1, len(x) // 2)
    return np.convolve(x, np.ones(window_size), mode='valid') / window_size

window_size = min(10, len(episode_df) // 2)  # Adaptive window size

# --- Create subplot grid ---
num_plots = len(reward_components)
num_cols = 2  # 2 columns for cleaner layout
num_rows = (num_plots + num_cols - 1) // num_cols
fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 4 * num_rows))
axes = axes.flatten() if num_plots > 1 else [axes]

# --- Plot each reward component ---
for i, (col, label, color) in enumerate(reward_components):
    ax = axes[i]
    data = episode_df[col].values
    
    # Calculate moving average
    if len(data) >= window_size:
        smoothed_data = moving_average(data, window_size)
        smoothed_index = episode_df['episode_index'].values[window_size - 1:]
        ax.plot(smoothed_index, smoothed_data, label=f'Rolling avg (window={window_size})', 
                color=color, linewidth=2)
    
    # Plot raw data
    ax.plot(episode_df['episode_index'], data, alpha=0.3, label='Raw', 
            color=color, linewidth=0.5)
    
    ax.set_title(label, fontsize=11, fontweight='bold')
    ax.set_xlabel("Episode Index")
    ax.set_ylabel("Value")
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at 0 for reward components
    if 'reward' in col:
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

# --- Clean up unused subplots ---
for j in range(num_plots, len(axes)):
    axes[j].set_visible(False)

fig.suptitle(f'Reward Components Analysis (RUN_ID={RUN_ID})', fontsize=14, fontweight='bold', y=0.995)
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

    