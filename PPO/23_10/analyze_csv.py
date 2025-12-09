from run_config import RUN_ID
import os
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import math

# get the path to the .csv file
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
root = os.path.join(repo_root, "metrics_23_10", f"run_{RUN_ID}", "all_agents_merged_sorted.csv")
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

# available columns: 'episode_index', 'steps', 'mean_reward_drift', 'mean_reward_progress',
# 'mean_reward_intrusion', 'sum_reward_drift', 'sum_reward_progress', 'sum_reward_intrusion',
# 'finished_at', 'pid', 'agent', and optionally 'mean_reward_proximity', 'sum_reward_proximity'
columns = df.columns

# Explicitly choose which columns to plot (will filter to those present)
cols_to_plot = [
    'steps',
    'mean_reward_drift', 'sum_reward_drift',
    'mean_reward_intrusion', 'sum_reward_intrusion',
    'mean_reward_progress', 'sum_reward_progress',
    'mean_reward_proximity', 'sum_reward_proximity',  # new proximity shaping (if present)
    'total_intrusions'
]
# Filter to only include columns that actually exist in the dataframe
cols_to_plot = [c for c in cols_to_plot if c in df.columns]


# moving average over 50 episodes for steps,mean_reward_drift, mean_reward_progress, mean_reward_intrusion
def moving_average(x, window_size):
    return np.convolve(x, np.ones(window_size), mode='valid') / window_size

# 25 - 75 % interquartile bound/range, kan je nog toevoegen1!!!???

# adjust 
window_size = 100

# --- 1. Set up the subplot grid ---
num_plots = len(cols_to_plot)
num_cols = 3
num_rows = (num_plots + num_cols - 1) // num_cols
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
axes = axes.flatten()

# --- 2. Loop through columns and plot on each subplot ---  
for i, col in enumerate(cols_to_plot):
    ax = axes[i]
    smoothed_data = moving_average(df[col], window_size)
    smoothed_index = df.index[window_size - 1:]

    ax.plot(smoothed_index, smoothed_data, label=f'Smoothed (window={window_size})')
    ax.plot(df.index, df[col], alpha=0.3, label='Original')

    ax.set_title(col)
    ax.set_xlabel("Index")
    ax.set_ylabel(col)
    ax.legend()
    ax.grid(True)

# --- 3. Clean up and adjust spacing ---
# Hide any unused subplots
for j in range(num_plots, len(axes)):
    axes[j].set_visible(False)

# ** NEW: Adjust the spacing between plots **
# Experiment with these values to get the look you want.
fig.subplots_adjust(hspace=0.4, wspace=0.3)

plt.show()

