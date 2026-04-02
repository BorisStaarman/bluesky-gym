import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Data Extraction
agents = [15, 20, 25, 30]
noise_labels = ['(0.0, 0.0)', '(3.5, 0.1)', '(5.25, 0.15)', '(7.0, 0.2)']

# Loss of Separation (LoS) Rate (%) 
# Rows correspond to noise levels, Columns correspond to agents [15, 20, 25, 30]
ppo_los = np.array([
    [0.0, 0.0, 0.1, 0.1],
    [0.0, 0.0, 0.1, 0.1],
    [0.0, 0.1, 0.2, 0.4],
    [0.3, 0.5, 0.6, 0.5]
])

mvp_los_old = np.array([
    [1.4, 1.8, 2.5, 4.0],
    [14.8, 19.0, 23.8, 29.9],
    [21.9, 30.5, 36.5, 44.2],
    [33.0, 42.2, 47.5, 56.2]
])

mvp_los = np.array([
    [3.1, 3.8, 7.2, 10.4],
    [9.4, 13.5, 19.2, 22.9],
    [12.6, 18.4, 23.8, 31.6],
    [17.2, 26.0, 33.2, 38.8]
])

# Colors and Markers
colors = ["#1f77b4", '#ff7f0e', '#2ca02c', '#d62728']
markers = ['o', 's', '^', 'D']

# Create a single, clear figure
fig, ax = plt.subplots(figsize=(10, 7))

x_noise = np.arange(len(noise_labels))

# Plot the data with thicker lines and larger markers for better readability
for idx, agent in enumerate(agents):
    # Plot MVP (Solid lines, showing baseline)
    ax.plot(x_noise, mvp_los[:, idx], linestyle='-', color=colors[idx], 
            marker=markers[idx], markersize=8, linewidth=2.5, 
            label=f'MVP: {agent} Ag.')
    
    # Plot PPO (Dashed lines, showing improvement)
    ax.plot(x_noise, ppo_los[:, idx], linestyle='--', color=colors[idx], 
            marker=markers[idx], markersize=8, linewidth=2.5, 
            label=f'PPO: {agent} Ag.')

# Formatting axes and labels
ax.set_xlabel('Noise Level (Positional $\sigma$, Velocity $\sigma$)', fontsize=14, labelpad=10)
ax.set_ylabel('Loss of Separation (LoS) Rate (%)', fontsize=14, labelpad=10)
ax.set_title('Sensitivity to Noise and Density: MVP vs PPO', fontsize=16, pad=15)

# Add baseline for No CR across the different noise levels
no_cr_los = [62.3, 72.0, 78.8, 82.8]
ax.plot(x_noise, no_cr_los, color='purple', linestyle=':', marker='x', markersize=8, linewidth=2.5, label='No CR')

ax.set_xticks(x_noise)
ax.set_xticklabels(noise_labels, rotation=35, fontsize=12)

# Logarithmic scaling but safely handling 0.0
ax.set_yscale('symlog', linthresh=0.1)
ax.set_ylim(bottom=0, top=100) # Maximum is 100%

# Explicitly set y-ticks to make values like 20%, 30%, etc. clear, but spaced out to avoid clutter
yticks = [0, 0.1, 0.5, 1, 5, 10, 20, 30, 50, 70, 100]
ax.set_yticks(yticks)

# Custom formatter to remove scientific notation (e.g., 10^1) and show standard numbers
formatter = ticker.FuncFormatter(lambda y, _: '{:g}'.format(y))
ax.yaxis.set_major_formatter(formatter)
ax.tick_params(axis='y', labelsize=12)

# Refined Grid
ax.grid(True, which="major", ls="-", alpha=0.4, color='gray')
ax.grid(True, which="minor", ls="--", alpha=0.2, color='gray')

# Reorder the legend so MVP and PPO are grouped cleanly
handles, labels = ax.get_legend_handles_labels()

# Create dummy lines for the legend to explain the line styles
import matplotlib.lines as mlines
dummy_mvp = mlines.Line2D([], [], color='black', linestyle='-', label='MVP (Solid)', linewidth=2.5)
dummy_ppo = mlines.Line2D([], [], color='black', linestyle='--', label='PPO (Dashed)', linewidth=2.5)
dummy_nocr = mlines.Line2D([], [], color='purple', linestyle=':', marker='x', label='No CR', linewidth=2.5)

# Keep only one set of handles for the colors (agents)
agent_handles = [h for h, l in zip(handles, labels) if 'MVP' in l]
agent_labels = [f'{agents[i]} Agents' for i in range(len(agents))]

# Combine them (with agents in reverse order)
ordered_handles = [dummy_nocr, dummy_mvp, dummy_ppo] + agent_handles[::-1]
ordered_labels = ['No CR', 'MVP (Solid)', 'PPO (Dashed)'] + agent_labels[::-1]

ax.legend(ordered_handles, ordered_labels,
          bbox_to_anchor=(1.03, 1), loc='upper left', 
          ncol=1, fontsize=11, title="Models & Agent Density", title_fontsize=12,
          frameon=True, shadow=True)

plt.tight_layout()


# ==========================================
# PLOT 2: ONLY PPO DATA (Linear Scale)
# ==========================================
fig2, ax2 = plt.subplots(figsize=(8, 6))

for idx, agent in enumerate(agents):
    # Plot ONLY PPO (Solid lines since it's the only model shown)
    ax2.plot(x_noise, ppo_los[:, idx], linestyle='-', color=colors[idx], 
             marker=markers[idx], markersize=9, linewidth=2.5, 
             label=f'{agent} Agents')

# Formatting axes and labels
ax2.set_xlabel('Noise Level (Positional $\sigma$, Velocity $\sigma$)', fontsize=14, labelpad=10)
ax2.set_ylabel('Loss of Separation (LoS) Rate (%)', fontsize=14, labelpad=10)
ax2.set_title('Sensitivity to Noise: PPO Model Only', fontsize=16, pad=15)

ax2.set_xticks(x_noise)
ax2.set_xticklabels(noise_labels, rotation=35, fontsize=12)

# Linear scaling
ax2.set_ylim(bottom=0, top=1.0) # Set a clean top limit since the max is around 0.6%
ax2.tick_params(axis='y', labelsize=12)

# Refined Grid
ax2.grid(True, which="major", ls="-", alpha=0.4, color='gray')

# Legend
ax2.legend(loc='upper left', fontsize=11, title="Agent Density", title_fontsize=12,
           frameon=True, shadow=True)

plt.tight_layout()

plt.show()
