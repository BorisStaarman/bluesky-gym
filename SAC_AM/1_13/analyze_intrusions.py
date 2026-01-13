"""
Simple script to analyze intrusion trends from CSV files
Run this after training to see if intrusions decreased over time
"""
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def analyze_intrusions(metrics_dir="metrics", run_id=None):
    """Analyze intrusion trends from CSV files"""
    
    # First try to find the merged CSV file
    if run_id:
        # Build path relative to script directory (same folder as the script)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        merged_file = os.path.join(script_dir, metrics_dir, f"run_{run_id}", "all_agents_merged_sorted.csv")
        
        if os.path.exists(merged_file):
            print(f"Using merged file: {merged_file}")
            combined_df = pd.read_csv(merged_file)
            print(f"Loaded {len(combined_df)} rows from merged file")
        else:
            print(f"Merged file not found at: {merged_file}")
            print(f"Falling back to individual agent files")
            csv_pattern = os.path.join(script_dir, metrics_dir, f"run_{run_id}", "pid_*", "*.csv")
            csv_files = glob.glob(csv_pattern)
            combined_df = None
    else:
        # Find all CSV files from all runs
        csv_pattern = os.path.join(metrics_dir, "run_*", "pid_*", "*.csv")
        csv_files = glob.glob(csv_pattern)
        combined_df = None
    
    # If we didn't load the merged file, read individual files
    if combined_df is None:
        if not csv_files:
            print(f"No CSV files found in {metrics_dir}")
            return
        
        all_data = []
        # Read all agent data
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                agent_name = os.path.basename(file).replace('.csv', '')
                if 'agent' not in df.columns:
                    df['agent'] = agent_name
                all_data.append(df)
            except Exception as e:
                print(f"Error reading {file}: {e}")
        
        if not all_data:
            print("No valid data found")
            return
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
    
    print(f"Total rows in combined data: {len(combined_df)}")
    print(f"Unique episode_index in raw data: {combined_df['episode_index'].nunique()}")
    print(f"Episode_index range: {combined_df['episode_index'].min()} to {combined_df['episode_index'].max()}")
    
    # Create unique environment episode ID (episode_index is shared across workers)
    if 'pid' in combined_df.columns:
        combined_df['env_episode_id'] = combined_df['episode_index'].astype(str) + '_' + combined_df['pid']
        print(f"\n✅ Found {combined_df['pid'].nunique()} workers (PIDs)")
        print(f"✅ Created env_episode_id: {combined_df['env_episode_id'].nunique()} unique environment episodes")
    else:
        print("\n⚠️  Warning: No 'pid' column found - treating episode_index as unique")
        combined_df['env_episode_id'] = combined_df['episode_index'].astype(str)
    
    # Filter to only completed episodes (those with finished_at timestamp)
    if 'finished_at' in combined_df.columns:
        episodes_before = combined_df['env_episode_id'].nunique()
        combined_df = combined_df[combined_df['finished_at'].notna()]
        episodes_after = combined_df['env_episode_id'].nunique()
        print(f"\nFiltered to completed episodes: {episodes_before} -> {episodes_after}")
    else:
        print("\nWarning: No 'finished_at' column found - cannot filter incomplete episodes!")
    
    # Count agents per environment episode to filter out incomplete episodes
    agents_per_episode = combined_df.groupby('env_episode_id')['agent'].count()
    max_agents = agents_per_episode.max()
    
    print(f"\nAgent count distribution per environment episode:")
    print(f"  Max agents per episode: {max_agents}")
    print(f"  Min agents per episode: {agents_per_episode.min()}")
    print(f"  Mean agents per episode: {agents_per_episode.mean():.1f}")
    
    # Only filter out episodes with very few agents (< 3), keep episodes with at least half the agents
    min_agents_threshold = max(3, max_agents * 0.5)  # At least 3 agents, or 50% of max
    valid_episodes = agents_per_episode[agents_per_episode >= min_agents_threshold].index
    
    print(f"Total episodes: {len(agents_per_episode)}, Valid episodes: {len(valid_episodes)}")
    print(f"Max agents per episode: {max_agents}, Threshold: {min_agents_threshold:.0f}")
    
    # Filter data to only valid episodes
    combined_df = combined_df[combined_df['env_episode_id'].isin(valid_episodes)]
    
    print(f"Columns in combined_df: {list(combined_df.columns)}")
    
    # Use total_intrusions column (this is the actual count of intrusions per agent per episode)
    intrusions_col = 'total_intrusions'
    if intrusions_col not in combined_df.columns:
        print(f"Warning: '{intrusions_col}' not found. Looking for alternative...")
        # Find any column with 'intrusion' in the name
        for col in combined_df.columns:
            if 'intrusion' in col.lower():
                intrusions_col = col
                break
        if intrusions_col not in combined_df.columns:
            raise KeyError("No column containing 'intrusion' found in data!")
    
    print(f"Using intrusions column: {intrusions_col}")
    
    # Sort episodes chronologically by finished_at timestamp
    if 'finished_at' in combined_df.columns:
        # Create a mapping of env_episode_id to its earliest finished_at time
        episode_times = combined_df.groupby('env_episode_id')['finished_at'].min().sort_values()
        # Create a sequential episode number based on chronological order
        episode_order = {ep_id: idx for idx, ep_id in enumerate(episode_times.index, 1)}
        combined_df['episode_number'] = combined_df['env_episode_id'].map(episode_order)
        group_by_col = 'episode_number'
        print(f"✅ Sorted episodes chronologically by finished_at timestamp")
    else:
        group_by_col = 'env_episode_id'
        print("⚠️  No finished_at column - using env_episode_id order")

    # Group by environment episode to get mean intrusions per episode across all agents
    episode_stats = combined_df.groupby(group_by_col).agg({
        intrusions_col: ['mean', 'sum', 'std'],
        'steps': 'mean'
    }).round(2)
    
    # Sort by index to ensure chronological order
    episode_stats = episode_stats.sort_index()

    # 50-episode rolling averages (smoothing)
    episode_stats[(intrusions_col, 'mean_roll50')] = (
        episode_stats[(intrusions_col, 'mean')]
        .rolling(window=100, min_periods=1)
        .mean()
        .round(2)
    )
    episode_stats[(intrusions_col, 'sum_roll50')] = (
        episode_stats[(intrusions_col, 'sum')]
        .rolling(window=100, min_periods=1)
        .mean()
        .round(2)
    )

    print("Episode Statistics:")
    print("=" * 50)
    print(episode_stats.head(10))

    # Plot intrusion trends
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Mean intrusions per episode (averaged across all agents in that episode)
    axes[0].plot(
        episode_stats.index,
        episode_stats[(intrusions_col, 'mean')],
        alpha=0.3, linestyle='-', color='red', linewidth=0.5, label='Mean per Episode'
    )
    axes[0].plot(
        episode_stats.index,
        episode_stats[(intrusions_col, 'mean_roll50')],
        linestyle='-', color='darkorange', linewidth=2, label='50-ep Rolling Mean'
    )
    axes[0].set_title(f'Mean Intrusions per Episode (averaged across agents) - {len(episode_stats)} episodes')
    axes[0].set_xlabel('Episode Number (chronological)')
    axes[0].set_ylabel('Mean Intrusions per Agent')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Total intrusions per episode (sum across all agents)
    axes[1].plot(
        episode_stats.index,
        episode_stats[(intrusions_col, 'sum')],
        alpha=0.3, linestyle='-', color='darkred', linewidth=0.5, label='Total per Episode'
    )
    axes[1].plot(
        episode_stats.index,
        episode_stats[(intrusions_col, 'sum_roll50')],
        linestyle='-', color='firebrick', linewidth=2, label='50-ep Rolling Total'
    )
    axes[1].set_title(f'Total Intrusions per Episode (sum across all agents) - {len(episode_stats)} episodes')
    axes[1].set_xlabel('Episode Number (chronological)')
    axes[1].set_ylabel('Total Intrusions')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    # Summary statistics
    print(f"\n" + "="*60)
    print(f"INTRUSION SUMMARY:")
    print(f"="*60)
    print(f"Total environment episodes analyzed: {len(episode_stats)}")
    print(f"\nMean intrusions per agent per episode:")
    print(f"  Overall average: {episode_stats[(intrusions_col, 'mean')].mean():.4f}")
    print(f"  First 50 episodes avg: {episode_stats[(intrusions_col, 'mean')].head(50).mean():.4f}")
    print(f"  Last 50 episodes avg: {episode_stats[(intrusions_col, 'mean')].tail(50).mean():.4f}")
    
    print(f"\nTotal intrusions per episode (sum across all agents):")
    print(f"  Overall average: {episode_stats[(intrusions_col, 'sum')].mean():.2f}")
    print(f"  First 50 episodes avg: {episode_stats[(intrusions_col, 'sum')].head(50).mean():.2f}")
    print(f"  Last 50 episodes avg: {episode_stats[(intrusions_col, 'sum')].tail(50).mean():.2f}")
    
    # Calculate improvement
    mean_improvement = episode_stats[(intrusions_col, 'mean')].head(50).mean() - episode_stats[(intrusions_col, 'mean')].tail(50).mean()
    sum_improvement = episode_stats[(intrusions_col, 'sum')].head(50).mean() - episode_stats[(intrusions_col, 'sum')].tail(50).mean()
    
    if mean_improvement > 0:
        print(f"\n✅ IMPROVEMENT: {mean_improvement:.4f} fewer intrusions per agent (first 50 vs last 50)")
        print(f"✅ IMPROVEMENT: {sum_improvement:.2f} fewer total intrusions per episode (first 50 vs last 50)")
    else:
        print(f"\n❌ REGRESSION: {abs(mean_improvement):.4f} more intrusions per agent (first 50 vs last 50)")
        print(f"❌ REGRESSION: {abs(sum_improvement):.2f} more total intrusions per episode (first 50 vs last 50)")
    
    print(f"\n50-episode rolling mean (first 5 values): {episode_stats[(intrusions_col, 'mean_roll50')].head().tolist()}")
    print(f"50-episode rolling mean (last 5 values): {episode_stats[(intrusions_col, 'mean_roll50')].tail().tolist()}")
    print(f"="*60)

if __name__ == "__main__":
    from run_config import RUN_ID
    analyze_intrusions(run_id=RUN_ID)