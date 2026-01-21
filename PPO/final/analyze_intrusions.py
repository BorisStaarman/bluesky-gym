"""
Simple script to analyze intrusion trends from CSV files
Run this after training to see if intrusions decreased over time
"""
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def analyze_intrusions(metrics_dir="metrics_29_10", run_id=None):
    """Analyze intrusion trends from CSV files"""
    
    # First try to find the merged CSV file
    if run_id:
        # Build path relative to script directory (go up 2 levels from script to repo root)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(script_dir))
        merged_file = os.path.join(repo_root, metrics_dir, f"run_{run_id}", "all_agents_merged_sorted.csv")
        
        if os.path.exists(merged_file):
            print(f"Using merged file: {merged_file}")
            combined_df = pd.read_csv(merged_file)
            print(f"Loaded {len(combined_df)} rows from merged file")
        else:
            print(f"Merged file not found at: {merged_file}")
            print(f"Falling back to individual agent files")
            csv_pattern = os.path.join(repo_root, metrics_dir, f"run_{run_id}", "pid_*", "*.csv")
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
    print(f"Unique episodes in raw data: {combined_df['episode_index'].nunique()}")
    print(f"Episode range: {combined_df['episode_index'].min()} to {combined_df['episode_index'].max()}")
    
    # Filter to only completed episodes (those with finished_at timestamp)
    if 'finished_at' in combined_df.columns:
        episodes_before = combined_df['episode_index'].nunique()
        combined_df = combined_df[combined_df['finished_at'].notna()]
        episodes_after = combined_df['episode_index'].nunique()
        print(f"\nFiltered to completed episodes: {episodes_before} -> {episodes_after}")
    else:
        print("\nWarning: No 'finished_at' column found - cannot filter incomplete episodes!")
    
    # Count agents per episode to filter out incomplete episodes
    agents_per_episode = combined_df.groupby('episode_index')['agent'].count()
    max_agents = agents_per_episode.max()
    
    print(f"\nAgent count distribution:")
    print(f"  Max agents per episode: {max_agents}")
    print(f"  Min agents per episode: {agents_per_episode.min()}")
    print(f"  Mean agents per episode: {agents_per_episode.mean():.1f}")
    
    # Only filter out episodes with very few agents (< 2), keep the rest
    min_agents_threshold = max(2, max_agents * 0.1)  # At least 2 agents, or 10% of max
    valid_episodes = agents_per_episode[agents_per_episode >= min_agents_threshold].index
    
    print(f"Total episodes: {len(agents_per_episode)}, Valid episodes: {len(valid_episodes)}")
    print(f"Max agents per episode: {max_agents}, Threshold: {min_agents_threshold:.0f}")
    
    # Filter data to only valid episodes
    combined_df = combined_df[combined_df['episode_index'].isin(valid_episodes)]
    
    # Group by episode to get mean intrusions per episode across all agents
    episode_stats = combined_df.groupby('episode_index').agg({
        'total_intrusions': ['mean', 'sum', 'std'],
        'steps': 'mean'
    }).round(2)

    # 10-episode rolling averages (smoothing)
    episode_stats[('total_intrusions', 'mean_roll10')] = (
        episode_stats[('total_intrusions', 'mean')]
        .rolling(window=50, min_periods=1)
        .mean()
        .round(2)
    )
    episode_stats[('total_intrusions', 'sum_roll10')] = (
        episode_stats[('total_intrusions', 'sum')]
        .rolling(window=50, min_periods=1)
        .mean()
        .round(2)
    )
    
    print("Episode Statistics:")
    print("=" * 50)
    print(episode_stats.head(10))
    
    # Plot intrusion trends
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Mean intrusions per episode
    axes[0].plot(
        episode_stats.index,
        episode_stats[('total_intrusions', 'mean')],
        marker='o', linestyle='-', color='red', label='Mean per Episode'
    )
    axes[0].plot(
        episode_stats.index,
        episode_stats[('total_intrusions', 'mean_roll10')],
        linestyle='--', color='darkorange', label='10-ep Rolling Mean'
    )
    axes[0].set_title('Mean Intrusions per Episode (All Agents)')
    axes[0].set_xlabel('Episode Index')
    axes[0].set_ylabel('Mean Intrusions')
    axes[0].grid(True)
    axes[0].legend()
    
    # Total intrusions per episode
    axes[1].plot(
        episode_stats.index,
        episode_stats[('total_intrusions', 'sum')],
        marker='s', linestyle='-', color='darkred', label='Total per Episode'
    )
    axes[1].plot(
        episode_stats.index,
        episode_stats[('total_intrusions', 'sum_roll10')],
        
        #episode_stats['sum_roll10'],
        linestyle='--', color='firebrick', label='10-ep Rolling Total'
    )
    axes[1].set_title('Total Intrusions per Episode (All Agents Combined)')
    axes[1].set_xlabel('Episode Index')
    axes[1].set_ylabel('Total Intrusions')
    axes[1].grid(True)
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Summary statistics
    print(f"\nIntrusion Summary:")
    print(f"Total episodes: {len(episode_stats)}")
    print(f"Average intrusions per episode: {episode_stats[('total_intrusions', 'mean')].mean():.2f}")
    print(f"First 5 episodes avg: {episode_stats[('total_intrusions', 'mean')].head().mean():.2f}")
    print(f"Last 5 episodes avg: {episode_stats[('total_intrusions', 'mean')].tail().mean():.2f}")
    print(f"10-ep rolling mean (first 5): {episode_stats[('total_intrusions', 'mean_roll10')].head().tolist()}")
    print(f"10-ep rolling mean (last 5): {episode_stats[('total_intrusions', 'mean_roll10')].tail().tolist()}")
    
    improvement = episode_stats[('total_intrusions', 'mean')].head().mean() - episode_stats[('total_intrusions', 'mean')].tail().mean()
    print(f"Improvement: {improvement:.2f} fewer intrusions per episode")

if __name__ == "__main__":
    from run_config import RUN_ID
    analyze_intrusions(run_id=RUN_ID)