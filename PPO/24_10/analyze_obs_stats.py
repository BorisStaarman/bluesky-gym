"""
Analyze observation statistics from obs_stats.csv files.
Reads the CSV and displays min, max, mean, std for each feature.
"""
import pandas as pd
import os
import sys

def analyze_obs_stats(csv_path):
    """Load and display observation statistics from a CSV file."""
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        return
    
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Display basic info
    print(f"\n{'='*80}")
    print(f"Observation Statistics Analysis")
    print(f"File: {os.path.abspath(csv_path)}")
    print(f"{'='*80}\n")
    
    if 'samples' in df.columns and not df.empty:
        print(f"Total samples: {df['samples'].iloc[0]}")
        if 'episode_index' in df.columns:
            print(f"Episode index: {df['episode_index'].iloc[0]}")
    print()
    
    # Display table with all features
    print(f"{'Feature':<15} {'Index':>6} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12}")
    print(f"{'-'*15} {'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    
    for _, row in df.iterrows():
        feature = row['feature']
        index = int(row['index']) if 'index' in row else -1
        min_val = float(row['min'])
        max_val = float(row['max'])
        mean_val = float(row['mean'])
        std_val = float(row['std']) if 'std' in row else 0.0
        
        print(f"{feature:<15} {index:>6} {min_val:>12.4f} {max_val:>12.4f} {mean_val:>12.4f} {std_val:>12.4f}")
    
    print(f"\n{'='*80}\n")
    
    # Summary by feature group
    print("Summary by feature group:")
    print(f"{'-'*80}")
    
    # Group features by prefix (before underscore)
    df['group'] = df['feature'].str.split('_').str[0]
    
    for group_name in df['group'].unique():
        group_df = df[df['group'] == group_name]
        print(f"\n{group_name.upper()}:")
        print(f"  Overall min: {group_df['min'].min():>10.4f}")
        print(f"  Overall max: {group_df['max'].max():>10.4f}")
        print(f"  Mean of means: {group_df['mean'].mean():>10.4f}")
        print(f"  Mean of stds: {group_df['std'].mean():>10.4f}")
        
        # Flag features outside [-1, 1] range
        outside = group_df[(group_df['min'] < -1.5) | (group_df['max'] > 1.5)]
        if not outside.empty:
            print(f"  ⚠️  Features with values far outside [-1,1]:")
            for _, row in outside.iterrows():
                print(f"      {row['feature']}: [{row['min']:.4f}, {row['max']:.4f}]")
    
    print(f"\n{'='*80}\n")
    
    return df


def find_latest_obs_stats(base_dir=None):
    """Find the most recent obs_stats.csv file in the metrics directory."""
    if base_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(script_dir))
        base_dir = os.path.join(repo_root, "metrics_24_10")
    
    matches = []
    for root, dirs, files in os.walk(base_dir):
        if "obs_stats.csv" in files:
            path = os.path.join(root, "obs_stats.csv")
            mtime = os.path.getmtime(path)
            matches.append((mtime, path))
    
    if not matches:
        return None
    
    # Return the most recently modified file
    matches.sort(reverse=True)
    return matches[0][1]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Use provided path
        csv_path = sys.argv[1]
    else:
        # Try to find the latest obs_stats.csv
        print("Searching for latest obs_stats.csv...")
        csv_path = find_latest_obs_stats()
        
        if csv_path is None:
            print("No obs_stats.csv found in bluesky-gym/metrics_24_10/")
            print("\nUsage: python analyze_obs_stats.py [path/to/obs_stats.csv]")
            sys.exit(1)
        
        print(f"Found: {csv_path}\n")
    
    df = analyze_obs_stats(csv_path)
