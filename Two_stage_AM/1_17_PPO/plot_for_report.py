"""
Quick plotting tool for evaluation_progress.csv written by main.py's fixed-seed eval.

Shows avg intrusions (left y-axis) and waypoint reached % (right y-axis) vs training iteration.
"""
import os
import sys
import pandas as pd 
import matplotlib.pyplot as plt

from run_config import RUN_ID



def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(script_dir))
    csv_path = os.path.join(script_dir, "metrics", f"run_{RUN_ID}", "evaluation_progress.csv")

    if not os.path.exists(csv_path):
        print(f"No evaluation_progress.csv found at: {csv_path}\n"
              f"Make sure main.py ran with periodic eval enabled (EVALUATION_INTERVAL).")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if df.empty:
        print("evaluation_progress.csv is empty")
        sys.exit(0)

    # Ensure iteration sorted
    df = df.sort_values("iteration").reset_index(drop=True)

    # Create figure with single subplot and twin y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left y-axis: Average intrusions
    color1 = 'crimson'
    ax1.set_xlabel('Training Iteration', fontsize=12)
    ax1.set_ylabel('Loss of Separation', color=color1, fontsize=12)
    ax1.plot(df["iteration"], df["avg_intrusions"], marker="o", color=color1, linewidth=2, label="Average Intrusions", zorder=3)
    
    # Add IQR shaded region for intrusions if available
    if 'intrusions_q25' in df.columns and 'intrusions_q75' in df.columns:
        ax1.fill_between(df["iteration"], df["intrusions_q25"], df["intrusions_q75"], 
                         color=color1, alpha=0.15, label="IQR (25th-75th percentile)")
    
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Right y-axis: Waypoint success rate
    ax2 = ax1.twinx()
    color2 = 'darkgreen'
    ax2.set_ylabel('Waypoint Success Rate (%)', color=color2, fontsize=12)
    ax2.plot(df["iteration"], df["waypoint_rate"] * 100.0, marker="s", color=color2, linewidth=2, label="Waypoint Success Rate (%)", zorder=3)
    
    # Add IQR shaded region for waypoint rate if available
    if 'waypoint_q25' in df.columns and 'waypoint_q75' in df.columns:
        ax2.fill_between(df["iteration"], df["waypoint_q25"], df["waypoint_q75"], 
                         color=color2, alpha=0.15, label="IQR (25th-75th percentile)")
    
    ax2.tick_params(axis='y', labelcolor=color2)

    # Add combined legend at the top center
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    all_lines = lines1 + lines2
    all_labels = labels1 + labels2
    # place legend above the plot, centered, with two columns if space allows
    ax1.legend(all_lines, all_labels, loc='upper center', bbox_to_anchor=(0.5, 1.08),
               ncol=2, frameon=False)

    fig.suptitle(f"Evaluation during training", fontsize=14, fontweight='bold')
    # make room at top for the legend
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Print a compact tail summary
    last = df.tail(1).iloc[0]
    print("\nLast eval row:")
    print(
        f"iter={int(last['iteration'])} | avg_intr={last['Loss of Separation']:.2f} | "
        f"wp_rate={last['waypoint_rate']*100:.1f}% | avg_len={last['avg_length']:.1f} | "
        f"avg_rew={last['avg_reward']:.2f}"
    )


if __name__ == "__main__":
    main()
