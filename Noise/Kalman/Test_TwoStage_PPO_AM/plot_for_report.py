"""
Quick plotting tool for evaluation_progress.csv written by main.py's fixed-seed eval.

Shows avg intrusions (left y-axis) and waypoint reached % (right y-axis) vs training iteration.
"""
import os
import sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from run_config import RUN_ID



def interquartile_mean(values: pd.Series) -> float:
    if values.empty:
        return float("nan")
    sorted_values = np.sort(values.to_numpy(dtype=float))
    lower = int(np.floor(0.25 * len(sorted_values)))
    upper = int(np.ceil(0.75 * len(sorted_values)))
    trimmed = sorted_values[lower:upper]
    if trimmed.size == 0:
        trimmed = sorted_values
    return float(trimmed.mean())


def load_plot_data(script_dir: str, run_id: str) -> tuple[pd.DataFrame, bool]:
    raw_csv_path = os.path.join(script_dir, "metrics", f"run_{run_id}", "evaluation_progress_raw.csv")
    if os.path.exists(raw_csv_path):
        raw_df = pd.read_csv(raw_csv_path)
        if not raw_df.empty:
            grouped = raw_df.groupby("iteration", as_index=False).agg(
                intrusions_iqm=("intrusions", interquartile_mean),
                intrusions_q25=("intrusions", lambda s: float(np.percentile(s, 25))),
                intrusions_q75=("intrusions", lambda s: float(np.percentile(s, 75))),
                waypoint_iqm=("waypoint_rate_pct", interquartile_mean),
                waypoint_q25=("waypoint_rate_pct", lambda s: float(np.percentile(s, 25))),
                waypoint_q75=("waypoint_rate_pct", lambda s: float(np.percentile(s, 75))),
            )
            return grouped.sort_values("iteration").reset_index(drop=True), True

    csv_path = os.path.join(script_dir, "metrics", f"run_{run_id}", "evaluation_progress.csv")
    if not os.path.exists(csv_path):
        print(
            f"No evaluation_progress.csv found at: {csv_path}\n"
            f"Make sure main.py ran with periodic eval enabled (EVALUATION_INTERVAL)."
        )
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if df.empty:
        print("evaluation_progress.csv is empty")
        sys.exit(0)

    df = df.sort_values("iteration").reset_index(drop=True)
    df["intrusions_iqm"] = df["avg_intrusions"]
    df["waypoint_iqm"] = df["waypoint_rate"] * 100.0
    if "intrusions_q25" not in df.columns or "intrusions_q75" not in df.columns:
        df["intrusions_q25"] = df["avg_intrusions"]
        df["intrusions_q75"] = df["avg_intrusions"]
    if "waypoint_q25" not in df.columns or "waypoint_q75" not in df.columns:
        waypoint_pct = df["waypoint_rate"] * 100.0
        df["waypoint_q25"] = waypoint_pct
        df["waypoint_q75"] = waypoint_pct
    return df, False


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df, has_true_iqm = load_plot_data(script_dir, RUN_ID)

    # Create figure with single subplot and twin y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left y-axis: Average intrusions
    color1 = 'crimson'
    ax1.set_xlabel('Training Iteration', fontsize=12)
    ax1.set_ylabel('Loss of Separation', color=color1, fontsize=12)
    intrusions_label = "IQM Intrusions" if has_true_iqm else "Average Intrusions"
    ax1.plot(df["iteration"], df["intrusions_iqm"], marker="o", color=color1, linewidth=2, label=intrusions_label, zorder=3)
    
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
    waypoint_label = "IQM Waypoint Success Rate (%)" if has_true_iqm else "Waypoint Success Rate (%)"
    ax2.plot(df["iteration"], df["waypoint_iqm"], marker="s", color=color2, linewidth=2, label=waypoint_label, zorder=3)
    
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

    title = "Evaluation during training"
    if has_true_iqm:
        title = "Evaluation during training (IQM)"
    fig.suptitle(title, fontsize=14, fontweight='bold')
    # make room at top for the legend
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Print a compact tail summary
    last = df.tail(1).iloc[0]
    print("\nLast eval row:")
    print(
        f"iter={int(last['iteration'])} | intrusions={last['intrusions_iqm']:.2f} | "
        f"wp_rate={last['waypoint_iqm']:.1f}%"
    )

    if not has_true_iqm:
        print("\nNote: using aggregate means with IQR bands because evaluation_progress_raw.csv is not available for this run.")


if __name__ == "__main__":
    main()
