"""
Quick plotting tool for evaluation_progress.csv written by main.py's fixed-seed eval.

Shows 4 time series vs training iteration:
- avg_intrusions (lower is better)
- waypoint_rate in % (higher is better)
- avg_reward (trend; scale depends on your penalties)
- avg_length (episode length)

Includes a 5-iteration rolling average overlay for each metric.
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

from run_config import RUN_ID



def main(window: int = 5):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "metrics_23_10", f"run_{RUN_ID}", "evaluation_progress.csv")

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

    # Rolling averages
    roll = df.rolling(window=window, min_periods=1).mean()

    # Prepare figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    # 1) Avg intrusions
    axes[0].plot(df["iteration"], df["avg_intrusions"], marker="o", alpha=0.4, label="raw")
    axes[0].plot(df["iteration"], roll["avg_intrusions"], color="crimson", label=f"roll-{window}")
    axes[0].set_title("Avg intrusions per episode")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("intrusions")
    axes[0].grid(True)
    axes[0].legend()

    # 2) Waypoint rate (%)
    axes[1].plot(df["iteration"], df["waypoint_rate"] * 100.0, marker="o", alpha=0.4, label="raw")
    axes[1].plot(df["iteration"], roll["waypoint_rate"] * 100.0, color="darkgreen", label=f"roll-{window}")
    axes[1].set_title("Waypoint success rate (%)")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("%")
    axes[1].grid(True)
    axes[1].legend()

    # 3) Avg reward
    axes[2].plot(df["iteration"], df["avg_reward"], marker="o", alpha=0.4, label="raw")
    axes[2].plot(df["iteration"], roll["avg_reward"], color="slateblue", label=f"roll-{window}")
    axes[2].set_title("Average reward (eval)")
    axes[2].set_xlabel("iteration")
    axes[2].set_ylabel("reward")
    axes[2].grid(True)
    axes[2].legend()

    # 4) Avg length
    axes[3].plot(df["iteration"], df["avg_length"], marker="o", alpha=0.4, label="raw")
    axes[3].plot(df["iteration"], roll["avg_length"], color="sienna", label=f"roll-{window}")
    axes[3].set_title("Average episode length (steps)")
    axes[3].set_xlabel("iteration")
    axes[3].set_ylabel("steps")
    axes[3].grid(True)
    axes[3].legend()

    fig.suptitle(f"Evaluation progress (RUN_ID={RUN_ID})")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Print a compact tail summary
    last = df.tail(1).iloc[0]
    print("\nLast eval row:")
    print(
        f"iter={int(last['iteration'])} | avg_intr={last['avg_intrusions']:.2f} | "
        f"wp_rate={last['waypoint_rate']*100:.1f}% | avg_len={last['avg_length']:.1f} | "
        f"avg_rew={last['avg_reward']:.2f}"
    )


if __name__ == "__main__":
    # Optional: pass a rolling window as first arg
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    main(window=w)
