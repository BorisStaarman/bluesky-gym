"""
Plot training reward vs iteration from the saved training_metrics.pkl written by main.py.
"""
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np

from run_config import RUN_ID

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    metrics_file = os.path.join(script_dir, "metrics", f"run_{RUN_ID}", "training_metrics.pkl")
    
    if not os.path.exists(metrics_file):
        print(f"No training_metrics.pkl found at: {metrics_file}")
        print("This file is created at the end of training by main.py.")
        return
    
    with open(metrics_file, 'rb') as f:
        metrics = pickle.load(f)
    reward_history = metrics.get('reward_history', [])
    
    if not reward_history:
        print("No reward data available")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = list(range(1, len(reward_history) + 1))
    ax.plot(iterations, reward_history, linewidth=2, color='steelblue', marker='o', markersize=0, alpha=0.7)
    
    ax.set_xlabel('Training Iteration', fontsize=12)
    ax.set_ylabel('Episode Return Mean', fontsize=12)
    ax.set_title('Training Reward Progress', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    plt.show()
    
    mean_reward = np.mean(reward_history)
    final_reward = reward_history[-1]
    best_reward = np.max(reward_history)
    
    print(f"\nTraining Reward Statistics:")
    print(f"  Iterations: {len(reward_history)}")
    print(f"  Final reward: {final_reward:.2f}")
    print(f"  Best reward: {best_reward:.2f}")
    print(f"  Mean reward: {mean_reward:.2f}")

if __name__ == "__main__":
    main()
