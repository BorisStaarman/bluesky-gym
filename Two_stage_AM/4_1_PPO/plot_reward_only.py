"""
Plot training reward vs iteration from the saved training plot data.
"""
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np

from run_config import RUN_ID

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try to load from saved metrics (if available)
    metrics_file = os.path.join(script_dir, "metrics", f"run_{RUN_ID}", "training_metrics.pkl")
    
    if os.path.exists(metrics_file):
        with open(metrics_file, 'rb') as f:
            metrics = pickle.load(f)
        reward_history = metrics.get('reward_history', [])
    else:
        print(f"No training_metrics.pkl found at: {metrics_file}")
        print("This file is created at the end of training.")
        return
    
    if not reward_history:
        print("No reward data available")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = list(range(1, len(reward_history) + 1))
    ax.plot(iterations, reward_history, linewidth=2, color='steelblue', alpha=0.7)
    
    ax.set_xlabel('Training Iteration', fontsize=12)
    ax.set_ylabel('Episode Return Mean', fontsize=12)
    ax.set_title('training reward', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add some statistics
    mean_reward = np.mean(reward_history)
    final_reward = reward_history[-1]
    best_reward = np.max(reward_history)
    
    fig.tight_layout()
    plt.show()
    
    print(f"\nTraining Reward Statistics:")
    print(f"  Iterations: {len(reward_history)}")
    print(f"  Final reward: {final_reward:.2f}")
    print(f"  Best reward: {best_reward:.2f}")
    print(f"  Mean reward: {mean_reward:.2f}")

if __name__ == "__main__":
    main()
