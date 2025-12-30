"""
Plot attention mechanism metrics from saved CSV file.
This allows you to analyze the health of the attention mechanism after training.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from run_config import RUN_ID

# Path to the attention metrics CSV
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, f"attention_metrics_run_{RUN_ID}.csv")

if not os.path.exists(csv_path):
    print(f"❌ Attention metrics CSV not found at: {csv_path}")
    print(f"   Make sure to run training first to generate the metrics file.")
    exit(1)

# Load the CSV
df = pd.read_csv(csv_path)

print(f"✅ Loaded attention metrics from: {csv_path}")
print(f"   Iterations: {len(df)}")
print(f"\n📊 Summary Statistics:")
print(df.describe())

# Create figure with subplots
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Plot 1: Attention Sharpness
axes[0].plot(df['iteration'], df['attention_sharpness'], marker='o', linestyle='-', color='purple', alpha=0.7)
axes[0].set_title('Attention Sharpness (Mean of Max Attention Weights)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Training Iteration')
axes[0].set_ylabel('Attention Sharpness')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=1.0/19, color='r', linestyle='--', alpha=0.5, label='Uniform (1/19 ≈ 0.053)')
axes[0].legend()

# Add annotation
mean_sharpness = df['attention_sharpness'].mean()
axes[0].text(0.02, 0.98, f'Mean: {mean_sharpness:.4f}', 
            transform=axes[0].transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 2: Weight Norms
axes[1].plot(df['iteration'], df['wq_weight_norm'], marker='o', linestyle='-', label='W_q (Query)', alpha=0.7)
axes[1].plot(df['iteration'], df['wk_weight_norm'], marker='s', linestyle='-', label='W_k (Key)', alpha=0.7)
axes[1].plot(df['iteration'], df['wv_weight_norm'], marker='^', linestyle='-', label='W_v (Value)', alpha=0.7)
axes[1].set_title('Attention Layer Weight Norms (First Head)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Training Iteration')
axes[1].set_ylabel('Weight Norm (L2)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Gradient Norms (if available)
# Filter out NaN values for gradient norms
wq_grad = df['wq_grad_norm'].dropna()
wk_grad = df['wk_grad_norm'].dropna()
wv_grad = df['wv_grad_norm'].dropna()

if len(wq_grad) > 0:
    axes[2].plot(df['iteration'], df['wq_grad_norm'], marker='o', linestyle='-', label='W_q (Query)', alpha=0.7)
    axes[2].plot(df['iteration'], df['wk_grad_norm'], marker='s', linestyle='-', label='W_k (Key)', alpha=0.7)
    axes[2].plot(df['iteration'], df['wv_grad_norm'], marker='^', linestyle='-', label='W_v (Value)', alpha=0.7)
    axes[2].set_title('Attention Layer Gradient Norms (First Head)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Training Iteration')
    axes[2].set_ylabel('Gradient Norm (L2)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale('log')  # Log scale often better for gradients
else:
    axes[2].text(0.5, 0.5, 'No Gradient Data Available\n(May need backward pass during metrics collection)', 
                ha='center', va='center', transform=axes[2].transAxes, fontsize=12)
    axes[2].set_title('Attention Layer Gradient Norms (First Head)', fontsize=14, fontweight='bold')

plt.tight_layout()

# Save the plot
output_path = os.path.join(script_dir, f"attention_analysis_run_{RUN_ID}.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n📊 Attention analysis plot saved to: {output_path}")

# Show the plot
plt.show()

print("\n✅ Analysis complete!")
print("\n💡 Interpretation Tips:")
print("   • Attention Sharpness > 0.2: Model is focusing on specific neighbors (good)")
print("   • Attention Sharpness ≈ 0.05: Model is averaging uniformly (may indicate issues)")
print("   • Weight Norms should be stable (not growing/shrinking unboundedly)")
print("   • Gradient Norms help detect vanishing/exploding gradients")
