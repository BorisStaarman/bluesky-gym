import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import defaultdict, deque

script_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up correctly from .../Noise/Kalman/PPO_Symmetric_Autoencoder to root bluesky-gym
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import bluesky as bs
import bluesky_gym.envs.common.functions as fn
from bluesky_gym.envs.ma_env_two_stage_AM_PPO_NOISE_autoencoder import SectorEnv, AE_DELTA_NORM, AE_VEL_NORM

NM2KM = 1.852

def get_raw_ae_mse(env, agent_id):
    """Helper to recalculate AE MSE without the np.clip() cap applied in the Env."""
    window = env._obs_windows.get(agent_id)
    raw = np.array(list(window), dtype=np.float32)
    
    ae_input = np.zeros_like(raw)
    ae_input[0, 2] = raw[0, 2] / AE_VEL_NORM
    ae_input[0, 3] = raw[0, 3] / AE_VEL_NORM
    for t in range(1, env._ae_window_size):
        ae_input[t, 0] = (raw[t, 0] - raw[t - 1, 0]) / AE_DELTA_NORM
        ae_input[t, 1] = (raw[t, 1] - raw[t - 1, 1]) / AE_DELTA_NORM
        ae_input[t, 2] = raw[t, 2] / AE_VEL_NORM
        ae_input[t, 3] = raw[t, 3] / AE_VEL_NORM
        
    x = torch.tensor(ae_input.flatten(), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        x_rec = env._ae_model(x)
        
    return float(torch.mean((x - x_rec) ** 2).item())

def main():
    print("Starting Autoencoder Performance Setup...")
    
    # Find the pre-trained autoencoder path
    ae_path = os.path.join(script_dir, "autoencoder_pretrained.pt")
    if not os.path.exists(ae_path):
        print(f"Error: Could not find autoencoder model at {ae_path}")
        print("Make sure you place the matching 'autoencoder_pretrained.pt' in the same folder.")
        return

    # Initiate evaluating environment 
    n_agents = 20
    env = SectorEnv(
        render_mode=None, 
        n_agents=n_agents, 
        run_id="AE_perf_evaluation",
        autoencoder_path=ae_path
    )
    
    obs, info = env.reset()
    
    true_errors = []
    ae_signals = []
    
    steps_to_collect = 2000
    print(f"Running simulation directly for {steps_to_collect} frame steps to gather noise data...")
    
    # Store true trajectories to compare window-vs-window
    true_history = defaultdict(lambda: deque(maxlen=env._ae_window_size))
    
    # Collect real environment data
    for step in range(steps_to_collect):
        # We need an action dictionary to progress the environment.
        # SectorEnv expects continuous actions [-1, 1] for [heading, velocity]
        actions = {agent_id: np.array([0.0, 0.0], dtype=np.float32) for agent_id in env._agent_ids}
        
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        
        # Test performance across all active agents
        for agent_id in env.agents:
            if agent_id not in bs.traf.id:
                continue
            ac_idx = bs.traf.id.index(agent_id)
            
            # Fetch True precise position from the BlueSky simulation backend
            true_loc = fn.latlong_to_nm(
                env.center, 
                np.array([bs.traf.lat[ac_idx], bs.traf.lon[ac_idx]])
            ) * NM2KM * 1000.0  # Converted to meters
            
            true_history[agent_id].append(true_loc)
            
            # Only record if the agent's sliding window has completely filled up
            window = env._obs_windows.get(agent_id)
            
            if window is not None and len(window) >= env._ae_window_size and len(true_history[agent_id]) == env._ae_window_size:
                
                # Get the 5-step noisy trajectory
                raw_window = np.array(list(window)) # shape (5, 4) where [0:2] is position
                noisy_locs = raw_window[:, 0:2] # shape (5, 2)
                
                # Get the 5-step true trajectory
                true_locs = np.array(list(true_history[agent_id])) # shape (5, 2)
                
                # Calculate the AVERAGE positional noise error over the whole 5-step sequence
                # This matches what the AE is actually looking at
                average_window_error = np.mean(np.linalg.norm(noisy_locs - true_locs, axis=1))
                
                # Use our new function to get the raw unclipped MSE
                raw_mse_signal = get_raw_ae_mse(env, agent_id)
                
                true_errors.append(average_window_error)
                ae_signals.append(raw_mse_signal)
                
        # Automatically handle early episode resets so we can fulfill all desired steps
        if terminateds.get("__all__", False) or truncateds.get("__all__", False):
            env.reset()
            true_history.clear() # Clear tracking
            
    env.close()
    
    if not true_errors:
        print("Warning: No valid data point found. The sliding window size might be too large.")
        return

    # Convert collected logs to numpy for matplotlib
    true_errors = np.array(true_errors)
    ae_signals = np.array(ae_signals)
    
    print(f"Simulation Complete. Successfully gathered {len(true_errors)} comparative data points.")
    
    # Generate scatter plot
    plt.figure(figsize=(9, 6))
    
    plt.scatter(true_errors, ae_signals, alpha=0.3, s=20, marker='o', color='purple', edgecolors='none')
    
    plt.title('Autoencoder reconstruction error validation', fontsize=14, fontweight="bold")
    plt.xlabel('Positional error', fontsize=12)
    plt.ylabel('Reconstruction signal', fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Set the lower limit of the Y-axis to exactly 0 to prevent dipping
    plt.ylim(bottom=0)
    # Also set X-axis bottom to 0 for a completely clean corner
    plt.xlim(left=0)

    plt.tight_layout()
    
    # Store graph properly and show figure window
    save_path = os.path.join(script_dir, "ae_perf_scatter.png")
    plt.savefig(save_path, dpi=200)
    print(f"Success! Final Plot stored at: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()