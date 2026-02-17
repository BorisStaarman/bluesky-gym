"""
Train LSTM Denoiser with MVP-Generated Trajectories
=====================================================
Collects realistic trajectory data by running the BlueSky environment with
MVP (Minimum Vector Projection) controller, then trains the LSTM denoiser
on this expert data.

This approach ensures the training data perfectly matches deployment scenarios:
- Realistic collision avoidance maneuvers
- Multi-agent interactions
- Actual drone dynamics from BlueSky simulator
- Same distribution as what LSTM will see during RL training

Data Specs:
    Inputs  (X):  Sliding window of last 10 timesteps, each [x, y, vx, vy]
    Noise Profile: Zero-mean Gaussian added to Ground Truth:
        Position (x, y):  σ = 3.5 m
        Velocity (vx, vy): σ = 0.1 m/s
    Targets (Y):  Ground Truth (clean) state at most recent timestep t

The state values are in *normalized* form (same as the environment):
    x  / 8500.0,  y  / 8000.0,  vx / 36.0,  vy / 36.0

Usage:
    python train_denoiser.py --n_episodes 200 --n_agents 20
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import io

# Add script dir so lstm_denoiser can be found
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from lstm_denoiser import LSTMDenoiser
from ma_env import SectorEnv  # Import clean environment (no noise yet)
from run_config import RUN_ID


# ──────────────────────────────────────────────────────────────────────
# 1.  HYPER-PARAMETERS
# ──────────────────────────────────────────────────────────────────────
SEQ_LEN = 10           # sliding window length
INPUT_DIM = 4          # [x, y, vx, vy]
HIDDEN_DIM = 128       # LSTM hidden units
NUM_LAYERS = 2         # stacked LSTM layers
MLP_HIDDEN = 64        # MLP head hidden width
OUTPUT_DIM = 4         # cleaned [x, y, vx, vy]

# Noise in *physical* units (meters, m/s)
POS_NOISE_STD_M = 3.5     # σ for x, y in meters
VEL_NOISE_STD_MS = 0.1    # σ for vx, vy in m/s

# Normalization constants (must match the environment)
X_NORM = 8500.0
Y_NORM = 8000.0
V_NORM = 36.0

# Derived noise σ in normalized space
NOISE_STD_NORM = np.array([
    POS_NOISE_STD_M / X_NORM,
    POS_NOISE_STD_M / Y_NORM,
    VEL_NOISE_STD_MS / V_NORM,
    VEL_NOISE_STD_MS / V_NORM,
], dtype=np.float32)

# Training
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 100
PATIENCE = 15          # early-stopping patience

# Data generation from MVP controller
N_EPISODES = 200        # number of episodes to collect
N_AGENTS = 20          # agents per episode (matches actual deployment)
MAX_STEPS_PER_EPISODE = 300  # max steps per episode


# ──────────────────────────────────────────────────────────────────────
# 2.  TRAJECTORY COLLECTION FROM MVP CONTROLLER
# ──────────────────────────────────────────────────────────────────────

@contextmanager
def suppress_output():
    """Suppress BlueSky verbose output."""
    null_out = io.StringIO()
    null_err = io.StringIO()
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = null_out
        sys.stderr = null_err
        with redirect_stdout(null_out), redirect_stderr(null_err):
            yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        null_out.close()
        null_err.close()


def collect_mvp_trajectories(n_episodes: int, n_agents: int, verbose: bool = True) -> list:
    """
    Collect clean trajectories by running MVP controller in BlueSky environment.
    
    Returns
    -------
    list of dict
        Each dict contains:
        - 'agent_id': str
        - 'trajectory': np.ndarray of shape (T, 4) [x, y, vx, vy] normalized
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"  COLLECTING MVP TRAJECTORIES")
        print(f"{'='*70}")
        print(f"  Episodes: {n_episodes}")
        print(f"  Agents per episode: {n_agents}")
        print(f"  Expected trajectories: ~{n_episodes * n_agents}")
    
    all_trajectories = []
    
    # Create clean environment (NO noise injection)
    env = SectorEnv(
        n_agents=n_agents,
        run_id=f"lstm_training_{RUN_ID}",
        render_mode=None,
    )
    
    for ep in range(n_episodes):
        with suppress_output():
            obs, info = env.reset()
        
        # Track trajectories for each agent
        episode_trajectories = {agent_id: [] for agent_id in env.agents}
        
        done = {agent: False for agent in env.agents}
        truncated = {agent: False for agent in env.agents}
        step = 0
        
        while not all(done.values()) and not all(truncated.values()) and step < MAX_STEPS_PER_EPISODE:
            # Get MVP actions for all agents
            actions = {}
            for agent_id in env.agents:
                if not done[agent_id] and not truncated[agent_id]:
                    with suppress_output():
                        mvp_action = env._calculate_mvp_action(agent_id)
                    actions[agent_id] = mvp_action
            
            # Extract clean ownship state BEFORE stepping
            # obs vector: [cos_drift, sin_drift, airspeed, dx, dy, vx, vy, ...]
            # Indices [3, 4, 5, 6] = [x, y, vx, vy] (already normalized)
            for agent_id in env.agents:
                if not done[agent_id] and not truncated[agent_id]:
                    ownship_state = obs[agent_id][3:7]  # [x, y, vx, vy]
                    episode_trajectories[agent_id].append(ownship_state.copy())
            
            # Step environment
            with suppress_output():
                obs, rewards, done, truncated, infos = env.step(actions)
            step += 1
        
        # Save trajectories that are long enough (at least SEQ_LEN + 10 steps)
        for agent_id, traj_list in episode_trajectories.items():
            if len(traj_list) >= SEQ_LEN + 10:  # Need enough for sliding windows
                traj_array = np.array(traj_list, dtype=np.float32)
                all_trajectories.append({
                    'agent_id': agent_id,
                    'episode': ep,
                    'trajectory': traj_array,
                })
        
        if verbose and (ep + 1) % 20 == 0:
            print(f"  Collected {ep + 1}/{n_episodes} episodes "
                  f"({len(all_trajectories)} trajectories so far)")
    
    env.close()
    
    if verbose:
        print(f"\n✅ Collection complete!")
        print(f"   Total trajectories: {len(all_trajectories)}")
        lengths = [t['trajectory'].shape[0] for t in all_trajectories]
        print(f"   Avg trajectory length: {np.mean(lengths):.1f} steps")
        print(f"   Length range: [{np.min(lengths)}, {np.max(lengths)}]")
    
    return all_trajectories


def add_noise(clean: np.ndarray, rng: np.random.Generator, noise_scale_factor: float = 1.0) -> np.ndarray:
    """
    Add zero-mean Gaussian noise to a (length, 4) clean trajectory.
    
    Parameters
    ----------
    noise_scale_factor : float
        Multiplier for noise std (for data augmentation)
    """
    noise = rng.normal(0.0, 1.0, size=clean.shape).astype(np.float32)
    noise *= NOISE_STD_NORM * noise_scale_factor  # broadcast (4,) across columns
    return clean + noise


def make_dataset_from_trajectories(trajectories: list, seq_len: int, seed: int = 42):
    """
    Build sliding-window dataset from MVP-collected trajectories.

    Parameters
    ----------
    trajectories : list of dict
        Each dict has 'trajectory': np.ndarray of shape (T, 4)
    seq_len : int
        Sliding window length
    seed : int
        Random seed for noise generation

    Returns
    -------
    X : ndarray (N, seq_len, 4)  — noisy windows
    Y : ndarray (N, 4)           — clean target at last timestep of each window
    """
    rng = np.random.default_rng(seed)
    X_all, Y_all = [], []

    for traj_data in trajectories:
        clean = traj_data['trajectory']  # (T, 4) clean trajectory
        
        # Data augmentation: vary noise levels to improve robustness
        # 70% normal noise, 20% higher noise, 10% lower noise
        noise_factor = 1.0
        rand_val = rng.random()
        if rand_val < 0.20:
            noise_factor = 1.5  # 50% more noise
        elif rand_val < 0.30:
            noise_factor = 0.5  # 50% less noise
        
        noisy = add_noise(clean, rng, noise_scale_factor=noise_factor)

        # Slide windows of length seq_len
        traj_len = clean.shape[0]
        for t in range(seq_len, traj_len):
            window = noisy[t - seq_len : t]   # (seq_len, 4) — noisy observations up to t-1
            target = clean[t - 1]              # (4,) — clean state at timestep t-1 (last in window)
            X_all.append(window)
            Y_all.append(target)

    X = np.array(X_all, dtype=np.float32)
    Y = np.array(Y_all, dtype=np.float32)
    return X, Y


# ──────────────────────────────────────────────────────────────────────
# 3.  PYTORCH DATASET
# ──────────────────────────────────────────────────────────────────────
class DenoiserDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ──────────────────────────────────────────────────────────────────────
# 4.  TRAINING LOOP
# ──────────────────────────────────────────────────────────────────────
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_denoiser] Device: {device}")

    # --- Collect MVP trajectories ---
    print(f"\n[train_denoiser] Collecting trajectories from MVP controller...")
    trajectories = collect_mvp_trajectories(
        n_episodes=args.n_episodes,
        n_agents=args.n_agents,
        verbose=True
    )
    
    # --- Generate dataset from trajectories ---
    print(f"\n[train_denoiser] Building training dataset (seq_len={args.seq_len})...")
    X, Y = make_dataset_from_trajectories(trajectories, args.seq_len, seed=42)
    print(f"[train_denoiser] Dataset: X={X.shape}, Y={Y.shape}")

    # Train / validation split (80/20)
    n = len(X)
    idx = np.random.RandomState(0).permutation(n)
    split = int(0.8 * n)
    X_train, Y_train = X[idx[:split]], Y[idx[:split]]
    X_val,   Y_val   = X[idx[split:]], Y[idx[split:]]

    train_ds = DenoiserDataset(X_train, Y_train)
    val_ds   = DenoiserDataset(X_val, Y_val)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, pin_memory=(device == "cuda"))
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                          num_workers=0, pin_memory=(device == "cuda"))

    # --- Model ---
    model = LSTMDenoiser(
        input_dim=INPUT_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        mlp_hidden=args.mlp_hidden,
        output_dim=OUTPUT_DIM,
        seq_len=args.seq_len,
        dropout=0.2,  # Increased dropout to reduce overfitting
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[train_denoiser] Model params: {total_params:,}")

    criterion = nn.MSELoss()
    # Add L2 regularization (weight decay) to reduce overfitting
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True,  # Reduced patience
    )

    # --- Training ---
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    save_dir = os.path.join(script_dir, "denoiser_models")
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "lstm_denoiser_best.pt")

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        train_loss = epoch_loss / len(train_ds)

        # --- Validate ---
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss_sum += criterion(pred, yb).item() * xb.size(0)
        val_loss = val_loss_sum / len(val_ds)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"Train MSE: {train_loss:.8f} | Val MSE: {val_loss:.8f}")

        # Early stopping
        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            patience_counter = 0
            model.save(best_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[train_denoiser] Early stopping at epoch {epoch} "
                      f"(best val MSE = {best_val_loss:.8f})")
                break

    print(f"\n[train_denoiser] Best validation MSE: {best_val_loss:.8f}")
    print(f"[train_denoiser] Model saved to: {best_path}")

    # --- Quick evaluation on validation set ---
    model_best = LSTMDenoiser.load(best_path, device=device)
    evaluate_model(model_best, X_val, Y_val, device)

    # --- Plot loss curves ---
    plot_path = os.path.join(save_dir, "training_loss.png")
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train MSE")
    plt.plot(val_losses, label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("LSTM Denoiser Training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"[train_denoiser] Loss plot saved to: {plot_path}")

    return best_path


# ──────────────────────────────────────────────────────────────────────
# 5.  EVALUATION
# ──────────────────────────────────────────────────────────────────────
def evaluate_model(model, X_val, Y_val, device="cpu"):
    """
    Evaluate the denoiser quality: compare noisy-last-step vs LSTM-denoised
    against the clean ground truth.
    """
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X_val, dtype=torch.float32).to(device)
        pred = model(xb).cpu().numpy()

    # Noisy baseline: just take the last timestep of the noisy window
    noisy_last = X_val[:, -1, :]  # (N, 4)
    clean = Y_val                 # (N, 4)

    # Per-feature RMSE (in normalized space)
    feature_names = ["x", "y", "vx", "vy"]
    unnorm = np.array([X_NORM, Y_NORM, V_NORM, V_NORM])

    print("\n" + "=" * 65)
    print(f"{'Feature':<8} | {'Noisy RMSE (phys)':<20} | {'LSTM  RMSE (phys)':<20} | {'Improvement':<12}")
    print("-" * 65)

    for i, name in enumerate(feature_names):
        noisy_rmse_norm = np.sqrt(np.mean((noisy_last[:, i] - clean[:, i]) ** 2))
        lstm_rmse_norm  = np.sqrt(np.mean((pred[:, i]       - clean[:, i]) ** 2))
        noisy_rmse_phys = noisy_rmse_norm * unnorm[i]
        lstm_rmse_phys  = lstm_rmse_norm  * unnorm[i]
        improvement = (1 - lstm_rmse_phys / noisy_rmse_phys) * 100 if noisy_rmse_phys > 0 else 0
        unit = "m" if i < 2 else "m/s"
        print(f"  {name:<6} | {noisy_rmse_phys:>12.4f} {unit:<6} | "
              f"{lstm_rmse_phys:>12.4f} {unit:<6} | {improvement:>8.1f}%")

    # Overall RMSE
    noisy_overall = np.sqrt(np.mean((noisy_last - clean) ** 2))
    lstm_overall  = np.sqrt(np.mean((pred       - clean) ** 2))
    print("-" * 65)
    print(f"  {'Total':<6} | {noisy_overall:>12.8f} (norm)  | "
          f"{lstm_overall:>12.8f} (norm)  | "
          f"{(1 - lstm_overall / noisy_overall) * 100:>8.1f}%")
    print("=" * 65)


# ──────────────────────────────────────────────────────────────────────
# 6.  MAIN
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM Denoiser with MVP Trajectories")
    parser.add_argument("--n_episodes", type=int,   default=N_EPISODES,
                       help="Number of episodes to collect from MVP")
    parser.add_argument("--n_agents",   type=int,   default=N_AGENTS,
                       help="Number of agents per episode")
    parser.add_argument("--seq_len",    type=int,   default=SEQ_LEN,
                       help="LSTM sequence length")
    parser.add_argument("--hidden_dim", type=int,   default=HIDDEN_DIM,
                       help="LSTM hidden dimension")
    parser.add_argument("--num_layers", type=int,   default=NUM_LAYERS,
                       help="Number of LSTM layers")
    parser.add_argument("--mlp_hidden", type=int,   default=MLP_HIDDEN,
                       help="MLP head hidden dimension")
    parser.add_argument("--batch_size", type=int,   default=BATCH_SIZE,
                       help="Training batch size")
    parser.add_argument("--lr",         type=float, default=LEARNING_RATE,
                       help="Learning rate")
    parser.add_argument("--epochs",     type=int,   default=EPOCHS,
                       help="Maximum training epochs")
    parser.add_argument("--patience",   type=int,   default=PATIENCE,
                       help="Early stopping patience")
    args = parser.parse_args()

    train(args)
