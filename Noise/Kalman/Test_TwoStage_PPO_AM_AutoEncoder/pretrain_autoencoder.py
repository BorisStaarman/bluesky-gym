"""
Autoencoder Pre-training Pipeline
===================================
Phase 2 of the AE integration: collect noise-free trajectory data from the
BlueSky simulator, train the FlightAutoencoder on it, and save the frozen
model so it can be loaded by SectorEnv during RL training.

Run from the project root:
    python Noise/Kalman/Test_TwoStage_PPO_AM_AutoEncoder/pretrain_autoencoder.py

The script will:
  1. Spin up a noise-free SectorEnv and let 20 drones fly random episodes
     under the MVP teacher controller (no RL needed).
  2. Build sliding windows of 5 consecutive normalised [x, y, vx, vy] frames.
  3. Train the FlightAutoencoder (MSE loss) until convergence.
  4. Save the model to  autoencoder_pretrained.pt  next to this script.
  5. Run a quick validation that compares reconstruction error on clean
     vs. noisy data to verify the model can distinguish them.
"""

import os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt

# --- path setup so imports work regardless of cwd ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from bluesky_gym.autoencoder import FlightAutoencoder, AE_WINDOW_SIZE, AE_FEATURES, AE_INPUT_DIM, AE_DELTA_NORM, AE_VEL_NORM

# We import the env for data collection but need BlueSky
import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy
import bluesky_gym.envs.common.functions as fn

# ── Constants (must match env) ──────────────────────────────────────────────
NM2KM = 1.852
CENTER = np.array([52.362566, 4.881444])
N_AGENTS = 20

# ── Hyperparameters ─────────────────────────────────────────────────────────
NUM_EPISODES       = 60       # episodes to collect
MAX_STEPS_PER_EP   = 300      # steps per episode (matches env MAX_STEPS)
BATCH_SIZE         = 256
LEARNING_RATE      = 1e-3
NUM_EPOCHS         = 80       # training epochs over the full dataset
VALIDATION_SPLIT   = 0.1      # 10 % held out for validation
POS_NOISE_STD      = 3.5      # metres  (same as env)
VEL_NOISE_STD      = 0.1      # m/s     (same as env)

# Where to save
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "autoencoder_pretrained.pt")
PLOT_SAVE_PATH  = os.path.join(SCRIPT_DIR, "ae_training_loss.png")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 – Collect CLEAN trajectory data straight from BlueSky
# ═══════════════════════════════════════════════════════════════════════════

def _window_to_ae_input(raw_window: np.ndarray) -> np.ndarray:
    """Convert a (AE_WINDOW_SIZE, 4) array of raw physical values [x_m, y_m, vx_ms, vy_ms]
    into the delta-normalised AE input (AE_WINDOW_SIZE, 4).

    Timestep 0: anchor frame with Δx=Δy=0, velocities normalised.
    Timesteps 1+: position deltas normalised by AE_DELTA_NORM, velocities by AE_VEL_NORM.
    """
    T = len(raw_window)
    out = np.zeros((T, 4), dtype=np.float32)
    out[0, 2] = raw_window[0, 2] / AE_VEL_NORM
    out[0, 3] = raw_window[0, 3] / AE_VEL_NORM
    for t in range(1, T):
        out[t, 0] = (raw_window[t, 0] - raw_window[t - 1, 0]) / AE_DELTA_NORM
        out[t, 1] = (raw_window[t, 1] - raw_window[t - 1, 1]) / AE_DELTA_NORM
        out[t, 2] = raw_window[t, 2] / AE_VEL_NORM
        out[t, 3] = raw_window[t, 3] / AE_VEL_NORM
    return out


def collect_clean_data(num_episodes: int = NUM_EPISODES,
                       max_steps: int = MAX_STEPS_PER_EP) -> np.ndarray:
    """Run the simulator with the MVP teacher and record noise-free normalised
    ownship states for every agent at every timestep.

    Returns
    -------
    windows : np.ndarray, shape (N_windows, AE_WINDOW_SIZE, AE_FEATURES)
        Each row is a sliding window of 5 consecutive [x_norm, y_norm, vx_norm, vy_norm].
    """
    from bluesky_gym.envs.ma_env_two_stage_AM_PPO_NOISE_autoencoder import SectorEnv

    print(f"\n{'='*60}")
    print(f"  STEP 1: Collecting clean trajectory data")
    print(f"  Episodes: {num_episodes}  |  Max steps: {max_steps}")
    print(f"  Agents per episode: {N_AGENTS}")
    print(f"  Features: delta-normalised [Δx/{AE_DELTA_NORM}m, Δy/{AE_DELTA_NORM}m, vx/{AE_VEL_NORM}m/s, vy/{AE_VEL_NORM}m/s]")
    print(f"{'='*60}\n")

    env = SectorEnv(
        n_agents=N_AGENTS,
        run_id="ae_data_collection",
        metrics_base_dir=os.path.join(SCRIPT_DIR, "metrics_ae"),
    )

    all_windows: list[np.ndarray] = []

    for ep in range(num_episodes):
        obs, infos = env.reset()

        # Per-agent buffer of raw normalised frames (built up over the episode)
        agent_buffers: dict[str, list[np.ndarray]] = {a: [] for a in env.agents}

        for step in range(max_steps):
            if not env.agents:
                break

            # ── Record CLEAN state for each alive agent ──
            for agent in list(env.agents):
                try:
                    ac_idx = bs.traf.id2idx(agent)
                    # True (noise-free) position in metres from centre
                    true_loc = fn.latlong_to_nm(
                        env.center,
                        np.array([bs.traf.lat[ac_idx], bs.traf.lon[ac_idx]])
                    ) * NM2KM * 1000
                    # True velocity components
                    hdg = bs.traf.hdg[ac_idx]
                    gs  = bs.traf.gs[ac_idx]
                    vx  = np.cos(np.deg2rad(hdg)) * gs
                    vy  = np.sin(np.deg2rad(hdg)) * gs

                    # Store RAW physical values — delta conversion happens when building windows
                    frame = np.array([true_loc[0], true_loc[1], vx, vy], dtype=np.float32)
                    agent_buffers[agent].append(frame)
                except Exception:
                    pass  # agent may have been removed

            # ── Teacher (MVP) action → step the env ──
            actions = {}
            for agent in list(env.agents):
                actions[agent] = env._calculate_mvp_action(agent)
            obs, _, term, trunc, _ = env.step(actions)

            # Remove finished agents from our buffer tracking
            done_agents = [a for a in list(agent_buffers) if a not in env.agents]
            for a in done_agents:
                # Before discarding, extract any complete windows and convert to delta features
                buf = agent_buffers.pop(a, [])
                for i in range(len(buf) - AE_WINDOW_SIZE + 1):
                    raw_w = np.stack(buf[i : i + AE_WINDOW_SIZE])
                    all_windows.append(_window_to_ae_input(raw_w))

        # End-of-episode: extract windows from remaining agents
        for a, buf in agent_buffers.items():
            for i in range(len(buf) - AE_WINDOW_SIZE + 1):
                raw_w = np.stack(buf[i : i + AE_WINDOW_SIZE])
                all_windows.append(_window_to_ae_input(raw_w))

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  Episode {ep+1}/{num_episodes} done — "
                  f"windows so far: {len(all_windows):,}")

    env.close()

    windows = np.array(all_windows, dtype=np.float32)  # (N, 5, 4)
    print(f"\n  Total clean windows collected: {windows.shape[0]:,}")
    print(f"  Window shape: {windows.shape}")
    return windows


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 – Train the Autoencoder
# ═══════════════════════════════════════════════════════════════════════════

def train_autoencoder(windows: np.ndarray) -> FlightAutoencoder:
    """Train FlightAutoencoder on clean sliding-window data.

    Parameters
    ----------
    windows : np.ndarray, shape (N, AE_WINDOW_SIZE, AE_FEATURES)

    Returns
    -------
    model : FlightAutoencoder  (trained, on CPU)
    """
    print(f"\n{'='*60}")
    print(f"  STEP 2: Training FlightAutoencoder")
    print(f"  Input dim: {AE_INPUT_DIM}  |  Samples: {windows.shape[0]:,}")
    print(f"  Epochs: {NUM_EPOCHS}  |  Batch size: {BATCH_SIZE}  |  LR: {LEARNING_RATE}")
    print(f"{'='*60}\n")

    # Flatten windows: (N, 5, 4) → (N, 20)
    X = windows.reshape(windows.shape[0], -1)

    # Train / validation split
    n_val = max(1, int(len(X) * VALIDATION_SPLIT))
    indices = np.random.permutation(len(X))
    X_train = torch.tensor(X[indices[n_val:]], dtype=torch.float32)
    X_val   = torch.tensor(X[indices[:n_val]],  dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, X_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, X_val),
                              batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlightAutoencoder(input_dim=AE_INPUT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    print(f"  Device: {device}")
    print(f"  Model architecture:\n{model}\n")

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, NUM_EPOCHS + 1):
        # ── Train ──
        model.train()
        epoch_loss = 0.0
        for xb, _ in train_loader:
            xb = xb.to(device)
            x_rec = model(xb)
            loss = criterion(x_rec, xb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        train_losses.append(epoch_loss / len(X_train))

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device)
                x_rec = model(xb)
                val_loss += criterion(x_rec, xb).item() * xb.size(0)
        val_losses.append(val_loss / len(X_val))

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{NUM_EPOCHS}  "
                  f"train_mse={train_losses[-1]:.6f}  "
                  f"val_mse={val_losses[-1]:.6f}  "
                  f"best_val={best_val_loss:.6f}")

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.cpu()
    model.eval()

    # ── Plot ──
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="train MSE")
    plt.plot(val_losses,   label="val MSE")
    plt.xlabel("Epoch");  plt.ylabel("MSE Loss")
    plt.title("Autoencoder Pre-training Loss")
    plt.legend();  plt.grid(True);  plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH)
    plt.close()
    print(f"\n  Loss curve saved to: {PLOT_SAVE_PATH}")
    print(f"  Best validation MSE: {best_val_loss:.6f}")
    return model


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 – Validate: clean vs noisy reconstruction error
# ═══════════════════════════════════════════════════════════════════════════

def validate_clean_vs_noisy(model: FlightAutoencoder,
                            clean_windows: np.ndarray,
                            n_samples: int = 2000):
    """Quick sanity check: the AE should produce *low* MSE on clean data and
    *higher* MSE when Gaussian sensor noise is injected.

    ``clean_windows`` is in delta-normalised form (output of ``_window_to_ae_input``),
    shape (N, AE_WINDOW_SIZE, AE_FEATURES).
    """

    print(f"\n{'='*60}")
    print(f"  STEP 3: Validation — clean vs noisy reconstruction error")
    print(f"{'='*60}\n")

    rng = np.random.default_rng(42)
    idx = rng.choice(len(clean_windows), size=min(n_samples, len(clean_windows)), replace=False)
    subset_ae = clean_windows[idx]                         # (n, 5, 4) delta-normalised

    # ── Clean MSE ──
    X_clean = torch.tensor(subset_ae.reshape(len(subset_ae), -1), dtype=torch.float32)
    with torch.no_grad():
        mse_clean = model.reconstruction_mse(X_clean).numpy()

    # ── Noisy version ──
    # Reconstruct approximate raw physical windows from delta-normalised form,
    # inject sensor noise, then re-encode as delta features.
    T = AE_WINDOW_SIZE
    noisy_ae_list = []
    for w in subset_ae:                                    # w: (5, 4) delta-normalised
        raw = np.zeros((T, 4), dtype=np.float32)
        raw[0, 2] = w[0, 2] * AE_VEL_NORM
        raw[0, 3] = w[0, 3] * AE_VEL_NORM
        for t in range(1, T):
            raw[t, 0] = raw[t - 1, 0] + w[t, 0] * AE_DELTA_NORM
            raw[t, 1] = raw[t - 1, 1] + w[t, 1] * AE_DELTA_NORM
            raw[t, 2] = w[t, 2] * AE_VEL_NORM
            raw[t, 3] = w[t, 3] * AE_VEL_NORM
        # Add the same sensor noise as the env
        raw[:, 0] += rng.normal(0, POS_NOISE_STD, T).astype(np.float32)
        raw[:, 1] += rng.normal(0, POS_NOISE_STD, T).astype(np.float32)
        raw[:, 2] += rng.normal(0, VEL_NOISE_STD, T).astype(np.float32)
        raw[:, 3] += rng.normal(0, VEL_NOISE_STD, T).astype(np.float32)
        noisy_ae_list.append(_window_to_ae_input(raw))

    noisy_ae = np.stack(noisy_ae_list)                      # (n, 5, 4)
    X_noisy = torch.tensor(noisy_ae.reshape(len(noisy_ae), -1), dtype=torch.float32)
    with torch.no_grad():
        mse_noisy = model.reconstruction_mse(X_noisy).numpy()

    print(f"  Clean  MSE — mean: {mse_clean.mean():.6f}  std: {mse_clean.std():.6f}")
    print(f"  Noisy  MSE — mean: {mse_noisy.mean():.6f}  std: {mse_noisy.std():.6f}")
    ratio = mse_noisy.mean() / max(mse_clean.mean(), 1e-10)
    print(f"  Ratio (noisy / clean): {ratio:.2f}x")

    if ratio > 2.0:
        print("  ✅  Good separation — the AE can distinguish noisy from clean data.")
    else:
        print("  ⚠️  Weak separation — consider more training data or tuning architecture.")

    # ── Histogram ──
    AE_MSE_SCALE = 0.08   # must match the constant in the env file
    X_MAX = 0.15          # clip x-axis so the clean peak is clearly visible

    # Bin edges shared across both distributions so bars are comparable
    bin_edges = np.linspace(0, X_MAX, 80)

    fig, ax1 = plt.subplots(figsize=(9, 5))

    # Plot noisy first (orange) so clean (blue) overlaps on top
    ax1.hist(mse_noisy, bins=bin_edges, alpha=0.55, color="tab:orange",
             label=f"Noisy  (μ={mse_noisy.mean():.5f})")
    ax1.hist(mse_clean, bins=bin_edges, alpha=0.75, color="tab:blue",
             label=f"Clean  (μ={mse_clean.mean():.5f})")

    # Clip boundary at s_AE = 1.0 (MSE = 0.05)
    ax1.axvline(AE_MSE_SCALE, color="gray", linestyle=":", linewidth=1.5,
                label=f"s_AE = 1.0  (MSE = {AE_MSE_SCALE:.2f})")

    ax1.set_xlim(0, X_MAX)
    ax1.set_xlabel("Reconstruction MSE  (raw AE output)", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Secondary x-axis: the normalised s_AE value that reaches the policy network
    ax2 = ax1.twiny()
    ax2.set_xlim(0, X_MAX / AE_MSE_SCALE)   # same physical range, different units
    ax2.set_xlabel(r"Policy input $s_{\mathrm{AE}}$ = MSE / 0.05  (clipped to [0, 1])",
                   fontsize=11, labelpad=8)

    plt.tight_layout()
    hist_path = os.path.join(SCRIPT_DIR, "ae_clean_vs_noisy.png")
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"  Histogram saved to: {hist_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0 = time.time()

    # 1. Collect
    clean_windows = collect_clean_data()

    # 2. Train
    model = train_autoencoder(clean_windows)

    # 3. Save (torch.save the full model so env.load_autoencoder() works directly)
    torch.save(model, MODEL_SAVE_PATH)
    print(f"\n  Model saved to: {MODEL_SAVE_PATH}")

    # 4. Validate
    validate_clean_vs_noisy(model, clean_windows)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Pre-training complete in {elapsed/60:.1f} minutes.")
    print(f"  Use this in main.py:")
    print(f'    env.load_autoencoder("{MODEL_SAVE_PATH}")')
    print(f"{'='*60}\n")
