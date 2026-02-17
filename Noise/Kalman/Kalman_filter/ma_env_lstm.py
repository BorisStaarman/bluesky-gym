"""
LSTM-Denoised Multi-Agent Environment
=======================================
Wraps the base SectorEnv to:
    1. Inject zero-mean Gaussian noise into ownship & intruder observations
       (Position σ=3.5 m, Velocity σ=0.1 m/s)
    2. Maintain a per-agent sliding window of the last `seq_len` timesteps
       for the 4 ownship state features [x, y, vx, vy]
    3. Feed each window through a trained LSTM denoiser and replace
       the noisy ownship features with the cleaned estimates

Usage in main.py:
    from ma_env_lstm import SectorEnvLSTM
    register_env("sector_env", lambda config: SectorEnvLSTM(**config))
"""

import os
import sys
import numpy as np
import torch
from collections import defaultdict

# Ensure local imports work
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from ma_env import SectorEnv, NUM_AC_STATE
from lstm_denoiser import LSTMDenoiser
from kalman_denoiser import KalmanDenoiser

# ── Noise parameters (physical units) ────────────────────────────────
POS_NOISE_STD_M  = 3.5    # σ for x, y in meters
VEL_NOISE_STD_MS = 0.1    # σ for vx, vy in m/s

# ── Normalization constants (must match ma_env.py) ───────────────────
X_NORM = 8500.0
Y_NORM = 8000.0
V_NORM = 36.0

# Noise σ in *normalized* observation space
NOISE_STD_OWNSHIP = np.array([
    POS_NOISE_STD_M  / X_NORM,   # x
    POS_NOISE_STD_M  / Y_NORM,   # y
    VEL_NOISE_STD_MS / V_NORM,   # vx
    VEL_NOISE_STD_MS / V_NORM,   # vy
], dtype=np.float32)

# Intruder relative features also get noise (twice: own + intruder sensor)
# The relative position/velocity noise compounds ≈ sqrt(2) * σ
NOISE_STD_INTRUDER_REL = np.array([
    0.0,                                             # distance (recomputed from noisy dx,dy)
    POS_NOISE_STD_M * np.sqrt(2) / X_NORM,          # rel_dx
    POS_NOISE_STD_M * np.sqrt(2) / Y_NORM,          # rel_dy
    VEL_NOISE_STD_MS * np.sqrt(2) / V_NORM,         # rel_dvx
    VEL_NOISE_STD_MS * np.sqrt(2) / V_NORM,         # rel_dvy
], dtype=np.float32)

# Default denoiser model path
DEFAULT_DENOISER_PATH = os.path.join(script_dir, "denoiser_models", "lstm_denoiser_best.pt")

# LSTM window length
SEQ_LEN = 3  # SHORTER - match training!


class SectorEnvLSTM(SectorEnv):
    """
    Extended SectorEnv that adds sensor noise and optionally denoises
    ownship observations with a pre-trained LSTM.
    """

    def __init__(
        self,
        render_mode=None,
        n_agents=20,
        run_id="default",
        # --- Noise / LSTM settings (passed via env_config) ---
        denoiser_path: str | None = None,
        use_denoiser: bool = True,
        noise_enabled: bool = True,
        seq_len: int = SEQ_LEN,
        pos_noise_std: float = POS_NOISE_STD_M,
        vel_noise_std: float = VEL_NOISE_STD_MS,
        add_intruder_noise: bool = True,
        # ---- pass-through to parent ----
        **kwargs,
    ):
        super().__init__(
            render_mode=render_mode,
            n_agents=n_agents,
            run_id=run_id,
            **kwargs,
        )

        self.noise_enabled = noise_enabled
        self.add_intruder_noise = add_intruder_noise
        self.seq_len = seq_len

        # Recompute noise std in case the user overrides them
        self._noise_std_own = np.array([
            pos_noise_std / X_NORM,
            pos_noise_std / Y_NORM,
            vel_noise_std / V_NORM,
            vel_noise_std / V_NORM,
        ], dtype=np.float32)

        self._noise_std_intr = np.array([
            0.0,
            pos_noise_std * np.sqrt(2) / X_NORM,
            pos_noise_std * np.sqrt(2) / Y_NORM,
            vel_noise_std * np.sqrt(2) / V_NORM,
            vel_noise_std * np.sqrt(2) / V_NORM,
        ], dtype=np.float32)

        # --- LSTM/Kalman denoiser ---
        self.use_denoiser = use_denoiser
        self._denoiser = None
        self._denoiser_type = None  # 'lstm' or 'kalman'
        # Always use CPU for Ray workers to avoid CUDA device errors
        self._denoiser_device = "cpu"

        if use_denoiser:
            path = denoiser_path or DEFAULT_DENOISER_PATH
            if os.path.isfile(path):
                try:
                    # Check file extension to determine denoiser type
                    if path.endswith('.npz'):
                        # Kalman filter (no training needed)
                        self._denoiser = KalmanDenoiser.load(path)
                        self._denoiser_type = 'kalman'
                        print(f"[SectorEnvLSTM] Loaded Kalman filter from {path}")
                    else:
                        # LSTM denoiser
                        self._denoiser = LSTMDenoiser.load(path, device=self._denoiser_device)
                        self._denoiser.eval()
                        self._denoiser_type = 'lstm'
                        print(f"[SectorEnvLSTM] Loaded LSTM denoiser from {path} (device={self._denoiser_device})")
                except Exception as e:
                    print(f"[SectorEnvLSTM] ERROR loading denoiser: {e}")
                    print(f"[SectorEnvLSTM] Running with noise but NO denoiser.")
                    self.use_denoiser = False
            else:
                print(f"[SectorEnvLSTM] WARNING: denoiser file not found at {path}")
                print(f"[SectorEnvLSTM] Running with noise but NO denoiser.")
                self.use_denoiser = False

        # Per-agent sliding window buffers:  agent_id -> deque-like list of (4,) arrays
        self._obs_windows: dict[str, list[np.ndarray]] = defaultdict(list)

        # RNG for reproducible noise
        self._noise_rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Override reset
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        obs, infos = super().reset(seed=seed, options=options)
        # Clear all sliding windows on episode reset
        self._obs_windows.clear()
        # Apply noise (and denoiser if ready) to initial observations
        obs = self._apply_noise_pipeline(obs)
        return obs, infos

    # ------------------------------------------------------------------
    # Override step
    # ------------------------------------------------------------------
    def step(self, actions):
        obs, rewards, terminateds, truncateds, infos = super().step(actions)
        # Apply noise → buffer → denoise pipeline
        obs = self._apply_noise_pipeline(obs)
        return obs, rewards, terminateds, truncateds, infos

    # ------------------------------------------------------------------
    # Noise + LSTM pipeline
    # ------------------------------------------------------------------
    def _apply_noise_pipeline(self, observations: dict) -> dict:
        """
        For each agent observation vector:
            1. Add Gaussian noise to ownship [x, y, vx, vy] and intruder features
            2. Append noisy ownship features to sliding window
            3. If window is full and denoiser is loaded, replace ownship
               features with the LSTM-denoised estimate
        """
        if not self.noise_enabled:
            return observations

        noised_obs = {}
        for agent_id, obs_vec in observations.items():
            obs = obs_vec.copy()

            # ── Ownship features: indices [3,4,5,6] → [dx, dy, vx, vy] ──
            # The obs vector layout: [cos_drift, sin_drift, airspeed, dx, dy, vx, vy, ...]
            #   dx at index 3, dy at index 4, vx at index 5, vy at index 6
            ownship_state_idx = [3, 4, 5, 6]   # x, y, vx, vy (all normalized)
            clean_own = obs[ownship_state_idx].copy()

            # Add noise to ownship features
            noise_own = self._noise_rng.normal(0.0, self._noise_std_own).astype(np.float32)
            obs[ownship_state_idx] += noise_own

            # ── Intruder features: blocks of 5 starting at index 7 ──
            if self.add_intruder_noise:
                intruder_start = 7
                for j in range(NUM_AC_STATE):
                    base = intruder_start + j * 5
                    if base + 5 > len(obs):
                        break
                    # Only add noise if the intruder slot is non-zero (not padding)
                    if np.any(obs[base:base + 5] != 0.0):
                        noise_intr = self._noise_rng.normal(
                            0.0, self._noise_std_intr
                        ).astype(np.float32)
                        obs[base:base + 5] += noise_intr
                        # Recompute distance (index 0 of the 5-block) from noisy dx, dy
                        noisy_dx = obs[base + 1]
                        noisy_dy = obs[base + 2]
                        obs[base] = float(np.hypot(noisy_dx, noisy_dy))

            # ── Sliding window management ──
            noisy_own = obs[ownship_state_idx].copy()   # (4,)
            window = self._obs_windows[agent_id]
            window.append(noisy_own)
            # Keep only the last seq_len entries
            if len(window) > self.seq_len:
                self._obs_windows[agent_id] = window[-self.seq_len:]
                window = self._obs_windows[agent_id]

            # ── Denoise if window is full ──
            if self.use_denoiser and self._denoiser is not None and len(window) == self.seq_len:
                window_arr = np.stack(window, axis=0)  # (seq_len, 4)
                denoised = self._denoiser.denoise(window_arr)  # (4,)
                # Replace the ownship state features with denoised values
                obs[ownship_state_idx] = denoised

            noised_obs[agent_id] = obs

        return noised_obs
