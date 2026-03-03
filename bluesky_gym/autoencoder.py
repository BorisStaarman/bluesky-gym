"""
Autoencoder for physical-inconsistency detection in UAV sensor data.

Architecture follows MAFTRL (Luo et al., 2020):
  - MLP Encoder–Decoder with a bottleneck at ~half the input dim.
  - Input: flattened sliding window of 5 clean timesteps × 4 features = 20.
  - Trained on noise-free trajectory data so that reconstruction error
    is near-zero for physically valid motion and spikes under sensor noise.

Usage:
    # Training
    from bluesky_gym.autoencoder import FlightAutoencoder, AE_INPUT_DIM
    model = FlightAutoencoder(input_dim=AE_INPUT_DIM)
    ...train on clean data...
    torch.save(model, "ae_pretrained.pt")

    # Inference (inside SectorEnv)
    env.load_autoencoder("ae_pretrained.pt")
"""

import torch
import torch.nn as nn

# Must match the constants in ma_env_two_stage_AM_PPO_NOISE_autoencoder.py
AE_WINDOW_SIZE = 5
AE_FEATURES = 4          # per timestep: [Δx_norm, Δy_norm, vx_norm, vy_norm]
AE_INPUT_DIM = AE_WINDOW_SIZE * AE_FEATURES  # 20

# Normalization for delta-based AE features (physical units)
# Chosen so that clean motion ≈ 0.6 and per-step noise ≈ 0.33 in normalised space,
# giving a noise-to-signal ratio of ~55 % which is clearly detectable.
AE_DELTA_NORM = 15.0   # m   — one timestep of motion at ~max speed; scales Δpos
AE_VEL_NORM   = 15.0   # m/s — ~max UAV speed; scales vx / vy


class FlightAutoencoder(nn.Module):
    """MLP Autoencoder for UAV trajectory reconstruction.

    Input features (per timestep in the 5-step window):
        [Δx / AE_DELTA_NORM,  Δy / AE_DELTA_NORM,  vx / AE_VEL_NORM,  vy / AE_VEL_NORM]
    Timestep 0 uses Δx = Δy = 0 (anchor frame, no predecessor).

    Why deltas instead of absolute positions?
        - Clean motion delta/step ≈ 9 m → 9/15 = 0.60 normalised
        - Noise delta between steps ≈ √2 × 3.5 m = 4.95 m → 4.95/15 = 0.33 normalised
        - Noise-to-signal ratio ≈ 55 % — clearly separable vs. the old 0.004 % ratio.

    Topology:  20 → 8 → 3 → 8 → 20
    (aggressive bottleneck forces the model to only encode the smooth-flight
     manifold; noisy windows lie off-manifold and produce high MSE.)

    Parameters
    ----------
    input_dim : int
        Flat input size (default ``AE_INPUT_DIM = 20``).
    hidden_dim : int
        First encoder/decoder hidden layer width (default 8).
    bottleneck_dim : int
        Latent dimension (default 3).
    """

    def __init__(
        self,
        input_dim: int = AE_INPUT_DIM,
        hidden_dim: int = 8,
        bottleneck_dim: int = 3,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            # No activation – reconstruction can take any range
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct *x*.

        Parameters
        ----------
        x : Tensor, shape ``(batch, input_dim)``
        
        Returns
        -------
        Tensor, same shape as *x*.
        """
        return self.decoder(self.encoder(x))

    def reconstruction_mse(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-sample MSE between *x* and its reconstruction.

        Parameters
        ----------
        x : Tensor, shape ``(batch, input_dim)``

        Returns
        -------
        Tensor, shape ``(batch,)``
        """
        x_rec = self.forward(x)
        return ((x - x_rec) ** 2).mean(dim=1)
