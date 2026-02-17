"""
LSTM Denoiser Model
====================
Takes a sliding window of noisy sensor observations [x, y, vx, vy] over the
last `seq_len` timesteps and outputs a single 'cleaned' estimate of the
current state (Position x, y and Velocity vx, vy).

Architecture:  LSTM  →  MLP head  →  4-dim output [x, y, vx, vy]
Loss:          Mean Squared Error (MSE)

The model operates on **normalized** values (same normalization used in the
environment observation vector):
    x  → / 8500.0   (meters from center)
    y  → / 8000.0
    vx → / 36.0     (m/s)
    vy → / 36.0
"""

import torch
import torch.nn as nn
import os
import numpy as np


class LSTMDenoiser(nn.Module):
    """
    LSTM-based denoising model.

    Parameters
    ----------
    input_dim : int
        Number of features per timestep (default 4: x, y, vx, vy).
    hidden_dim : int
        Number of LSTM hidden units (64 or 128).
    num_layers : int
        Number of stacked LSTM layers.
    mlp_hidden : int
        Width of the MLP head hidden layer.
    output_dim : int
        Dimension of the output (default 4: cleaned x, y, vx, vy).
    seq_len : int
        Length of the input sliding window (default 10).
    dropout : float
        Dropout between LSTM layers (only active if num_layers > 1).
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 2,
        mlp_hidden: int = 64,
        output_dim: int = 4,
        seq_len: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.seq_len = seq_len

        # --- LSTM encoder ---
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,           # (batch, seq, feature)
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # --- MLP head ---
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, input_dim)
            Sliding window of noisy observations.

        Returns
        -------
        Tensor of shape (batch, output_dim)
            Cleaned estimate for the most recent timestep.
        """
        # LSTM forward — we only need the output at the last timestep
        lstm_out, _ = self.lstm(x)          # (batch, seq_len, hidden_dim)
        last_hidden = lstm_out[:, -1, :]    # (batch, hidden_dim)

        # MLP head
        cleaned = self.mlp(last_hidden)     # (batch, output_dim)
        return cleaned

    # ------------------------------------------------------------------
    # Convenience: save / load
    # ------------------------------------------------------------------
    def save(self, path: str):
        """Save model weights + config to *path*."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "input_dim": self.input_dim,
                    "hidden_dim": self.hidden_dim,
                    "num_layers": self.num_layers,
                    "mlp_hidden": self.mlp[0].out_features,   # recover from first Linear output
                    "output_dim": self.output_dim,
                    "seq_len": self.seq_len,
                },
            },
            path,
        )
        print(f"[LSTMDenoiser] Saved to {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "LSTMDenoiser":
        """Load a previously saved model."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        model = cls(
            input_dim=cfg["input_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            mlp_hidden=cfg.get("mlp_hidden", 64),
            output_dim=cfg["output_dim"],
            seq_len=cfg["seq_len"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        print(f"[LSTMDenoiser] Loaded from {path} (device={device})")
        return model

    # ------------------------------------------------------------------
    # Inference helper (single-sample, numpy in → numpy out)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def denoise(self, window: np.ndarray) -> np.ndarray:
        """
        Denoise a single window of observations.

        Parameters
        ----------
        window : ndarray of shape (seq_len, input_dim)
            The most recent `seq_len` noisy observations.

        Returns
        -------
        ndarray of shape (output_dim,)
            Cleaned state estimate.
        """
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # (1, seq, feat)
        x = x.to(next(self.parameters()).device)
        out = self.forward(x)           # (1, output_dim)
        return out.squeeze(0).cpu().numpy()
