"""
Dynamics-Aware LSTM Denoiser
==============================
Instead of just learning to map noisy → clean, this LSTM learns the
**drone dynamics model**: position changes should match velocity.

Key insight: position[t] - position[t-1] ≈ velocity[t-1] * dt

Architecture:
    Inputs:  [x, y, vx, vy] over seq_len timesteps (noisy)
    Outputs: [x, y, vx, vy] at current time (denoised)
    Loss:    MSE + Velocity Consistency Loss

Velocity Consistency Loss:
    predicted_dx = x_pred[t] - x_pred[t-1]
    expected_dx = vx_pred[t-1] * dt
    consistency_loss = MSE(predicted_dx, expected_dx)
"""

import torch
import torch.nn as nn
import numpy as np
import os


class LSTMDenoiserDynamics(nn.Module):
    """
    LSTM denoiser that enforces velocity-position consistency.
    
    Parameters
    ----------
    input_dim : int
        Features per timestep (4: x, y, vx, vy)
    hidden_dim : int
        LSTM hidden size
    num_layers : int
        Number of LSTM layers
    output_dim : int
        Output features (4: x, y, vx, vy)
    seq_len : int
        Sequence length
    dt : float
        Timestep duration in seconds (default 1.0)
    dropout : float
        Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,  # Smaller than before!
        num_layers: int = 1,   # Just 1 layer
        output_dim: int = 4,
        seq_len: int = 10,
        dt: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.dt = dt
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (batch, seq_len, 4)
        
        Returns
        -------
        Tensor (batch, 4) - denoised state at t
        """
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden)
        out = self.mlp(last_hidden)  # (batch, 4)
        return out
    
    def compute_dynamics_loss(
        self,
        pred_current: torch.Tensor,
        input_sequence: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute velocity consistency loss.
        
        The predicted position should be consistent with the predicted velocity
        integrated over dt from the previous timestep.
        
        Parameters
        ----------
        pred_current : Tensor (batch, 4)
            Predicted [x, y, vx, vy] at time t
        input_sequence : Tensor (batch, seq_len, 4)
            Input noisy sequence
            
        Returns
        -------
        Tensor (scalar)
            Dynamics consistency loss
        """
        # Get previous timestep (noisy, but best we have)
        prev_state = input_sequence[:, -2, :]  # (batch, 4) at t-1
        
        # Predicted position at t
        pred_x, pred_y = pred_current[:, 0], pred_current[:, 1]
        # Predicted velocity at t
        pred_vx, pred_vy = pred_current[:, 2], pred_current[:, 3]
        
        # Previous position
        prev_x, prev_y = prev_state[:, 0], prev_state[:, 1]
        # Velocity at t-1 (use current prediction as proxy)
        prev_vx, prev_vy = prev_state[:, 2], prev_state[:, 3]
        
        # Expected position change based on velocity
        # x[t] ≈ x[t-1] + vx[t-1] * dt
        expected_dx = prev_vx * self.dt
        expected_dy = prev_vy * self.dt
        
        # Actual predicted position change
        actual_dx = pred_x - prev_x
        actual_dy = pred_y - prev_y
        
        # Consistency loss
        dx_loss = torch.mean((actual_dx - expected_dx) ** 2)
        dy_loss = torch.mean((actual_dy - expected_dy) ** 2)
        
        return dx_loss + dy_loss
    
    def save(self, path: str):
        """Save model weights + config."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "input_dim": self.input_dim,
                    "hidden_dim": self.hidden_dim,
                    "num_layers": self.num_layers,
                    "output_dim": self.output_dim,
                    "seq_len": self.seq_len,
                    "dt": self.dt,
                },
            },
            path,
        )
    
    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "LSTMDenoiserDynamics":
        """Load a saved model."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        model = cls(
            input_dim=cfg["input_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            output_dim=cfg["output_dim"],
            seq_len=cfg["seq_len"],
            dt=cfg.get("dt", 1.0),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        return model
    
    @torch.no_grad()
    def denoise(self, window: np.ndarray) -> np.ndarray:
        """
        Denoise a single window.
        
        Parameters
        ----------
        window : ndarray (seq_len, 4)
        
        Returns
        -------
        ndarray (4,) - denoised state
        """
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
        x = x.to(next(self.parameters()).device)
        out = self.forward(x)
        return out.squeeze(0).cpu().numpy()
