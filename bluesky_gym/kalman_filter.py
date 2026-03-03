"""
Kalman Filter Denoiser
=======================
Optimal state estimator for linear Gaussian systems.

For constant-velocity drone motion:
    State: [x, y, vx, vy]
    Dynamics: x[t] = x[t-1] + vx[t-1] * dt
              vx[t] = vx[t-1] + process_noise
    Measurements: [x, y, vx, vy] with Gaussian noise

The Kalman filter is mathematically optimal for this problem - it provides
the best possible estimate given:
1. Linear dynamics model
2. Gaussian noise
3. Known noise statistics

No training required - just configure with known noise levels.
"""

import numpy as np
import os


class KalmanDenoiser:
    """
    Kalman filter for denoising drone state observations.
    
    Parameters
    ----------
    dt : float
        Timestep duration in seconds (default 1.0)
    pos_noise_std : float
        Position measurement noise std in meters (default 3.5)
    vel_noise_std : float
        Velocity measurement noise std in m/s (default 0.1)
    process_noise_std : float
        Process noise std for velocity changes (default 0.05 m/s^2)
    x_norm : float
        X normalization constant (default 8500.0)
    y_norm : float
        Y normalization constant (default 8000.0)
    v_norm : float
        Velocity normalization constant (default 36.0)
    """
    
    def __init__(
        self,
        dt: float = 1.0,
        pos_noise_std: float = 3.5,
        vel_noise_std: float = 0.1,
        process_noise_std: float = 1.0,  # Optimal for maneuvering drones (tuned)
        x_norm: float = 8500.0,
        y_norm: float = 8000.0,
        v_norm: float = 36.0,
    ):
        self.dt = dt
        self.x_norm = x_norm
        self.y_norm = y_norm
        self.v_norm = v_norm
        
        # Convert noise to normalized space
        self.pos_noise_x = pos_noise_std / x_norm
        self.pos_noise_y = pos_noise_std / y_norm
        self.vel_noise = vel_noise_std / v_norm
        self.process_noise = process_noise_std / v_norm  # Velocity change noise
        
        # State: [x, y, vx, vy]
        self.state_dim = 4
        
        # Initialize state and covariance (will be reset per sequence)
        self.x = None  # State estimate
        self.P = None  # Covariance matrix
        
        self._build_matrices()
    
    def _build_matrices(self):
        """Build Kalman filter matrices."""
        # State transition matrix (constant velocity model)
        # x[t] = x[t-1] + vx[t-1] * dt
        # vx[t] = vx[t-1]
        self.F = np.array([
            [1.0, 0.0, self.dt * (self.v_norm / self.x_norm), 0.0],  # x = x + vx*dt
            [0.0, 1.0, 0.0, self.dt * (self.v_norm / self.y_norm)],  # y = y + vy*dt
            [0.0, 0.0, 1.0, 0.0],                                     # vx = vx
            [0.0, 0.0, 0.0, 1.0],                                     # vy = vy
        ], dtype=np.float32)
        
        # Measurement matrix (observe all 4 states directly)
        self.H = np.eye(4, dtype=np.float32)
        
        # Process noise covariance (models acceleration/maneuvers)
        # Drones do collision avoidance maneuvers, so velocity changes significantly
        # Need higher process noise to track maneuvers vs constant velocity assumption
        pos_process = (self.process_noise * self.dt)**2  # Position uncertainty from velocity changes
        self.Q = np.diag([
            pos_process * (self.v_norm / self.x_norm)**2,  # x uncertainty from vx changes
            pos_process * (self.v_norm / self.y_norm)**2,  # y uncertainty from vy changes
            self.process_noise**2,  # vx (acceleration/maneuvers)
            self.process_noise**2,  # vy (acceleration/maneuvers)
        ]).astype(np.float32)
        
        # Measurement noise covariance
        self.R = np.diag([
            self.pos_noise_x**2,
            self.pos_noise_y**2,
            self.vel_noise**2,
            self.vel_noise**2,
        ]).astype(np.float32)
    
    def reset(self, initial_measurement: np.ndarray):
        """
        Initialize filter with first measurement.
        
        Parameters
        ----------
        initial_measurement : ndarray (4,)
            First observation [x, y, vx, vy]
        """
        self.x = initial_measurement.copy()
        # Initial covariance: REDUCED for faster convergence
        # Trust initial measurement more to avoid poor estimates in early timesteps
        self.P = np.diag([
            (self.pos_noise_x)**2,  # Match measurement noise (3.5m → normalized)
            (self.pos_noise_y)**2,  # Match measurement noise
            (self.vel_noise)**2,    # Match measurement noise (0.1 m/s → normalized)
            (self.vel_noise)**2,    # Match measurement noise
        ]).astype(np.float32)
    
    def predict(self):
        """Prediction step: propagate state forward using dynamics model."""
        # State prediction
        self.x = self.F @ self.x
        
        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, measurement: np.ndarray):
        """
        Update step: incorporate new measurement.
        
        Parameters
        ----------
        measurement : ndarray (4,)
            New observation [x, y, vx, vy]
        """
        # Innovation (measurement residual)
        y = measurement - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # State update
        self.x = self.x + K @ y
        
        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self.state_dim) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
    
    def denoise(self, window: np.ndarray) -> np.ndarray:
        """
        Denoise a sequence of observations.
        
        This runs the Kalman filter forward through the window,
        returning the filtered estimate at the final timestep.
        
        Parameters
        ----------
        window : ndarray (seq_len, 4)
            Sequence of noisy observations [x, y, vx, vy]
        
        Returns
        -------
        ndarray (4,)
            Filtered state estimate at final timestep
        """
        # Initialize with first observation
        self.reset(window[0])
        
        # Process all measurements in sequence
        for t in range(1, len(window)):
            self.predict()
            self.update(window[t])
        
        return self.x.copy()
    
    def denoise_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Denoise an entire sequence, returning filtered estimates at each timestep.
        
        Parameters
        ----------
        sequence : ndarray (T, 4)
            Full sequence of noisy observations
        
        Returns
        -------
        ndarray (T, 4)
            Filtered estimates at each timestep
        """
        filtered = np.zeros_like(sequence)
        
        # Initialize
        self.reset(sequence[0])
        filtered[0] = self.x.copy()
        
        # Filter through sequence
        for t in range(1, len(sequence)):
            self.predict()
            self.update(sequence[t])
            filtered[t] = self.x.copy()
        
        return filtered
    
    def save(self, path: str):
        """
        Save Kalman filter configuration.
        
        Note: Kalman filter doesn't have learnable parameters,
        so we just save the configuration.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        config = {
            'type': 'kalman',
            'dt': self.dt,
            'x_norm': self.x_norm,
            'y_norm': self.y_norm,
            'v_norm': self.v_norm,
            'pos_noise_x': self.pos_noise_x * self.x_norm,  # Save in physical units
            'pos_noise_y': self.pos_noise_y * self.y_norm,
            'vel_noise': self.vel_noise * self.v_norm,
            'process_noise': self.process_noise * self.v_norm,
        }
        np.savez(path, **config)
        print(f"[KalmanDenoiser] Saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "KalmanDenoiser":
        """Load a saved Kalman filter configuration."""
        data = np.load(path, allow_pickle=True)
        
        # Compute average position noise from x and y
        pos_noise = (data['pos_noise_x'] + data['pos_noise_y']) / 2.0
        
        kalman = cls(
            dt=float(data['dt']),
            pos_noise_std=float(pos_noise),
            vel_noise_std=float(data['vel_noise']),
            process_noise_std=float(data['process_noise']),
            x_norm=float(data['x_norm']),
            y_norm=float(data['y_norm']),
            v_norm=float(data['v_norm']),
        )
        print(f"[KalmanDenoiser] Loaded from {path}")
        return kalman


class KalmanDenoiserBatch:
    """
    Batch-processing wrapper for Kalman filter.
    Maintains separate filter state for each sequence in batch.
    """
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.base_filter = KalmanDenoiser(**kwargs)
    
    def denoise_batch(self, windows: np.ndarray) -> np.ndarray:
        """
        Denoise a batch of windows.
        
        Parameters
        ----------
        windows : ndarray (batch, seq_len, 4)
        
        Returns
        -------
        ndarray (batch, 4)
        """
        batch_size = windows.shape[0]
        results = np.zeros((batch_size, 4), dtype=np.float32)
        
        for i in range(batch_size):
            kf = KalmanDenoiser(**self.kwargs)
            results[i] = kf.denoise(windows[i])
        
        return results
