"""
Position-Only Kalman Filter Denoiser
=====================================
Kalman filter that only uses position measurements, ignoring noisy velocity.

Key differences from standard Kalman filter:
1. Measurements: [x, y] only (not [x, y, vx, vy])
2. Velocity is ESTIMATED from position changes, not measured
3. Better for scenarios where position is critical and velocity noise is high

This should give better position estimates since we don't corrupt the filter
with noisy velocity measurements.
"""

import numpy as np
import os


class KalmanDenoiserPositionOnly:
    """
    Position-only Kalman filter - measures position, estimates velocity.
    
    Parameters
    ----------
    dt : float
        Timestep duration in seconds (default 1.0)
    pos_noise_std : float
        Position measurement noise std in meters (default 3.5)
    process_noise_pos : float
        Process noise for position (default 0.1 m)
    process_noise_vel : float
        Process noise for velocity changes (default 1.0 m/s²)
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
        process_noise_pos: float = 0.1,
        process_noise_vel: float = 1.0,
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
        self.process_noise_pos = process_noise_pos / x_norm  # Position process noise
        self.process_noise_vel = process_noise_vel / v_norm  # Velocity process noise
        
        # State: [x, y, vx, vy] but we only MEASURE [x, y]
        self.state_dim = 4
        self.measurement_dim = 2  # Only position!
        
        # Initialize state and covariance (will be reset per sequence)
        self.x = None  # State estimate
        self.P = None  # Covariance matrix
        
        self._build_matrices()
    
    def _build_matrices(self):
        """Build Kalman filter matrices."""
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1.0, 0.0, self.dt * (self.v_norm / self.x_norm), 0.0],
            [0.0, 1.0, 0.0, self.dt * (self.v_norm / self.y_norm)],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)
        
        # Measurement matrix - we only observe POSITION
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],  # Measure x
            [0.0, 1.0, 0.0, 0.0],  # Measure y
        ], dtype=np.float32)
        
        # Process noise covariance
        # Allows for small position drift and velocity changes
        self.Q = np.diag([
            self.process_noise_pos**2,  # x position
            self.process_noise_pos**2,  # y position
            self.process_noise_vel**2,  # vx acceleration
            self.process_noise_vel**2,  # vy acceleration
        ]).astype(np.float32)
        
        # Measurement noise covariance - POSITION ONLY
        self.R = np.diag([
            self.pos_noise_x**2,
            self.pos_noise_y**2,
        ]).astype(np.float32)
    
    def reset(self, initial_position: np.ndarray, initial_velocity: np.ndarray = None):
        """
        Initialize filter with first measurement.
        
        Parameters
        ----------
        initial_position : ndarray (2,)
            First position observation [x, y]
        initial_velocity : ndarray (2,), optional
            Initial velocity estimate [vx, vy]. If None, assumes zero velocity.
        """
        if initial_velocity is None:
            initial_velocity = np.zeros(2, dtype=np.float32)
        
        self.x = np.concatenate([initial_position, initial_velocity])
        
        # Initial covariance: high uncertainty
        self.P = np.diag([
            (5.0 / self.x_norm)**2,   # Initial position uncertainty
            (5.0 / self.y_norm)**2,
            (2.0 / self.v_norm)**2,   # Higher velocity uncertainty since we don't measure it
            (2.0 / self.v_norm)**2,
        ]).astype(np.float32)
    
    def predict(self):
        """Prediction step: propagate state forward using dynamics model."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, position_measurement: np.ndarray):
        """
        Update step: incorporate new position measurement.
        
        Parameters
        ----------
        position_measurement : ndarray (2,)
            New position observation [x, y]
        """
        # Innovation (measurement residual)
        y = position_measurement - self.H @ self.x
        
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
        Denoise a sequence of observations (uses position only).
        
        Parameters
        ----------
        window : ndarray (seq_len, 4)
            Sequence of noisy observations [x, y, vx, vy]
            Note: We only use [x, y] columns
        
        Returns
        -------
        ndarray (4,)
            Filtered state estimate [x, y, vx, vy] at final timestep
        """
        # Initialize with first position, use noisy velocity as initial guess
        self.reset(window[0, :2], window[0, 2:])
        
        # Process all measurements in sequence
        for t in range(1, len(window)):
            self.predict()
            self.update(window[t, :2])  # Only use position!
        
        return self.x.copy()
    
    def denoise_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Denoise an entire sequence, returning filtered estimates at each timestep.
        
        Parameters
        ----------
        sequence : ndarray (T, 4)
            Full sequence of noisy observations [x, y, vx, vy]
        
        Returns
        -------
        ndarray (T, 4)
            Filtered estimates at each timestep
        """
        filtered = np.zeros_like(sequence)
        
        # Initialize
        self.reset(sequence[0, :2], sequence[0, 2:])
        filtered[0] = self.x.copy()
        
        # Filter through sequence
        for t in range(1, len(sequence)):
            self.predict()
            self.update(sequence[t, :2])  # Only use position!
            filtered[t] = self.x.copy()
        
        return filtered
    
    def save(self, path: str):
        """Save Kalman filter configuration."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        config = {
            'type': 'kalman_position_only',
            'dt': self.dt,
            'x_norm': self.x_norm,
            'y_norm': self.y_norm,
            'v_norm': self.v_norm,
            'pos_noise': self.pos_noise_x * self.x_norm,  # Save in physical units
            'process_noise_pos': self.process_noise_pos * self.x_norm,
            'process_noise_vel': self.process_noise_vel * self.v_norm,
        }
        np.savez(path, **config)
        print(f"[KalmanDenoiserPositionOnly] Saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "KalmanDenoiserPositionOnly":
        """Load a saved Kalman filter configuration."""
        data = np.load(path, allow_pickle=True)
        
        kalman = cls(
            dt=float(data['dt']),
            pos_noise_std=float(data['pos_noise']),
            process_noise_pos=float(data['process_noise_pos']),
            process_noise_vel=float(data['process_noise_vel']),
            x_norm=float(data['x_norm']),
            y_norm=float(data['y_norm']),
            v_norm=float(data['v_norm']),
        )
        print(f"[KalmanDenoiserPositionOnly] Loaded from {path}")
        return kalman


class KalmanDenoiserPositionOnlyBatch:
    """Batch-processing wrapper for position-only Kalman filter."""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.base_filter = KalmanDenoiserPositionOnly(**kwargs)
    
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
            kf = KalmanDenoiserPositionOnly(**self.kwargs)
            results[i] = kf.denoise(windows[i])
        
        return results
