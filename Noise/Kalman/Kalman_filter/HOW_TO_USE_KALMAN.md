# 🎯 Complete Guide: Using the Kalman Filter

## Everything You Need to Know

The Kalman filter is in `kalman_denoiser.py` and has **two main classes**:
1. `KalmanDenoiser` - For single sequences
2. `KalmanDenoiserBatch` - For multiple sequences at once

---

## 📦 How to Import

```python
from kalman_denoiser import KalmanDenoiser, KalmanDenoiserBatch
```

That's it! No other dependencies needed (besides numpy).

---

## 🔧 How to Create a Kalman Filter

### Basic Creation (with optimal tuned parameters)

```python
# Create with default optimal values
kalman = KalmanDenoiser(
    dt=1.0,                    # Time between measurements (seconds)
    pos_noise_std=3.5,        # Position measurement noise (meters)
    vel_noise_std=0.1,        # Velocity measurement noise (m/s)
    process_noise_std=1.0,    # Acceleration/maneuver noise (m/s²) - OPTIMAL!
    x_norm=8500.0,            # X-axis scale (meters)
    y_norm=8000.0,            # Y-axis scale (meters)
    v_norm=36.0,              # Velocity scale (m/s)
)
```

### Load a Saved Filter

```python
# Load previously saved configuration
kalman = KalmanDenoiser.load("denoiser_models/kalman_denoiser.npz")
```

---

## 📊 Data Format Requirements

**CRITICAL:** Your data must be in **NORMALIZED** space!

### What is Normalized Space?

```python
# Physical space (meters, m/s)
x_physical = 4250.0  # meters
y_physical = 4000.0  # meters
vx_physical = 18.0   # m/s
vy_physical = -12.0  # m/s

# Normalized space (0 to 1 range)
x_normalized = x_physical / 8500.0   # = 0.5
y_normalized = y_physical / 8000.0   # = 0.5
vx_normalized = vx_physical / 36.0   # = 0.5
vy_normalized = vy_physical / 36.0   # = -0.333
```

### Data Shape

```python
# Single observation (one timestep)
observation = np.array([x_norm, y_norm, vx_norm, vy_norm])  # Shape: (4,)

# Sequence (multiple timesteps)
sequence = np.array([
    [x_norm_t0, y_norm_t0, vx_norm_t0, vy_norm_t0],
    [x_norm_t1, y_norm_t1, vx_norm_t1, vy_norm_t1],
    [x_norm_t2, y_norm_t2, vx_norm_t2, vy_norm_t2],
    # ... more timesteps
])  # Shape: (T, 4) where T = number of timesteps

# Batch of sequences (for processing many at once)
batch = np.array([
    [[x, y, vx, vy], [x, y, vx, vy], ...],  # Sequence 1
    [[x, y, vx, vy], [x, y, vx, vy], ...],  # Sequence 2
    # ... more sequences
])  # Shape: (batch_size, seq_len, 4)
```

---

## 🚀 Method 1: Denoise a Sequence (Most Common)

**Use Case:** You have a full trajectory and want filtered estimates at each timestep.

```python
from kalman_denoiser import KalmanDenoiser
import numpy as np

# Your noisy trajectory (MUST BE NORMALIZED!)
noisy_trajectory = np.array([
    [0.50, 0.50, 0.01, 0.01],   # t=0: [x, y, vx, vy]
    [0.51, 0.51, 0.02, 0.01],   # t=1
    [0.52, 0.52, 0.01, 0.02],   # t=2
    [0.53, 0.53, 0.02, 0.01],   # t=3
    # ... more timesteps
])  # Shape: (timesteps, 4)

# Create filter
kalman = KalmanDenoiser(
    dt=1.0,
    pos_noise_std=3.5,
    vel_noise_std=0.1,
    process_noise_std=1.0,  # Your optimal value!
)

# Denoise the entire sequence
clean_trajectory = kalman.denoise_sequence(noisy_trajectory)

# Result: clean_trajectory has same shape as input
# Shape: (timesteps, 4) - filtered estimate at EACH timestep
print(clean_trajectory.shape)  # (timesteps, 4)

# Use the results
for t in range(len(clean_trajectory)):
    x_clean, y_clean, vx_clean, vy_clean = clean_trajectory[t]
    print(f"Timestep {t}: position=({x_clean:.3f}, {y_clean:.3f})")
```

---

## 🎯 Method 2: Denoise a Window (Get Final Estimate)

**Use Case:** You have recent observations and only want the current best estimate.

```python
from kalman_denoiser import KalmanDenoiser
import numpy as np

# Recent observations (e.g., last 3 timesteps)
recent_window = np.array([
    [0.50, 0.50, 0.01, 0.01],   # t-2 (oldest)
    [0.51, 0.51, 0.02, 0.01],   # t-1
    [0.52, 0.52, 0.01, 0.02],   # t (current/newest)
])  # Shape: (window_size, 4)

# Create filter
kalman = KalmanDenoiser()

# Get denoised estimate at the FINAL timestep only
current_estimate = kalman.denoise(recent_window)

# Result: single estimate vector
# Shape: (4,) - [x, y, vx, vy] at the current time
print(current_estimate.shape)  # (4,)

x_now, y_now, vx_now, vy_now = current_estimate
print(f"Current clean estimate: pos=({x_now:.3f}, {y_now:.3f}), vel=({vx_now:.3f}, {vy_now:.3f})")
```

---

## ⚡ Method 3: Batch Processing (Multiple Sequences)

**Use Case:** You have many trajectories and want to process them efficiently.

```python
from kalman_denoiser import KalmanDenoiserBatch
import numpy as np

# Many sequences (e.g., from multiple drones or episodes)
batch_of_sequences = np.array([
    # Sequence 1
    [[0.50, 0.50, 0.01, 0.01],
     [0.51, 0.51, 0.02, 0.01],
     [0.52, 0.52, 0.01, 0.02]],
    
    # Sequence 2
    [[0.30, 0.40, 0.03, 0.02],
     [0.31, 0.41, 0.03, 0.02],
     [0.32, 0.42, 0.03, 0.02]],
    
    # ... more sequences
])  # Shape: (num_sequences, window_size, 4)

# Create batch processor
kalman_batch = KalmanDenoiserBatch(
    dt=1.0,
    pos_noise_std=3.5,
    vel_noise_std=0.1,
    process_noise_std=1.0,
)

# Process all sequences at once
final_estimates = kalman_batch.denoise_batch(batch_of_sequences)

# Result: final estimate for EACH sequence
# Shape: (num_sequences, 4)
print(final_estimates.shape)  # (100, 4) if you had 100 sequences

# Use results for each sequence
for i, estimate in enumerate(final_estimates):
    x, y, vx, vy = estimate
    print(f"Sequence {i}: final estimate = ({x:.3f}, {y:.3f}, {vx:.3f}, {vy:.3f})")
```

---

## 🔄 Method 4: Real-Time / Step-by-Step Processing

**Use Case:** Processing data as it arrives (streaming/real-time).

```python
from kalman_denoiser import KalmanDenoiser
import numpy as np

# Scenario: You receive measurements one at a time

kalman = KalmanDenoiser()

# First measurement arrives
first_measurement = np.array([0.50, 0.50, 0.01, 0.01])
kalman.reset(first_measurement)
current_estimate = kalman.x.copy()
print(f"After measurement 1: {current_estimate}")

# Second measurement arrives
second_measurement = np.array([0.51, 0.51, 0.02, 0.01])
kalman.predict()  # Predict next state
kalman.update(second_measurement)  # Incorporate measurement
current_estimate = kalman.x.copy()
print(f"After measurement 2: {current_estimate}")

# Third measurement arrives
third_measurement = np.array([0.52, 0.52, 0.01, 0.02])
kalman.predict()
kalman.update(third_measurement)
current_estimate = kalman.x.copy()
print(f"After measurement 3: {current_estimate}")

# Continue as more measurements arrive...
```

---

## 💾 Saving and Loading

### Save a Configuration

```python
kalman = KalmanDenoiser(process_noise_std=1.0)  # Your tuned value
kalman.save("my_kalman_config.npz")
```

### Load and Use

```python
# Load previously saved config
kalman = KalmanDenoiser.load("my_kalman_config.npz")

# Use it immediately
clean_estimate = kalman.denoise(noisy_window)
```

---

## 🌐 Complete Real-World Example

### Example: Denoising Drone Trajectories in Your Simulation

```python
import numpy as np
from kalman_denoiser import KalmanDenoiser

class DroneController:
    def __init__(self):
        # Create Kalman filter with optimal parameters
        self.kalman = KalmanDenoiser(
            dt=1.0,
            pos_noise_std=3.5,
            vel_noise_std=0.1,
            process_noise_std=1.0,  # Your tuned optimal value!
        )
        
        # Normalization constants (must match Kalman filter)
        self.X_NORM = 8500.0
        self.Y_NORM = 8000.0
        self.V_NORM = 36.0
    
    def normalize_state(self, x_m, y_m, vx_ms, vy_ms):
        """Convert from physical units to normalized."""
        return np.array([
            x_m / self.X_NORM,
            y_m / self.Y_NORM,
            vx_ms / self.V_NORM,
            vy_ms / self.V_NORM,
        ])
    
    def denormalize_state(self, state_norm):
        """Convert from normalized back to physical units."""
        return np.array([
            state_norm[0] * self.X_NORM,
            state_norm[1] * self.Y_NORM,
            state_norm[2] * self.V_NORM,
            state_norm[3] * self.V_NORM,
        ])
    
    def get_clean_estimate(self, recent_observations):
        """
        Get denoised estimate from recent noisy observations.
        
        Parameters
        ----------
        recent_observations : list of tuples
            Recent observations [(x_m, y_m, vx_ms, vy_ms), ...]
            in physical units (meters, m/s)
        
        Returns
        -------
        tuple
            Clean estimate (x_m, y_m, vx_ms, vy_ms) in physical units
        """
        # Normalize observations
        normalized_window = np.array([
            self.normalize_state(*obs) for obs in recent_observations
        ])
        
        # Run Kalman filter
        clean_normalized = self.kalman.denoise(normalized_window)
        
        # Convert back to physical units
        clean_physical = self.denormalize_state(clean_normalized)
        
        return tuple(clean_physical)


# Usage in your simulation
controller = DroneController()

# Simulate receiving noisy measurements over time
noisy_measurements = [
    (4250.0 + np.random.randn()*3.5, 
     4000.0 + np.random.randn()*3.5,
     18.0 + np.random.randn()*0.1,
     -12.0 + np.random.randn()*0.1)
    for _ in range(5)
]

# Get clean estimate
x_clean, y_clean, vx_clean, vy_clean = controller.get_clean_estimate(
    noisy_measurements
)

print(f"Clean position: ({x_clean:.2f}m, {y_clean:.2f}m)")
print(f"Clean velocity: ({vx_clean:.2f}m/s, {vy_clean:.2f}m/s)")

# Use clean estimate for collision detection, control, etc.
```

---

## 📋 Quick Reference Cheat Sheet

| **What You Want** | **Method to Use** | **Input Shape** | **Output Shape** |
|-------------------|-------------------|-----------------|------------------|
| Full trajectory cleaned | `denoise_sequence()` | `(T, 4)` | `(T, 4)` |
| Just current estimate | `denoise()` | `(window_len, 4)` | `(4,)` |
| Many sequences at once | `KalmanDenoiserBatch.denoise_batch()` | `(N, T, 4)` | `(N, 4)` |
| Real-time streaming | `reset()` → `predict()` → `update()` loop | `(4,)` per step | `(4,)` per step |

---

## ⚠️ Common Mistakes to Avoid

### ❌ WRONG: Using physical units
```python
# This will give WRONG results!
noisy_data = np.array([[4250, 4000, 18, -12]])  # Physical units
clean = kalman.denoise(noisy_data)  # ❌ WRONG!
```

### ✅ CORRECT: Normalize first
```python
# Normalize to [0, 1] range
noisy_data_norm = np.array([[4250/8500, 4000/8000, 18/36, -12/36]])
clean_norm = kalman.denoise(noisy_data_norm)  # ✅ CORRECT
# Then denormalize back if needed
clean_physical = clean_norm * np.array([8500, 8000, 36, 36])
```

### ❌ WRONG: Wrong shape
```python
# Missing sequence dimension
data = np.array([0.5, 0.5, 0.01, 0.01])  # Shape: (4,)
clean = kalman.denoise(data)  # ❌ ERROR: needs shape (T, 4)
```

### ✅ CORRECT: Add sequence dimension
```python
data = np.array([[0.5, 0.5, 0.01, 0.01]])  # Shape: (1, 4)
clean = kalman.denoise(data)  # ✅ CORRECT
```

---

## 🎓 Summary

**To use the Kalman filter in any method:**

1. **Import**: `from kalman_denoiser import KalmanDenoiser`

2. **Create once** (reuse for multiple calls):
   ```python
   kalman = KalmanDenoiser(process_noise_std=1.0)  # Your optimal value
   ```

3. **Normalize your data**:
   ```python
   data_norm = data_physical / np.array([8500, 8000, 36, 36])
   ```

4. **Call the appropriate method**:
   - Full trajectory → `kalman.denoise_sequence(trajectory)`
   - Current estimate → `kalman.denoise(recent_window)`
   - Batch → `KalmanDenoiserBatch().denoise_batch(batch)`

5. **Denormalize results**:
   ```python
   result_physical = result_norm * np.array([8500, 8000, 36, 36])
   ```

That's it! You now know everything to use the Kalman filter anywhere in your code! 🚀
