# Complete Guide: Integrating SAC_AM Attention Model into BlueSky Simulation

## Table of Contents
1. [Overview](#overview)
2. [What is Different About SAC_AM](#what-is-different)
3. [Critical Requirements](#critical-requirements)
4. [Step-by-Step Integration](#step-by-step-integration)
5. [Model Architecture Deep Dive](#model-architecture)
6. [Observation Format Requirements](#observation-format)
7. [Code Implementation](#code-implementation)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The SAC_AM model uses an **attention mechanism** to process observations, which is fundamentally different from your previous SAC models. This guide explains exactly what needs to be implemented in your BlueSky simulation environment to use this model.

### Key Difference from Regular SAC Models

**Regular SAC Models:**
```
Observation (flat vector) → MLP layers → Actions
```

**SAC_AM (Attention Model):**
```
Observation → Split into [Ownship | Neighbors] 
           → Attention mechanism (focus on important neighbors)
           → Concatenate [Ownship Embedding | Attention Vector]
           → MLP layers → Actions
```

---

## What is Different About SAC_AM

### 1. **Observation Structure is CRITICAL**

The model **requires** observations in a very specific format:

```python
observation = [
    # Ownship features (3 values)
    cos_drift,           # Cosine of drift angle
    sin_drift,           # Sine of drift angle  
    airspeed_normalized, # Normalized airspeed
    
    # Neighbor 1 features (7 values) - sorted by distance, closest first
    dx_norm,             # Normalized relative X position
    dy_norm,             # Normalized relative Y position
    dvx_norm,            # Normalized relative X velocity
    dvy_norm,            # Normalized relative Y velocity
    cos_track,           # Cosine of relative track angle
    sin_track,           # Sine of relative track angle
    dist_norm,           # Normalized distance
    
    # Neighbor 2 features (7 values)
    dx_norm, dy_norm, dvx_norm, dvy_norm, cos_track, sin_track, dist_norm,
    
    # ... repeat for all N neighbors (N = 24 in your case)
    
    # If fewer than N neighbors exist, pad with zeros
    0, 0, 0, 0, 0, 0, 0,  # Padding for missing neighbors
]
```

**Total size:** 3 + (7 × 24) = 171 dimensions

### 2. **Attention Mechanism Processing**

Once the observation enters the model:

1. **Separation**: Model splits observation into ownship (first 3) and neighbors (remaining 168)
2. **Reshape**: Neighbors are reshaped from flat (168,) to (24, 7) - 24 neighbors with 7 features each
3. **Embedding**: 
   - Ownship: 3D → 128D embedding
   - Each neighbor: 7D → 128D embedding
4. **Attention Calculation** (like Transformer attention):
   - Query (Q): From ownship embedding
   - Keys (K): From all neighbor embeddings
   - Values (V): From all neighbor embeddings
   - Attention scores: Q · K^T / sqrt(128)
   - Mask out padded neighbors (where all features = 0)
   - Softmax to get attention weights (sum to 1.0)
5. **Context Vector**: Weighted sum of Values using attention weights
6. **Concatenation**: [Ownship Embedding | Context Vector] → 256D
7. **MLP**: Process through hidden layers → Actions

### 3. **What the Attention Does**

The attention mechanism allows the model to:
- **Focus** on the most relevant neighbors (usually the closest or most conflicting)
- **Ignore** far-away or irrelevant aircraft
- **Adapt** dynamically - attention weights change based on the situation
- **Handle variable numbers** of neighbors (padding is automatically masked)

---

## Critical Requirements

### ✅ Must Have

1. **Correct Observation Shape**: 171 dimensions (3 + 7×24)
2. **Correct Observation Order**: 
   - First 3 values = ownship state
   - Next 168 values = neighbors (24 × 7), sorted by distance
3. **Normalization**: Same normalization constants as training
4. **Neighbor Sorting**: Neighbors MUST be sorted by distance (closest first)
5. **Padding**: If fewer than 24 neighbors, pad with zeros

### ⚠️ Common Mistakes to Avoid

- ❌ Wrong observation order (neighbors before ownship)
- ❌ Not sorting neighbors by distance
- ❌ Different normalization than training
- ❌ Wrong number of neighbors (not 24)
- ❌ Including the ownship aircraft in the neighbor list
- ❌ Not padding when fewer than 24 neighbors exist

---

## Step-by-Step Integration

### Step 1: Copy Required Files

Copy these files to your BlueSky plugin directory:

```
bluesky/plugins/models_boris/
├── SAC_AM_7.pt                    # The exported model weights
├── attention_model_M.py           # The model architecture definition
└── sac_am_inference.py            # Inference wrapper (you'll create this)
```

### Step 2: Create the Inference Wrapper

This wrapper handles loading the model and running inference:

```python
# sac_am_inference.py
import torch
import torch.nn as nn
import numpy as np
from attention_model_M import AttentionSACModel
from gymnasium import spaces

class SACAttentionInference:
    """
    Inference wrapper for SAC with Attention Mechanism
    """
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        
        # Define observation and action spaces (must match training)
        obs_dim = 171  # 3 (ownship) + 7*24 (neighbors)
        action_dim = 2  # [heading_change, speed_change]
        
        obs_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(action_dim,), 
            dtype=np.float32
        )
        
        # Create the model architecture
        model_config = {
            "custom_model_config": {
                "hidden_dims": [256, 256],  # Must match training
                "is_critic": False,
            }
        }
        
        # For SAC, num_outputs = 2 * action_dim (mean + log_std)
        self.model = AttentionSACModel(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=2 * action_dim,  # 4 outputs for SAC
            model_config=model_config,
            name="policy"
        )
        
        # Load trained weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)
        
        print(f"✅ Loaded SAC Attention Model from {model_path}")
        print(f"   Observation space: {obs_dim}")
        print(f"   Action space: {action_dim}")
    
    def get_action(self, observation, deterministic=True):
        """
        Get action from observation.
        
        Args:
            observation: numpy array of shape (171,)
            deterministic: if True, use mean; if False, sample from distribution
            
        Returns:
            action: numpy array of shape (2,) with [heading_change, speed_change]
        """
        # Convert to tensor
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Forward pass through model
            input_dict = {"obs": obs_tensor}
            outputs, _ = self.model(input_dict, [], None)
            
            # outputs shape: (1, 4) = [mean_heading, mean_speed, log_std_heading, log_std_speed]
            action_dim = outputs.shape[1] // 2
            means = outputs[:, :action_dim]
            log_stds = outputs[:, action_dim:]
            
            if deterministic:
                # Use mean action (no exploration)
                action = means
            else:
                # Sample from distribution (with exploration)
                stds = torch.exp(log_stds)
                action = means + stds * torch.randn_like(means)
            
            # Clip to action space bounds [-1, 1]
            action = torch.clamp(action, -1.0, 1.0)
        
        return action.cpu().numpy()[0]
    
    def get_attention_weights(self):
        """
        Get the attention weights from the last forward pass.
        Useful for visualization/debugging.
        
        Returns:
            attention_weights: numpy array of shape (24,) showing attention to each neighbor
        """
        if hasattr(self.model, '_last_attn_weights'):
            return self.model._last_attn_weights[0, 0, :]  # Shape: (24,)
        return None
```

### Step 3: Prepare Observations in BlueSky

In your BlueSky plugin, you need to create observations in the correct format:

```python
def create_observation_for_attention_model(ac_idx, traf, center):
    """
    Create observation vector for SAC_AM attention model.
    
    Args:
        ac_idx: Index of ownship aircraft
        traf: BlueSky traffic object
        center: Center point [lat, lon] for relative coordinates
        
    Returns:
        observation: numpy array of shape (171,)
    """
    # Constants from training (MUST MATCH TRAINING VALUES)
    NUM_NEIGHBORS = 24
    MAX_SCENARIO_DIM_M = 50000.0
    MAX_RELATIVE_VEL_MS = 200.0
    DISTANCE_CENTER_M = 25000.0
    DISTANCE_SCALE_M = 25000.0
    AIRSPEED_CENTER_KTS = 250.0
    AIRSPEED_SCALE_KTS = 100.0
    MpS2Kt = 1.94384
    NM2KM = 1.852
    
    # 1. OWNSHIP FEATURES (3 values)
    wpt_lat, wpt_lon = get_waypoint_for_aircraft(ac_idx)  # Your function
    wpt_qdr, _ = bs.tools.geo.kwikqdrdist(
        traf.lat[ac_idx], traf.lon[ac_idx], 
        wpt_lat, wpt_lon
    )
    ac_hdg = traf.hdg[ac_idx]
    drift = bound_angle_180(ac_hdg - wpt_qdr)  # Your function
    
    cos_drift = np.cos(np.deg2rad(drift))
    sin_drift = np.sin(np.deg2rad(drift))
    airspeed_kts = traf.tas[ac_idx] * MpS2Kt
    airspeed_norm = (airspeed_kts - AIRSPEED_CENTER_KTS) / AIRSPEED_SCALE_KTS
    
    ownship_features = np.array([cos_drift, sin_drift, airspeed_norm], dtype=np.float32)
    
    # 2. OWNSHIP POSITION AND VELOCITY (for relative calculations)
    ac_loc = latlong_to_nm(center, [traf.lat[ac_idx], traf.lon[ac_idx]]) * NM2KM * 1000
    vx = np.cos(np.deg2rad(ac_hdg)) * traf.gs[ac_idx]
    vy = np.sin(np.deg2rad(ac_hdg)) * traf.gs[ac_idx]
    
    # 3. NEIGHBOR FEATURES (24 neighbors × 7 features each)
    candidates = []
    
    for i in range(len(traf.id)):
        if i == ac_idx:
            continue  # Skip self
        
        # Calculate relative position
        int_loc = latlong_to_nm(center, [traf.lat[i], traf.lon[i]]) * NM2KM * 1000
        dx = float(int_loc[0] - ac_loc[0])
        dy = float(int_loc[1] - ac_loc[1])
        
        # Calculate relative velocity
        int_hdg = traf.hdg[i]
        vxi = np.cos(np.deg2rad(int_hdg)) * traf.gs[i]
        vyi = np.sin(np.deg2rad(int_hdg)) * traf.gs[i]
        dvx = float(vxi - vx)
        dvy = float(vyi - vy)
        
        # Calculate relative track angle
        trk = np.arctan2(dvy, dvx)
        
        # Calculate distance
        d_now = float(np.hypot(dx, dy))
        
        candidates.append({
            'distance': d_now,
            'dx': dx,
            'dy': dy,
            'dvx': dvx,
            'dvy': dvy,
            'cos_trk': np.cos(trk),
            'sin_trk': np.sin(trk),
        })
    
    # 4. SORT BY DISTANCE (CRITICAL!)
    candidates.sort(key=lambda x: x['distance'])
    
    # 5. TAKE TOP 24 NEIGHBORS
    top_neighbors = candidates[:NUM_NEIGHBORS]
    
    # 6. CREATE NEIGHBOR FEATURE VECTOR
    neighbor_features = []
    for neighbor in top_neighbors:
        # Normalize features
        dx_norm = neighbor['dx'] / MAX_SCENARIO_DIM_M
        dy_norm = neighbor['dy'] / MAX_SCENARIO_DIM_M
        dvx_norm = neighbor['dvx'] / MAX_RELATIVE_VEL_MS
        dvy_norm = neighbor['dvy'] / MAX_RELATIVE_VEL_MS
        dist_norm = (neighbor['distance'] - DISTANCE_CENTER_M) / DISTANCE_SCALE_M
        
        # Add 7 features for this neighbor
        neighbor_features.extend([
            dx_norm, 
            dy_norm, 
            dvx_norm, 
            dvy_norm, 
            neighbor['cos_trk'], 
            neighbor['sin_trk'], 
            dist_norm
        ])
    
    # 7. PAD WITH ZEROS if fewer than 24 neighbors
    while len(neighbor_features) < NUM_NEIGHBORS * 7:
        neighbor_features.extend([0.0] * 7)
    
    neighbor_features = np.array(neighbor_features[:NUM_NEIGHBORS * 7], dtype=np.float32)
    
    # 8. CONCATENATE: [Ownship | Neighbors]
    observation = np.concatenate([ownship_features, neighbor_features])
    
    # 9. SANITIZE (handle NaN/Inf)
    observation = np.nan_to_num(observation, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return observation
```

### Step 4: Use the Model in BlueSky

```python
# In your BlueSky plugin initialization
class YourPlugin:
    def __init__(self):
        # Load the model
        model_path = "plugins/models_boris/SAC_AM_7.pt"
        self.inference_model = SACAttentionInference(model_path, device='cpu')
        
    def update(self, traf):
        """Called every simulation step"""
        for ac_idx in range(len(traf.id)):
            # Skip if aircraft doesn't need control
            if not self.needs_control(traf.id[ac_idx]):
                continue
            
            # 1. Create observation
            obs = create_observation_for_attention_model(
                ac_idx, traf, self.center
            )
            
            # 2. Get action from model
            action = self.inference_model.get_action(obs, deterministic=True)
            
            # 3. Scale actions to actual commands
            D_HEADING = 20.0  # Max heading change (degrees) - must match training
            D_VELOCITY = 50.0  # Max speed change (knots) - must match training
            
            heading_change = action[0] * D_HEADING
            speed_change = action[1] * D_VELOCITY
            
            # 4. Apply commands
            new_heading = bound_angle_360(traf.hdg[ac_idx] + heading_change)
            new_speed = max(100, min(400, traf.tas[ac_idx] * 1.94384 + speed_change))
            
            bs.stack.stack(f"HDG {traf.id[ac_idx]} {new_heading}")
            bs.stack.stack(f"SPD {traf.id[ac_idx]} {new_speed}")
            
            # 5. Optional: Get attention weights for visualization
            attn_weights = self.inference_model.get_attention_weights()
            if attn_weights is not None:
                # You can visualize which neighbors the aircraft is focusing on
                top_3_indices = np.argsort(attn_weights)[-3:][::-1]
                print(f"{traf.id[ac_idx]} paying attention to neighbors: {top_3_indices}")
```

---

## Model Architecture Deep Dive

### Internal Processing Flow

```
INPUT: observation [171]
  ↓
SPLIT:
  ownship_state [3]
  neighbor_states [168] → reshape to [24, 7]
  ↓
EMBEDDINGS:
  ownship_embed = LeakyReLU(Linear(3 → 128))     # Shape: [128]
  neighbor_embeds = LeakyReLU(Linear(7 → 128))   # Shape: [24, 128]
  ↓
ATTENTION (Multi-Head style):
  Query = W_q(ownship_embed)                      # Shape: [1, 128]
  Keys = W_k(neighbor_embeds).transpose()         # Shape: [128, 24]
  Values = W_v(neighbor_embeds)                   # Shape: [24, 128]
  ↓
  Scores = Query · Keys / sqrt(128)               # Shape: [1, 24]
  ↓
  Mask padding (where all neighbor features = 0)
  ↓
  Alpha = Softmax(Scores)                         # Shape: [1, 24]
  ↓
  Context = Alpha · Values                        # Shape: [1, 128]
  ↓
  Attention_Vector = tanh(Linear(Context))        # Shape: [128]
  ↓
CONCATENATION:
  Combined = [ownship_embed | Attention_Vector]   # Shape: [256]
  ↓
MLP (Hidden Layers):
  h1 = LeakyReLU(Linear(256 → 256))
  h2 = LeakyReLU(Linear(256 → 256))
  ↓
OUTPUT LAYER:
  output = Linear(256 → 4)                        # [mean_h, mean_v, log_std_h, log_std_v]
```

### Parameter Count Breakdown

```
Ownship FC:       3 × 128 + 128 bias      =    512 params
Intruder FC:      7 × 128 + 128 bias      =  1,024 params
W_q:             128 × 128                 = 16,384 params
W_k:             128 × 128                 = 16,384 params
W_v:             128 × 128                 = 16,384 params
Attn Output:     128 × 128 + 128 bias     = 16,512 params
Hidden Layer 1:  256 × 256 + 256 bias     = 65,792 params
Hidden Layer 2:  256 × 256 + 256 bias     = 65,792 params
Output Layer:    256 × 4 + 4 bias         =  1,028 params
                                          ---------------
TOTAL:                                    ≈ 200K params
```

---

## Observation Format Requirements

### Detailed Breakdown

```python
# Index 0-2: Ownship State
obs[0] = cos(drift_angle)           # Range: [-1, 1]
obs[1] = sin(drift_angle)           # Range: [-1, 1]
obs[2] = (airspeed - 250) / 100     # Normalized around 250 kts

# Index 3-9: Neighbor 1 (Closest)
obs[3] = dx / 50000                 # Normalized relative X position
obs[4] = dy / 50000                 # Normalized relative Y position
obs[5] = dvx / 200                  # Normalized relative X velocity
obs[6] = dvy / 200                  # Normalized relative Y velocity
obs[7] = cos(relative_track)        # Range: [-1, 1]
obs[8] = sin(relative_track)        # Range: [-1, 1]
obs[9] = (distance - 25000) / 25000 # Normalized distance

# Index 10-16: Neighbor 2 (2nd Closest)
# ... same 7 features ...

# Index 17-23: Neighbor 3 (3rd Closest)
# ... same 7 features ...

# ... continues for all 24 neighbors ...

# Index 164-170: Neighbor 24 (or padding if fewer neighbors)
obs[164] = 0  # Padding
obs[165] = 0
obs[166] = 0
obs[167] = 0
obs[168] = 0
obs[169] = 0
obs[170] = 0
```

### Why This Order Matters

The attention mechanism **expects neighbors to be sorted by distance** because:
1. The model was trained this way
2. Closer aircraft are typically more relevant for conflict resolution
3. The model learns patterns based on this ordering
4. Changing the order will produce incorrect/unpredictable behavior

### Normalization Constants (CRITICAL - Must Match Training)

```python
# These values MUST match what was used during training
# Check your training environment configuration!

MAX_SCENARIO_DIM_M = 50000.0      # Max scenario dimension (meters)
MAX_RELATIVE_VEL_MS = 200.0       # Max relative velocity (m/s)
DISTANCE_CENTER_M = 25000.0       # Center value for distance normalization
DISTANCE_SCALE_M = 25000.0        # Scale for distance normalization
AIRSPEED_CENTER_KTS = 250.0       # Center value for airspeed (knots)
AIRSPEED_SCALE_KTS = 100.0        # Scale for airspeed (knots)
```

**If these don't match training, the model will fail!**

---

## Code Implementation

### Complete Example: BlueSky Plugin

```python
# bluesky/plugins/sac_attention_plugin.py

import numpy as np
import bluesky as bs
from bluesky import stack, traf
from bluesky.tools import geo
from bluesky.tools.aero import nm, kts

# Import your inference wrapper
from .models_boris.sac_am_inference import SACAttentionInference

class SACAttentionPlugin:
    """
    BlueSky plugin for SAC with Attention Mechanism
    """
    def __init__(self):
        self.active = False
        self.model = None
        self.center = [52.3676, 4.9041]  # Amsterdam coordinates (or your center)
        
        # Action scaling (must match training)
        self.D_HEADING = 20.0  # degrees
        self.D_VELOCITY = 50.0  # knots
        
        # Normalization constants (must match training)
        self.MAX_SCENARIO_DIM_M = 50000.0
        self.MAX_RELATIVE_VEL_MS = 200.0
        self.DISTANCE_CENTER_M = 25000.0
        self.DISTANCE_SCALE_M = 25000.0
        self.AIRSPEED_CENTER_KTS = 250.0
        self.AIRSPEED_SCALE_KTS = 100.0
        self.NUM_NEIGHBORS = 24
        
        # Register stack commands
        stack.command("SACAM", self.enable, "ON/OFF", "Enable SAC Attention Model")
        
    def enable(self, flag=True):
        """Enable/disable the plugin"""
        if flag and self.model is None:
            # Load model on first enable
            model_path = "plugins/models_boris/SAC_AM_7.pt"
            self.model = SACAttentionInference(model_path, device='cpu')
            print("✅ SAC Attention Model loaded")
        
        self.active = flag
        return True
    
    def update(self):
        """Called every simulation step"""
        if not self.active or self.model is None:
            return
        
        # Process each aircraft
        for ac_idx in range(traf.ntraf):
            # Get observation
            obs = self.create_observation(ac_idx)
            
            # Get action from model
            action = self.model.get_action(obs, deterministic=True)
            
            # Scale and apply action
            heading_change = action[0] * self.D_HEADING
            speed_change = action[1] * self.D_VELOCITY
            
            new_heading = (traf.hdg[ac_idx] + heading_change) % 360
            current_speed_kts = traf.tas[ac_idx] * nm / kts
            new_speed = np.clip(current_speed_kts + speed_change, 100, 400)
            
            # Send commands to BlueSky
            stack.stack(f"HDG {traf.id[ac_idx]} {new_heading:.1f}")
            stack.stack(f"SPD {traf.id[ac_idx]} {new_speed:.1f}")
    
    def create_observation(self, ac_idx):
        """Create observation for attention model"""
        # 1. Ownship features
        # Get waypoint (simplified - adapt to your scenario)
        wpt_idx = traf.ap.route[ac_idx].iactwp
        if wpt_idx >= 0:
            wpt_lat = traf.ap.route[ac_idx].wplat[wpt_idx]
            wpt_lon = traf.ap.route[ac_idx].wplon[wpt_idx]
        else:
            wpt_lat, wpt_lon = traf.lat[ac_idx], traf.lon[ac_idx]
        
        wpt_qdr, _ = geo.kwikqdrdist(
            traf.lat[ac_idx], traf.lon[ac_idx],
            wpt_lat, wpt_lon
        )
        
        drift = self.bound_angle_180(traf.hdg[ac_idx] - wpt_qdr)
        cos_drift = np.cos(np.radians(drift))
        sin_drift = np.sin(np.radians(drift))
        
        airspeed_kts = traf.tas[ac_idx] * nm / kts
        airspeed_norm = (airspeed_kts - self.AIRSPEED_CENTER_KTS) / self.AIRSPEED_SCALE_KTS
        
        ownship_features = np.array([cos_drift, sin_drift, airspeed_norm], dtype=np.float32)
        
        # 2. Ownship position and velocity
        ac_loc = self.latlong_to_meters(self.center, [traf.lat[ac_idx], traf.lon[ac_idx]])
        vx = np.cos(np.radians(traf.hdg[ac_idx])) * traf.gs[ac_idx]
        vy = np.sin(np.radians(traf.hdg[ac_idx])) * traf.gs[ac_idx]
        
        # 3. Collect neighbor information
        candidates = []
        for i in range(traf.ntraf):
            if i == ac_idx:
                continue
            
            int_loc = self.latlong_to_meters(self.center, [traf.lat[i], traf.lon[i]])
            dx = int_loc[0] - ac_loc[0]
            dy = int_loc[1] - ac_loc[1]
            
            vxi = np.cos(np.radians(traf.hdg[i])) * traf.gs[i]
            vyi = np.sin(np.radians(traf.hdg[i])) * traf.gs[i]
            dvx = vxi - vx
            dvy = vyi - vy
            
            trk = np.arctan2(dvy, dvx)
            d_now = np.hypot(dx, dy)
            
            candidates.append({
                'distance': d_now,
                'dx': dx,
                'dy': dy,
                'dvx': dvx,
                'dvy': dvy,
                'cos_trk': np.cos(trk),
                'sin_trk': np.sin(trk),
            })
        
        # 4. Sort by distance
        candidates.sort(key=lambda x: x['distance'])
        top_neighbors = candidates[:self.NUM_NEIGHBORS]
        
        # 5. Create neighbor features
        neighbor_features = []
        for neighbor in top_neighbors:
            dx_norm = neighbor['dx'] / self.MAX_SCENARIO_DIM_M
            dy_norm = neighbor['dy'] / self.MAX_SCENARIO_DIM_M
            dvx_norm = neighbor['dvx'] / self.MAX_RELATIVE_VEL_MS
            dvy_norm = neighbor['dvy'] / self.MAX_RELATIVE_VEL_MS
            dist_norm = (neighbor['distance'] - self.DISTANCE_CENTER_M) / self.DISTANCE_SCALE_M
            
            neighbor_features.extend([
                dx_norm, dy_norm, dvx_norm, dvy_norm,
                neighbor['cos_trk'], neighbor['sin_trk'], dist_norm
            ])
        
        # 6. Pad if necessary
        while len(neighbor_features) < self.NUM_NEIGHBORS * 7:
            neighbor_features.extend([0.0] * 7)
        
        neighbor_features = np.array(neighbor_features[:self.NUM_NEIGHBORS * 7], dtype=np.float32)
        
        # 7. Concatenate
        observation = np.concatenate([ownship_features, neighbor_features])
        observation = np.nan_to_num(observation, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return observation
    
    @staticmethod
    def bound_angle_180(angle):
        """Bound angle to [-180, 180]"""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
    
    @staticmethod
    def latlong_to_meters(center, point):
        """Convert lat/lon to meters relative to center"""
        qdr, dist = geo.kwikqdrdist(center[0], center[1], point[0], point[1])
        x = np.cos(np.radians(qdr)) * dist * nm
        y = np.sin(np.radians(qdr)) * dist * nm
        return np.array([x, y])


# Initialize plugin
plugin = SACAttentionPlugin()
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. **Model produces strange actions**

**Symptoms:** Aircraft making unexpected maneuvers, actions seem random

**Causes:**
- ✗ Observation format is wrong
- ✗ Normalization constants don't match training
- ✗ Neighbors not sorted by distance
- ✗ Wrong number of neighbors

**Solution:**
```python
# Add debugging to verify observation
obs = create_observation(ac_idx)
print(f"Observation shape: {obs.shape}")  # Should be (171,)
print(f"Ownship features: {obs[:3]}")      # Check if reasonable
print(f"First neighbor: {obs[3:10]}")      # Check if reasonable
```

#### 2. **Model always outputs same action**

**Symptoms:** All aircraft do the same thing

**Causes:**
- ✗ Observation is all zeros
- ✗ Model not loaded correctly
- ✗ Model in wrong mode

**Solution:**
```python
# Verify model is in eval mode
model.eval()

# Check if observation has variation
print(f"Obs min: {obs.min()}, max: {obs.max()}, mean: {obs.mean()}")
```

#### 3. **Import errors for attention_model_M**

**Symptoms:** `ModuleNotFoundError: No module named 'attention_model_M'`

**Solution:**
```python
import sys
import os
sys.path.insert(0, os.path.join(bs.settings.plugin_path, 'models_boris'))
```

#### 4. **Attention weights don't make sense**

**Symptoms:** Model paying attention to wrong neighbors

**Causes:**
- ✗ Neighbors not sorted by distance
- ✗ Padding not handled correctly

**Solution:**
```python
# Verify sorting
candidates.sort(key=lambda x: x['distance'])
print(f"Distances: {[c['distance'] for c in candidates[:5]]}")  # Should be increasing

# Check attention weights
attn = model.get_attention_weights()
print(f"Top 3 attention indices: {np.argsort(attn)[-3:]}")
print(f"Top 3 attention values: {np.sort(attn)[-3:]}")
```

#### 5. **Model runs but performance is poor**

**Symptoms:** Aircraft not avoiding conflicts effectively

**Causes:**
- ✗ Action scaling different from training (D_HEADING, D_VELOCITY)
- ✗ Deterministic vs stochastic action selection
- ✗ Different scenario characteristics than training

**Solution:**
```python
# Verify action scaling
D_HEADING = 20.0   # Must match training
D_VELOCITY = 50.0  # Must match training

# Try deterministic actions first
action = model.get_action(obs, deterministic=True)
```

---

## Testing Checklist

Before deploying, verify:

- [ ] Observation shape is exactly (171,)
- [ ] Ownship features are first 3 values
- [ ] Neighbors are sorted by distance (closest first)
- [ ] Normalization constants match training
- [ ] Padding is applied when < 24 neighbors
- [ ] Action scaling (D_HEADING, D_VELOCITY) matches training
- [ ] Model loads without errors
- [ ] Actions are in expected range [-1, 1] before scaling
- [ ] Attention weights sum to approximately 1.0
- [ ] Model produces different actions for different observations

---

## Summary

**Key Takeaways:**

1. **Observation Format is Critical**: 3 ownship + (24 × 7) neighbors = 171 dimensions
2. **Sorting Matters**: Neighbors MUST be sorted by distance
3. **Normalization Matters**: Use the EXACT same constants as training
4. **Attention is Automatic**: Once observation is correct, attention works automatically
5. **Visualization**: Use attention weights to debug what the model is focusing on

**What Makes SAC_AM Different:**
- ✅ Handles variable numbers of neighbors (via padding + masking)
- ✅ Focuses on relevant conflicts (via attention mechanism)
- ✅ More interpretable (can visualize attention weights)
- ✅ Better scalability to complex scenarios

**What Stays the Same:**
- ✅ Action space (heading change, speed change)
- ✅ Action scaling (D_HEADING, D_VELOCITY)
- ✅ Integration with BlueSky (commands, updates)
- ✅ Model loading and inference process

Good luck with the integration! If you encounter issues, check the observation format first - that's where 90% of problems occur.
