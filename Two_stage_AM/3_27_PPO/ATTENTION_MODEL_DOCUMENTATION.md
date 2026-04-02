# Multi-Head Additive Attention Model with Learnable Temperature
## Complete Architecture Documentation for Simulation Integration

**Version:** 1_17_PPO (January 2026)  
**Paper Reference:** Groot et al. 2025 - Multi-Agent Conflict Resolution with Attention Mechanisms

---

## TABLE OF CONTENTS
1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
3. [The Learnable Temperature Parameter](#the-learnable-temperature-parameter)
4. [Input Format](#input-format)
5. [Forward Pass Step-by-Step](#forward-pass-step-by-step)
6. [Output Format](#output-format)
7. [Integration Guide for Simulation Software](#integration-guide)
8. [PyTorch State Dict Structure](#pytorch-state-dict-structure)

---

## OVERVIEW

This model uses a **3-head additive attention mechanism** to process multi-agent observations for conflict-free navigation. The key innovation in version 1_17_PPO is the **learnable temperature parameter** that controls attention sharpness during training.

### Key Features:
- **3 independent attention heads** (experts) that learn different aspects of the scene
- **Additive attention** (not dot-product) using tanh activation
- **Learnable temperature** parameter that adjusts attention focus dynamically
- **Separate value network** for PPO training (not needed for inference)
- **Masking support** for variable numbers of intruders

---

## MODEL ARCHITECTURE

### High-Level Structure

```
Input (Observation Vector)
    ↓
[Ownship State: 7 features] + [Intruder States: N × 5 features]
    ↓
Split into: Ownship (7) | Intruders (N, 5)
    ↓
┌─────────────────────────────────────────────┐
│   3-HEAD ADDITIVE ATTENTION MECHANISM       │
│                                             │
│  For each head h = 1, 2, 3:                │
│    1. Project Ownship → Query (5)          │
│    2. Project Intruders → Keys (N, 5)      │
│    3. Project Intruders → Values (N, 5)    │
│    4. Energy = tanh(Q + K)                  │
│    5. Scores = Energy · v_att               │
│    6. Scores = Scores × Temperature  ← NEW! │
│    7. Mask padding slots                    │
│    8. Weights = softmax(Scores)             │
│    9. Context = Σ(Weights × Values)         │
│                                             │
│  Output: [Context_h1, Context_h2, Context_h3]│
└─────────────────────────────────────────────┘
    ↓
Concatenate: [Ownship (7), Context (15)] = 22 features
    ↓
┌─────────────────────────────────────────────┐
│          POLICY NETWORK (ACTOR)             │
│                                             │
│  Hidden Layer 1: 22 → 512 (LeakyReLU)      │
│  Hidden Layer 2: 512 → 512 (LeakyReLU)     │
│  Output Layer: 512 → 2 (action means)      │
│                                             │
└─────────────────────────────────────────────┘
    ↓
Output: [heading_change_mean, speed_change_mean]
+ [log_std_heading, log_std_speed] (for exploration)
```

---

## THE LEARNABLE TEMPERATURE PARAMETER

### What is it?

The **temperature parameter** (τ) is a **learnable scalar** that scales the attention scores before the softmax operation. This is a crucial addition in version 1_17_PPO.

### Mathematical Formulation

```
Standard attention (old version):
    Scores = Energy · v_att
    Weights = softmax(Scores)

With learnable temperature (NEW):
    Scores = Energy · v_att
    Scores_scaled = Scores × |τ|  ← Temperature scaling
    Weights = softmax(Scores_scaled)
```

**Note:** We take `|τ|` (absolute value) to ensure the parameter stays positive during training.

### Why is it important?

The temperature controls **attention sharpness**:

| Temperature | Effect | Attention Distribution |
|------------|--------|------------------------|
| τ > 3.0 | **Sharp attention** | Focuses strongly on 1-2 closest threats |
| τ ≈ 1.0 | **Balanced attention** | Spreads focus across multiple threats |
| τ < 0.5 | **Diffuse attention** | Nearly uniform distribution (less selective) |

### Training Dynamics

During training, the model learns the optimal temperature:
- **Stage 1 (Imitation):** Temperature initialized to 3.0 (mimics teacher's sharp focus)
- **Stage 2 (RL):** Temperature adjusts based on reward feedback
  - If ignoring distant threats causes collisions → τ decreases (broader attention)
  - If over-reacting to distant threats reduces efficiency → τ increases (sharper focus)

### Implementation in Code

```python
# Initialization (in __init__)
self.temperature = nn.Parameter(torch.ones(1) * 3.0)

# Forward pass (in attention calculation)
scores_h = torch.matmul(energy_h, self.v_att_heads[h])  # (Batch, N, 1)
scores_h = scores_h.transpose(1, 2)  # (Batch, 1, N)
scores_h = scores_h * torch.abs(self.temperature)  # ← Temperature scaling
alpha_h = F.softmax(scores_h, dim=-1)  # Attention weights
```

### For Simulation Integration

**If you're loading a trained model for inference:**

The temperature value is **stored in the state_dict** as:
```
state_dict['temperature']: Tensor([2.847])  # Example final value
```

You **MUST** apply the temperature scaling when computing attention weights, exactly as shown above. Skipping this will cause the model to behave differently than during training.

---

## INPUT FORMAT

The model expects a **flat observation vector** with this exact structure:

```
[Ownship (7 features) | Intruder_1 (5) | Intruder_2 (5) | ... | Intruder_N (5)]
```

### Ownship State (7 features)
| Index | Feature | Description | Normalization |
|-------|---------|-------------|---------------|
| 0 | cos_drift | cos(heading_error_to_waypoint) | [-1, 1] |
| 1 | sin_drift | sin(heading_error_to_waypoint) | [-1, 1] |
| 2 | speed | Normalized airspeed | (speed - 35) / 10 knots |
| 3 | x | Ownship x-position from center | Normalized by scenario size |
| 4 | y | Ownship y-position from center | Normalized by scenario size |
| 5 | vx | Ownship x-velocity | m/s (raw) |
| 6 | vy | Ownship y-velocity | m/s (raw) |

### Intruder State (5 features per intruder)
| Index | Feature | Description | Normalization |
|-------|---------|-------------|---------------|
| 0 | rel_x | Relative x-position to ownship | Normalized by scenario size |
| 1 | rel_y | Relative y-position to ownship | Normalized by scenario size |
| 2 | rel_vx | Relative x-velocity to ownship | m/s (raw) |
| 3 | rel_vy | Relative y-velocity to ownship | m/s (raw) |
| 4 | distance | Euclidean distance to ownship | Normalized |

**Intruders are sorted by distance (closest first).**

**Padding:** If fewer than N intruders exist, pad remaining slots with zeros.

---

## FORWARD PASS STEP-BY-STEP

### Step 1: Input Parsing

```python
# Input: (Batch, 7 + N × 5)
ownship_state = input[:, :7]  # (Batch, 7)
intruder_flat = input[:, 7:7 + N×5]  # (Batch, N×5)
intruder_states = intruder_flat.view(-1, N, 5)  # (Batch, N, 5)
```

### Step 2: Multi-Head Attention (For each head h = 1, 2, 3)

#### 2a. Linear Projections
```python
# Query: Ownship (7) → (5)
Q_h = W_q_h(ownship_state)  # (Batch, 5)
Q_h = Q_h.unsqueeze(1)  # (Batch, 1, 5) for broadcasting

# Keys: Intruders (N, 5) → (N, 5)
K_h = W_k_h(intruder_states)  # (Batch, N, 5)

# Values: Intruders (N, 5) → (N, 5)
V_h = W_v_h(intruder_states)  # (Batch, N, 5)
```

#### 2b. Energy Calculation (Additive Attention)
```python
# Additive mechanism: tanh(Q + K)
energy_h = torch.tanh(Q_h + K_h)  # (Batch, N, 5)
```

#### 2c. Score Calculation
```python
# Project to scalar using scoring vector v_att
scores_h = energy_h @ v_att_h  # (Batch, N, 1)
scores_h = scores_h.transpose(1, 2)  # (Batch, 1, N)

# *** APPLY TEMPERATURE SCALING ***
scores_h = scores_h * |temperature|  # (Batch, 1, N)
```

#### 2d. Masking (Handle Padding)
```python
# Detect padding: if intruder row is all zeros
is_padding = (intruder_states.abs().sum(dim=2) < 1e-6)  # (Batch, N)

# Mask padded slots with -inf so softmax ignores them
scores_h = scores_h.masked_fill(is_padding.unsqueeze(1), float('-inf'))
```

#### 2e. Attention Weights
```python
alpha_h = softmax(scores_h, dim=-1)  # (Batch, 1, N)
alpha_h = nan_to_num(alpha_h, nan=0.0)  # Safety for all-padding case
```

#### 2f. Context Vector
```python
# Weighted sum: Context = Σ(alpha × V)
context_h = (alpha_h @ V_h).squeeze(1)  # (Batch, 5)
```

### Step 3: Concatenate Heads
```python
# Stack all 3 head outputs
context = [context_h1, context_h2, context_h3]  # Each (Batch, 5)
context_vector = torch.cat(context, dim=1)  # (Batch, 15)
```

### Step 4: Policy Network
```python
# Combine ownship and attention context
x = torch.cat([ownship_state, context_vector], dim=1)  # (Batch, 22)

# Hidden layers
x = LeakyReLU(Linear_512(x))  # (Batch, 512)
x = LeakyReLU(Linear_512(x))  # (Batch, 512)

# Output layer
action_means = Linear_2(x)  # (Batch, 2)
```

### Step 5: Add Exploration Noise (Training Only)
```python
# For stochastic actions during training
log_std = model.log_std  # Learnable parameter (2,)
output = [action_means, log_std]  # (Batch, 4)
```

---

## OUTPUT FORMAT

### For Inference (Deterministic Actions)
```python
output = [heading_change_mean, speed_change_mean]  # (2,)
```

Values are in **normalized action space** [-1, 1]:
- `heading_change`: Multiply by 45° to get actual heading change
- `speed_change`: Multiply by 3.33 knots to get actual speed change

### For Training (Stochastic Actions)
```python
output = [mean_heading, mean_speed, log_std_heading, log_std_speed]  # (4,)
```

Sample action: `action = mean + std × noise`, where `std = exp(log_std)`

---

## INTEGRATION GUIDE FOR SIMULATION SOFTWARE

### Option 1: Load PyTorch Model Directly

```python
import torch
import numpy as np

# 1. Load state dict
state_dict = torch.load('Two_stage_AM_Stage2_iter88.pt')

# 2. Create model instance (need attention_model_A.py)
from attention_model_A import AttentionSACModel
from gymnasium import spaces

obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7 + 5*19,), dtype=np.float32)
action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

model = AttentionSACModel(
    obs_space, 
    action_space, 
    num_outputs=2,
    model_config={
        'custom_model_config': {
            'hidden_dims': [512, 512],
            'is_critic': False,
            'n_agents': 20
        }
    }, 
    name="policy"
)

# 3. Load weights
model.load_state_dict(state_dict)
model.eval()  # Set to evaluation mode

# 4. Inference
def get_action(observation):
    with torch.no_grad():
        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        output, _ = model.forward({'obs': obs_tensor}, state=[], seq_lens=None)
        action_means = output[0, :2].numpy()  # First 2 outputs
    return action_means
```

### Option 2: Manual Implementation (No PyTorch Dependency)

If you cannot use PyTorch in your simulation software, you must:

1. **Extract all weights** from the state_dict and convert to your format
2. **Implement the attention mechanism** exactly as described above
3. **CRITICAL:** Remember to apply the temperature scaling:
   ```
   scores_scaled = scores * abs(temperature_value)
   ```
4. **Implement softmax** with masking for padding
5. **Implement the feedforward network** with LeakyReLU activation

**Temperature value from trained model:**
```python
temperature = state_dict['temperature'].item()  # e.g., 2.847
```

---

## PYTORCH STATE DICT STRUCTURE

When you load the `.pt` file, it contains these parameters:

### Attention Mechanism (3 heads × 3 matrices each)

```
temperature: Tensor([2.847])  ← NEW PARAMETER!

W_q_heads.0.weight: (5, 7)  # Head 1 - Query projection
W_q_heads.0.bias: (5,)
W_k_heads.0.weight: (5, 5)  # Head 1 - Key projection
W_k_heads.0.bias: (5,)
W_v_heads.0.weight: (5, 5)  # Head 1 - Value projection
W_v_heads.0.bias: (5,)
v_att_heads.0: (5, 1)       # Head 1 - Scoring vector

W_q_heads.1.weight: (5, 7)  # Head 2 - Query projection
W_q_heads.1.bias: (5,)
... (same structure for head 2)

W_q_heads.2.weight: (5, 7)  # Head 3 - Query projection
W_q_heads.2.bias: (5,)
... (same structure for head 3)
```

### Policy Network (Actor)

```
hidden_layers.0.weight: (512, 22)  # First hidden layer
hidden_layers.0.bias: (512,)
hidden_layers.1.weight: (512, 512) # Second hidden layer
hidden_layers.1.bias: (512,)
final_layer.weight: (2, 512)       # Output layer
final_layer.bias: (2,)
log_std: (2,)                      # Exploration parameter
```

### Value Network (Not needed for inference)

```
value_branch.0.weight: (512, 22)
value_branch.0.bias: (512,)
value_branch.2.weight: (512, 512)
value_branch.2.bias: (512,)
value_branch.4.weight: (1, 512)
value_branch.4.bias: (1,)
```

---

## CRITICAL NOTES FOR SIMULATION INTEGRATION

### ✅ DO:
1. **Apply temperature scaling** to attention scores before softmax
2. **Mask padding slots** by setting scores to -inf
3. **Use LeakyReLU(0.2)** for hidden layer activations
4. **Sort intruders by distance** before feeding to model
5. **Normalize inputs** exactly as during training

### ❌ DON'T:
1. **Skip temperature scaling** - this will break attention focus
2. **Use ReLU instead of LeakyReLU** - model expects negative slopes
3. **Forget to mask padding** - will cause softmax to fail
4. **Change normalization scales** - model is sensitive to input distribution
5. **Mix up coordinate systems** - ensure consistent reference frame

---

## TESTING YOUR IMPLEMENTATION

To verify correct integration:

1. **Load a known checkpoint**
2. **Feed the same observation** to both PyTorch model and your implementation
3. **Compare outputs** - should match within 1e-5 tolerance
4. **Check attention weights** - visualize which intruders get focus
5. **Monitor temperature** - should be ~2.5-3.5 for trained Stage 2 models

### Example Test Case:
```python
# Test observation (1 ownship + 2 intruders, rest padded)
obs = np.array([
    # Ownship: cos_drift, sin_drift, speed, x, y, vx, vy
    0.9, 0.1, 0.5, 100.0, 50.0, 15.0, 2.0,
    
    # Intruder 1: rel_x, rel_y, rel_vx, rel_vy, dist
    -50.0, 30.0, 5.0, -3.0, 58.3,
    
    # Intruder 2: rel_x, rel_y, rel_vx, rel_vy, dist
    80.0, -40.0, -2.0, 8.0, 89.4,
    
    # Remaining 17 intruders: all zeros (padding)
    *([0.0] * (5 * 17))
], dtype=np.float32)

# Expected output (example, will vary by model):
# action ≈ [0.234, -0.156]  (heading_change, speed_change)
```

---

## QUESTIONS?

**Q: Do I need to implement the value network for inference?**  
A: No, only the policy (actor) network is needed. The value network is only used during PPO training.

**Q: What if my simulation has a different number of agents?**  
A: The model can handle variable numbers of intruders (1-19) thanks to masking. Just pad unused slots with zeros.

**Q: Can I use a different temperature value?**  
A: You **must** use the trained temperature from the state_dict. Changing it will degrade performance significantly.

**Q: What's the difference between Stage 1 and Stage 2 models?**  
A: Stage 1 learns from a teacher (behavior cloning), Stage 2 is fine-tuned with RL. Stage 2 is optimized for reward maximization and should perform better in novel scenarios.

---

## VERSION HISTORY

- **1_17_PPO (Current):** Added learnable temperature parameter
- **1_13_PPO:** Fixed temperature at 1.0 (no learning)
- **1_9_PPO:** Original implementation without temperature scaling

**⚠️ Models from different versions are NOT compatible! Always use the correct architecture version for your checkpoint.**

---

**Document End** - Good luck with your integration! 🚀
