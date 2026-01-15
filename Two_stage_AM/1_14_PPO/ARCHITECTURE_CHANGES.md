# Modified Attention Architecture: Ownship Bypass Branch

## Summary
This version implements a **bypass branch** for ownship features in the attention mechanism to prevent waypoint drift information from being filtered out during high intruder density scenarios.

## Key Changes

### 1. Architecture Modification (`attention_model_A.py`)

#### Before (Original):
```
Input: [Ownship (7) | Intruders (N×5)]
       ↓
Query (Q): Ownship (7) → Linear → (5)
Keys (K): Intruders (N×5) → Linear → (N×5)
Values (V): Intruders (N×5) → Linear → (N×5)
       ↓
Attention: Q attends to K,V
       ↓
Context: (15) from 3 heads
       ↓
Concatenate: [Ownship (7) + Context (15)] = 22
       ↓
Policy/Value Networks
```

#### After (Modified with Bypass):
```
Input: [Ownship (7) | Intruders (N×5)]
       ↓
       ├─────────────────────┐ BYPASS
       │                     ↓
       │              Ownship (7) [raw, unprocessed]
       │
       └→ Intruders (N×5)
              ↓
          Query (Q): Mean(Intruders) → Linear → (5)
          Keys (K): Intruders → Linear → (N×5)
          Values (V): Intruders → Linear → (N×5)
              ↓
          Attention: Q attends to K,V
              ↓
          Context: (15) from 3 heads
              ↓
Concatenate: [Ownship (7, bypassed) + Context (15)] = 22
       ↓
Policy/Value Networks
```

### 2. Specific Code Changes

#### Query Computation (Line ~165):
**Before:**
```python
query_h = self.W_q_heads[h](ownship_state).unsqueeze(1)
```

**After:**
```python
intruder_mean = intruder_states.mean(dim=1)  # (Batch, 5)
query_h = self.W_q_heads[h](intruder_mean).unsqueeze(1)  # (Batch, 1, 5)
```

#### W_q Projection Layer (Line ~51):
**Before:**
```python
# W_q: Projects Ownship (7) -> Head Dim (5)
self.W_q_heads = nn.ModuleList([
    nn.Linear(self.ownship_dim, self.head_dim, bias=True)
    for _ in range(self.num_heads)
])
```

**After:**
```python
# W_q: Projects INTRUDER (5) -> Head Dim (5) [CHANGED FROM OWNSHIP]
self.W_q_heads = nn.ModuleList([
    nn.Linear(self.intruder_dim, self.head_dim, bias=True)
    for _ in range(self.num_heads)
])
```

## Motivation

### Problem with Original Architecture:
- When intruder density is high, attention mechanism focuses heavily on collision avoidance
- Ownship features (including waypoint drift) pass through attention query projection
- Attention weights can suppress or dilute the waypoint drift signal
- Result: Poor waypoint tracking in congested scenarios

### Solution with Bypass:
- **Ownship features bypass attention entirely** - they flow directly to policy network
- Attention mechanism focuses purely on **intruder-to-intruder relationships**
- Waypoint drift signal is **always preserved** at full strength
- Clearer separation of concerns:
  - **Ownship (bypassed)**: Navigation, waypoint tracking, speed control
  - **Attention context**: Collision avoidance, neighbor awareness

## Expected Benefits

1. **Improved Waypoint Tracking**: Waypoint drift signal no longer filtered by attention
2. **Better High-Density Performance**: Attention focuses on collision context without affecting navigation
3. **More Interpretable**: Clear separation between navigation and collision avoidance features
4. **Stable Training**: Reduced interference between different objectives

## Training Requirements

⚠️ **IMPORTANT**: This architectural change requires **retraining from Stage 1**!

The query projection layer has changed from 7→5 (ownship) to 5→5 (intruder), making pre-trained weights incompatible.

### Training Procedure:
1. **Stage 1**: Imitation learning with new bypass architecture (~75 iterations)
2. **Stage 2**: RL fine-tuning on top of imitated policy (~100 iterations)

### Files Modified:
- `attention_model_A.py`: Core architecture changes
- `main.py`: Added documentation header

### Files Ready (No Changes Needed):
- `main.py`: Training loop already configured
- `bluesky_gym/envs/ma_env_two_stage_AM_PPO.py`: Environment unchanged
- Callbacks, metrics tracking: All compatible with new architecture

## Testing the Changes

After training completes, compare performance metrics:

### Expected Improvements:
- ✅ Higher waypoint success rate in high-density scenarios
- ✅ Lower average drift from optimal path
- ✅ Similar or better collision avoidance (intrusion count)
- ✅ More stable entropy/exploration during Stage 2

### Metrics to Monitor:
- Waypoint reached rate (should increase)
- Average episode length (should be similar or shorter)
- Total intrusions (should remain low)
- Training reward (should improve faster in Stage 2)
- VF explained variance (should stabilize quicker)

## Rollback Instructions

If the bypass architecture performs worse, revert to original:

1. Restore original `attention_model_A.py` from `1_9_PPO` or `1_7`
2. Delete checkpoint directories: `models/sectorcr_ma_sac/*`
3. Retrain from Stage 1 with original architecture

---

**Created**: January 12, 2026  
**Version**: 1_12 (Bypass Branch)  
**Previous Version**: 1_9_PPO (Original Architecture)
