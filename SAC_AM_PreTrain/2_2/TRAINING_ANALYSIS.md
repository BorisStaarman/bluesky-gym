# SAC Training Analysis & PPO Integration

## Executive Summary

Your SAC training was experiencing instability during the burn-in (pretraining) phase. I've integrated the successful learning rate strategy from your PPO implementation (`Two_stage_AM/1_17_PPO`) which achieved stable imitation learning.

---

## 🎯 Key Changes Made

### 1. **Burn-in Learning Rates (Pretraining Phase)**

**BEFORE (Unstable):**
- Actor: `7.5e-7` (extremely conservative)
- Attention: `1e-4`
- Temperature: `1e-4`
- Critic: `1e-5`

**AFTER (PPO-Aligned):**
- **All components: `1e-4`** (unified rate)

**Rationale:**
- PPO Stage 1 proved that unified `1e-4` is stable for imitation learning
- SAC burn-in is analogous: offline learning from expert demonstrations
- The extreme asymmetry (7.5e-7 vs 1e-4) was causing gradient imbalance
- Critic at `1e-5` was too slow to learn expert value estimates

---

### 2. **Main Training Phase Learning Rates**

Updated all schedules to provide smoother transitions from burn-in:

#### Actor (Navigation Base Parameters)
```python
[0, 5e-6]      # Start 20x lower than burn-in (conservative)
[500, 1e-5]    # Gradual ramp-up
[1500, 2e-5]   # Peak for collision avoidance
[2500, 1e-5]   # Reduce for fine-tuning
[3500, 5e-6]   # Final stabilization
```

#### Attention Mechanism
```python
[0, 5e-5]      # Start at 50% of burn-in (smoother transition)
[1000, 3e-5]   # Maintain during collision learning
[2500, 2e-5]   # Stability phase
[3500, 1e-5]   # Final fine-tuning
```

#### Temperature (Attention Sharpness)
```python
[0, 5e-5]      # Match attention schedule
[1000, 3e-5]
[2500, 2e-5]
[3500, 1e-5]
```

#### Critic (Q-Network)
```python
[0, 1e-4]      # Continue at burn-in rate (needs to adapt to online data)
[1000, 5e-5]   # Stabilize as Q-function converges
[2500, 3e-5]   # Fine-tuning
[3500, 2e-5]   # Minimal updates
```

---

## 📊 Reward Scale Analysis

Your current reward structure:

### Per-Step Rewards (300 steps/episode)
| Component | Value | Episode Impact | Notes |
|-----------|-------|----------------|-------|
| **Step Penalty** | -0.01 | -3.0 total | Efficiency incentive ✅ |
| **Waypoint Reached** | +15.0 | +15.0 per agent | Strong positive signal ✅ |
| **Progress** | +10.0 scale | ±1-5 typical | Guidance reward ✅ |
| **Intrusion** | -150.0 | -150 per violation | ⚠️ **TOO LARGE FOR SAC** |
| **Proximity** | -4.0 max | -0.5 to -4.0 | Good warning signal ✅ |
| **Boundary** | -3.0 | -3.0 if violated | Reasonable ✅ |

### ⚠️ **Critical Issue: Intrusion Penalty**

**Current:** `-150.0` per timestep during intrusion  
**Problem:** This creates Q-value instability in SAC

**Typical Episode Breakdown:**
- **Good agent:** +15 (waypoint) + 5 (progress) - 3 (steps) = **+17**
- **Agent with 1 intrusion (5 steps):** +15 - 750 (intrusion) - 3 = **-738**
- **Ratio:** 43:1 penalty-to-reward ratio!

#### Why This Breaks SAC:

1. **Q-Value Scale:** SAC's critic learns Q-values in the range of expected returns
   - With `-150/step`, Q-values for bad states ≈ -750 to -1500
   - With `+15/waypoint`, Q-values for good states ≈ +10 to +20
   - **Imbalance:** 50:1 to 100:1 scale difference

2. **Gradient Instability:**
   - Large negative Q-values → large gradients during TD error calculation
   - Actor receives massive gradient spikes when near intrusion states
   - Leads to: policy oscillation, collapsed attention, NaN losses

3. **Exploration Breakdown:**
   - Agent becomes "fear-frozen" - too scared to explore optimal trajectories
   - Alpha (entropy term) fights against massive Q-penalties
   - Results in conservative, suboptimal navigation

#### Recommended Fix:

```python
# Option 1: Moderate Penalty (Recommended)
INTRUSION_PENALTY = -15.0  # 10x reduction, matches waypoint magnitude

# Option 2: Scaled Penalty (Alternative)
INTRUSION_PENALTY = -30.0  # 5x reduction, still strong signal

# Option 3: Per-Episode Penalty (Most Stable)
# Add intrusion count to terminal reward: -50 per intrusion
# Remove per-step penalty entirely
```

**Rationale:**
- SAC works best when reward magnitudes are similar (within 10x)
- `-15` matches the positive `+15` waypoint reward (balanced)
- `-30` still provides strong avoidance signal but more stable
- Collision avoidance should come from:
  1. Proximity warnings (-0.5 to -4.0 gradual penalty) ✅
  2. Moderate intrusion penalty (-15 to -30)
  3. Teacher demonstrations (MVP already avoids collisions)

---

## 🔍 Comparison: PPO vs SAC

### PPO Stage 1 (Imitation Learning)
- **Learning Rate:** `1e-4` (unified)
- **Loss Function:** MSE between student and teacher actions
- **Batch Size:** 32,000 transitions
- **Optimization:** 10 SGD iterations per batch
- **Result:** ✅ Stable convergence

### SAC Burn-in (Your System)
- **Learning Rate:** NOW `1e-4` (unified, matching PPO)
- **Loss Function:** SAC critic loss (TD error)
- **Batch Size:** 4,096 transitions from expert buffer
- **Optimization:** Continuous updates (1,500 iterations)
- **Expected Result:** ✅ Should match PPO stability

### Key Differences (Why SAC Needs Lower Reward Scale):

1. **Value Function:**
   - PPO: Learns state value V(s) ≈ 0-30 range
   - SAC: Learns action-value Q(s,a) with entropy bonus
   - SAC's Q-values include cumulative future rewards → more sensitive to scale

2. **Entropy Regularization:**
   - PPO: Entropy bonus in objective (controllable via coefficient)
   - SAC: Alpha (temperature) multiplies with entropy in Q-target
   - Large penalties fight against entropy → alpha scheduling breaks

3. **Target Networks:**
   - PPO: No target networks
   - SAC: Soft updates to target Q-networks (τ=0.001)
   - Unstable Q-values → unstable target updates → training collapse

---

## 📈 Expected Training Behavior

### Phase 1: Burn-in (Iterations 0-1500)
**With New LRs:**
- All components learn at `1e-4` from expert buffer
- **Expected:** Smooth MSE decrease, stable Q-values
- **Monitor:** Mean Q should converge to +10 to +15 range
- **Waypoint Rate:** Should reach 85-95% by iteration 1000

### Phase 2: Online Learning (Iterations 1501-3500)
**With Adjusted LR Schedule:**
- Actor: `5e-6 → 2e-5` (gradual ramp-up)
- Critic: `1e-4 → 2e-5` (continues learning, then stabilizes)
- **Expected:** Smooth transition, improving collision avoidance
- **Monitor:** Intrusion count should decrease, waypoint rate should stay ≥90%

---

## 🚀 Recommended Next Steps

### 1. **Immediate: Fix Intrusion Penalty**
```python
# In ma_env_SAC_AM.py, line ~41
INTRUSION_PENALTY = -15.0  # Changed from -150.0
```

### 2. **Test Burn-in Stability**
```bash
# Run only burn-in phase (set TOTAL_ITERS=0 after burn-in)
# Monitor: mean_q, waypoint_rate, critic_loss
python main.py
```

**Success Criteria:**
- Mean Q stabilizes at +10 to +20
- Waypoint rate reaches 90%+
- Critic loss decreases smoothly (no spikes)

### 3. **Full Training Run**
If burn-in is stable, proceed with full training:
- Watch for alpha schedule (should decay from 0.2 → 0.06)
- Monitor intrusion count (should decrease after iteration 1000)
- Check attention temperature (should stabilize at 2-4 range)

---

## 🔧 Troubleshooting

### If Burn-in Still Unstable:

**Symptom:** Waypoint rate stays low (<70%)
- **Check:** Expert buffer quality - is MVP teacher working correctly?
- **Try:** Increase burn-in iterations to 2000-2500

**Symptom:** Q-values explode (>100 or <-500)
- **Check:** Reward scale (intrusion penalty still too large)
- **Try:** Reduce all rewards by 10x globally

**Symptom:** NaN losses during burn-in
- **Check:** Batch sampling (might be hitting empty states)
- **Try:** Increase `num_steps_sampled_before_learning_starts` to 10000

### If Main Training Unstable:

**Symptom:** Performance collapse after burn-in
- **Check:** LR transition (5e-6 might be too low)
- **Try:** Start main phase at `1e-5` instead of `5e-6`

**Symptom:** Intrusion count increases
- **Check:** Alpha schedule (might be too high → over-exploration)
- **Try:** Set `FREEZE_UNTIL=2000` (delay entropy decay)

---

## 📚 References

- **PPO Source:** `Two_stage_AM/1_17_PPO/main.py` (lines 109-115, 258-280)
- **SAC Rewards:** `bluesky_gym/envs/ma_env_SAC_AM.py` (lines 38-47)
- **Learning Rate Logic:** `SAC_AM_PreTrain/2_2/main.py` (lines 75-120, 150-220)

---

## ✅ Summary Checklist

- [x] Unified burn-in LR to `1e-4` (matches PPO Stage 1)
- [x] Smoothed main training LR schedules
- [x] Added proper LR transition points (4 stages instead of 3)
- [ ] **TODO: Reduce intrusion penalty from -150 to -15**
- [ ] **TODO: Test burn-in phase in isolation**
- [ ] **TODO: Monitor Q-value scale during training**

---

**Last Updated:** 2026-02-02  
**Changes By:** GitHub Copilot (Claude Sonnet 4.5)  
**Integrated from:** PPO 1_17 successful training configuration
