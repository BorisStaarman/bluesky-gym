# 🎯 Quick Reference: SAC Learning Rates

## Learning Rate Configuration Summary

### 🔵 BURN-IN PHASE (Offline Expert Learning)
**Duration:** 1500 iterations  
**All Components:** `1e-4` (unified, matches PPO Stage 1)

| Component | Learning Rate | Notes |
|-----------|---------------|-------|
| Actor (Base) | **1e-4** | ↑ Increased from 7.5e-7 |
| Attention (W_q, W_k, W_v) | **1e-4** | Maintained |
| Temperature | **1e-4** | Maintained |
| Critic (Q-networks) | **1e-4** | ↑ Increased from 1e-5 |

**Rationale:** Unified LR proved stable in PPO pretraining. SAC burn-in is analogous (offline learning).

---

### 🟢 MAIN TRAINING PHASE (Online RL)
**Duration:** Iterations 1501-3500  
**Strategy:** Gradual LR decay with 4 transition points

#### Actor (Navigation Parameters)
```
Iteration    LR        Purpose
─────────────────────────────────────────────
0-500        5e-6      Conservative post-burn-in
500-1500     1e-5      Ramp-up for learning
1500-2500    2e-5      Peak collision avoidance
2500-3500    1e-5      Fine-tuning
3500+        5e-6      Final stabilization
```

#### Attention Mechanism
```
Iteration    LR        Purpose
─────────────────────────────────────────────
0-1000       5e-5      Smooth transition (50% of burn-in)
1000-2500    3e-5      Maintain during collision learning
2500-3500    2e-5      Stability phase
3500+        1e-5      Fine-tuning
```

#### Temperature (Attention Sharpness)
```
Iteration    LR        Purpose
─────────────────────────────────────────────
0-1000       5e-5      Match attention schedule
1000-2500    3e-5      Moderate learning
2500-3500    2e-5      Stability
3500+        1e-5      Final refinement
```

#### Critic (Q-Function)
```
Iteration    LR        Purpose
─────────────────────────────────────────────
0-1000       1e-4      Continue burn-in rate (adapt to online data)
1000-2500    5e-5      Q-function stabilization
2500-3500    3e-5      Fine-tuning
3500+        2e-5      Minimal updates
```

---

## 💡 Key Insights

### Why Unified Burn-in LR Works:
1. **PPO proved it:** Stage 1 imitation at `1e-4` was stable
2. **Offline learning:** No environment dynamics → can use higher LR
3. **Gradient balance:** All components learn at same pace
4. **Critic needs to catch up:** Previous `1e-5` was too slow

### Why Lower Main Phase LRs:
1. **Online uncertainty:** Environment samples are noisy
2. **Q-value stability:** Large LRs → unstable TD targets
3. **Policy robustness:** Conservative updates prevent collapse
4. **Gradual adaptation:** 4 stages provide smooth transitions

---

## 🎚️ Reward Scale Changes

### CRITICAL FIX: Intrusion Penalty
```python
# OLD (Unstable)
INTRUSION_PENALTY = -150.0  # 10:1 penalty-to-reward ratio

# NEW (Balanced)
INTRUSION_PENALTY = -15.0   # 1:1 with waypoint reward
```

### Why This Matters:
- **SAC Q-values:** Scale with cumulative rewards
- **Old scale:** Q(good) ≈ +15, Q(bad) ≈ -750 → 50:1 imbalance
- **New scale:** Q(good) ≈ +15, Q(bad) ≈ -15 → 1:1 balance
- **Result:** Stable gradients, better exploration

### Reward Magnitudes (Per Episode):
| Event | Old | New | Episode Impact |
|-------|-----|-----|----------------|
| Waypoint | +15 | +15 | +15 (1 per agent) |
| Intrusion (5 steps) | -750 | -75 | 50x → 5x penalty |
| Progress | +5 | +5 | +5 cumulative |
| Step penalty | -3 | -3 | -3 total |
| **Typical Good** | +17 | +17 | Baseline |
| **Typical Bad** | -738 | -63 | 43:1 → 3.7:1 |

---

## 📊 Monitoring Checklist

### During Burn-in (Iter 0-1500):
- [ ] Mean Q converges to +10 to +20
- [ ] Waypoint rate reaches 90%+ by iter 1000
- [ ] Critic loss decreases smoothly
- [ ] No NaN values in losses
- [ ] Attention sharpness stabilizes (0.3-0.6)

### During Main Training (Iter 1501-3500):
- [ ] Intrusion count decreases
- [ ] Waypoint rate stays ≥90%
- [ ] Alpha decays from 0.2 → 0.06
- [ ] Temperature stabilizes at 2-4
- [ ] Q-values stay within -50 to +50 range

---

## 🚨 Warning Signs

| Symptom | Possible Cause | Fix |
|---------|----------------|-----|
| WP rate <70% in burn-in | Expert buffer quality | Check MVP teacher |
| Q-values >100 or <-500 | Reward scale too large | Reduce all by 10x |
| NaN losses | Numerical instability | Check log_std clamping |
| Performance collapse | LR too high after burn-in | Reduce actor start LR to 1e-6 |
| Intrusions increase | Over-exploration | Delay alpha decay (`FREEZE_UNTIL=2000`) |

---

## 🔄 Comparison: PPO vs SAC (This System)

| Aspect | PPO (Stage 1) | SAC (Burn-in) | SAC (Main) |
|--------|---------------|---------------|------------|
| **LR** | 1e-4 (unified) | 1e-4 (unified) | 5e-6 to 2e-5 (actor) |
| **Duration** | 80 iters | 1500 iters | 3500 iters |
| **Loss** | MSE (actions) | TD error (Q) | SAC objective |
| **Batch Size** | 32K | 4K | 4K |
| **Data Source** | Online (teacher) | Offline (buffer) | Online (env) |
| **Result** | ✅ Stable | 🎯 Should be stable | ⏳ Testing |

---

**Files Modified:**
- `SAC_AM_PreTrain/2_2/main.py` (lines 75-120)
- `bluesky_gym/envs/ma_env_SAC_AM.py` (line 40)

**Last Updated:** 2026-02-02
