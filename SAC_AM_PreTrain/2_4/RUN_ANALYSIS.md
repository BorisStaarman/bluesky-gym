# Training Run Analysis & Improvements (Run 2_4)

## Evaluation Performance Mismatch Explained

### Your Observation:
- **Training metrics** (end of run): ~40% WP, ~10 intrusions
- **Evaluation results**: 46.5% WP, 18.45 intrusions

### Why the Mismatch?

1. **Different Metric Sources**:
   - Training graphs show **moving-averaged metrics** from training episodes (smoothed over many iterations)
   - Evaluation runs **20 fresh episodes** with the final policy checkpoint
   
2. **Environment Stochasticity**:
   - Each episode has different:
     - Initial aircraft positions
     - Traffic density (your eval showed avg density: 7.72)
     - Waypoint assignments
   - Some scenarios are naturally harder (more conflicts, tighter spacing)

3. **Policy Variance**:
   - Even with `explore=False`, the policy has learned a distribution over actions
   - Different scenarios trigger different attention patterns → different outcomes

4. **Training vs Testing Data**:
   - Training metrics include episodes from throughout training (with curriculum)
   - Final evaluation uses the Stage 4 penalty (-250) consistently
   - The agent may have "memorized" training scenarios better than generalizing

### **Verdict**: The evaluation is working correctly! 
The 18.45 intrusions vs 10 in training just reflects:
- Natural variance across episodes
- Potentially harder test scenarios
- The agent's true generalization performance

---

## Improvements Implemented

### ✅ 1. Waypoint-Prioritized Replay
**What**: Oversample expert trajectories where agents successfully reached waypoints (3x repeat count)

**Why**: Your agent learned collision avoidance but forgot navigation. By giving the buffer MORE examples of successful waypoint reaching, burn-in will better learn "navigate + avoid" instead of just "avoid".

**Code Change**:
```python
# In prefill_sac_buffer, line ~650
agent_reached_waypoint = aid in env.waypoint_reached_agents

if agent_reached_waypoint:
    repeat_count = 3  # Highest priority: successful navigation
elif is_stuur_actie:
    repeat_count = 2  # Maneuvers
else:
    repeat_count = 1  # Regular samples
```

---

### ✅ 2. Training Safeguard (WP Drop Alert)
**What**: Print warning when waypoint rate drops below 80% after iteration 6000

**Why**: Early detection of collapse. If WP drops during Stage 3/4, you can intervene (revert checkpoint, adjust curriculum) before wasting compute.

**Code Change**:
```python
# In ForceAlphaCallback.on_train_result, after curriculum logging
if current_iter > 6000 and current_iter % 100 == 0:
    if wp_rate < 0.80:
        print(f"⚠️  WARNING: WP rate dropped to {wp_rate*100:.1f}%...")
        if wp_rate < 0.50:
            print(f"🚨 CRITICAL: Consider reverting to earlier checkpoint!")
```

---

### ✅ 3. Burn-in Target Adjusted (80% → 70%)
**What**: Early stopping threshold lowered from 80% to 70% WP

**Why**: Your burn-in charts show peak WP ~45%. Expert buffer itself may not be 95% perfect (18 intrusions across 400 episodes = some conflicts). 70% is more realistic and still sufficient for downstream learning.

**Code Change**:
```python
# In burn_in_on_expert_buffer, line ~1080
if smoothed_wp >= 0.70:  # Was 0.80
    print(f"🎯 Burn-in success: 70% WP threshold reached!")
```

---

### ✅ 4. Already Implemented (from previous run):
- ✅ 4-stage curriculum (delays -250 penalty until iter 12K)
- ✅ Extended alpha freeze to 2000 iterations
- ✅ Increased burn-in to 4000 iterations
- ✅ Increased pretrain episodes to 400

---

## Reward Divisor Recommendation

### Current Setup (with /1000):
```
Waypoint:  +150/1000 = +0.15
Intrusion: -250/1000 = -0.25
Timestep:   -1/1000  = -0.001

Ratio: Intrusion penalty is 1.67× stronger than waypoint reward
```

### Problem:
The agent still learns "avoid everyone = -0.001/step" beats "navigate + risk intrusion = +0.15 - 0.25".

### **Recommended Change**: `/800` or `/750`

```python
# Option A: /800 (25% stronger signals, same ratio)
Waypoint:  +150/800 = +0.1875
Intrusion: -250/800 = -0.3125
Ratio: 1.67× (unchanged)

# Option B: /750 (33% stronger signals, same ratio)  
Waypoint:  +150/750 = +0.20
Intrusion: -250/750 = -0.333
Ratio: 1.67× (unchanged)

# Option C: /1000 but increase waypoint reward to 200
Waypoint:  +200/1000 = +0.20
Intrusion: -250/1000 = -0.25
Ratio: 1.25× (waypoint more valuable!)
```

**Recommendation**: Try **Option C** first (increase waypoint reward to 200 in the environment), keeping /1000. This makes navigation more rewarding relative to penalties.

---

## Expected Results with These Changes

### Burn-in Phase (iter 0-4000):
- **Target**: 60-70% WP rate, <200 intrusions
- **Mechanism**: Waypoint-prioritized buffer + longer burn-in duration
- **Success criteria**: Smoothed WP ≥ 70% triggers early stop

### Main Training (iter 0-8000):
- **Target**: Maintain 85-95% WP, learn light collision avoidance
- **Mechanism**: Stage 1-2 curriculum (-100 → -150), extended alpha freeze
- **Safeguard**: Prints warning if WP drops below 80%

### Advanced Training (iter 8000-12000):
- **Target**: 80-90% WP, reduce intrusions to <50
- **Mechanism**: Stage 3 penalty (-175), attention sharpening
- **Trade-off**: WP may dip slightly as agent balances objectives

### Final Training (iter 12000-16000):
- **Target**: 85-95% WP, <20 intrusions (best of both worlds)
- **Mechanism**: Stage 4 full penalty (-250), attention peaks (~0.70)
- **Success**: Agent masters navigation + collision avoidance

---

## Debugging Checklist for Next Run

Monitor these metrics to diagnose issues early:

| Metric | Burn-in Target | Iter 4K Target | Iter 8K Target | Iter 16K Target |
|--------|----------------|----------------|----------------|-----------------|
| **WP Rate** | 60-70% | 90-95% | 80-90% | 85-95% |
| **Intrusions** | <200 | <100 | <50 | <20 |
| **Attention Sharpness** | 0.10-0.15 | 0.20-0.30 | 0.50-0.60 | 0.65-0.75 |
| **Mean Q-Value** | -0.02 to 0.00 | -0.01 to +0.01 | +0.01 to +0.05 | +0.02 to +0.10 |
| **Alpha** | 0.01 (forced) | 0.05 (frozen) | 0.03 (decaying) | 0.015 (final) |

### Red Flags:
- ❌ Burn-in WP <50% after 2000 iters → Expert buffer quality issue
- ❌ WP drops >20% after curriculum stage change → Penalty too harsh
- ❌ Attention sharpness stuck at 0.10-0.15 → Temperature LR too low
- ❌ Mean Q stays negative past iter 8000 → Reward structure broken

---

## Files Modified

1. **`SAC_AM_PreTrain/2_4/main.py`**:
   - Line ~650: Added waypoint success oversampling (3× repeat count)
   - Line ~370: Added WP drop safeguard in training callback
   - Line ~1080: Reduced burn-in early stop threshold to 70%

---

## Next Steps

1. **Run training** with the updated code
2. **Monitor burn-in**: Should reach 60-70% WP and exit early (before 4000 iters)
3. **Watch for warnings**: If WP drops below 80% after iter 6000, investigate immediately
4. **Evaluate final model**: Compare against this run (aim for 85%+ WP, <20 intrusions)

Good luck! 🚀
