# Kalman Filter Parameter Guide

## 1. **Process Noise** (`process_noise_std`) - MOST IMPORTANT ⭐

**What it models:** How much velocity changes between timesteps (acceleration/maneuvers)

**Physical meaning:** 
- Process noise = σ_a (standard deviation of acceleration)
- If process_noise = 2.0 m/s², this means: "Velocity can change by roughly 2 m/s per second"

**Effect on filter:**
- **Low (0.1-1.0):** Filter trusts constant-velocity model
  - ✅ Smoother position estimates
  - ❌ Lags behind maneuvers, velocity estimates get worse
  
- **High (5.0-20.0):** Filter expects large velocity changes
  - ✅ Tracks maneuvers better, velocity estimates improve
  - ❌ Less position smoothing, more sensitive to noise

**Tune this when:**
- Your drones accelerate frequently (collision avoidance!)
- Velocity estimates are poor
- You see the "trade-off" between position and velocity

**Current issue:** Your values (0.1-3.0) are too low for maneuvering drones!

---

## 2. **Measurement Noise** (`pos_noise_std`, `vel_noise_std`)

**What it models:** Sensor measurement uncertainty

**Physical meaning:**
- `pos_noise_std = 3.5m`: GPS/position sensor has ±3.5m standard deviation
- `vel_noise_std = 0.1 m/s`: Velocity sensor has ±0.1 m/s standard deviation

**Effect on filter:**
- **Low measurement noise:** Filter trusts measurements more
- **High measurement noise:** Filter trusts model (prediction) more

**Tune this when:**
- You're uncertain about actual sensor specifications
- You want to be conservative (set slightly higher than actual)
- Testing "what if" scenarios with different sensors

**Your case:** You said these are good, so leave them as-is!

---

## 3. **Initial Covariance** (`P` matrix, lines 127-132 in kalman_denoiser.py)

**What it models:** Uncertainty about the very first measurement

**Current values:**
```python
Position uncertainty: 5.0 m  (± 5 meters)
Velocity uncertainty: 0.5 m/s (± 0.5 m/s)
```

**Effect on filter:**
- Only affects **first few timesteps** (2-5 steps)
- Filter quickly converges to optimal uncertainty
- **Minimal impact** on overall performance

**Tune this when:**
- Your first measurement is particularly noisy/unreliable
- You have prior knowledge about initial state quality
- You care a lot about the first 3-5 timesteps

**Priority:** LOW (unless you have specific requirements)

---

## 4. **Time Step** (`dt`)

**What it models:** Duration between measurements (seconds)

**Current:** `dt = 1.0 second`

**Effect on filter:**
- Affects how process noise propagates
- Position prediction: `x[t] = x[t-1] + vx[t-1] * dt`
- Higher dt → more uncertainty accumulates

**Tune this when:**
- Your actual measurement rate changes
- You're testing different sampling rates

**Your case:** If measurements are actually 1 second apart, leave as 1.0

---

## 5. **Normalization Constants** (`x_norm`, `y_norm`, `v_norm`)

**What they do:** Convert physical units to normalized [0, 1] scale for numerical stability

**Current:**
- `x_norm = 8500 m` (workspace width)
- `y_norm = 8000 m` (workspace height)  
- `v_norm = 36 m/s` (typical drone speed)

**Effect on filter:**
- Helps keep matrix operations numerically stable
- Should match your actual data scales

**Tune this when:**
- Your workspace size changes
- Drone speeds are very different
- You get numerical instability warnings

**Priority:** LOW (current values look reasonable)

---

## **Recommended Tuning Strategy**

### **Step 0: MOST IMPORTANT - Should You Even Measure Velocity? 🤔**

**Key Insight:** If velocity noise is high relative to position noise (in normalized space), 
you might get BETTER results by **NOT using velocity measurements at all**!

**Check your noise levels:**
- Position noise: 3.5m / 8500m = 0.0004 (normalized)
- Velocity noise: 0.1 m/s / 36 m/s = 0.0028 (normalized)
- **Velocity noise is 7× larger!**

**Test position-only Kalman:**
```bash
python tune_kalman_position_only.py --n_episodes 50
```
This will compare:
- Standard Kalman: measures [x, y, vx, vy]
- Position-Only Kalman: measures [x, y] only, estimates velocity from position changes

**When to use Position-Only:**
- ✅ Position is critical (collision avoidance!)
- ✅ Velocity measurements are noisy relative to position
- ✅ You care more about position accuracy than velocity accuracy

### **Step 1: Fix Process Noise (CRITICAL)**
```bash
python tune_kalman_advanced.py --n_episodes 50 --fine_tune
```
Based on your results: Optimal process_noise ≈ 1.0 m/s² (not 5-15 as initially expected!)

This means your drones **don't maneuver as aggressively** - constant velocity is a good model.

### **Step 2: (Optional) Adjust Measurement Noise**
If you're uncertain about sensor specs, test ±50% around current values:
- Position: 2.5m, 3.5m, 5.0m
- Velocity: 0.05, 0.1, 0.15 m/s

### **Step 3: (Rarely Needed) Tune Initial Covariance**
Only if first few timesteps matter a lot. Test:
- Initial position: 2m, 5m, 10m
- Initial velocity: 0.2, 0.5, 1.0 m/s

---

## **Understanding the Trade-offs**

```
High Process Noise (maneuvering drones)
    ↓
✅ Better velocity tracking
✅ Adapts quickly to maneuvers  
❌ Less position smoothing
❌ More measurement noise passes through

Low Process Noise (constant velocity)
    ↓
✅ Maximum position smoothing
✅ Filters out measurement noise well
❌ Poor velocity tracking
❌ Lags behind actual maneuvers
```

**For collision-avoidance drones:** You need HIGH process noise!

**BUT YOUR RESULTS SHOW:** Process noise = 1.0 is optimal (not 5-15)!
This means your drones actually follow constant-velocity quite well.

---

## **Position-Only Kalman Filter - A Better Option?**

### **The Problem with Standard Kalman**

Standard Kalman uses **both** position and velocity measurements:
- Measurement: `[x_noisy, y_noisy, vx_noisy, vy_noisy]`
- The filter tries to fuse all four noisy measurements

**But what if velocity measurements are noisier than position?**
- Position noise: 3.5m / 8500m = 0.0004 (normalized)
- Velocity noise: 0.1 m/s / 36 m/s = 0.0028 (normalized)  
- **Velocity is 7× noisier in normalized space!**

### **Position-Only Solution**

Only measure position, **estimate** velocity from position changes:
- Measurement: `[x_noisy, y_noisy]` only
- State: `[x, y, vx, vy]` (velocity is internal/hidden)
- Velocity estimated from: vx ≈ (x[t] - x[t-1]) / dt

**Benefits:**
- ✅ No noisy velocity measurements corrupting the estimate
- ✅ Filter infers velocity from smoother position estimates
- ✅ Better position accuracy (what matters for collision avoidance!)

**Trade-offs:**
- ⚠️ Velocity estimates may be less accurate (but they already were!)
- ⚠️ Requires slightly more tuning (process noise for velocity changes)

### **When to Use Position-Only**

Use when:
1. **Position is critical** (collision avoidance, control)
2. **Velocity measurements are noisy** (relative to position)
3. **Velocity from your sensor doesn't help much** (your +0.6% improvement!)

### **Test It Yourself**

```bash
python tune_kalman_position_only.py --n_episodes 50 --standard_process_noise 1.0
```

This will:
1. Tune the position-only filter
2. Compare it directly with your standard Kalman (process_noise=1.0)
3. Show if you gain position accuracy by ignoring velocity measurements
4. Save the better filter automatically

**Expected outcome based on your results:**
- Standard Kalman: 38.1% position improvement, 0.6% velocity improvement
- Position-Only: Likely 39-40% position improvement (no noisy velocity to corrupt it!)

---

## **Your Current Problem**

```
Results show:
- Position improvement: 38% ✅ (good!)
- Velocity improvement: -205% ❌ (TERRIBLE!)

Why: Process noise too low → filter assumes constant velocity → 
     can't track maneuvers → velocity estimates lag reality
     
Solution: Increase process noise to 5-20 m/s²
```

---

## **Quick Reference Table**

| Parameter | Current | Effect | When to Tune | Priority |
|-----------|---------|--------|--------------|----------|
| `process_noise_std` | 0.5-1.0 | Velocity tracking vs smoothing | Maneuvering drones | **HIGH** ⭐⭐⭐ |
| `pos_noise_std` | 3.5 m | Trust measurements vs model | Uncertain sensor specs | Medium |
| `vel_noise_std` | 0.1 m/s | Trust measurements vs model | Uncertain sensor specs | Medium |
| Initial position cov | 5.0 m | First few timesteps only | Special initialization | Low |
| Initial velocity cov | 0.5 m/s | First few timesteps only | Special initialization | Low |
| `dt` | 1.0 s | Time between measurements | Different sampling rate | Low |
| Normalization | 8500/8000/36 | Numerical stability | Data scale changes | Low |

