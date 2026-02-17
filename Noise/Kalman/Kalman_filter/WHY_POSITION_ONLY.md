# Position-Only Kalman Filter - Why It Might Be Better

## The Key Insight

You asked: **"Can't we just predict position and ignore velocity?"** 

**YES! This is actually a great idea!** Here's why:

## The Problem

Your **velocity measurements are much noisier than position** (in normalized space):
- Position noise: 3.5m / 8500m = **0.0004**
- Velocity noise: 0.1 m/s / 36 m/s = **0.0028**
- **Velocity is 7× noisier!**

Your standard Kalman filter results:
- Position improvement: **38.1%** ✅
- Velocity improvement: **+0.6%** ❌ (barely helps!)

This suggests: **Velocity measurements aren't helping much, they might be hurting position estimates!**

## The Solution: Position-Only Kalman Filter

### What It Does

**Standard Kalman:**
```
Measures: [x, y, vx, vy] ← all four noisy values
Estimates: [x, y, vx, vy]
```

**Position-Only Kalman:**
```
Measures: [x, y] only ← just position
Estimates: [x, y, vx, vy] ← velocity inferred from position changes
```

The filter **still estimates velocity**, but it does so by tracking how position changes over time, rather than using noisy velocity measurements.

### Why This Works

1. **Velocity from position changes is cleaner**
   - Filter tracks: vx ≈ (x[t] - x[t-1]) / dt
   - Based on filtered (smooth) positions, not noisy measurements

2. **No corruption from noisy velocity measurements**
   - Standard Kalman tries to fuse noisy position + noisy velocity
   - Position-Only just uses cleaner position measurements

3. **Position is what matters for collision avoidance**
   - You care about WHERE drones are
   - Velocity is secondary (just helps predict future position)

## Your Question: "Stdev of velocity is smaller than position"

You're right that in **physical units**:
- Position: 3.5m
- Velocity: 0.1 m/s

But the Kalman filter works in **normalized space** where:
- Position: 3.5m / 8500m ≈ 0.0004
- Velocity: 0.1 m/s / 36 m/s ≈ 0.0028

So velocity noise is actually **7× larger** when properly scaled!

## How to Test

Run this command:
```bash
cd Noise/LSTM/10_2
python tune_kalman_position_only.py --n_episodes 50 --standard_process_noise 1.0
```

This will:
1. ✅ Test position-only filter with different settings
2. ✅ Compare it directly with your standard Kalman (process_noise=1.0)
3. ✅ Show detailed breakdown of which is better
4. ✅ Create visualizations
5. ✅ Save the winner automatically

## Expected Results

Based on your current results, I expect:

| Filter Type | Position Improvement | Velocity Improvement |
|-------------|---------------------|---------------------|
| Standard Kalman | 38.1% | +0.6% |
| **Position-Only** | **~39-40%** | ~0-5% |

**Prediction:** Position-only will be **1-2% better for position** (the critical metric!)

## The Process Noise Question

### Standard Kalman
- Process noise = 1.0 m/s² means "velocity can change by ~1 m/s per second"
- This is the acceleration/maneuver rate

### Position-Only Kalman
- Has **two** process noise parameters:
  - `process_noise_pos`: How much position drifts (keep low ~0.1m)
  - `process_noise_vel`: How much velocity changes (tune ~0.5-2.0 m/s²)

The tuning script will find the optimal values automatically!

## Summary: Your Original Intuition Was Correct!

You asked: "Isn't it better to only predict position?"

**YES!** Because:
1. ✅ Position measurements are cleaner (relatively)
2. ✅ Velocity measurements barely help (+0.6%)
3. ✅ Velocity from noisy measurements might corrupt position estimates
4. ✅ Position is what matters for collision avoidance
5. ✅ Filter can estimate velocity from position changes instead

**Bottom line:** Your noisy velocity measurements are probably **hurting more than helping**!

## What About "Constant Velocity Assumption"?

Great question! The Kalman filter **does assume constant velocity** in its model:
- Prediction: "velocity should stay the same"
- Reality: "velocity changes due to maneuvers/acceleration"

**But it still estimates velocity because:**
- You observe changing positions over time
- The filter infers: "If position changed by Δx, velocity must be ≈ Δx/Δt"
- Process noise allows velocity to change (models acceleration)

**Position-only just does this more cleanly:**
- Standard: Fuses model prediction + noisy velocity measurement
- Position-only: Only uses model prediction based on position changes

## Next Steps

1. **Run the test:**
   ```bash
   python tune_kalman_position_only.py --n_episodes 50
   ```

2. **Check the results:**
   - See if position-only beats standard Kalman for position accuracy
   - Look at the visualization: `denoiser_models/kalman_comparison.png`

3. **Use the winner:**
   - Script automatically saves the better filter
   - If position-only wins: `kalman_denoiser_position_only.npz`
   - If standard wins: stick with `kalman_denoiser.npz` (process_noise=1.0)

Let me know the results! 🚀
