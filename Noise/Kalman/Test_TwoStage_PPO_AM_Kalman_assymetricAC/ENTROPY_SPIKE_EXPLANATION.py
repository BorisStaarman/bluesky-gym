"""
Entropy Spike Analysis - Iteration ~10-15
==========================================

OBSERVATION:
-----------
Looking at your training plots:
1. Iterations 0-10 (Stage 1 + Warm-up): Entropy is very low, ~0 or slightly negative
2. Around iteration 10-15: Large positive spike to ~2.5  
3. After iteration 15: Gradual decline to ~1.5-2.0

WHY THIS HAPPENS:
----------------

Stage 1 (Iterations 0-~80):
- Uses imitation learning with entropy_coeff = 0.0 (disabled)
- Log_std initialized at -2.5 → std = exp(-2.5) = 0.082
- Policy is VERY deterministic (mimicking teacher exactly)
- Entropy ≈ 0.5 * log(2πe * 0.082²) * 2 ≈ -4.0 (very low)

Stage 2 Warm-up (Iterations ~1-10):  
- Keeps log_std = -2.5 to maintain deterministic policy
- Critic learns to evaluate this deterministic behavior
- Entropy stays very low/negative
- Sometimes dips even lower if policy becomes MORE deterministic during training

Stage 2 Fine-tuning Starts (Iteration 11):
- Code executes: model.log_std.fill_(0.0)  
- Log_std jumps from -2.5 → 0.0
- Std jumps from 0.082 → 1.0 (12× increase!)
- Entropy jumps to: 0.5 * log(2πe * 1.0²) * 2 ≈ 2.84

This is the SPIKE you see in the plot!

Gradual Decline After Spike:
- Policy learns to balance exploration vs exploitation
- Log_std gradually adjusts (PPO learns optimal std)
- Entropy settles to ~1.5-2.0 (healthy exploration level)

IS THIS A PROBLEM?
-----------------

NO! This is EXPECTED and INTENTIONAL behavior:

✓ Warm-up phase NEEDS deterministic policy (low entropy)
  → Allows critic to learn value function for teacher-like behavior
  → Prevents policy from wandering during critic initialization

✓ Fine-tuning phase NEEDS exploration (higher entropy)
  → Allows policy to discover improvements beyond imitation
  → Prevents getting stuck in local optima

✓ The spike is just the TRANSITION between these phases
  → Not a bug, it's the warm-up completing successfully!

WHAT IF ENTROPY SPIKE IS TOO LARGE?
-----------------------------------

If you want a smoother transition, you can:

1. Gradually increase log_std instead of sudden jump:
   ```python
   # Instead of: model.log_std.fill_(0.0)
   # Use gradual increase over N iterations:
   for i in range(WARMUP_ITERATIONS+1, WARMUP_ITERATIONS+6):
       target_log_std = -2.5 + (i - WARMUP_ITERATIONS) * (0.0 - (-2.5)) / 5
       model.log_std.fill_(target_log_std)
       result = algo.train()
   ```

2. Use a smaller final log_std value:
   ```python
   model.log_std.fill_(-1.0)  # std = 0.37 instead of 1.0
   ```

3. Enable entropy_coeff in Stage 2 config to regulate exploration:
   ```python
   "entropy_coeff": 0.001,  # Already enabled in your config!
   ```
   This penalizes high entropy, preventing excessive exploration.

CURRENT VERDICT:
---------------

Your entropy behavior is NORMAL and HEALTHY:
- Low during warm-up (deterministic for critic learning) ✓
- Spike at transition (enabling exploration) ✓  
- Gradual decline during fine-tuning (learning optimal exploration) ✓
- Final value ~1.5-2.0 (good balance) ✓

The spike is not hurting performance - it's part of the two-stage training strategy.

FOCUS YOUR ATTENTION ON:
-----------------------

Instead of worrying about entropy, investigate:
1. ✓ Kalman filter issues (FIXED: first-step noise, process noise)
2. Waypoint success rate improvement (should improve after Kalman fixes)
3. Value function explained variance (should be >0.6 for good critic)
4. Episode reward trends (should increase steadily in Stage 2)

The entropy spike is a red herring - your real issue was the Kalman filter configuration!
"""

print(__doc__)
