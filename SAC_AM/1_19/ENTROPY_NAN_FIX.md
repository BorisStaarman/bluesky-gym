# Entropy NaN Issue in SAC - Diagnosis & Fix

## Root Cause
The entropy becomes NaN in RLlib's SAC implementation due to numerical instability in the **TanhGaussian distribution**'s entropy calculation.

### Why This Happens:
1. **Tanh Squashing**: SAC uses `tanh()` to squash unbounded Gaussian samples to [-1, 1] for bounded action spaces
2. **Entropy Formula**: Entropy includes `log(1 - tanh(a)²)` term from the change-of-variables formula:
   ```
   Entropy = Gaussian_entropy - E[log(1 - tanh(a)²)]
   ```
3. **Numerical Issue**: When `tanh(a) ≈ ±1`, then `1 - tanh(a)² ≈ 0`, causing `log(0) = -inf` or NaN

### Common Triggers:
- Policy becomes very deterministic (low std → actions pushed to boundaries)
- Actions consistently near boundaries (±1)
- Training instabilities causing extreme action values
- Very small log_std values (< -10) in policy network
- Gradient explosion/vanishing in actor network

## What The Code Does Now

### 1. Enhanced Debugging (Iterations 1-10)
- Prints all available `learner_info` keys to see what RLlib provides
- Shows raw `policy_entropy` value and type
- Checks if entropy array contains NaN values
- Inspects model's `log_std` parameter if available
- Computes approximate Gaussian entropy (pre-tanh) as reference

### 2. Robust NaN Handling
```python
def to_scalar(val):
    # Filters out NaN values from arrays before taking mean
    # Returns NaN if all values are NaN
```

### 3. Numerical Stability Added
- `clip_actions=True` in SAC config to prevent extreme action values
- Keeps entropy as NaN in display for visibility (better than hiding the issue)

## Solutions to Try (In Priority Order)

### ✅ Solution 1: Check RLlib Logs (What We Just Added)
Run training and check the debug output for:
```
[Debug Iter 1] Available learner_info keys: [...]
[Model] log_std: [...], std: [...]
[WARNING] log_std is very negative (< -10), causing numerical issues!
```

If log_std < -10, the policy is too deterministic → entropy calculation breaks.

### Solution 2: Increase Initial Exploration (If log_std is too small)
```python
initial_alpha = 1.0,  # Higher alpha = more exploration (current: 0.5)
```

### Solution 3: Reduce Alpha Learning Rate (If alpha decays too fast)
```python
alpha_lr=1e-6,  # Slower alpha updates (current: 0 → 1e-5 → 1e-6)
```

### Solution 4: Add Log Std Constraints (In custom callback)
Add to `ForceAlphaCallback.on_train_result()`:
```python
# Prevent log_std from becoming too negative
policy = algorithm.get_policy("shared_policy")
if hasattr(policy, 'model') and hasattr(policy.model, 'log_std'):
    with torch.no_grad():
        # Clamp to reasonable range: std ∈ [exp(-5), exp(1)] = [0.0067, 2.72]
        policy.model.log_std.clamp_(-5, 1)
```

### Solution 5: Adjust Target Entropy (If using "auto" is problematic)
```python
# "auto" sets target_entropy = -action_dim (=-2 for 2D actions)
# Try a less negative value for more exploration:
target_entropy=-1.0,  # Instead of "auto" (which gives -2.0)
```

### Solution 6: Use Fixed Alpha (Disable entropy tuning)
```python
alpha_lr=0,  # Disable alpha learning
initial_alpha=0.1,  # Fixed value
```

## Monitoring Strategy

Track these metrics to diagnose the issue:

1. **Alpha (`alpha_value`)**: Should decay gradually from 0.5 → ~0.1
   - If drops too fast → entropy calculation may fail
   
2. **Mean Q (`mean_q`)**: Should be stable and negative (around -0.05)
   - Large fluctuations indicate training instability
   
3. **Actor Loss (`actor_loss`)**: Should be negative and stable
   - Spikes indicate gradient issues
   
4. **Log Std**: Check first 10 iterations
   - If < -10: Policy too deterministic
   - If > 5: Policy too random

## Impact on Reward-to-Entropy (R/E) Ratio

The R/E Ratio measures whether the agent prioritizes **task reward** over **exploration reward**:

### Formula
```
R/E Ratio = |mean_reward| / (α × H + ε)
```

Where:
- **α (Alpha)**: Temperature parameter (scales entropy in SAC objective)
- **H (Entropy)**: Policy entropy (actual randomness of actions)
- **α × H**: Weighted Entropy = the actual exploration signal in the loss
- **ε**: Small constant (1e-6) to prevent division by zero

### Interpretation
- **High R/E (> 1000)**: Agent strongly prefers task reward → good mission learning
- **Low R/E (< 100)**: Agent prefers exploration → may ignore mission
- **INVALID**: When entropy is NaN, R/E ratio cannot be computed

### Why Entropy Matters
- **Alpha alone is not enough**: α is just a scaling factor
- **Need actual entropy H**: Measures how random actions really are
- **If H = 0**: Agent collapsed to deterministic policy (no exploration regardless of α)
- **If H is NaN**: Cannot assess exploration vs reward balance

### When R/E Ratio is NaN
The code now displays: `R/E Ratio: INVALID (H=nan)`

This means:
- ❌ Cannot assess if agent ignores mission for exploration
- ✓ Must rely on **Alpha** as indirect exploration indicator
- ✓ Monitor **Reward** trend to see if agent is learning mission

## What to Expect

### Scenario A: Entropy NaN but Training is Stable
- **Symptom**: NaN entropy but rewards improve, alpha stable, losses smooth
- **Diagnosis**: RLlib reporting issue, not actual training problem
- **Action**: Ignore entropy, track alpha and rewards instead
- **Why**: The actual entropy is being used internally, just not reported correctly

### Scenario B: Entropy NaN and Training Unstable
- **Symptom**: NaN entropy + rewards collapse + Q-values explode
- **Diagnosis**: True numerical instability in policy
- **Action**: Apply Solution 4 (clamp log_std) + Solution 3 (slower alpha decay)

### Scenario C: Entropy NaN from Start
- **Symptom**: NaN entropy from iteration 1
- **Diagnosis**: Issue with custom model or RLlib version
- **Action**: Check if `policy_entropy` key exists in learner_info (debug will show this)

## Current Status ✅
- ✅ Added comprehensive debugging (iterations 1-10)
- ✅ Added robust NaN filtering in `to_scalar()`
- ✅ Added fallback entropy computation for reference
- ✅ Added `clip_actions=True` for numerical stability
- ✅ Entropy displays as "nan" for visibility
- ✅ Log std inspection when NaN detected

## Next Steps

1. **Run Training** - Check debug output in first 10 iterations
2. **Analyze Logs** - Look for WARNING messages about log_std
3. **If log_std < -10** - Apply Solution 4 (clamp log_std)
4. **If alpha decays too fast** - Apply Solution 3 (reduce alpha_lr)
5. **If NaN persists after iter 100** - Consider Solutions 5 or 6

## References
- [RLlib SAC Docs](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#sac)
- [rlkit TanhGaussian Implementation](https://github.com/rail-berkeley/rlkit) - Shows epsilon for numerical stability
- [Spinning Up SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html) - Explains entropy term
- [Original SAC Paper](https://arxiv.org/abs/1801.01290) - Section 3.2 on entropy regularization
