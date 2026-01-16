# Modified Attention Architecture: Ownship Bypass Branch (SAC Version)

## Summary
This SAC_AM version implements the same **bypass branch** for ownship features as the Two_stage_AM version, preventing waypoint drift information from being filtered out during high intruder density scenarios.

## Key Changes

### Architecture Modification (Same as Two_stage_AM)

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
Policy/Value Networks (SAC Actor/Critic)
```

## Files Modified

### 1. **attention_model_A.py**
- ✅ Added comprehensive header documentation
- ✅ Changed `W_q_heads` from `ownship_dim (7)` to `intruder_dim (5)`
- ✅ Updated `forward()` to compute query from mean of intruder features
- ✅ Bypass ownship directly to concatenation step
- ✅ Same dimensions maintained (22 features total)

### 2. **main.py**
- ✅ Added header documentation noting the architectural modification
- ✅ Existing SAC training loop is fully compatible (no changes needed)

## Differences from Two_stage_AM Version

The architectural change is **identical**, but the training algorithm differs:

| Aspect | Two_stage_AM | SAC_AM |
|--------|-------------|---------|
| **Architecture** | Bypass branch | Bypass branch (same) |
| **Algorithm** | PPO with 2-stage training | SAC continuous training |
| **Stage 1** | Imitation learning (~75 iters) | N/A |
| **Stage 2** | RL fine-tuning (~100 iters) | SAC from scratch (~18000 iters) |
| **Exploration** | Entropy coefficient schedule | Automatic temperature tuning |
| **Value Function** | Shared value network | Separate Q-networks + V-network |

## Training Requirements

⚠️ **IMPORTANT**: This architectural change requires **retraining from scratch**!

The query projection layer has changed from 7→5 (ownship) to 5→5 (intruder), making pre-trained weights incompatible.

### Training Procedure (SAC):
```bash
cd SAC_AM/1_12
python main.py
```

This will train using SAC (Soft Actor-Critic) with:
- Automatic entropy temperature tuning
- Twin Q-networks for stability
- Continuous exploration throughout training
- No staged training (unlike Two_stage_AM)

### Expected Training Time:
- **SAC iterations**: ~18,000 (configurable via `TOTAL_ITERS`)
- **Environment**: Same as Two_stage_AM (20 agents, polygon airspace)
- **Observation space**: Same 112-dim vector (7 ownship + 19×5 intruders)
- **Action space**: Same 2-dim continuous (heading change, speed change)

## Expected Benefits (Same as Two_stage_AM)

1. **Improved Waypoint Tracking**: Waypoint drift signal no longer filtered by attention
2. **Better High-Density Performance**: Attention focuses on collision context without affecting navigation
3. **More Interpretable**: Clear separation between navigation and collision avoidance features
4. **Stable Training**: Reduced interference between different objectives

## Comparison with Two_stage_AM

After training both versions, you can compare:

### Metrics to Compare:
- **Waypoint success rate**: Should improve in both
- **Intrusion count**: Should remain low in both
- **Training stability**: SAC may be more stable (automatic temperature tuning)
- **Sample efficiency**: Two_stage_AM may be faster (imitation learning jumpstart)
- **Final performance**: Should be similar if both train to convergence

### When to Use Each:
- **Use Two_stage_AM (PPO)** when:
  - You have expert demonstrations (MVP solver)
  - Want faster initial convergence via imitation
  - Prefer staged training approach
  
- **Use SAC_AM (SAC)** when:
  - Want pure RL without imitation
  - Need automatic exploration tuning
  - Prefer continuous off-policy learning
  - Have more computational budget

## Rollback Instructions

If the bypass architecture performs worse, revert to original:

1. Restore original `attention_model_A.py` from a previous SAC_AM folder (if available)
2. Delete checkpoint directories: `models/*`
3. Set `FORCE_RETRAIN = True` in main.py
4. Retrain from scratch with original architecture

---

**Created**: January 12, 2026  
**Version**: SAC_AM/1_12 (Bypass Branch)  
**Algorithm**: SAC (Soft Actor-Critic)  
**Architecture**: Same bypass as Two_stage_AM/1_12
