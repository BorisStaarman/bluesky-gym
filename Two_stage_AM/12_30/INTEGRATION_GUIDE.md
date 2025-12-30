# Stage 1 Behavioral Cloning with Attention Mechanism - Integration Guide

## Overview
This integration combines a multi-head attention mechanism with behavioral cloning (imitation learning) for Stage 1 of your two-stage RL training pipeline.

## What Changed

### 1. **Main Training Script (`main.py`)**
- **Import**: Now imports from `ma_env_two_stage_AM.py` (the correct environment with MVP teacher)
- **Model Registration**: Registered `AttentionSACModel` as `"attention_sac"` with RLlib
- **Stage 1 Configuration**:
  ```python
  "model": {
      "custom_model": "attention_sac",
      "custom_model_config": {
          "hidden_dims": [256, 256],  # Downstream MLP layers
          "is_critic": False,         # Actor network for imitation
      }
  }
  ```
- **Attention Metrics**: Added logging for attention weights and model health metrics

### 2. **Attention Model (`attention_model_A.py`)**
- Compatible with PPO (used for behavioral cloning)
- No changes needed - already perfectly designed for this task!
- Architecture:
  - **Input**: Observation vector `[ownship(7), intruders(5*N)]`
  - **Attention**: 3 independent heads, each projects to 5-dim space
  - **Output**: Concatenated attention context (15-dim) → MLP → Action (2-dim)

### 3. **Environment (`ma_env_two_stage_AM.py`)**
- Already provides teacher actions via `_calculate_mvp_action()`
- Stores teacher actions in `infos["teacher_action"]` during step
- Observation format matches attention model expectations perfectly

## How It Works

### Stage 1: Behavioral Cloning (Current Focus)
1. **Environment Step**:
   - Agent takes action from neural network
   - Environment calculates optimal MVP action (teacher)
   - Both stored: `obs`, `teacher_action`

2. **Data Collection**:
   - Callback (`MVPDataBridgeCallback`) extracts teacher actions from infos
   - Injects them into training batch as `"teacher_targets"`

3. **Neural Network Forward Pass**:
   - Observation → **Attention Model** → Action prediction
   - Attention mechanism learns to focus on relevant nearby aircraft
   - 3 attention heads capture different aspects of the scenario

4. **Loss Calculation**:
   - MSE between predicted action and teacher action
   - `Loss = ||NN_action - MVP_action||²`
   - Backpropagation trains both attention weights and downstream MLP

5. **Training Loop**:
   - The attention mechanism weights (`W_q`, `W_k`, `W_v`) are trained
   - The scoring vectors (`v_att`) are trained
   - The downstream MLP is trained
   - All parameters update to minimize imitation loss

### Stage 2: RL Fine-Tuning (Future)
- Uses the same attention model (pre-trained weights)
- Switches to standard PPO loss (policy gradient)
- Continues learning from experience

## Observation Structure

The environment provides observations in the exact format the attention model expects:

```python
# Observation Vector: [Ownship(7), Intruder1(5), Intruder2(5), ..., IntruderN(5)]

Ownship (7 features):
  [0] cos(drift)      # Cosine of drift angle from waypoint
  [1] sin(drift)      # Sine of drift angle from waypoint
  [2] airspeed        # Normalized airspeed
  [3] x               # Normalized x position
  [4] y               # Normalized y position
  [5] vx              # Normalized x velocity
  [6] vy              # Normalized y velocity

Each Intruder (5 features, sorted by distance):
  [0] distance        # Normalized distance to intruder
  [1] dx_rel          # Relative x position
  [2] dy_rel          # Relative y position
  [3] vx_rel          # Relative x velocity
  [4] vy_rel          # Relative y velocity
```

**Why This Format?**
- Attention model compares ownship (query) with each intruder (keys/values)
- Relative positions/velocities are rotation-invariant
- Distance sorting ensures closest threats get attention first
- Padding with zeros handles variable number of active agents

## Attention Mechanism Details

### Architecture
```
Input: [Ownship(7), Intruders(5*N)]
  ↓
Split: Ownship(7) | Intruder₁(5) | ... | IntruderN(5)
  ↓
For each of 3 attention heads:
  Query = W_q(Ownship)           → [5-dim]
  Keys = W_k(Intruders)          → [N × 5-dim]
  Values = W_v(Intruders)        → [N × 5-dim]
  
  Energy = tanh(Query + Keys)     → [N × 5-dim]
  Scores = v_att^T · Energy       → [N × 1]
  Attention = softmax(Scores)     → [N × 1]
  Context = Σ(Attention × Values) → [5-dim]
  ↓
Concatenate 3 heads: [Context₁(5) | Context₂(5) | Context₃(5)] → [15-dim]
  ↓
Combine: [Ownship(7) | AttentionContext(15)] → [22-dim]
  ↓
MLP: [22] → [256] → [256] → [2] (action)
```

### What Gets Trained?
1. **Attention Weights**: `W_q`, `W_k`, `W_v` for each of 3 heads (learns feature projections)
2. **Scoring Vectors**: `v_att` for each head (learns importance scoring)
3. **Downstream MLP**: Dense layers that process attention output
4. **All parameters** update via backpropagation from imitation loss

### Benefits
- **Selective Focus**: Learns to attend to threatening aircraft
- **Scalability**: Handles variable number of agents gracefully
- **Interpretability**: Can visualize attention weights to see what model focuses on
- **Multi-Perspective**: 3 heads capture different aspects (e.g., closest threat, converging traffic, future conflicts)

## Training Flow

### Stage 1 (100 iterations)
```
Episode Collection → MVP Teacher Calculates Actions → Store in Infos
                                     ↓
                        Callback Extracts Teacher Actions
                                     ↓
                          Training Batch with Targets
                                     ↓
            NN Forward Pass (with Attention) → Predicted Actions
                                     ↓
                Loss = MSE(Predicted, Teacher) ← TRAINS ATTENTION
                                     ↓
                    Backprop → Update All Weights
                                     ↓
                              Repeat
```

### What to Monitor
- **Imitation Loss**: Should decrease over iterations (target: < 0.01)
- **Attention Sharpness**: Mean of max attention weights (should be > 0.5)
- **Weight Norms**: `wq_weight_norm`, `wk_weight_norm`, `wv_weight_norm` (should be stable)
- **Gradient Norms**: `wq_grad_norm`, etc. (should not explode or vanish)

## How to Run

### 1. Test Integration (Recommended First)
```bash
cd Two_stage_AM/12_16_2
python test_integration.py
```

This validates:
- Environment loads correctly
- Attention model instantiates
- Forward pass works
- Observation structure is correct
- Teacher actions are available

### 2. Run Stage 1 Training
```bash
python main.py
```

**Key Parameters in `main.py`:**
- `N_AGENTS = 20`: Number of agents in environment
- `iterations_stage1 = 100`: Stage 1 training iterations
- `RUN_STAGE_2 = False`: Set to True to automatically transition to Stage 2

### 3. Monitor Training
Look for:
```
Stage 1 - Iter 1/100 | Imitation Loss: 0.425316
   ⭐ New best Stage 1 loss: 0.425316 (saved to stage1_best_weights)
Stage 1 - Iter 2/100 | Imitation Loss: 0.312445
...
```

Loss should decrease steadily. If it plateaus > 0.05, there may be an issue.

## Expected Results

### Good Training Indicators
- **Loss decreases**: From ~0.5 to < 0.01
- **Attention sharpness increases**: Agent learns to focus on key threats
- **Stable gradients**: No NaN or infinite values
- **Convergence**: Loss plateaus near zero after ~50 iterations

### Troubleshooting

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| Loss stays high (> 0.1) | Learning rate too low | Increase `lr` to 3e-4 |
| Loss oscillates wildly | Learning rate too high | Decrease `lr` to 5e-5 |
| NaN in attention weights | Gradient explosion | Add gradient clipping (already at 1.0) |
| All attention equal | No learning signal | Check teacher actions are varying |
| Slow convergence | Batch size too small | Increase `train_batch_size` |

## Code Modifications Summary

### Files Modified
1. ✅ `main.py` - Import, model config, attention metrics
2. ✅ `attention_model_A.py` - Value function stub for PPO compatibility
3. ✅ `ma_env_two_stage_AM.py` - No changes needed (already perfect!)

### Files Created
1. ✅ `test_integration.py` - Validation script
2. ✅ `INTEGRATION_GUIDE.md` - This document

## Next Steps

1. **Run Test**: `python test_integration.py`
2. **Train Stage 1**: `python main.py` (watch loss decrease)
3. **Analyze Attention**: Add visualization code to plot attention weights
4. **Stage 2**: Set `RUN_STAGE_2 = True` and run RL fine-tuning

## Questions & Answers

**Q: Does the attention mechanism train during Stage 1?**  
A: Yes! All parameters (attention weights, scoring vectors, MLP) train via backpropagation from the imitation loss.

**Q: Why use attention for imitation learning?**  
A: The attention mechanism learns which aircraft are most relevant for decision-making. This is better than fixed input ordering because:
- Focuses on nearby/threatening aircraft
- Handles variable number of agents
- More interpretable (can visualize what model "looks at")

**Q: Can I use a different teacher besides MVP?**  
A: Yes! Just modify `_calculate_mvp_action()` in the environment to return different actions.

**Q: What if I have more/fewer agents?**  
A: Change `N_AGENTS` in `main.py`. The attention mechanism automatically adapts to any number of intruders.

## Success Criteria

Stage 1 is successful when:
- ✅ Imitation loss < 0.01
- ✅ Attention sharpness > 0.5
- ✅ No NaN/Inf values
- ✅ Model checkpoint saved successfully

Then you're ready for Stage 2 RL fine-tuning!

---

**Author**: GitHub Copilot  
**Date**: December 19, 2025  
**Version**: 1.0
