# Quick Start Checklist

## Before Training

- [ ] Verify all dependencies are installed (ray, torch, gymnasium)
- [ ] Check BlueSky simulation is accessible
- [ ] Run integration test: `python test_integration.py`
- [ ] Verify all tests pass ✓

## Training Stage 1

- [ ] Review settings in `main.py`:
  - `N_AGENTS = 20` (number of agents)
  - `iterations_stage1 = 100` (training iterations)
  - `RUN_STAGE_2 = False` (disable Stage 2 for now)
  
- [ ] Start training: `python main.py`

- [ ] Monitor output for:
  - "✓ Attention model instantiated" message
  - Decreasing imitation loss
  - No error messages
  
- [ ] Expected output:
  ```
  Stage 1 - Iter 1/100 | Imitation Loss: 0.425316
  Stage 1 - Iter 2/100 | Imitation Loss: 0.312445
  ...
  ✓ Stage 1 Complete. Checkpoint saved: models/sectorcr_ma_sac/stage1_best_weights
  ```

## After Training

- [ ] Check results:
  - Imitation loss graph saved to `metrics/stage1_imitation_loss_*.png`
  - Model checkpoint saved to `models/sectorcr_ma_sac/stage1_best_weights/`
  
- [ ] Verify best loss is < 0.05

- [ ] If Stage 2 needed:
  - Set `RUN_STAGE_2 = True` in `main.py`
  - Run again: `python main.py` (will load Stage 1 weights)

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | Training script (run this) |
| `attention_model_A.py` | Attention mechanism model |
| `ma_env_two_stage_AM.py` | Environment with MVP teacher |
| `test_integration.py` | Validation tests |
| `INTEGRATION_GUIDE.md` | Detailed documentation |

## Troubleshooting

**Problem**: Import errors  
**Solution**: Ensure you're in the correct directory and PYTHONPATH includes bluesky_gym

**Problem**: High imitation loss (> 0.1 after 50 iters)  
**Solution**: Check teacher actions are being calculated correctly

**Problem**: NaN values  
**Solution**: Reduce learning rate or check observation normalization

**Problem**: Slow training  
**Solution**: Increase `num_env_runners` or use GPU

## Success Metrics

✅ Integration test passes  
✅ Training runs without errors  
✅ Imitation loss < 0.01  
✅ Attention metrics logged  
✅ Model checkpoint saved  

Then you're ready for Stage 2 RL fine-tuning!
