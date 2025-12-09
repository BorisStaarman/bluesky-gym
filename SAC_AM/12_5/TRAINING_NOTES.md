# Training Configuration Notes

## Current Model Status

### Old Trained Model (existing checkpoint)
- **Neighbors tracked**: 4 per agent
- **Observation space**: 3 + 7 * 4 = 31 dimensions
- **Alpha values**: 4 per agent
- **To evaluate**: Set `NUM_AC_STATE = 4` in `ma_env_SAC_AM.py`

### New Model Training (for 25 agents with full attention)
- **Neighbors tracked**: 24 per agent (all other agents)
- **Observation space**: 3 + 7 * 24 = 171 dimensions
- **Alpha values**: 24 per agent
- **To train**: Set `NUM_AC_STATE = N_AGENTS - 1` in `ma_env_SAC_AM.py`

## Steps to Train New Model

1. **Update environment configuration**:
   ```python
   # In bluesky_gym/envs/ma_env_SAC_AM.py line 35:
   NUM_AC_STATE = N_AGENTS - 1  # Track all other agents
   ```

2. **Delete or backup old checkpoint**:
   ```powershell
   # Move old model to backup
   Move-Item "SAC_AM/12_3/models" "SAC_AM/12_3/models_backup_4neighbors"
   ```

3. **Start training**:
   ```powershell
   python main.py
   ```

4. **Evaluate new model**:
   ```powershell
   python evaluate.py
   ```
   The visualization will now show 24 alpha values per agent (one for each neighbor).

## Why This Matters

- **Model architecture changes** with the number of neighbors
- The attention mechanism learns different patterns for 4 vs 24 neighbors
- Cannot mix checkpoints trained with different observation spaces
- Training with more neighbors captures richer interactions but takes longer
