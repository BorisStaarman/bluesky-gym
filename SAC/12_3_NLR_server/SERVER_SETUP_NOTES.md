# Server Setup Notes for Running on Remote Machine

## Changes Already Made ✅

1. **Matplotlib Backend** - Set to 'Agg' (non-interactive) for headless servers
2. **Figure Saving** - Training plot now saves to PNG file instead of showing
3. **File Paths** - Using relative paths with `script_dir`

## Additional Things to Check Before Running

### 1. Environment Variables & Display
The code should work on headless servers now, but verify:
- pygame won't try to initialize display (already handled by render_mode=None)
- No DISPLAY environment variable issues

### 2. Python Dependencies
Make sure the remote machine has all dependencies:
```bash
pip install ray[rllib] torch numpy matplotlib pygame bluesky-simulator gymnasium
```

### 3. CPU/GPU Configuration
Current setting: `num_env_runners=os.cpu_count() - 1`
- This will use all available CPU cores minus 1
- Adjust if the server has many cores or if you want to reserve resources
- GPU is set to 0 (CPU only), change if remote machine has GPU

### 4. Ray Configuration
Consider adding these to handle potential memory issues on servers:
```python
ray.init(
    num_cpus=None,  # Use all available
    object_store_memory=10 * 1024 * 1024 * 1024,  # 10GB object store
    _temp_dir="/tmp/ray"  # Custom temp directory if needed
)
```

### 5. Output & Logging
- Console output will be saved if you redirect: `python main.py > training.log 2>&1`
- Consider using `nohup` for long-running jobs: `nohup python main.py > training.log 2>&1 &`
- Metrics are already being saved to CSV files in the metrics directory

### 6. Checkpointing & Recovery
- Checkpoints are saved to `models/sectorcr_ma_sac/`
- If training crashes, set `FORCE_RETRAIN = False` to resume
- Monitor disk space - checkpoints can be large with many agents

### 7. Progress Monitoring
Since you can't see real-time output, you can:
- `tail -f training.log` to watch progress
- Check metrics CSV files periodically
- View the saved PNG plot after training

### 8. File Paths to Verify
All these should be relative (already done):
- `METRICS_DIR = os.path.join(script_dir, "metrics")`
- `CHECKPOINT_DIR = os.path.join(script_dir, "models/sectorcr_ma_sac")`
- Figure saves to script_dir

## Recommended Workflow

1. **Before pushing to GitHub:**
   - Test locally with small TOTAL_ITERS (e.g., 10) to verify everything works
   - Commit and push all changes

2. **On the remote machine:**
   ```bash
   git clone <your-repo-url>
   cd bluesky-gym/SAC/12_3_NLR_server
   pip install -r requirements.txt  # Create this if needed
   
   # Test run
   python main.py
   
   # Production run with logging
   nohup python main.py > training_$(date +%Y%m%d_%H%M%S).log 2>&1 &
   ```

3. **Monitor progress:**
   ```bash
   # Watch the log file
   tail -f training_*.log
   
   # Check if still running
   ps aux | grep python
   
   # Monitor metrics
   ls -lh metrics/run_*/
   ```

4. **After training:**
   - Download the best checkpoint
   - Download metrics CSV files
   - Download the training plot PNG
   - Run analysis on your local machine

## Performance Tips

- Start with fewer agents (10-15) for faster testing
- Use `EVALUATION_INTERVAL = None` to skip mid-training evaluations (faster)
- Increase `train_batch_size` if server has more RAM
- Consider using `num_gpus=1` if available (much faster)

## Troubleshooting

**If Ray workers crash:**
- Check RAM usage (too many agents/workers)
- Reduce `num_env_runners`
- Reduce replay buffer capacity

**If training is slow:**
- Check CPU usage (should be near 100% on all cores)
- Increase `sample_timeout_s` if episodes are very long
- Profile with Ray dashboard: `ray.init(include_dashboard=True)`

**If disk fills up:**
- Reduce `EVALUATION_INTERVAL` or disable it
- Delete old checkpoints manually
- Reduce replay buffer capacity
