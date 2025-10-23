# Checklist for Copying to a New Folder

When you copy this folder (e.g., from `23_10` to `24_10`), update these **TWO** locations:

## 1. In `main.py` (line ~33):
```python
METRICS_DIR = "metrics_24_10"  # Update the date here!
```

## 2. In `bluesky_gym/envs/ma_env.py` (line ~70):
```python
METRICS_BASE_DIR = "metrics_24_10"  # Update the date here!
```

**That's it!** The metrics will now be saved to the correct directory matching your new folder name.

---

## What was fixed?
- The metrics directory is now configurable via `metrics_base_dir` parameter
- Training script passes `METRICS_DIR` to all environments
- Evaluation function uses the same `METRICS_DIR`
- Removed class-level state that could cause cross-contamination
- Each instance now properly cleans up only its own PID directory

## Why this matters:
Before this fix, copying the folder would still write metrics to the old `metrics_17_10` or `metrics_23_10` directory, mixing data from different experiments. Now each copy writes to its own metrics directory.
