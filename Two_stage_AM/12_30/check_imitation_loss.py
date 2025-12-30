# code to plot the imitation loss over training iterations
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import run_config


# path to the CSV file containing training metrics
script_dir = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.join(script_dir, "metrics")

# --- Path for model ---
CHECKPOINT_DIR = os.path.join(script_dir, f"models/stage1_imitation_loss_{run_config.RUN_ID}")

# plot the 