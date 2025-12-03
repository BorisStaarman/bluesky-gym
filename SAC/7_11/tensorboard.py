import os
import subprocess
import webbrowser
import time

# ---- ADAPT THIS IF NEEDED ----
# This will load the RLlib logging directory used by your main training script.
# If you're using `algo.logdir`, logs are inside ~/ray_results/* by default.
LOGDIR = os.path.expanduser("~/ray_results")

# If you want to use your own metrics folder instead, change this:
# LOGDIR = r"C:\Users\boris\Documents\bsgym\bluesky-gym\SAC\5_11\metrics"

print(f"\n📊 TensorBoard will read logs from:\n   {LOGDIR}")

# Start TensorBoard
print("\n🚀 Launching TensorBoard...")
proc = subprocess.Popen(["tensorboard", "--logdir", LOGDIR])

# Wait a moment and try to auto-open browser
time.sleep(2)

url = "http://localhost:6006"
print(f"\n🌍 Open your browser and go to:\n   {url}\n")
try:
    webbrowser.open(url)
except:
    pass

# Keep terminal open
print("🔄 TensorBoard is running. Press Ctrl+C to stop.\n")
proc.wait()
