"""Quick test to verify Kalman filter import works"""

import sys
import os

# Add the path to test
sys.path.insert(0, r'c:\Users\boris\Documents\bsgym\bluesky-gym')

try:
    print("Testing import from bluesky_gym.envs...")
    from bluesky_gym.kalman_filter import KalmanDenoiser
    print("✅ SUCCESS! Import worked.")
    
    print("\nCreating KalmanDenoiser instance...")
    kalman = KalmanDenoiser(process_noise_std=1.0)
    print("✅ SUCCESS! Instance created.")
    print(f"   Kalman filter configured with process_noise_std=1.0")
    
except ImportError as e:
    print(f"❌ ImportError: {e}")
    print(f"\nPython path:")
    for p in sys.path:
        print(f"  - {p}")
        
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")
