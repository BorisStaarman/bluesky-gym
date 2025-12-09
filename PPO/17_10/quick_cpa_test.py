"""
Quick CPA risk validation - add this to your main.py if you want
"""
from bluesky_gym.envs.ma_env_ppo import SectorEnv

def quick_cpa_test():
    """Quick test of CPA risk function"""
    print("Quick CPA Risk Test:")
    
    # Test cases: (dx, dy, dvx, dvy, expected_risk_level)
    test_cases = [
        (0, 100, 0, -20, "HIGH"),    # head-on collision
        (50, 0, 0, 10, "LOW"),       # parallel paths
        (30, 0, -5, 0, "MEDIUM"),    # close approach
        (200, 0, 0, 5, "LOW"),       # distant parallel
    ]
    
    print("Scenario                    | Risk  | Expected")
    print("-" * 45)
    
    for dx, dy, dvx, dvy, expected in test_cases:
        risk = SectorEnv._cpa_risk(dx, dy, dvx, dvy)
        scenario = f"dx={dx:3}, dy={dy:3}, dvx={dvx:3}, dvy={dvy:3}"
        print(f"{scenario:<25} | {risk:.3f} | {expected}")
    
    print()

# Add this call in your main.py before training starts:
# quick_cpa_test()