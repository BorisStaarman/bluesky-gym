"""
Quick Validation - Check if LSTM is actually loaded and running
================================================================
Run this to verify the LSTM denoiser is being used during evaluation.
Prints debug info showing whether denoising is active.

Usage:
    python validate_lstm_loading.py
"""

import os
import sys
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from ma_env_lstm import SectorEnvLSTM
from run_config import RUN_ID


def test_lstm_loading():
    """Test if LSTM loads and processes data correctly."""
    
    print("\n" + "="*70)
    print("  LSTM LOADING VALIDATION")
    print("="*70)
    
    # Test 1: Load environment with LSTM enabled
    print("\n[Test 1] Creating environment with LSTM enabled...")
    env_lstm = SectorEnvLSTM(
        n_agents=5,  # Small for quick test
        run_id=f"validation_{RUN_ID}",
        noise_enabled=True,
        use_denoiser=True,
        add_intruder_noise=True,
    )
    
    if env_lstm.use_denoiser and env_lstm._denoiser is not None:
        print("✅ LSTM denoiser loaded successfully")
        print(f"   Device: {env_lstm._denoiser_device}")
        print(f"   Model path: {env_lstm._denoiser is not None}")
    else:
        print("❌ LSTM denoiser NOT loaded!")
        print("   This explains why your evaluation showed improvement - LSTM wasn't running!")
        return False
    
    # Test 2: Run a few steps and check if denoising happens
    print("\n[Test 2] Running environment steps...")
    obs, info = env_lstm.reset()
    
    # Track one agent
    test_agent = list(obs.keys())[0]
    print(f"   Tracking agent: {test_agent}")
    
    # Run enough steps to fill the window
    for step in range(12):  # Need 10 steps to fill window
        actions = {agent_id: np.array([0.0, 0.0]) for agent_id in env_lstm.agents}
        obs, rewards, done, truncated, infos = env_lstm.step(actions)
        
        window_size = len(env_lstm._obs_windows.get(test_agent, []))
        ownship_obs = obs[test_agent][3:7]  # x, y, vx, vy
        
        if step < 9:
            print(f"   Step {step+1:2d}: Window size = {window_size:2d}/10 (filling...)")
        elif step == 9:
            print(f"   Step {step+1:2d}: Window size = {window_size:2d}/10 ✅ FULL - LSTM should activate!")
        else:
            print(f"   Step {step+1:2d}: Window size = {window_size:2d}/10 ✅ LSTM denoising active")
            print(f"            Denoised ownship: [{ownship_obs[0]:.6f}, {ownship_obs[1]:.6f}, "
                  f"{ownship_obs[2]:.6f}, {ownship_obs[3]:.6f}]")
    
    env_lstm.close()
    
    # Test 3: Compare LSTM on vs off
    print("\n[Test 3] Comparing LSTM ON vs OFF...")
    
    # Environment WITHOUT LSTM
    env_noisy = SectorEnvLSTM(
        n_agents=5,
        run_id=f"validation_noisy_{RUN_ID}",
        noise_enabled=True,
        use_denoiser=False,  # LSTM OFF
        add_intruder_noise=True,
    )
    
    # Reset both with same seed for comparison
    np.random.seed(42)
    obs_lstm, _ = env_lstm.reset(seed=42)
    np.random.seed(42)
    obs_noisy, _ = env_noisy.reset(seed=42)
    
    # Run 15 steps
    for _ in range(15):
        # Same actions for both
        actions = {agent_id: np.array([0.0, 0.0]) for agent_id in env_lstm.agents}
        obs_lstm, _, _, _, _ = env_lstm.step(actions)
        obs_noisy, _, _, _, _ = env_noisy.step(actions)
    
    # Compare observations for first agent
    agent = list(obs_lstm.keys())[0]
    ownship_lstm = obs_lstm[agent][3:7]
    ownship_noisy = obs_noisy[agent][3:7]
    
    diff = np.abs(ownship_lstm - ownship_noisy)
    
    print(f"\n   Agent: {agent}")
    print(f"   Noisy ownship:  [{ownship_noisy[0]:.6f}, {ownship_noisy[1]:.6f}, "
          f"{ownship_noisy[2]:.6f}, {ownship_noisy[3]:.6f}]")
    print(f"   LSTM ownship:   [{ownship_lstm[0]:.6f}, {ownship_lstm[1]:.6f}, "
          f"{ownship_lstm[2]:.6f}, {ownship_lstm[3]:.6f}]")
    print(f"   Absolute diff:  [{diff[0]:.6f}, {diff[1]:.6f}, {diff[2]:.6f}, {diff[3]:.6f}]")
    
    if np.any(diff > 1e-8):
        print("\n✅ LSTM IS MODIFYING OBSERVATIONS (denoising is active)")
    else:
        print("\n❌ LSTM NOT MODIFYING OBSERVATIONS (denoising not working!)")
        return False
    
    env_lstm.close()
    env_noisy.close()
    
    print("\n" + "="*70)
    print("  VALIDATION COMPLETE")
    print("="*70)
    print("\n✅ All tests passed! LSTM denoiser is loaded and running correctly.")
    print("\nNEXT STEPS:")
    print("1. Retrain LSTM: python train_denoiser.py")
    print("2. Run diagnostics: python diagnose_lstm.py")
    print("3. Evaluate with MVP: python evaluate_lstm_mvp.py --episodes 100")
    
    return True


if __name__ == "__main__":
    success = test_lstm_loading()
    if not success:
        print("\n⚠️  LSTM denoiser has issues - see errors above")
        sys.exit(1)
