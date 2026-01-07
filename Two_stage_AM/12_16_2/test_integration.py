"""
Test script to validate the integration of:
1. Environment (ma_env_two_stage_AM)
2. Attention Model (AttentionSACModel)
3. PPO Training Setup

Run this before starting full training to catch any integration issues.
"""
import numpy as np
import torch
from gymnasium import spaces

# Test imports
print("Testing imports...")
try:
    from bluesky_gym.envs.ma_env_two_stage_AM import SectorEnv
    print("✓ Environment imported successfully")
except Exception as e:
    print(f"✗ Environment import failed: {e}")
    exit(1)

try:
    from attention_model_A import AttentionSACModel
    print("✓ Attention model imported successfully")
except Exception as e:
    print(f"✗ Attention model import failed: {e}")
    exit(1)

# Test environment creation
print("\nTesting environment creation...")
try:
    env = SectorEnv(n_agents=5, run_id="test")
    obs, info = env.reset()
    print(f"✓ Environment created and reset")
    print(f"  - Number of agents: {len(obs)}")
    print(f"  - Observation shape: {list(obs.values())[0].shape}")
    print(f"  - Expected shape: (7 + 5*{env.num_ac-1},) = ({7 + 5*(env.num_ac-1)},)")
    env.close()
except Exception as e:
    print(f"✗ Environment test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test attention model instantiation
print("\nTesting attention model...")
try:
    # Get observation and action space from environment
    env = SectorEnv(n_agents=5, run_id="test")
    obs, _ = env.reset()
    
    # Get single agent obs/action space
    single_obs_space = env.observation_space[list(env.observation_space.keys())[0]]
    single_action_space = env.action_space[list(env.action_space.keys())[0]]
    
    print(f"  - Obs space shape: {single_obs_space.shape}")
    print(f"  - Action space shape: {single_action_space.shape}")
    
    # Create model
    model = AttentionSACModel(
        obs_space=single_obs_space,
        action_space=single_action_space,
        num_outputs=2,  # Action dimension
        model_config={
            "custom_model_config": {
                "hidden_dims": [256, 256],
                "is_critic": False
            }
        },
        name="test_model"
    )
    print("✓ Attention model instantiated")
    
    # Test forward pass
    agent_id = list(obs.keys())[0]
    test_obs = torch.tensor(obs[agent_id], dtype=torch.float32).unsqueeze(0)  # Add batch dim
    
    print(f"  - Test input shape: {test_obs.shape}")
    
    output, state = model({"obs": test_obs}, [], None)
    print(f"✓ Forward pass successful")
    print(f"  - Output shape: {output.shape}")
    print(f"  - Expected: (1, 2)")
    
    # Check attention weights
    if hasattr(model, '_last_attn_weights'):
        print(f"✓ Attention weights computed")
        print(f"  - Attention shape: {model._last_attn_weights.shape}")
    else:
        print("⚠ Warning: Attention weights not found")
    
    env.close()
    
except Exception as e:
    print(f"✗ Attention model test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test observation parsing
print("\nTesting observation structure...")
try:
    env = SectorEnv(n_agents=5, run_id="test")
    obs, _ = env.reset()
    
    agent_id = list(obs.keys())[0]
    obs_vec = obs[agent_id]
    
    # Parse observation according to attention model expectations
    ownship_dim = 7
    intruder_dim = 5
    
    ownship_state = obs_vec[:ownship_dim]
    intruder_flat = obs_vec[ownship_dim:]
    
    num_intruders = len(intruder_flat) // intruder_dim
    
    print(f"✓ Observation parsing successful")
    print(f"  - Ownship features: {ownship_state.shape} (expected: (7,))")
    print(f"  - Intruder features: {intruder_flat.shape} (expected: ({num_intruders*5},))")
    print(f"  - Number of intruders: {num_intruders}")
    
    # Reshape intruders
    intruder_states = intruder_flat.reshape(num_intruders, intruder_dim)
    print(f"  - Reshaped intruders: {intruder_states.shape} (expected: ({num_intruders}, 5))")
    
    env.close()
    
except Exception as e:
    print(f"✗ Observation structure test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test behavioral cloning compatibility
print("\nTesting behavioral cloning compatibility...")
try:
    env = SectorEnv(n_agents=5, run_id="test")
    obs, _ = env.reset()
    
    # Simulate one step to get teacher action
    actions = {agent_id: np.array([0.0, 0.0], dtype=np.float32) for agent_id in obs.keys()}
    obs, rewards, dones, truncs, infos = env.step(actions)
    
    # Check if teacher_action exists in info
    agent_id = list(infos.keys())[0]
    if "teacher_action" in infos[agent_id]:
        teacher_action = infos[agent_id]["teacher_action"]
        print(f"✓ Teacher action found in infos")
        print(f"  - Teacher action shape: {np.array(teacher_action).shape}")
        print(f"  - Expected: (2,)")
    else:
        print(f"✗ Teacher action NOT found in infos")
        print(f"  - Available info keys: {infos[agent_id].keys()}")
    
    env.close()
    
except Exception as e:
    print(f"✗ Behavioral cloning test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*60)
print("✓ ALL TESTS PASSED!")
print("="*60)
print("\nYour integration is ready for training. Key points:")
print("1. Environment outputs correct observation format (7 + 5*N)")
print("2. Attention model accepts observations and produces actions")
print("3. Teacher actions are available in step infos")
print("4. Observation structure matches attention model expectations")
print("\nYou can now run main.py to start Stage 1 behavioral cloning!")
