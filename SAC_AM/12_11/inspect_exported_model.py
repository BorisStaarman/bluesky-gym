"""
Script to inspect what's inside an exported SAC_AM model file.
This shows the exact structure and parameters that get saved.
"""
import torch
import sys

# Add the SAC_AM directory to path
sac_am_dir = r"C:\Users\boris\Documents\bsgym\bluesky-gym\SAC_AM\12_5"
if sac_am_dir not in sys.path:
    sys.path.insert(0, sac_am_dir)

def inspect_checkpoint(checkpoint_path):
    """Inspect what's inside a checkpoint before exporting"""
    print("="*80)
    print("INSPECTING CHECKPOINT (Ray RLlib format)")
    print("="*80)
    
    # This would require Ray to be initialized and the algorithm loaded
    # For now, we'll just show the structure
    print("\nA Ray checkpoint contains:")
    print("  1. Algorithm configuration (hyperparameters, settings)")
    print("  2. Policy states (neural network weights)")
    print("  3. Optimizer states (Adam/RMSprop momentum, etc.)")
    print("  4. Training statistics")
    print("\nWhen we call algo.get_policy('shared_policy').model.state_dict():")
    print("  -> We extract ONLY the neural network weights (policy.model)")
    print("  -> This is the part needed for inference/deployment")

def inspect_exported_model(model_path):
    """Inspect what's inside an exported .pt file"""
    print("\n" + "="*80)
    print("INSPECTING EXPORTED MODEL (.pt file)")
    print("="*80)
    
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        
        print(f"\n📦 Model file: {model_path}")
        print(f"   Total number of parameter tensors: {len(state_dict)}\n")
        
        print("📊 MODEL ARCHITECTURE:")
        print("-" * 80)
        
        total_params = 0
        
        # Group by component
        components = {
            "Attention Mechanism": [],
            "Hidden Layers (MLP)": [],
            "Output Layer": [],
        }
        
        for name, tensor in state_dict.items():
            num_params = tensor.numel()
            total_params += num_params
            
            info = f"   {name:50s} {str(tuple(tensor.shape)):20s} ({num_params:,} params)"
            
            if any(x in name for x in ['ownship_fc', 'intruder_fc', 'W_q', 'W_k', 'W_v', 'attn']):
                components["Attention Mechanism"].append(info)
            elif 'hidden_layers' in name:
                components["Hidden Layers (MLP)"].append(info)
            elif 'final_layer' in name:
                components["Output Layer"].append(info)
        
        # Print grouped
        for component, params in components.items():
            if params:
                print(f"\n🔹 {component}:")
                for p in params:
                    print(p)
        
        print("\n" + "="*80)
        print(f"TOTAL PARAMETERS: {total_params:,}")
        print("="*80)
        
        # Explain what each component does
        print("\n📚 WHAT EACH COMPONENT DOES:")
        print("-" * 80)
        print("""
🔹 Attention Mechanism:
   - ownship_fc: Encodes the agent's own state (drift, speed) into 128D embedding
   - intruder_fc: Encodes each neighbor's state (position, velocity) into 128D embedding
   - W_q, W_k, W_v: Query, Key, Value projection matrices (like Transformer attention)
   - attn_output_proj: Projects attention output to final attention vector
   
   → This calculates which neighbors are most important and focuses on them

🔹 Hidden Layers (MLP):
   - Takes concatenated [ownship embedding + attention vector] and processes it
   - Usually 2-3 layers of 256 neurons each
   - Extracts high-level features for decision making
   
   → This processes the attended information to make decisions

🔹 Output Layer:
   - For ACTOR: Outputs mean and log_std for action distribution (heading, speed changes)
   - For CRITIC: Outputs Q-value estimate
   
   → This produces the final action or value prediction
        """)
        
        print("\n💡 KEY DIFFERENCE FROM REGULAR SAC:")
        print("-" * 80)
        print("""
Regular SAC models just take the observation and pass it through MLP layers.

SAC_AM (Attention Model) does this:
1. Separates ownship state from neighbor states
2. Uses attention mechanism to FOCUS on important neighbors
3. Then passes [ownship + focused_attention] through MLP

This allows the agent to:
- Handle variable number of neighbors
- Ignore far-away aircraft
- Focus computational resources on relevant conflicts
        """)
        
    except FileNotFoundError:
        print(f"❌ File not found: {model_path}")
        print("   Run the export script first to create the .pt file")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

if __name__ == "__main__":
    # Show what's conceptually in the checkpoint
    inspect_checkpoint("models/sectorcr_ma_sac/best_iter_00011")
    
    # Try to inspect exported model if it exists
    exported_path = r"C:\Users\boris\BS_setup\bluesky-master\plugins\models_boris\SAC_AM_7.pt"
    inspect_exported_model(exported_path)
