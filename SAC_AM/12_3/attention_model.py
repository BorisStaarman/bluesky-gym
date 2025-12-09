import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from gymnasium import spaces

class AttentionSACModel(TorchModelV2, nn.Module):
    """
    Implementation of the D2MAV-A Attention Architecture for RLlib SAC.
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # 1. Configuration Dimensions
        self.ownship_dim = 3
        self.intruder_dim = 7
        
        # Calculate N agents based on observation space
        total_obs_dim = obs_space.shape[0]
        self.num_intruders = (total_obs_dim - self.ownship_dim) // self.intruder_dim
        self.expected_intruder_size = self.num_intruders * self.intruder_dim

        # --- Read Config ---
        custom_config = model_config.get("custom_model_config", {})
        hidden_layer_sizes = custom_config.get("hidden_dims", [256, 256])
        self.is_critic = custom_config.get("is_critic", False)
        
        # Get Action Dimension
        if isinstance(action_space, spaces.Box):
            self.action_dim = int(np.product(action_space.shape))
        else:
            self.action_dim = 2 
        
        # Attention internal size
        self.attn_dim = 128 

        # 2. Pre-processing & Attention Layers
        self.ownship_fc = nn.Linear(self.ownship_dim, self.attn_dim)
        self.intruder_fc = nn.Linear(self.intruder_dim, self.attn_dim)
        
        # Attention Weights
        self.attn_W = nn.Linear(self.attn_dim, self.attn_dim, bias=False)
        
        # [FIX 3]: Add Equation 13 projection (Context -> Attention Vector)
        self.attn_output_proj = nn.Linear(self.attn_dim, self.attn_dim)

        # 3. Dynamic Hidden Layers
        # [FIX 2]: Input dim uses self.attn_dim (Ownship Embedding), NOT self.ownship_dim
        # Base = Ownship Embedding (128) + Attention Vector (128)
        input_dim = self.attn_dim + self.attn_dim
        
        if self.is_critic:
            input_dim += self.action_dim
            print(f"[AttentionModel] Initialized as CRITIC (Input Dim: {input_dim})")
        else:
            print(f"[AttentionModel] Initialized as ACTOR (Input Dim: {input_dim})")
        
        self.hidden_layers = nn.ModuleList()
        current_dim = input_dim
        for h_dim in hidden_layer_sizes:
            self.hidden_layers.append(nn.Linear(current_dim, h_dim))
            current_dim = h_dim

        # 4. Final Output Layer
        self.final_layer = nn.Linear(current_dim, num_outputs if num_outputs else current_dim)
        self._last_output_dim = num_outputs if num_outputs else current_dim

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        # 1. Input Handling
        inputs = input_dict["obs"]
        
        # A. Extract Ownship
        ownship_state = inputs[:, :self.ownship_dim]
        
        # B. Extract Intruders
        intruder_end_idx = self.ownship_dim + self.expected_intruder_size
        intruder_flat = inputs[:, self.ownship_dim : intruder_end_idx]
        
        # The env sends neighbors in agent-first format: [A1F1, A1F2, ..., A1F7, A2F1, A2F2, ..., A2F7, ...]
        # Reshape to (Batch, Num_Agents, Features_per_Agent)
        intruder_states = intruder_flat.view(-1, self.num_intruders, self.intruder_dim)

        # 2. Encode / Pre-process
        # own_embed: (Batch, 128)
        own_embed = F.leaky_relu(self.ownship_fc(ownship_state), negative_slope=0.2)
        # int_embed: (Batch, N, 128)
        int_embed = F.leaky_relu(self.intruder_fc(intruder_states), negative_slope=0.2)
        
        # 3. Attention Mechanism (Eq 10-12)
        # Query: (Batch, 1, 128)
        query = self.attn_W(own_embed).unsqueeze(1)
        # Keys: (Batch, 128, N)
        keys = int_embed.transpose(1, 2)
        
        # Scores: (Batch, 1, N)
        scores = torch.bmm(query, keys)
        scores = scores / (self.attn_dim ** 0.5) # Stability Scaling

        # Masking padding (If all features of an intruder are 0, it is padding)
        is_padding = intruder_states.abs().sum(dim=2) < 1e-6
        scores = scores.masked_fill(is_padding.unsqueeze(1), float('-inf'))
        
        alpha = F.softmax(scores, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        
        self._last_attn_weights = alpha.detach().cpu().numpy()
        
        # Context Vector (Eq 12): (Batch, 128)
        context_vector = torch.bmm(alpha, int_embed).squeeze(1)
        
        # [FIX 3]: Apply Eq 13 (Linear + Tanh) to get final Attention Vector
        attention_vector = torch.tanh(self.attn_output_proj(context_vector))
        
        # 4. Concatenation Strategy
        if self.is_critic:
            actions = inputs[:, intruder_end_idx:]
            if actions.shape[1] == 0:
                 actions = torch.zeros(inputs.shape[0], self.action_dim, device=inputs.device)
            
            # [FIX 2]: Concatenate own_embed (not ownship_state) with attention_vector
            x = torch.cat([own_embed, attention_vector, actions], dim=1)
        else:
            # [FIX 2]: Concatenate own_embed (not ownship_state) with attention_vector
            x = torch.cat([own_embed, attention_vector], dim=1)
        
        # 5. Dynamic Hidden Layers
        for layer in self.hidden_layers:
            x = F.leaky_relu(layer(x), negative_slope=0.2)
            
        out = self.final_layer(x)
        
        return out, state

    @override(TorchModelV2)
    def value_function(self):
        return torch.zeros(1)