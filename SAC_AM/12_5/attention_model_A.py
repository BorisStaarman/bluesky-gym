import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from gymnasium import spaces

class AdditiveAttentionSACModel(TorchModelV2, nn.Module):
    """
    Implementation of the Additive (Bahdanau) Attention Architecture.
    Matches the 'ADD' method described in the paper.
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
        
        if isinstance(action_space, spaces.Box):
            self.action_dim = int(np.product(action_space.shape))
        else:
            self.action_dim = 2 
        
        # Attention internal size
        self.attn_dim = 128 

        # 2. Pre-processing Layers
        self.ownship_fc = nn.Linear(self.ownship_dim, self.attn_dim)
        self.intruder_fc = nn.Linear(self.intruder_dim, self.attn_dim)
        
        # --- ATTENTION LAYERS (ADDITIVE SPECIFIC) ---
        # We need Wq and Wk to project inputs before adding them
        self.W_q = nn.Linear(self.attn_dim, self.attn_dim, bias=False)
        self.W_k = nn.Linear(self.attn_dim, self.attn_dim, bias=False)
        self.W_v = nn.Linear(self.attn_dim, self.attn_dim, bias=False)
        
        # The "v" vector (Parameter) that projects the combined tanh activation to a scalar score
        # This replaces the dot product.
        self.v_att = nn.Parameter(torch.Tensor(self.attn_dim, 1))
        # Initialize v_att randomly (Xavier uniform is standard)
        nn.init.xavier_uniform_(self.v_att)

        # Output projection
        self.attn_output_proj = nn.Linear(self.attn_dim, self.attn_dim)

        # 3. Dynamic Hidden Layers
        input_dim = self.attn_dim + self.attn_dim
        
        if self.is_critic:
            input_dim += self.action_dim
        
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
        ownship_state = inputs[:, :self.ownship_dim]
        intruder_end_idx = self.ownship_dim + self.expected_intruder_size
        intruder_flat = inputs[:, self.ownship_dim : intruder_end_idx]
        intruder_states = intruder_flat.view(-1, self.num_intruders, self.intruder_dim)

        # 2. Embed
        own_embed = F.leaky_relu(self.ownship_fc(ownship_state), negative_slope=0.2)
        int_embed = F.leaky_relu(self.intruder_fc(intruder_states), negative_slope=0.2)
        
        # 3. Additive Attention Mechanism
        # ---------------------------------------------------------
        
        # A. Project Query, Keys, Values
        # Query: (Batch, 1, 128)
        query = self.W_q(own_embed).unsqueeze(1)    
        # Keys:  (Batch, N, 128) - Note: We keep dimensions aligned for addition
        keys  = self.W_k(int_embed)                 
        # Values: (Batch, N, 128)
        values = self.W_v(int_embed)                

        # B. Calculate Energy (The "Neural Net" part)
        # Equation: tanh(Q + K)
        # We broadcast Query (1, 128) across Keys (N, 128)
        # Result: (Batch, N, 128)
        energy = torch.tanh(query + keys) 
        
        # C. Calculate Scores (Project to Scalar)
        # Equation: energy * v
        # (Batch, N, 128) x (128, 1) -> (Batch, N, 1)
        scores = torch.matmul(energy, self.v_att)
        
        # Transpose to (Batch, 1, N) to match Softmax format
        scores = scores.transpose(1, 2)
        
        # ---------------------------------------------------------

        # Masking padding
        is_padding = intruder_states.abs().sum(dim=2) < 1e-6
        scores = scores.masked_fill(is_padding.unsqueeze(1), float('-inf'))
        
        # Softmax
        alpha = F.softmax(scores, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        
        self._last_attn_weights = alpha.detach().cpu().numpy()
        
        # Context Vector: (Batch, 1, N) x (Batch, N, 128) -> (Batch, 1, 128)
        context_vector = torch.bmm(alpha, values).squeeze(1)
        
        # Output Projection + Tanh (Eq 13 in paper)
        attention_vector = torch.tanh(self.attn_output_proj(context_vector))
        
        # 4. Concatenation & Network
        if self.is_critic:
            actions = inputs[:, intruder_end_idx:]
            if actions.shape[1] == 0:
                 actions = torch.zeros(inputs.shape[0], self.action_dim, device=inputs.device)
            x = torch.cat([own_embed, attention_vector, actions], dim=1)
        else:
            x = torch.cat([own_embed, attention_vector], dim=1)
        
        for layer in self.hidden_layers:
            x = F.leaky_relu(layer(x), negative_slope=0.2)
            
        out = self.final_layer(x)
        
        return out, state

    @override(TorchModelV2)
    def value_function(self):
        return torch.zeros(1)