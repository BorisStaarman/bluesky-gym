import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from gymnasium import spaces

class AttentionSACModel(TorchModelV2, nn.Module):
    """
    Implementation of Multi-Head Additive (Bahdanau) Attention Architecture.
    Uses 3 attention heads as described in the paper.
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # 1. Configuration Dimensions
        self.ownship_dim = 7
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
        
        # Multi-head attention configuration
        self.attn_dim = 128
        self.num_heads = 3
        self.head_dim = self.attn_dim // self.num_heads  # 128 / 3 = 42 (with 2 extra)
        
        # Adjust to make it divisible: use 42 per head, total = 126
        self.head_dim = 42
        self.total_attn_dim = self.head_dim * self.num_heads  # 126

        # 2. Pre-processing Layers
        self.ownship_fc = nn.Linear(self.ownship_dim, self.total_attn_dim)
        self.intruder_fc = nn.Linear(self.intruder_dim, self.total_attn_dim)
        
        # --- MULTI-HEAD ATTENTION LAYERS (ADDITIVE SPECIFIC) ---
        # Create separate projection matrices for each head
        self.W_q_heads = nn.ModuleList([
            nn.Linear(self.head_dim, self.head_dim, bias=False) 
            for _ in range(self.num_heads)
        ])
        self.W_k_heads = nn.ModuleList([
            nn.Linear(self.head_dim, self.head_dim, bias=False) 
            for _ in range(self.num_heads)
        ])
        self.W_v_heads = nn.ModuleList([
            nn.Linear(self.head_dim, self.head_dim, bias=False) 
            for _ in range(self.num_heads)
        ])
        
        # Scoring vector for each head
        self.v_att_heads = nn.ParameterList([
            nn.Parameter(torch.Tensor(self.head_dim, 1)) 
            for _ in range(self.num_heads)
        ])
        
        # Initialize all scoring vectors
        for v_att in self.v_att_heads:
            nn.init.xavier_uniform_(v_att)

        # Output projection to combine all heads
        self.attn_output_proj = nn.Linear(self.total_attn_dim, self.attn_dim)
        
        # Project ownship embedding to standard attn_dim (128) for concatenation
        self.ownship_output_proj = nn.Linear(self.total_attn_dim, self.attn_dim)

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
        own_embed = F.leaky_relu(self.ownship_fc(ownship_state), negative_slope=0.2)  # (Batch, 126)
        int_embed = F.leaky_relu(self.intruder_fc(intruder_states), negative_slope=0.2)  # (Batch, N, 126)
        
        # 3. Multi-Head Additive Attention Mechanism
        # ---------------------------------------------------------
        
        # Split embeddings into heads: (Batch, 126) -> 3 heads of (Batch, 42)
        batch_size = own_embed.shape[0]
        num_intruders = int_embed.shape[1]
        
        # Reshape for multi-head: (Batch, num_heads, head_dim)
        own_embed_heads = own_embed.view(batch_size, self.num_heads, self.head_dim)
        # Reshape intruders: (Batch, N, num_heads, head_dim) -> (Batch, num_heads, N, head_dim)
        int_embed_heads = int_embed.view(batch_size, num_intruders, self.num_heads, self.head_dim)
        int_embed_heads = int_embed_heads.permute(0, 2, 1, 3)  # (Batch, num_heads, N, head_dim)
        
        # Process each head independently
        context_heads = []
        attention_weights_all_heads = []
        
        # do for each head
        for h in range(self.num_heads):
            # Extract this head's embeddings
            own_h = own_embed_heads[:, h, :]  # (Batch, head_dim)
            int_h = int_embed_heads[:, h, :, :]  # (Batch, N, head_dim)
            
            # A. Project Query, Keys, Values for this head
            query_h = self.W_q_heads[h](own_h).unsqueeze(1)  # (Batch, 1, head_dim)
            keys_h = self.W_k_heads[h](int_h)  # (Batch, N, head_dim)
            values_h = self.W_v_heads[h](int_h)  # (Batch, N, head_dim)
            
            # B. Calculate Energy: tanh(Q + K)
            energy_h = torch.tanh(query_h + keys_h)  # (Batch, N, head_dim)
            
            # C. Calculate Scores: energy * v
            scores_h = torch.matmul(energy_h, self.v_att_heads[h])  # (Batch, N, 1)
            scores_h = scores_h.transpose(1, 2)  # (Batch, 1, N)
            
            # Masking padding (same for all heads)
            is_padding = intruder_states.abs().sum(dim=2) < 1e-6  # (Batch, N)
            scores_h = scores_h.masked_fill(is_padding.unsqueeze(1), float('-inf'))
            
            # D. Softmax to get attention weights
            alpha_h = F.softmax(scores_h, dim=-1)  # (Batch, 1, N)
            alpha_h = torch.nan_to_num(alpha_h, nan=0.0)
            
            attention_weights_all_heads.append(alpha_h)
            
            # E. Context Vector for this head
            context_h = torch.bmm(alpha_h, values_h).squeeze(1)  # (Batch, head_dim)
            context_heads.append(context_h)
        
        # Concatenate all heads: 3 heads × 42 dim = 126 total
        context_vector = torch.cat(context_heads, dim=1)  # (Batch, 126)
        
        # Store attention weights for visualization (shape: num_heads × Batch × 1 × N)
        # Average across heads for backward compatibility
        avg_attention = torch.stack(attention_weights_all_heads, dim=0).mean(dim=0)
        self._last_attn_weights = avg_attention.detach().cpu().numpy()
        # Also store per-head weights for detailed analysis
        self._last_attn_weights_per_head = [alpha.detach().cpu().numpy() for alpha in attention_weights_all_heads]
        
        # ---------------------------------------------------------
        
        # Output Projection + Tanh (projects from 126 -> 128)
        attention_vector = torch.tanh(self.attn_output_proj(context_vector))
        
        # Project ownship embedding from 126 -> 128 for consistent concatenation
        ownship_vector = torch.tanh(self.ownship_output_proj(own_embed))
        
        # 4. Concatenation & Network (now both are 128 dimensions)
        if self.is_critic:
            actions = inputs[:, intruder_end_idx:]
            if actions.shape[1] == 0:
                 actions = torch.zeros(inputs.shape[0], self.action_dim, device=inputs.device)
            x = torch.cat([ownship_vector, attention_vector, actions], dim=1)
        else:
            x = torch.cat([ownship_vector, attention_vector], dim=1)
        
        for layer in self.hidden_layers:
            x = F.leaky_relu(layer(x), negative_slope=0.2)
            
        out = self.final_layer(x)
        
        return out, state

    @override(TorchModelV2)
    def value_function(self):
        return torch.zeros(1)