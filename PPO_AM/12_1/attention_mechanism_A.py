import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override

class AttentionModel(TorchModelV2, nn.Module):
    """
    Implementation of Multi-Head Additive (Bahdanau) Attention Architecture for PPO.
    Uses 3 attention heads as described in the paper.
    Adapted from SAC version with PPO-specific actor-critic structure.
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # --- CONFIGURATION ---
        # Observation is flat: 7 ownship + 7*19 intruders = 140 dims
        original_space = obs_space.original_space if hasattr(obs_space, "original_space") else obs_space
        
        # For flat observation space
        self.ownship_dim = 7  # cos_drift, sin_drift, airspeed, dx, dy, vx, vy
        self.intruder_dim = 7  # same features per intruder
        self.num_intruders = 19  # NUM_AC_STATE from environment
        
        # --- Read Config ---
        custom_config = model_config.get("custom_model_config", {})
        hidden_layer_sizes = custom_config.get("hidden_dims", [256, 256])
        
        # Multi-head attention configuration
        self.attn_dim = 128
        self.num_heads = 3
        self.head_dim = 42  # 126 total / 3 heads
        self.total_attn_dim = self.head_dim * self.num_heads  # 126

        # --- LAYERS ---
        
        # 1. Pre-processing Layers
        self.ownship_fc = nn.Linear(self.ownship_dim, self.total_attn_dim)
        self.intruder_fc = nn.Linear(self.intruder_dim, self.total_attn_dim)
        
        # 2. Multi-Head Attention Layers (Additive/Bahdanau)
        # Create separate projection matrices for each head
        self.W_q_heads = nn.ModuleList([
            nn.Linear(self.head_dim, self.head_dim, bias=True) 
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

        # 3. Shared Backbone Layers
        input_dim = self.attn_dim + self.attn_dim  # ownship + attention context
        
        self.hidden_layers = nn.ModuleList()
        current_dim = input_dim
        for h_dim in hidden_layer_sizes:
            self.hidden_layers.append(nn.Linear(current_dim, h_dim))
            current_dim = h_dim

        # 4. PPO-specific: Actor and Critic Heads
        self.actor_head = nn.Linear(current_dim, num_outputs)
        self.critic_head = nn.Linear(current_dim, 1)

        self._value_out = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        # 1. Reshape flat observation into structured format
        # Observation is flat: [7 ownship + 7*19 intruders] = 140 dims
        flat_obs = input_dict["obs"]  # (Batch, 140)
        batch_size = flat_obs.shape[0]
        
        # Split into ownship (first 7) and intruders (remaining 133)
        ownship_state = flat_obs[:, :self.ownship_dim]  # (Batch, 7)
        intruder_flat = flat_obs[:, self.ownship_dim:]  # (Batch, 133)
        
        # Reshape intruders into (Batch, N, 7)
        intruder_states = intruder_flat.view(batch_size, self.num_intruders, self.intruder_dim)
        
        # Create mask: intruders are padding if all features are zero
        mask = (intruder_states.abs().sum(dim=2) != 0).float()  # (Batch, N)

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
        
        # Convert mask to boolean: 0 means padding (should be masked)
        is_padding = (mask == 0)  # (Batch, N)
        
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
            
            # D. Masking padding (use PPO's mask)
            scores_h = scores_h.masked_fill(is_padding.unsqueeze(1), float('-inf'))
            
            # E. Softmax to get attention weights
            alpha_h = F.softmax(scores_h, dim=-1)  # (Batch, 1, N)
            alpha_h = torch.nan_to_num(alpha_h, nan=0.0)
            
            attention_weights_all_heads.append(alpha_h)
            
            # F. Context Vector for this head
            context_h = torch.bmm(alpha_h, values_h).squeeze(1)  # (Batch, head_dim)
            context_heads.append(context_h)
        
        # Concatenate all heads: 3 heads × 42 dim = 126 total
        context_vector = torch.cat(context_heads, dim=1)  # (Batch, 126)
        
        # Store attention weights for visualization
        avg_attention = torch.stack(attention_weights_all_heads, dim=0).mean(dim=0)
        self._last_attn_weights = avg_attention.detach().cpu().numpy()
        self._last_attn_weights_per_head = [alpha.detach().cpu().numpy() for alpha in attention_weights_all_heads]
        
        # ---------------------------------------------------------
        
        # 4. Output Projection + Tanh (projects from 126 -> 128)
        attention_vector = torch.tanh(self.attn_output_proj(context_vector))
        
        # Project ownship embedding from 126 -> 128 for consistent concatenation
        ownship_vector = torch.tanh(self.ownship_output_proj(own_embed))
        
        # 5. Concatenation & Shared Layers (now both are 128 dimensions)
        x = torch.cat([ownship_vector, attention_vector], dim=1)
        
        for layer in self.hidden_layers:
            x = F.leaky_relu(layer(x), negative_slope=0.2)
        
        # 6. PPO-specific: Compute both actor logits and critic value
        self._features = x
        logits = self.actor_head(self._features)
        self._value_out = self.critic_head(self._features)

        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        """PPO-specific: Return the value estimate from the critic head."""
        return self._value_out.reshape(-1)