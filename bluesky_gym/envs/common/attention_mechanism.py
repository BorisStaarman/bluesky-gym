import numpy as np
import gymnasium as gym
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

class AttentionModel(TorchModelV2, nn.Module):
    """
    Implementation of D2MAV-A (Deep Distributed Multi-Agent Variable - Attention)
    Paper: arXiv:2003.08353v2
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # --- CONFIGURATION ---
        original_space = obs_space.original_space if hasattr(obs_space, "original_space") else obs_space
        ownship_dim = original_space["ownship"].shape[0] 
        intruder_dim = original_space["intruders"].shape[1] 
        
        # Hyperparameters from Paper [cite: 395]
        hidden_dim = 128
        fc_dim = 256
        leaky_alpha = 0.2

        # --- LAYERS ---
        
        # 1. Encoders (Pre-Processing)
        self.ownship_encoder = nn.Sequential(
            nn.Linear(ownship_dim, hidden_dim),
            nn.LeakyReLU(leaky_alpha)
        )
        
        # Shared weights for all intruders
        self.intruder_encoder = nn.Sequential(
            nn.Linear(intruder_dim, hidden_dim),
            nn.LeakyReLU(leaky_alpha)
        )

        # 2. Attention Mechanism
        # W1: Transforms intruder embeddings for scoring
        self.attention_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # W2: Transforms context vector before concatenation
        self.attention_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # 3. Backbone (Shared Layers)
        # Input size is hidden_dim (Ownship) + hidden_dim (Attention Context)
        self.shared_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, fc_dim),
            nn.LeakyReLU(leaky_alpha),
            nn.Linear(fc_dim, fc_dim),
            nn.LeakyReLU(leaky_alpha)
        )

        # 4. Heads
        self.actor_head = nn.Linear(fc_dim, num_outputs)
        self.critic_head = nn.Linear(fc_dim, 1)

        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        # A. Unpack Dict Inputs
        ownship = input_dict["obs"]["ownship"]      # (Batch, Own_Feats)
        intruders = input_dict["obs"]["intruders"]  # (Batch, N, Int_Feats)
        mask = input_dict["obs"]["mask"]            # (Batch, N)

        # B. Encode
        # Add dimension to ownship for matrix math: (Batch, 1, 128)
        ownship_embed = self.ownship_encoder(ownship).unsqueeze(1)
        # Encode intruders: (Batch, N, 128)
        intruder_embed = self.intruder_encoder(intruders)

        # C. Attention Scores
        # Query (Ownship) dot Key (Intruders)
        key_layer = self.attention_layer(intruder_embed)
        scores = torch.matmul(ownship_embed, key_layer.transpose(-2, -1)) 
        scores = scores.squeeze(1) # (Batch, N)

        # D. Masking
        # Set score of padding to -1e9 so Softmax makes them 0
        scores = scores.masked_fill(mask == 0, -1e9)

        # E. Weights & Context
        attn_weights = torch.nn.functional.softmax(scores, dim=-1) # (Batch, N)
        # Weighted Sum: (Batch, 1, N) * (Batch, N, 128) -> (Batch, 1, 128)
        context_vector = torch.matmul(attn_weights.unsqueeze(1), intruder_embed)
        context_vector = context_vector.squeeze(1)

        # F. Post-Processing
        # Abstract understanding vector [cite: 43]
        attention_out = self.attention_output(context_vector)
        
        # Concatenate Ownship + Attention [cite: 316]
        # (Batch, 128) + (Batch, 128) -> (Batch, 256)
        combined_features = torch.cat([ownship_embed.squeeze(1), attention_out], dim=1)

        # G. Output Heads
        self._features = self.shared_layers(combined_features)
        logits = self.actor_head(self._features)
        self._value_out = self.critic_head(self._features)

        return logits, state

    def value_function(self):
        return self._value_out.reshape(-1)