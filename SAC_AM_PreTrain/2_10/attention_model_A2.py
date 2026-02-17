import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override

class AttentionSACModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.ownship_dim = 7  
        self.intruder_dim = 5 
        total_obs_dim = obs_space.shape[0]
        self.num_intruders = (total_obs_dim - self.ownship_dim) // self.intruder_dim
        
        custom_config = model_config.get("custom_model_config", {})
        hidden_layer_sizes = custom_config.get("hidden_dims", [256, 256])
        self.is_critic = custom_config.get("is_critic", False)
        self.action_dim = 2 

        # --- ATTENTION CONFIGURATION ---
        self.num_heads = 3
        self.head_dim = 5 
        self.total_attn_dim = self.head_dim * self.num_heads 

        # Onafhankelijke koppen voor Query, Key, Value
        self.W_q_heads = nn.ModuleList([nn.Linear(self.ownship_dim, self.head_dim) for _ in range(self.num_heads)])
        self.W_k_heads = nn.ModuleList([nn.Linear(self.intruder_dim, self.head_dim) for _ in range(self.num_heads)])
        self.W_v_heads = nn.ModuleList([nn.Linear(self.intruder_dim, self.head_dim) for _ in range(self.num_heads)])
        
        self.v_att_heads = nn.ParameterList([nn.Parameter(torch.Tensor(self.head_dim, 1)) for _ in range(self.num_heads)])
        
        # NIEUW: Schalingsfactor voor stabiele Softmax (root d_k)
        self.scale = np.sqrt(self.head_dim)

        # Learnable temperature (start op 3.0 zoals in Stage 1)
        self.temperature = nn.Parameter(torch.ones(1) * 3.0)

        # --- INITIALISATIE (CRUCIAAL) ---
        self._initialize_weights()

        # Input: Ownship (7) + Context (15)
        input_dim = self.ownship_dim + self.total_attn_dim
        if self.is_critic:
            input_dim += self.action_dim
        
        # Gebruik LayerNorm voor stabiliteit na Attention
        self.norm_layer = nn.LayerNorm(input_dim)

        self.hidden_layers = nn.ModuleList()
        current_dim = input_dim
        for h_dim in hidden_layer_sizes:
            self.hidden_layers.append(nn.Linear(current_dim, h_dim))
            current_dim = h_dim

        actual_output_dim = self.action_dim if not self.is_critic else num_outputs
        self.final_layer = nn.Linear(current_dim, actual_output_dim)

        if not self.is_critic:
            self.log_std = nn.Parameter(torch.zeros(self.action_dim))
            # Separate value branch voor PPO/SAC stabiliteit
            self.value_branch = nn.Sequential(
                nn.Linear(input_dim, hidden_layer_sizes[0]),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_layer_sizes[0], 1)
            )

    def _initialize_weights(self):
        """Voorkomt exploderende gewichten bij de start."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for v in self.v_att_heads:
            nn.init.xavier_uniform_(v)

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        inputs = input_dict["obs"]
        ownship_state = inputs[:, :self.ownship_dim]
        intruder_flat = inputs[:, self.ownship_dim : self.ownship_dim + (self.num_intruders * self.intruder_dim)]
        intruder_states = intruder_flat.view(-1, self.num_intruders, self.intruder_dim)

        context_heads = []
        attention_weights_all_heads = []

        for h in range(self.num_heads):
            # Projecties
            query_h = self.W_q_heads[h](ownship_state).unsqueeze(1) 
            keys_h = self.W_k_heads[h](intruder_states)
            values_h = self.W_v_heads[h](intruder_states)
            
            # Additive Attention Energy: tanh(Q + K)
            energy_h = torch.tanh(query_h + keys_h) 
            
            # Scores berekenen en SCHALEN voor stabiliteit
            scores_h = torch.matmul(energy_h, self.v_att_heads[h]).transpose(1, 2)
            # Schaling voorkomt verzadiging van Softmax
            scores_h = (scores_h / self.scale) * torch.abs(self.temperature)
            
            # Masking voor lege intruder slots
            is_padding = (intruder_states.abs().sum(dim=2) < 1e-6)
            scores_h = scores_h.masked_fill(is_padding.unsqueeze(1), float('-inf'))
            
            alpha_h = F.softmax(scores_h, dim=-1)
            alpha_h = torch.nan_to_num(alpha_h, nan=0.0)
            attention_weights_all_heads.append(alpha_h)
            
            context_h = torch.bmm(alpha_h, values_h).squeeze(1)
            context_heads.append(context_h)

        context_vector = torch.cat(context_heads, dim=1)
        
        # Integratie
        if self.is_critic:
            actions = inputs[:, -self.action_dim:]
            x = torch.cat([ownship_state, context_vector, actions], dim=1)
        else:
            x = torch.cat([ownship_state, context_vector], dim=1)
            self._features = x # Opslaan voor value_function

        # Layer Normalization toepassen voor stabiele gradiënten
        x = self.norm_layer(x)

        for layer in self.hidden_layers:
            x = F.leaky_relu(layer(x), negative_slope=0.2)
            
        out = self.final_layer(x)
        
        if not self.is_critic:
            batch_size = out.shape[0]
            log_std_expanded = self.log_std.unsqueeze(0).expand(batch_size, -1)
            out = torch.cat([out, log_std_expanded], dim=1)
        
        self._last_attn_weights = torch.stack(attention_weights_all_heads, dim=0).mean(dim=0).detach().cpu().numpy()
        return out, state

    @override(TorchModelV2)
    def value_function(self):
        return self.value_branch(self._features).squeeze(-1) if hasattr(self, '_features') else torch.zeros(1)

    @override(TorchModelV2)
    def metrics(self):
        # (Zelfde metrics als voorheen voor monitoring)
        stats = {'attention_temperature': float(self.temperature.item())}
        if hasattr(self, '_last_attn_weights'):
            stats['attention_sharpness'] = float(np.mean(np.max(self._last_attn_weights, axis=-1)))
        return stats