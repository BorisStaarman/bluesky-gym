import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from gymnasium import spaces

class AttentionSACModel(TorchModelV2, nn.Module):
    """
    Corrected Implementation of Multi-Head Additive Attention (Groot et al. 2025).
    - Uses 3 independent heads.
    - Projects directly from raw inputs (7->5 and 5->5).
    - Uses 'Relative' Intruder States (5 features).
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # 1. Configuration Dimensions (Per Paper Table 2)
        self.ownship_dim = 7  # [cos_drift, sin_drift, speed, x, y, vx, vy]
        self.intruder_dim = 5 # [rel_x, rel_y, rel_vx, rel_vy, dist] (MUST MATCH _get_observation)
        
        # Calculate N agents based on observation space size
        total_obs_dim = obs_space.shape[0]
        self.num_intruders = (total_obs_dim - self.ownship_dim) // self.intruder_dim
        self.expected_intruder_size = self.num_intruders * self.intruder_dim
        
        # --- Read Config ---
        custom_config = model_config.get("custom_model_config", {})
        hidden_layer_sizes = custom_config.get("hidden_dims", [256, 256])
        self.is_critic = custom_config.get("is_critic", False)
        
        # Asymmetric actor-critic tracking
        self._asymmetric_ac_enabled = False
        self._clean_obs_count = 0
        
        # Debug: Print action space info
        print(f"[AttentionModel __init__] action_space: {action_space}")
        print(f"[AttentionModel __init__] action_space type: {type(action_space)}")
        if isinstance(action_space, spaces.Box):
            print(f"[AttentionModel __init__] action_space.shape: {action_space.shape}")
        
        if isinstance(action_space, spaces.Box):
            self.action_dim = int(np.product(action_space.shape))
        else:
            self.action_dim = 2 
        
        # --- ATTENTION CONFIGURATION (STRICTLY PER PAPER) ---
        # "Each head is the same size as Y" -> head_dim = 5
        self.num_heads = 3
        self.head_dim = 5 
        self.total_attn_dim = self.head_dim * self.num_heads  # 3 * 5 = 15 features

        # 2. Multi-Head Additive Attention Layers
        # We create independent linear layers for each head to allow them to learn distinct features.
        
        # W_q: Projects Ownship (7) -> Head Dim (5)
        self.W_q_heads = nn.ModuleList([
            nn.Linear(self.ownship_dim, self.head_dim, bias=True) 
            for _ in range(self.num_heads)
        ])
        
        # W_k: Projects Intruder (5) -> Head Dim (5)
        self.W_k_heads = nn.ModuleList([
            nn.Linear(self.intruder_dim, self.head_dim, bias=True) 
            for _ in range(self.num_heads)
        ])
        
        # W_v: Projects Intruder (5) -> Head Dim (5)
        self.W_v_heads = nn.ModuleList([
            nn.Linear(self.intruder_dim, self.head_dim, bias=True) 
            for _ in range(self.num_heads)
        ])
        
        # Scoring vector 'v' for each head (projects tanh output to scalar)
        self.v_att_heads = nn.ParameterList([
            nn.Parameter(torch.Tensor(self.head_dim, 1)) 
            for _ in range(self.num_heads)
        ])
        
        # Initialize scoring vectors (Xavier)
        for v_att in self.v_att_heads:
            nn.init.xavier_uniform_(v_att)
        
        # Learnable temperature parameter for attention sharpness
        # Initialize to 3.0 to match Stage 1 pretraining weights
        self.temperature = nn.Parameter(torch.ones(1) * 3.0)
        print(f"[AttentionModel] Initialized learnable temperature: {self.temperature.item():.3f}")

        # 3. Main Network (Actor / Critic)
        # Input: Ownship State (7) + Attention Context (15) = 22 features
        input_dim = self.ownship_dim + self.total_attn_dim
        
        if self.is_critic:
            input_dim += self.action_dim
        
        self.hidden_layers = nn.ModuleList()
        current_dim = input_dim
        for h_dim in hidden_layer_sizes:
            self.hidden_layers.append(nn.Linear(current_dim, h_dim))
            current_dim = h_dim

        # Final Output Layer
        # For PPO with free_log_std=True, we should only output action means (action_dim)
        # Ignore num_outputs if it's incorrectly set to action_dim * 2
        actual_output_dim = self.action_dim if not self.is_critic else num_outputs
        self.final_layer = nn.Linear(current_dim, actual_output_dim)
        self._last_output_dim = actual_output_dim
        
        # Add log_std parameter for PPO (needed for custom models)
        # This allows us to control exploration by setting log_std directly
        if not self.is_critic:
            self.log_std = nn.Parameter(torch.zeros(self.action_dim))
            print(f"[AttentionModel] Added log_std parameter (shape={self.log_std.shape})")
        
        # Add separate value function network (critical for PPO!)
        # PPO needs this to compute advantages and VF explained variance
        if not self.is_critic:
            self.value_branch = nn.Sequential(
                nn.Linear(input_dim, hidden_layer_sizes[0]),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1]),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_layer_sizes[1], 1)
            )
            print(f"[AttentionModel] Added value_branch network for critic")
        
        # Debug: Print model configuration
        print(f"[AttentionModel] Initialized with:")
        print(f"  - num_outputs (from RLlib): {num_outputs}")
        print(f"  - actual_output_dim (used): {actual_output_dim}")
        print(f"  - action_dim: {self.action_dim}")
        print(f"  - num_intruders: {self.num_intruders}")
        print(f"  - obs_space.shape: {obs_space.shape}")
        print(f"  - is_critic: {self.is_critic}")

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        # 1. Input Handling
        inputs = input_dict["obs"]
        
        # Check if clean observations are available (for asymmetric actor-critic)
        # Clean obs should be stored by callback in input_dict
        if "clean_obs" in input_dict and input_dict["clean_obs"] is not None:
            clean_inputs = input_dict["clean_obs"]
            # Store clean inputs for value function computation
            self._clean_inputs = clean_inputs
        else:
            self._clean_inputs = None
        
        # 2. Compute attention features from (noisy) observations
        # Split: Ownship is first 7, Intruders are the rest
        ownship_state = inputs[:, :self.ownship_dim]  # (Batch, 7)
        
        intruder_end_idx = self.ownship_dim + self.expected_intruder_size
        intruder_flat = inputs[:, self.ownship_dim : intruder_end_idx]
        
        # Reshape Intruders: (Batch, N, 5)
        intruder_states = intruder_flat.view(-1, self.num_intruders, self.intruder_dim)

        # 3. Multi-Head Additive Attention Loop
        context_heads = []
        attention_weights_all_heads = []

        # Iterate through 3 independent experts (Heads)
        for h in range(self.num_heads):
            
            # --- A. Linear Projections (Directly from Raw States) ---
            # Q: Ownship (Batch, 7) -> (Batch, 5) -> Unsqueeze for broadcast (Batch, 1, 5)
            query_h = self.W_q_heads[h](ownship_state).unsqueeze(1) 
            
            # K, V: Intruders (Batch, N, 5) -> (Batch, N, 5)
            keys_h = self.W_k_heads[h](intruder_states)
            values_h = self.W_v_heads[h](intruder_states)
            
            # --- B. Calculate Energy ---
            # Equation: tanh(Q + K + b)
            # Note: The +b is implicit in the bias=True of the Linear layers above
            energy_h = torch.tanh(query_h + keys_h) # (Batch, N, 5)
            
            # --- C. Calculate Scores ---
            # Project vector energy to scalar score using v^T
            scores_h = torch.matmul(energy_h, self.v_att_heads[h]) # (Batch, N, 1)
            scores_h = scores_h.transpose(1, 2) # (Batch, 1, N)
            # Apply learnable temperature with constraint to keep positive
            scores_h = scores_h * torch.abs(self.temperature)
            
            # --- Masking ---
            # If an intruder slot is all zeros (padding), set score to -inf
            is_padding = (intruder_states.abs().sum(dim=2) < 1e-6) # (Batch, N)
            scores_h = scores_h.masked_fill(is_padding.unsqueeze(1), float('-inf'))
            
            # --- D. Attention Weights (Softmax) ---
            alpha_h = F.softmax(scores_h, dim=-1) # (Batch, 1, N)
            alpha_h = torch.nan_to_num(alpha_h, nan=0.0)
            
            # Store for visualization/debugging
            attention_weights_all_heads.append(alpha_h)
            
            # --- E. Weighted Sum (Context) ---
            # Context = Sum(alpha * V)
            context_h = torch.bmm(alpha_h, values_h).squeeze(1) # (Batch, 5)
            context_heads.append(context_h)

        # 4. Concatenate Heads
        # Result: (Batch, 15) -> [Head1_feat, Head2_feat, Head3_feat]
        context_vector = torch.cat(context_heads, dim=1)
        
        # Store debug data
        avg_attention = torch.stack(attention_weights_all_heads, dim=0).mean(dim=0)
        self._last_attn_weights = avg_attention.detach().cpu().numpy()
        self._last_attn_weights_per_head = [a.detach().cpu().numpy() for a in attention_weights_all_heads]

        # 5. Integrate into Actor/Critic
        # Concatenate Ownship (7) + Context (15) directly
        if self.is_critic:
            actions = inputs[:, intruder_end_idx:]
            if actions.shape[1] == 0:
                # print(f"[AttentionModel] forward: inputs shape: {inputs.shape}")
                actions = torch.zeros(inputs.shape[0], self.action_dim, device=inputs.device)
            # Critic Input: [Ownship, Context, Action]
                # print(f"[AttentionModel] forward: ownship_state shape: {ownship_state.shape}")
            x = torch.cat([ownship_state, context_vector, actions], dim=1)
        else:
            # print(f"[AttentionModel] forward: intruder_flat shape: {intruder_flat.shape}")
            # Actor Input: [Ownship, Context]
            x = torch.cat([ownship_state, context_vector], dim=1)
            # Store this for value function computation (will use clean obs if available)
            self._features = x
            # print(f"[AttentionModel] forward: intruder_states shape: {intruder_states.shape}")
        
        # 6. Main Dense Network
        for layer in self.hidden_layers:
            x = F.leaky_relu(layer(x), negative_slope=0.2)
            
        out = self.final_layer(x)
        
        # For actor (not critic), append log_std to output
        # PPO expects [mean, log_std] format when free_log_std is used
        if not self.is_critic and hasattr(self, 'log_std'):
            # Expand log_std to match batch size
            batch_size = out.shape[0]
            log_std_expanded = self.log_std.unsqueeze(0).expand(batch_size, -1)
            # Concatenate: [mean (action_dim), log_std (action_dim)]
            out = torch.cat([out, log_std_expanded], dim=1)
        
        if out.dim() == 1:
            out = out.unsqueeze(0)

        # Safety check for PPO/RLlib internal logic
        self._last_value = torch.zeros(out.shape[0], device=out.device) 
        
        return out, state
    
    def _compute_features(self, inputs):
        """
        Helper method to compute features from observations.
        Used by both forward() (actor) and value_function() (critic).
        
        Args:
            inputs: Observation tensor (batch, obs_dim)
        
        Returns:
            Concatenated features: [ownship_state, attention_context] (batch, 7+15)
        """
        # Split: Ownship is first 7, Intruders are the rest
        ownship_state = inputs[:, :self.ownship_dim]  # (Batch, 7)
        
        intruder_end_idx = self.ownship_dim + self.expected_intruder_size
        intruder_flat = inputs[:, self.ownship_dim : intruder_end_idx]
        
        # Reshape Intruders: (Batch, N, 5)
        intruder_states = intruder_flat.view(-1, self.num_intruders, self.intruder_dim)

        # Multi-Head Additive Attention Loop
        context_heads = []

        # Iterate through 3 independent experts (Heads)
        for h in range(self.num_heads):
            # Linear Projections
            query_h = self.W_q_heads[h](ownship_state).unsqueeze(1) 
            keys_h = self.W_k_heads[h](intruder_states)
            values_h = self.W_v_heads[h](intruder_states)
            
            # Calculate Energy
            energy_h = torch.tanh(query_h + keys_h)
            
            # Calculate Scores
            scores_h = torch.matmul(energy_h, self.v_att_heads[h])
            scores_h = scores_h.transpose(1, 2)
            scores_h = scores_h * torch.abs(self.temperature)
            
            # Masking
            is_padding = (intruder_states.abs().sum(dim=2) < 1e-6)
            scores_h = scores_h.masked_fill(is_padding.unsqueeze(1), float('-inf'))
            
            # Attention Weights (Softmax)
            alpha_h = F.softmax(scores_h, dim=-1)
            alpha_h = torch.nan_to_num(alpha_h, nan=0.0)
            
            # Weighted Sum (Context)
            context_h = torch.bmm(alpha_h, values_h).squeeze(1)
            context_heads.append(context_h)

        # Concatenate Heads
        context_vector = torch.cat(context_heads, dim=1)
        
        # Concatenate Ownship + Context
        features = torch.cat([ownship_state, context_vector], dim=1)
        
        return features

    @override(TorchModelV2)
    def value_function(self):
        """
        Compute state value for PPO.
        This uses a separate value network to evaluate the current state.
        
        For asymmetric actor-critic: Uses clean observations if available,
        otherwise falls back to noisy observations from forward pass.
        """
        if not hasattr(self, 'value_branch'):
            # Fallback for critic models
            if hasattr(self, '_last_value'):
                return self._last_value
            return torch.zeros(1)
        
        # ASYMMETRIC ACTOR-CRITIC: Use clean observations for value function if available
        if hasattr(self, '_clean_inputs') and self._clean_inputs is not None:
            # Recompute features with clean observations
            with torch.no_grad() if not self.training else torch.enable_grad():
                features = self._compute_features(self._clean_inputs)
            
            # Log when asymmetric AC is active (only first time to avoid spam)
            if not self._asymmetric_ac_enabled:
                self._asymmetric_ac_enabled = True
                print("[AttentionModel] 🔄 Asymmetric Actor-Critic ENABLED: Critic using clean observations")
            
            self._clean_obs_count += 1
        else:
            # Use stored features from last forward pass (noisy observations)
            if not hasattr(self, '_features'):
                # If forward hasn't been called yet, return zeros
                return torch.zeros(1)
            features = self._features
        
        # Compute value using the value branch
        value = self.value_branch(features).squeeze(-1)
        return value
    
    @override(TorchModelV2)
    def metrics(self):
        """
        Return custom metrics for TensorBoard logging to debug attention mechanism health.
        
        Metrics:
        - Attention Temperature: Current value of learnable temperature parameter
        - Attention Sharpness: Mean of max attention weight across batch (how focused is attention?)
        - Layer Health: Weight norms and gradient norms for first attention head (W_q, W_k, W_v)
        """
        stats = {}
        
        # 0. Learnable Temperature (for monitoring attention focus strategy)
        if hasattr(self, 'temperature'):
            stats['attention_temperature'] = float(self.temperature.item())
        
        # 0.5. Asymmetric Actor-Critic metrics
        if hasattr(self, '_asymmetric_ac_enabled'):
            stats['asymmetric_ac_enabled'] = 1.0 if self._asymmetric_ac_enabled else 0.0
        if hasattr(self, '_clean_obs_count'):
            stats['clean_obs_usage_count'] = float(self._clean_obs_count)
        
        # 1. Attention Sharpness
        if hasattr(self, '_last_attn_weights') and self._last_attn_weights is not None:
            try:
                # _last_attn_weights shape: (Batch, 1, N_intruders)
                # Get max attention weight per batch sample, then average
                max_attn_per_sample = np.max(self._last_attn_weights, axis=-1)  # (Batch, 1)
                mean_max_attn = float(np.mean(max_attn_per_sample))
                stats['attention_sharpness'] = mean_max_attn
            except Exception:
                pass
        
        # 2. Layer Health - First Attention Head (Index 0)
        # W_q, W_k, W_v weight norms and gradient norms
        
        # W_q (query) - first head
        if len(self.W_q_heads) > 0:
            wq_layer = self.W_q_heads[0]
            if wq_layer.weight is not None:
                stats['wq_weight_norm'] = float(torch.norm(wq_layer.weight).item())
                if wq_layer.weight.grad is not None:
                    stats['wq_grad_norm'] = float(torch.norm(wq_layer.weight.grad).item())
        
        # W_k (key) - first head
        if len(self.W_k_heads) > 0:
            wk_layer = self.W_k_heads[0]
            if wk_layer.weight is not None:
                stats['wk_weight_norm'] = float(torch.norm(wk_layer.weight).item())
                if wk_layer.weight.grad is not None:
                    stats['wk_grad_norm'] = float(torch.norm(wk_layer.weight.grad).item())
        
        # W_v (value) - first head
        if len(self.W_v_heads) > 0:
            wv_layer = self.W_v_heads[0]
            if wv_layer.weight is not None:
                stats['wv_weight_norm'] = float(torch.norm(wv_layer.weight).item())
                if wv_layer.weight.grad is not None:
                    stats['wv_grad_norm'] = float(torch.norm(wv_layer.weight.grad).item())
        
        return stats