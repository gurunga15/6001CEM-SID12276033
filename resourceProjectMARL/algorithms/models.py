"""
Custom PyTorch models for reinforcement learning policies.

This module provides custom PyTorch models for PPO that can be used with RLlib.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType


class ResourceCNNModel(TorchModelV2, nn.Module):
    """
    Custom CNN model for processing resource collection observations.
    
    This model handles both the spatial (grid) and non-spatial (agent state, env features)
    components of the observation for PPO agents.
    """
    
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        """
        initialise the model.
        
        Args:
            obs_space: Observation space
            action_space: Action space
            num_outputs: Number of output units
            model_config: Model configuration
            name: Model name
        """
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        
        # Check if the observation space is a Dict
        if not isinstance(obs_space, gym.spaces.Dict):
            # Handle flattened observations
            self.flat_obs = True
            self.obs_size = int(np.prod(obs_space.shape))
            
            # Create a simple fully connected network
            fcnet_hiddens = model_config.get("fcnet_hiddens", [256, 256, 128])
            fcnet_activation = model_config.get("fcnet_activation", "relu")
            
            # Activation function mapping
            if fcnet_activation == "relu":
                activation_fn = nn.ReLU
            elif fcnet_activation == "tanh":
                activation_fn = nn.Tanh
            else:
                activation_fn = nn.ReLU  # Default to ReLU
            
            # Build fully connected layers
            layers = []
            prev_layer_size = self.obs_size
            
            for size in fcnet_hiddens:
                layers.append(nn.Linear(prev_layer_size, size))
                layers.append(activation_fn())
                prev_layer_size = size
            
            # Final output layer
            self.fc_layers = nn.Sequential(*layers)
            self.policy_head = nn.Linear(prev_layer_size, num_outputs)
            self.value_head = nn.Linear(prev_layer_size, 1)
            
        else:
            # Handle dictionary observations
            self.flat_obs = False
            
            # Check for required keys
            required_keys = ["grid", "agent_state"]  # env_features is optional
            for key in required_keys:
                if key not in obs_space.spaces:
                    raise ValueError(f"Observation space must contain '{key}' key")
            
            # Get the input shapes
            self.grid_shape = obs_space.spaces["grid"].shape
            self.agent_state_shape = obs_space.spaces["agent_state"].shape
            
            # Get env_features shape if available
            self.env_features_shape = (0,)
            if "env_features" in obs_space.spaces:
                self.env_features_shape = obs_space.spaces["env_features"].shape
            
            # CNN configuration from model_config or use defaults
            cnn_filters = model_config.get("conv_filters", [
                [32, [3, 3], 1],
                [64, [3, 3], 1],
            ])
            
            # Fully connected layer sizes from model config or use defaults
            fcnet_hiddens = model_config.get("post_fcnet_hiddens", [256, 128])
            fcnet_activation = model_config.get("fcnet_activation", "relu")
            
            # Activation function mapping
            if fcnet_activation == "relu":
                activation_fn = nn.ReLU
            elif fcnet_activation == "tanh":
                activation_fn = nn.Tanh
            else:
                activation_fn = nn.ReLU  # Default to ReLU
            
            # Build CNN for spatial observations
            cnn_layers = []
            in_channels = self.grid_shape[2]  # channels in the observation
            spatial_size = self.grid_shape[0]  # assuming square grid
            
            # Add CNN layers
            for out_channels, kernel, stride in cnn_filters:
                cnn_layers.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel,
                        stride=stride,
                        padding=kernel[0] // 2,  # Same padding
                    )
                )
                cnn_layers.append(activation_fn())
                in_channels = out_channels
                # Update spatial size after conv+stride
                if isinstance(stride, int):
                    spatial_size = spatial_size // stride
                else:
                    spatial_size = spatial_size // stride[0]
            
            self.cnn = nn.Sequential(*cnn_layers)
            
            # Calculate flattened size of CNN output
            cnn_out_size = in_channels * spatial_size * spatial_size
            
            # FC layers for non-spatial features
            non_spatial_size = int(np.prod(self.agent_state_shape))
            if "env_features" in obs_space.spaces:
                non_spatial_size += int(np.prod(self.env_features_shape))
            
            # Combine features and pass through FC layers
            combined_size = cnn_out_size + non_spatial_size
            fc_layers = []
            prev_layer_size = combined_size
            
            # Create fully connected layers
            for size in fcnet_hiddens:
                fc_layers.append(nn.Linear(prev_layer_size, size))
                fc_layers.append(activation_fn())
                prev_layer_size = size
            
            self.fc_layers = nn.Sequential(*fc_layers)
            
            # Policy and value heads
            self.policy_head = nn.Linear(prev_layer_size, num_outputs)
            self.value_head = nn.Linear(prev_layer_size, 1)
        
        # Variable to hold the features for value function
        self._features = None
    
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        """
        Forward pass through the model.
        
        Args:
            input_dict: Input dictionary containing 'obs' tensor
            state: RNN hidden state (not used)
            seq_lens: RNN sequence lengths (not used)
            
        Returns:
            model_out: Output tensor
            state: New RNN state (unchanged)
        """
        # Extract the observations
        obs = input_dict["obs"]
        
        if self.flat_obs:
            # Handle flattened observations
            if isinstance(obs, dict):
                # If somehow we get a dict with flat_obs=True, concatenate all values
                features = []
                for k, v in obs.items():
                    features.append(v.reshape(v.shape[0], -1))
                combined_features = torch.cat(features, dim=1)
            else:
                # Direct flat observation
                combined_features = obs
            
            # Pass through fully connected layers
            features = self.fc_layers(combined_features)
            self._features = features
            
            # Policy head
            logits = self.policy_head(features)
            
        else:
            # Process spatial (grid) observations
            # Reshape to [batch, channels, height, width] for PyTorch CNN
            grid = obs["grid"].to(torch.float32).permute(0, 3, 1, 2)
            grid_features = self.cnn(grid)
            grid_features = grid_features.reshape(grid_features.shape[0], -1)  # Flatten
            
            # Process non-spatial features
            features_list = [grid_features]
            
            # Add agent state
            agent_state = obs["agent_state"].to(torch.float32)
            features_list.append(agent_state)
            
            # Add environmental features if available
            if "env_features" in obs and self.env_features_shape[0] > 0:
                env_features = obs["env_features"].to(torch.float32)
                features_list.append(env_features)
            
            # Combine all features
            combined_features = torch.cat(features_list, dim=1)
            
            # Pass through fully connected layers
            features = self.fc_layers(combined_features)
            self._features = features
            
            # Policy head
            logits = self.policy_head(features)
        
        return logits, state
    
    def value_function(self) -> TensorType:
        """
        Return the value function estimate for the most recent forward pass.
        
        Returns:
            Value function estimate
        """
        assert self._features is not None, "Must call forward() first"
        return self.value_head(self._features).squeeze(1) 