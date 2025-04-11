"""
PPO policy implementation for multi-agent resource collection.
"""

from typing import Dict, List, Any, Union, Optional, Tuple
import numpy as np
import gym
import gymnasium

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# Import core Ray modules
import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env  # Add import for environment registration

# Import Ray 2.x modules
from ray.rllib.algorithms.ppo import PPOConfig, PPO

# Import environment
try:
    from env.resourceEnv import ResourceCollectionEnv
except ImportError:
    try:
        from resourceProjectMARL.env.resourceEnv import ResourceCollectionEnv
    except ImportError:
        print("Warning: Could not import ResourceCollectionEnv. Environment must be registered separately.")
        ResourceCollectionEnv = None


class ResourceCollectionNetwork(TorchModelV2, nn.Module):
    """
    Custom neural network for resource collection agents with PPO.
    
    Features a shared backbone followed by separate policy and value heads.
    """
    
    def __init__(self, obsSpace, actionSpace, numOutputs, modelConfig, name):
        """
        Initialise the network.
        
        Args:
            obsSpace: Observation space
            actionSpace: Action space
            numOutputs: Number of outputs (number of actions)
            modelConfig: Model configuration
            name: Name of the model
        """
        TorchModelV2.__init__(self, obsSpace, actionSpace, numOutputs, modelConfig, name)
        nn.Module.__init__(self)
        
        # Process observation space to determine input size
        self.inputDim = int(np.product(obsSpace.shape))
        
        # Define base network
        self.baseNetwork = nn.Sequential(
            nn.Linear(self.inputDim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Policy head
        self.policyHead = nn.Linear(128, numOutputs)
        
        # Value head
        self.valueHead = nn.Linear(128, 1)
        
        # Current value output
        self._curValue = None
    
    def forward(self, inputDict, states, seqLens):
        """
        Forward pass of the network.
        
        Args:
            inputDict: Input dictionary containing observation tensor
            states: RNN hidden states (if any)
            seqLens: Sequence lengths for RNN
            
        Returns:
            Policy logits and updated RNN states
        """
        # Extract observation tensor
        obs = inputDict["obs_flat"].float()
        
        # Run through base network
        features = self.baseNetwork(obs)
        
        # Get action logits from policy head
        logits = self.policyHead(features)
        
        # Get value estimate from value head
        self._curValue = self.valueHead(features).squeeze(1)
        
        return logits, states
    
    def value_function(self):
        """Return the current value function estimate for the last input."""
        assert self._curValue is not None, "Must call forward first"
        return self._curValue


class CentralisedCriticNetwork(TorchModelV2, nn.Module):
    """
    Centralised critic network for MAPPO.
    
    Takes global state (concatenated observations) and outputs a value estimate.
    Handles partially observable environments by aggregating agent observations.
    """
    
    def __init__(self, obsSpace, actionSpace, numOutputs, modelConfig, name):
        """
        Initialise the centralised critic network.
        
        Args:
            obsSpace: Observation space (concatenated for all agents)
            actionSpace: Action space
            numOutputs: Number of outputs (typically 1 for value function)
            modelConfig: Model configuration
            name: Name of the model
        """
        TorchModelV2.__init__(self, obsSpace, actionSpace, numOutputs, modelConfig, name)
        nn.Module.__init__(self)
        
        # Process observation space to determine input size
        self.inputDim = int(np.product(obsSpace.shape))
        
        # Get custom model configuration
        customConfig = modelConfig.get("custom_model_config", {})
        
        # Number of agents for observation aggregation
        self.numAgents = customConfig.get("num_agents", 4)
        
        # Whether to use attention mechanism for aggregating observations
        self.useAttention = customConfig.get("use_attention", False)
        
        # For partially observable environments, create observation aggregation
        if self.useAttention:
            # Attention-based aggregation
            self.keyDim = 64
            self.valueDim = 64
            
            # Query, Key, Value projections
            self.queryProj = nn.Linear(self.inputDim, self.keyDim)
            self.keyProj = nn.Linear(self.inputDim, self.keyDim)
            self.valueProj = nn.Linear(self.inputDim, self.valueDim)
            
            # Output projection
            self.outputProj = nn.Linear(self.valueDim, 128)
            
            # Final critic layers
            self.criticLayers = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
        else:
            # Simple concatenation and MLP
            # Define critic network (deeper for centralised state representation)
            self.criticNetwork = nn.Sequential(
                nn.Linear(self.inputDim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
        
        # Current value output
        self._curValue = None
    
    def forward(self, inputDict, states, seqLens):
        """
        Forward pass of the centralised critic.
        
        Args:
            inputDict: Input dictionary containing concatenated observations
            states: RNN hidden states (if any)
            seqLens: Sequence lengths for RNN
            
        Returns:
            Dummy logits (not used) and updated RNN states
        """
        # Extract observation tensor
        obs = inputDict["obs_flat"].float()
        
        if self.useAttention:
            # Reshape observation tensor to [batch_size, num_agents, obs_dim]
            # Assuming the observations are concatenated in the order [agent1, agent2, ...]
            batchSize = obs.shape[0]
            agentObsDim = self.inputDim // self.numAgents
            
            # Reshape to [batch, num_agents, obs_dim]
            obs = obs.view(batchSize, self.numAgents, agentObsDim)
            
            # Compute attention
            queries = self.queryProj(obs)  # [batch, num_agents, key_dim]
            keys = self.keyProj(obs)       # [batch, num_agents, key_dim]
            values = self.valueProj(obs)   # [batch, num_agents, value_dim]
            
            # Compute attention scores
            scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.keyDim ** 0.5)
            attention = F.softmax(scores, dim=-1)  # [batch, num_agents, num_agents]
            
            # Apply attention to values
            context = torch.matmul(attention, values)  # [batch, num_agents, value_dim]
            
            # Average over agents
            context = context.mean(dim=1)  # [batch, value_dim]
            
            # Project to critic input
            critic_input = self.outputProj(context)
            
            # Compute value
            self._curValue = self.criticLayers(critic_input).squeeze(1)
        else:
            # Standard centralised critic (concatenated observations)
            self._curValue = self.criticNetwork(obs).squeeze(1)
        
        # Return dummy logits (not used for critic)
        return torch.zeros(obs.shape[0], 1, device=obs.device), states
    
    def value_function(self):
        """Return the current value function estimate for the last input."""
        assert self._curValue is not None, "Must call forward first"
        return self._curValue


class PPOPolicy:
    """
    PPO policy implementation for resource collection.
    
    Supports standard PPO and centralised critic MAPPO variants.
    """
    
    def __init__(
        self, 
        envName: str = "ResourceCollectionEnv",
        configOverrides: Dict[str, Any] = None,
        useCentralisedCritic: bool = True,
        sharedPolicy: bool = False,
        rewardType: str = "hybrid",
        hybridRewardMix: float = 0.5,  # 0.0 = fully global, 1.0 = fully individual
        useGpu: bool = False
    ):
        """
        Initialise the PPO policy.
        
        Args:
            envName: Name of the environment
            configOverrides: Override default PPO configuration
            useCentralisedCritic: Whether to use a centralised critic (MAPPO)
            sharedPolicy: Whether to share policy weights across agents
            rewardType: Type of reward structure ("individual", "global", or "hybrid")
            hybridRewardMix: Weight for hybrid rewards (0.0 = fully global, 1.0 = fully individual)
            useGpu: Whether to use GPU for training
        """
        # Register custom torch models
        ModelCatalog.register_custom_model("ResourceCollectionNetwork", ResourceCollectionNetwork)
        ModelCatalog.register_custom_model("CentralisedCriticNetwork", CentralisedCriticNetwork)
        
        # Validate reward parameters
        assert rewardType in ["individual", "global", "hybrid"], "rewardType must be 'individual', 'global', or 'hybrid'"
        assert 0.0 <= hybridRewardMix <= 1.0, "hybridRewardMix must be between 0.0 and 1.0"
        
        # Store reward configuration
        self.rewardType = rewardType
        self.hybridRewardMix = hybridRewardMix
        
        # Create configuration using Ray 2.x PPOConfig
        try:
            # initialise with PPOConfig
            self.config = PPOConfig()
            
            # Disable new API stack to maintain compatibility with ModelV2
            self.config = self.config.api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False
            )
            
            # Environment
            self.config = self.config.environment(env=envName)
            
            # Framework
            self.config = self.config.framework("torch")
            
            # Set training parameters directly on config
            self.config.model = {
                "custom_model": "ResourceCollectionNetwork",
                "vf_share_layers": False,
            }
            self.config.gamma = 0.99
            self.config.lambda_ = 0.95
            self.config.kl_coeff = 0.2
            self.config.clip_param = 0.2
            self.config.vf_clip_param = 10.0
            self.config.entropy_coeff = 0.01
            self.config.train_batch_size = 4000
            self.config.sgd_minibatch_size = 128
            self.config.num_sgd_iter = 10
            self.config.lr = 3e-4
            self.config.vf_loss_coeff = 0.5
            
            # Resources - set directly on config object
            self.config.num_gpus = 1 if useGpu else 0
            self.config.num_workers = 8
            
            # Evaluation
            self.config = self.config.evaluation(
                evaluation_duration=10,
                evaluation_duration_unit="episodes",
                evaluation_interval=5,
                evaluation_num_workers=1,  
                evaluation_config={
                    "explore": False
                }
            )
            
            # Convert to dict for easier manipulation with overrides
            self.config = self.config.to_dict()
            
        except Exception as e:
            print(f"Error initializing with PPOConfig: {e}")
            # Fallback to manual configuration
            self.config = {
                "env": envName,
                "framework": "torch",
                # Disable new API stack in fallback config too
                "enable_rl_module_and_learner": False,
                "enable_env_runner_and_connector_v2": False,
                "model": {
                    "custom_model": "ResourceCollectionNetwork",
                    "vf_share_layers": False,
                },
                "gamma": 0.99,
                "lambda_": 0.95,
                "kl_coeff": 0.2,
                "clip_param": 0.2,
                "vf_clip_param": 10.0,
                "entropy_coeff": 0.01,
                "train_batch_size": 4000,
                "sgd_minibatch_size": 128,
                "num_sgd_iter": 10,
                "lr": 3e-4,
                "vf_loss_coeff": 0.5,
                "num_workers": 8,
                "evaluation_duration": 10,
                "evaluation_duration_unit": "episodes",
                "evaluation_interval": 5,
                "evaluation_num_workers": 1,
                "evaluation_config": {
                    "explore": False
                }
            }
        
        # MAPPO configuration
        if useCentralisedCritic:
            self._configureMAPPO(sharedPolicy, useAttention=False)  # Default to no attention
        
        # Apply any overrides
        if configOverrides:
            self.config.update(configOverrides)
    
    def _configureMAPPO(self, sharedPolicy: bool, useAttention: bool = False):
        """
        Configure for Multi-Agent PPO with centralised critic.
        
        Args:
            sharedPolicy: Whether to share policy weights across agents
            useAttention: Whether to use attention mechanism for observation aggregation
        """
        # Define policy mapping function
        if sharedPolicy:
            def policyMappingFn(*args, **kwargs):
                return "shared_policy"
            
            policies = {
                "shared_policy": (None, None, None, {})
            }
        else:
            def policyMappingFn(agentId, *args, **kwargs):
                return agentId
            
            # Define the policies based on expected agent IDs
            policies = {f"agent_{i}": (None, None, None, {}) for i in range(10)}
        
        # Set multi-agent configuration
        self.config["multiagent"] = {
            "policies": policies,
            "policy_mapping_fn": policyMappingFn,
        }
        
        # Determine number of agents from env config
        numAgents = 4  # Default
        if "env_config" in self.config and "numAgents" in self.config["env_config"]:
            numAgents = self.config["env_config"]["numAgents"]
        
        # Centralised critic settings with enhanced configuration
        self.config["model"]["custom_model_config"] = {
            "use_centralised_critic": True,
            "centralised_critic_model": "CentralisedCriticNetwork",
            "num_agents": numAgents,
            "use_attention": useAttention,
            "reward_type": self.rewardType,
            "hybrid_reward_mix": self.hybridRewardMix,
            # Agent observation information for proper processing
            "agent_view_radius": self.config.get("env_config", {}).get("agentViewRadius", 5),
            "partial_observability": True,  # Assume the environment is partially observable
        }
    
    def getConfig(self) -> Dict[str, Any]:
        """
        Get the configuration dictionary.
        
        Returns:
            The PPO configuration dictionary
        """
        return self.config
    
    def getTrainer(self) -> Any:
        """
        Get a configured PPO trainer instance.
        
        Returns:
            PPO trainer instance
        """
        # Register the environment with Ray first
        def env_creator(env_config):
            """Creates an instance of the ResourceCollectionEnv."""
            if ResourceCollectionEnv is not None:
                return ResourceCollectionEnv(env_config)
            
            # Try importing at runtime if not available
            try:
                from env.resourceEnv import ResourceCollectionEnv as EnvClass
                return EnvClass(env_config)
            except ImportError:
                try:
                    from resourceProjectMARL.env.resourceEnv import ResourceCollectionEnv as EnvClass
                    return EnvClass(env_config)
                except ImportError:
                    raise ImportError(
                        "Could not import ResourceCollectionEnv. Make sure it's properly installed."
                    )
        
        # Register the environment
        register_env("ResourceCollectionEnv", env_creator)
        
        # Make sure the config uses the registered env name
        if isinstance(self.config, dict):
            self.config["env"] = "ResourceCollectionEnv"
        else:
            self.config.env = "ResourceCollectionEnv"
        
        try:
            # First attempt with normal configuration
            return PPO(config=self.config)
        except Exception as e:
            print(f"Error initializing PPO trainer: {e}")
            print("Attempting to suppress validation errors...")
            
            # Try again with validation turned off if we're using new Ray API
            if isinstance(self.config, dict):
                if "_validate_config" not in self.config:
                    self.config["_validate_config"] = False
            else:
                self.config = self.config.experimental(_validate_config=False)
                self.config = self.config.to_dict()
            
            return PPO(config=self.config)


# Helper function to create default PPO policy
def createDefaultPPOPolicy(
    useCentralisedCritic: bool = True, 
    numAgents: int = 4, 
    config: Dict[str, Any] = None,
    rewardType: str = "hybrid",
    hybridRewardMix: float = 0.5,
    useAttention: bool = False,
    useGpu: bool = False
) -> PPOPolicy:
    """
    Create a default PPO policy for resource collection.
    
    Args:
        useCentralisedCritic: Whether to use a centralised critic (MAPPO)
        numAgents: Number of agents in the environment
        config: Additional configuration overrides
        rewardType: Type of reward structure ("individual", "global", or "hybrid")
        hybridRewardMix: Weight for hybrid rewards (0.0 = fully global, 1.0 = fully individual)
        useAttention: Whether to use attention mechanism for centralised critic
        useGpu: Whether to use GPU for training
        
    Returns:
        Configured PPO policy
    """
    # Create base configuration
    configOverrides = {
        "env_config": {
            "numAgents": numAgents,
            "rewardType": rewardType,
            "hybridRewardMix": hybridRewardMix
        }
    }
    
    # Apply additional config overrides if provided
    if config is not None:
        # Update env_config separately to preserve numAgents and reward settings
        if "env_config" in config:
            # Keep our reward settings unless explicitly overridden
            if "rewardType" not in config["env_config"]:
                config["env_config"]["rewardType"] = rewardType
            if "hybridRewardMix" not in config["env_config"]:
                config["env_config"]["hybridRewardMix"] = hybridRewardMix
                
            configOverrides["env_config"].update(config["env_config"])
            del config["env_config"]
        
        # Update top-level config
        configOverrides.update(config)
    
    # Create policy with configured settings
    policy = PPOPolicy(
        configOverrides=configOverrides,
        useCentralisedCritic=useCentralisedCritic,
        sharedPolicy=True,  # Use shared policy for simpler training
        rewardType=rewardType,
        hybridRewardMix=hybridRewardMix,
        useGpu=useGpu
    )
    
    # Configure MAPPO settings if using centralised critic
    if useCentralisedCritic:
        policy._configureMAPPO(sharedPolicy=True, useAttention=useAttention)
    
    return policy 