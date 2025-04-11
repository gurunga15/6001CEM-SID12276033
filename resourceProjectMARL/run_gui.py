#!/usr/bin/env python

"""
This script loads trained PPO policies and visualises their behaviour.
"""

import os
import sys
import argparse
import time
import numpy as np
import gymnasium as gym
from pathlib import Path
import ray
from ray.rllib.algorithms.algorithm import Algorithm
import random
import json

# Add the project root to the path
scriptDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.dirname(scriptDir)
if parentDir not in sys.path:
    sys.path.insert(0, parentDir)

# Import environment and GUI modules
from env.resourceEnv import ResourceCollectionEnv
from gui.viewer import ResourceCollectionVisualiser
from env.utils import calculateGiniCoefficient, calculateJainFairnessIndex
from algorithms.ppoPolicies import createDefaultPPOPolicy

# For Gymnasium compatibility with Ray
sys.modules["gym"] = gym  # Monkey-patch for Ray compatibility


def main():
    parser = argparse.ArgumentParser(description="Visualise trained policy on ResourceCollectionEnv")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint directory")
    parser.add_argument("--grid-size", type=int, nargs=2, default=[15, 15], help="Grid size (width height)")
    parser.add_argument("--num-agents", type=int, default=4, help="Number of agents")
    parser.add_argument("--resource-types", type=str, nargs="+", default=["food", "wood", "stone"],
                       help="Resource types to include")
    parser.add_argument("--day-night", action="store_true", help="Enable day/night cycle")
    parser.add_argument("--weather", action="store_true", help="Enable weather effects")
    parser.add_argument("--seasons", action="store_true", help="Enable seasonal changes")
    parser.add_argument("--seed", type=int, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Initialise Ray if not already initialised
    if not ray.is_initialized():
        ray.init(num_cpus=1, include_dashboard=False)
    
    # Load algorithm based on checkpoint (default to PPO)
    algorithm = "ppo"
    
    # Create environment configuration
    config = {
        "gridSize": (args.grid_size[0], args.grid_size[1]),
        "numAgents": args.num_agents,
        "resourceTypes": args.resource_types,
        "dayNightCycle": args.day_night,
        "weatherEnabled": args.weather,
        "seasonEnabled": args.seasons,
        "renderMode": "rgb_array",
        "max_steps": 1000,  # Longer episodes for visualisation
        "use_flattened_obs": True,  # For PPO compatibility
        "rewardType": "hybrid",
        "hybridRewardMix": 0.5,  # Equal weight to individual and team rewards
    }
    
    print(f"Creating environment with configuration:")
    print(f"  Grid size: {config['gridSize']}")
    print(f"  Agents: {config['numAgents']}")
    print(f"  Resources: {config['resourceTypes']}")
    print(f"  Features: Day/Night={config['dayNightCycle']}, Weather={config['weatherEnabled']}, Seasons={config['seasonEnabled']}")
    
    # Create environment for evaluation
    env = ResourceCollectionEnv(config)
    
    # Initialise policy/algorithm
    if args.checkpoint:
        print("Loading environment configuration from checkpoint...")
        configPath = os.path.join(os.path.dirname(args.checkpoint), "params.json")
        if not os.path.exists(configPath):
            configPath = os.path.join(os.path.dirname(args.checkpoint), "..", "params.json")
        
        if os.path.exists(configPath):
            with open(configPath, "r") as f:
                checkpointConfig = json.load(f)
            
            if "env_config" in checkpointConfig:
                env_config = checkpointConfig["env_config"]
                # Copy key values from checkpoint config to current config
                print("Using checkpoint environment settings:")
                for key in ["numAgents", "gridSize", "resourceTypes", "rewardType", "hybridRewardMix"]:
                    if key in env_config:
                        config[key] = env_config[key]
                        print(f"  {key}: {env_config[key]}")
            
            # Recreate environment with updated config
            env = ResourceCollectionEnv(config)
        
        # Create policy
        ppo_policy = createDefaultPPOPolicy(
            useCentralisedCritic=True,
            numAgents=config["numAgents"],
            config={"env_config": config}
        )
        
        # Create and restore algorithm
        try:
            # Try restoring directly
            trainer = Algorithm.from_checkpoint(args.checkpoint)
        except Exception as e:
            print(f"Error loading checkpoint directly: {e}")
            print("Attempting to create and restore PPO algorithm...")
            
            # Create trainer and restore manually
            trainer = ppo_policy.getTrainer()
            trainer.restore(args.checkpoint)
        else:
            print("No checkpoint provided. Using randomly initialised policy.")
            # Create policy with default settings
        ppo_policy = createDefaultPPOPolicy(
            useCentralisedCritic=True,
            numAgents=config["numAgents"],
            config={"env_config": config}
        )
        trainer = ppo_policy.getTrainer()
    
    # Create the GUI viewer
    viewer = ResourceCollectionVisualiser(env, render_fps=5)
    
    # Function to get actions from trained policy
    def policyFn(obs_dict, state_dict=None):
        """Function that maps observations to actions using the trained policy."""
        actions = {}
        
        for agent_id, obs in obs_dict.items():
            # Use compute_single_action for each agent
            actions[agent_id] = trainer.compute_single_action(
                observation=obs,
                policy_id=agent_id,
                explore=True
            )
        
        # For testing:
        # return get_random_actions(env, obs_dict)
        
        return actions
    
    # Start the GUI viewer with the policy
    viewer.run(policy_fn=policyFn)
    
    # Cleanup
    ray.shutdown()


def get_random_actions(env, obs_dict):
    """Get random actions for all agents."""
    return {agent_id: env.action_spaces[agent_id].sample() for agent_id in obs_dict.keys()}


if __name__ == "__main__":
    main() 