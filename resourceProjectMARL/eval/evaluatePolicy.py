"""
Evaluation script for trained PPO policies.
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import csv
import random
from typing import Dict, List, Any, Optional, Tuple, Union
import pprint

# Add project root to path
scriptDir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(scriptDir)
sys.path.insert(0, project_dir)

# For Gymnasium compatibility with Ray
import gymnasium as gym
sys.modules["gym"] = gym  # Monkey-patch to keep RLlib happy with Gymnasium

# Import Ray
import ray
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.tune.registry import register_env

# Import environment and configurations
from env.resourceEnv import ResourceCollectionEnv
from env.config import DEFAULT_ENV_CONFIG

# Check if new-style Ray APIs are available
try:
    import ray.rllib.algorithms.ppo
    NEW_RAY_API = True
except ImportError:
    NEW_RAY_API = False
    
# Add the parent directory (resourceProjectMARL)
parent_dir = os.path.dirname(scriptDir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import directly from relative paths
import env.resourceEnv  # Register environment
from train.common import loadConfig
from algorithms.ppoPolicies import createDefaultPPOPolicy


def createTrainer(config_source: Union[str, dict], numAgents: int = 4) -> Any:
    """
    Create a PPO trainer.
    
    Args:
        config_source: Either a path to a checkpoint or a config dictionary
        numAgents: Number of agents
        
    Returns:
        Configured trainer instance
    """
    # Load configuration if a path is provided
    if isinstance(config_source, str):
        config_dir = os.path.dirname(config_source)
        config = loadConfig(os.path.join(config_dir, "config.pkl"))
        if not config:
            # Try parent directory
            config = loadConfig(os.path.join(os.path.dirname(config_dir), "config.pkl"))
    else:
        config = config_source
    
    # Create default config if none found
    if not config:
        config = {}
    
    # Extract environment config
    env_config = config.get("env_config", {})
    if "numAgents" not in env_config:
        env_config["numAgents"] = numAgents
    
    # Get centralised critic setting from config if available
    useCentralisedCritic = config.get("use_centralised_critic", True)
    
    # Create PPO policy
    policy = createDefaultPPOPolicy(
        useCentralisedCritic=useCentralisedCritic,
        numAgents=env_config.get("numAgents", numAgents)
    )
    
    # Override with loaded config
    policy.config.update(config)
    
    # Make sure we have the environment config
    policy.config["env_config"] = env_config
    
    return policy.getTrainer()


def loadConfigFromCheckpoint(checkpointDir: str) -> dict:
    """
    Load configuration from checkpoint directory.
    
    Args:
        checkpointDir: Path to the checkpoint directory
        
    Returns:
        Environment and algorithm configurations
    """
    # Look for config.py in parent directory
    parentDir = os.path.dirname(os.path.dirname(checkpointDir))
    
    # Try to load config.py from parent directory
    configPath = os.path.join(parentDir, "config.py")
    if not os.path.exists(configPath):
        # Try config.pkl for backward compatibility 
        configPath = os.path.join(parentDir, "config.pkl")
    
    if os.path.exists(configPath):
        return loadConfig(configPath)
    
    # Try to load config.py from checkpoint directory
    configPath = os.path.join(os.path.dirname(checkpointDir), "config.py")
    if not os.path.exists(configPath):
        # Try config.pkl for backward compatibility
        configPath = os.path.join(os.path.dirname(checkpointDir), "config.pkl")
    
    if os.path.exists(configPath):
        return loadConfig(configPath)
    
    # If config not found, return empty dict
    print(f"Warning: No configuration found for checkpoint {checkpointDir}")
    return {}


def evaluatePolicy(
    checkpointPath: str,
    numEpisodes: int = 10,
    renderMode: str = None,
    outputPath: str = None,
    seed: int = None,
    configPath: str = None
):
    """
    Evaluate a trained PPO policy.
    
    Args:
        checkpointPath: Path to the checkpoint to evaluate
        numEpisodes: Number of episodes to evaluate
        renderMode: Rendering mode ("human" or None)
        outputPath: Path to save evaluation results
        seed: Random seed for reproducibility
        configPath: Path to configuration file
    """
    # initialise Ray
    if not ray.is_initialized():
        ray.init(num_cpus=os.cpu_count() or 4, include_dashboard=False)
    
    # Load configuration
    config = loadConfig(configPath) if configPath else {}
    
    # Create trainer with PPO algorithm
    trainer = createTrainer(checkpointPath, config)
    
    # Load the checkpoint
    if os.path.exists(checkpointPath):
        trainer.restore(checkpointPath)
    else:
        raise ValueError(f"Checkpoint not found: {checkpointPath}")
    
    # Create environment
    env_config = config.get("env_config", {})
    env_config["renderMode"] = renderMode
    
    # Set seed if provided
    if seed is not None:
        env_config["seed"] = seed
        if "seed" not in config:
            config["seed"] = seed
    
    env = ResourceCollectionEnv(env_config)
    
    # Prepare for evaluation
    episode_rewards = []
    episode_lengths = []
    episode_metrics = []
    
    # Create output directory if needed
    if outputPath:
        os.makedirs(outputPath, exist_ok=True)
    
    # Record timestamps
    start_time = time.time()
    
    for episode in range(numEpisodes):
        # Set episode seed if global seed is provided
        episode_seed = seed + episode if seed is not None else None
        
        # Reset environment
        observations, infos = env.reset(seed=episode_seed)
        
        # initialise episode tracking
        episode_reward = 0
        episode_step = 0
        done = False
        terminated = {"__all__": False}
        truncated = {"__all__": False}
        
        # Agent tracking
        agent_rewards = {agent_id: 0.0 for agent_id in observations.keys()}
        agent_inventories = {agent_id: {} for agent_id in observations.keys()}
        
        while not (terminated.get("__all__", False) or truncated.get("__all__", False)):
            # Get actions from trainer
            actions = {}
            for agent_id, obs in observations.items():
                # Get action from policy
                if NEW_RAY_API:
                    # New API (Ray 2.0+)
                    policy_id = trainer.config["multiagent"]["policy_mapping_fn"](agent_id)
                    action = trainer.compute_single_action(obs, policy_id=policy_id, explore=False)
                else:
                    # Old API (Ray 1.x)
                    policy_id = trainer.config["multiagent"]["policy_mapping_fn"](agent_id)
                    action = trainer.compute_action(obs, policy_id=policy_id, explore=False)
                
                actions[agent_id] = action
            
            # Step environment
            observations, rewards, terminated, truncated, infos = env.step(actions)
            
            # Accumulate rewards
            step_reward = sum(rewards.values())
            episode_reward += step_reward
            
            # Track per-agent rewards
            for agent_id, reward in rewards.items():
                agent_rewards[agent_id] += reward
            
            # Track agent inventories
            for agent_id, info in infos.items():
                if "inventory" in info:
                    agent_inventories[agent_id] = info["inventory"]
            
            # Increment step counter
            episode_step += 1
        
        # Record episode results
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_step)
        
        # Extract metrics from last info
        metrics = {}
        for key in infos.get(list(infos.keys())[0], {}):
            if key in ["fairnessGini", "jainFairnessIndex", "resourcesPerStep", "collisionsPerStep", "resourceSpecialisation"]:
                metrics[key] = infos[list(infos.keys())[0]][key]
        
        # Add agent-specific metrics
        metrics["agent_rewards"] = agent_rewards
        metrics["agent_inventories"] = agent_inventories
        
        episode_metrics.append(metrics)
        
        print(f"Episode {episode+1}/{numEpisodes}: reward={episode_reward}, length={episode_step}")
    
    # Calculate evaluation statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    # Calculate average metrics
    avg_metrics = {}
    for key in episode_metrics[0].keys():
        if key not in ["agent_rewards", "agent_inventories"]:
            avg_metrics[key] = np.mean([m[key] for m in episode_metrics if key in m])
    
    # Calculate per-agent statistics
    agent_stats = {}
    for agent_id in agent_rewards.keys():
        agent_stats[agent_id] = {
            "mean_reward": np.mean([m["agent_rewards"][agent_id] for m in episode_metrics]),
            "std_reward": np.std([m["agent_rewards"][agent_id] for m in episode_metrics]),
            "resources_collected": sum([sum(m["agent_inventories"][agent_id].values()) for m in episode_metrics])
        }
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Checkpoint: {checkpointPath}")
    print(f"Episodes: {numEpisodes}")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean Episode Length: {mean_length:.2f}")
    print("\nMetrics:")
    for key, value in avg_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Print per-agent statistics
    print("\nPer-Agent Statistics:")
    for agent_id, stats in agent_stats.items():
        print(f"  {agent_id}: Reward={stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}, Resources={stats['resources_collected']}")
    
    # Save results if output path provided
    if outputPath:
        # Save summary
        summary = {
            "algorithm": "ppo",
            "checkpoint": checkpointPath,
            "episodes": numEpisodes,
            "seed": seed,
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "mean_length": float(mean_length),
            "metrics": avg_metrics,
            "agent_stats": {
                agent_id: {k: float(v) for k, v in stats.items()}
                for agent_id, stats in agent_stats.items()
            },
            "evaluation_time": time.time() - start_time
        }
        
        with open(os.path.join(outputPath, "summary.py"), "w") as f:
            f.write("# Auto-generated evaluation summary file\n")
            f.write("# Do not edit manually\n\n")
            f.write("SUMMARY = ")
            f.write(pprint.pformat(summary, indent=4, width=100))
            f.write("\n\n# Export the summary for easy import\ndef getSummary():\n    return SUMMARY\n")
        
        # Save episode data
        with open(os.path.join(outputPath, "episodes.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward", "length"] + list(avg_metrics.keys()))
            for i, (reward, length, metrics) in enumerate(zip(episode_rewards, episode_lengths, episode_metrics)):
                row = [i+1, reward, length]
                row.extend([metrics.get(key, "") for key in avg_metrics.keys()])
                writer.writerow(row)
        
        # Save per-agent data
        with open(os.path.join(outputPath, "agents.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["agent_id", "episode", "reward"] + [f"resource_{rtype}" for rtype in DEFAULT_ENV_CONFIG["resourceTypes"]])
            for i, metrics in enumerate(episode_metrics):
                for agent_id, reward in metrics["agent_rewards"].items():
                    row = [agent_id, i+1, reward]
                    inventory = metrics["agent_inventories"][agent_id]
                    row.extend([inventory.get(rtype, 0) for rtype in DEFAULT_ENV_CONFIG["resourceTypes"]])
                    writer.writerow(row)
        
        print(f"\nEvaluation results saved to {outputPath}")
    
    # Clean up
    env.close()
    
    return summary if outputPath else {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_length": mean_length,
        "metrics": avg_metrics,
        "agent_stats": agent_stats
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO policy")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--output", type=str, help="Path to save evaluation results")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    
    args = parser.parse_args()
    
    evaluatePolicy(
        checkpointPath=args.checkpoint,
        numEpisodes=args.episodes,
        renderMode="human" if args.render else None,
        outputPath=args.output,
        seed=args.seed,
        configPath=args.config
    ) 