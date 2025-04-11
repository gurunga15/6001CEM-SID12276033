"""
Rendering demo for resource collection environment with trained policies.
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add the project root to the path
scriptDir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(scriptDir))
sys.path.insert(0, project_dir)

# Add the parent directory (resourceProjectMARL)
parent_dir = os.path.dirname(scriptDir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import directly from relative paths
import env.resourceEnv  # Register environment
from env.resourceEnv import ResourceCollectionEnv
from .evaluatePolicy import evaluatePolicy
from train.common import loadConfig


def loadEnvironmentConfig(configPath):
    """
    Load environment configuration from a file.
    
    Args:
        configPath: Path to configuration file
        
    Returns:
        Environment configuration dictionary
    """
    if os.path.exists(configPath):
        config = loadConfig(configPath)
        return config.get('env_config', {})
    else:
        print(f"Warning: Configuration file not found at {configPath}")
        return {}


def renderDemo(
    checkpointPath=None,
    configPath=None,
    randomPolicy=False,
    delay=0.1,
    episodes=3,
    maxSteps=500
):
    """
    Render a demonstration of the resource collection environment.
    
    Args:
        checkpointPath: Path to a trained policy checkpoint
        configPath: Path to environment configuration file
        randomPolicy: Use random policy instead of trained policy
        delay: Delay between steps for visualisation (seconds)
        episodes: Number of episodes to run
        maxSteps: Maximum steps per episode
    """
    # Load environment config
    envConfig = {}
    if configPath:
        envConfig = loadEnvironmentConfig(configPath)
    
    # Set render mode
    envConfig['renderMode'] = 'human'
    
    if randomPolicy:
        # Run with random policy
        env = ResourceCollectionEnv(envConfig)
        
        for episode in range(episodes):
            print(f"Episode {episode+1}/{episodes}")
            obs, info = env.reset()
            done = {"__all__": False}
            step = 0
            
            while not done["__all__"] and step < maxSteps:
                # Random actions
                actions = {}
                for agent_id in obs:
                    actions[agent_id] = env.action_spaces[agent_id].sample()
                
                # Step environment
                obs, rewards, terminateds, truncateds, info = env.step(actions)
                
                # Update done flag
                done = {"__all__": terminateds.get("__all__", False) or truncateds.get("__all__", False)}
                for agent_id in terminateds:
                    if agent_id != "__all__":
                        done[agent_id] = terminateds.get(agent_id, False) or truncateds.get(agent_id, False)
                
                # Render and delay
                env.render()
                time.sleep(delay)
                step += 1
            
            print(f"Episode completed in {step} steps")
        
        env.close()
    
    else:
        # Run with trained policy
        if not checkpointPath:
            print("Error: Checkpoint path required for trained policy demo")
            return
        
        # Use evaluatePolicy with render enabled
        evaluatePolicy(
            checkpointPath=checkpointPath,
            numEpisodes=episodes,
            renderMode="human"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a demo of the resource collection environment")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint for trained policy")
    parser.add_argument("--config", type=str, help="Path to environment configuration file")
    parser.add_argument("--random", action="store_true", help="Use random policy instead of trained policy")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between steps (seconds)")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum steps per episode")
    
    args = parser.parse_args()
    
    renderDemo(
        checkpointPath=args.checkpoint,
        configPath=args.config,
        randomPolicy=args.random,
        delay=args.delay,
        episodes=args.episodes,
        maxSteps=args.max_steps
    ) 