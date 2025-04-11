"""
Utility functions for working with environment and training configurations.

This module provides functions to load, save, and manipulate configuration
parameters for the resource collection environment and training process.
"""

import os
import sys
import copy
from typing import Dict, Any, Optional, List, Tuple

# Add the project root to the path
scriptDir = os.path.dirname(os.path.abspath(__file__))
projectDir = os.path.dirname(scriptDir)
if projectDir not in sys.path:
    sys.path.insert(0, projectDir)

# Import the default environment configuration
from env.config import DEFAULT_ENV_CONFIG


def getDefaultEnvConfig() -> Dict[str, Any]:
    """
    Get a copy of the default environment configuration.
    
    Returns:
        Copy of the default environment configuration
    """
    return copy.deepcopy(DEFAULT_ENV_CONFIG)


def mergeConfigs(base_config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration dictionary
        overrides: Configuration dictionary with values to override
        
    Returns:
        Merged configuration dictionary
    """
    result = copy.deepcopy(base_config)
    
    def merge_dict(target, source):
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                merge_dict(target[key], value)
            else:
                # Override or add the value
                target[key] = value
    
    merge_dict(result, overrides)
    return result


def createEnvConfig(
    numAgents: int = None,
    gridSize: Tuple[int, int] = None,
    resourceTypes: List[str] = None,
    rewardType: str = None,
    hybridRewardMix: float = None,
    overrides: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create an environment configuration with common parameters.
    
    Args:
        numAgents: Number of agents in the environment
        gridSize: Size of the grid as (width, height)
        resourceTypes: List of resource types
        rewardType: Type of reward ('individual', 'shared', or 'hybrid')
        hybridRewardMix: Mix factor for hybrid rewards (0.0-1.0)
        overrides: Additional configuration overrides
        
    Returns:
        Environment configuration dictionary
    """
    config = getDefaultEnvConfig()
    
    # Apply specific parameter overrides
    if numAgents is not None:
        config["numAgents"] = numAgents
    
    if gridSize is not None:
        config["gridSize"] = gridSize
    
    if resourceTypes is not None:
        config["resourceTypes"] = resourceTypes
    
    if rewardType is not None:
        config["rewardType"] = rewardType
    
    if hybridRewardMix is not None:
        config["hybridRewardMix"] = hybridRewardMix
    
    # Apply general overrides
    if overrides:
        config = mergeConfigs(config, overrides)
    
    return config


def validateConfig(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check required fields
    required_fields = [
        "gridSize",
        "numAgents",
        "resourceTypes",
        "rewardType"
    ]
    
    for field in required_fields:
        if field not in config:
            errors.append(f"Required field '{field}' missing")
    
    # Validate numeric ranges
    if "numAgents" in config and (not isinstance(config["numAgents"], int) or config["numAgents"] <= 0):
        errors.append("Number of agents must be a positive integer")
    
    if "gridSize" in config:
        grid_size = config["gridSize"]
        if not isinstance(grid_size, tuple) or len(grid_size) != 2:
            errors.append("Grid size must be a tuple of (width, height)")
        elif not all(isinstance(x, int) and x > 0 for x in grid_size):
            errors.append("Grid dimensions must be positive integers")
    
    if "hybridRewardMix" in config:
        mix = config["hybridRewardMix"]
        if not isinstance(mix, (int, float)) or mix < 0 or mix > 1:
            errors.append("Hybrid reward mix must be a float between 0 and 1")
    
    # Validate reward type
    if "rewardType" in config and config["rewardType"] not in ["individual", "shared", "hybrid"]:
        errors.append("Reward type must be 'individual', 'shared', or 'hybrid'")
    
    return len(errors) == 0, errors


def getConfigSummary(config: Dict[str, Any]) -> str:
    """
    Get a human-readable summary of a configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        String summary of the configuration
    """
    env_config = config.get("env_config", config)
    
    summary = [
        "Environment Configuration Summary:",
        f"Number of Agents: {env_config.get('numAgents', 'Not specified')}",
        f"Grid Size: {env_config.get('gridSize', 'Not specified')}",
        f"Resource Types: {', '.join(env_config.get('resourceTypes', ['Not specified']))}",
        f"Reward Type: {env_config.get('rewardType', 'Not specified')}"
    ]
    
    if env_config.get("rewardType") == "hybrid":
        summary.append(f"Hybrid Reward Mix: {env_config.get('hybridRewardMix', 'Not specified')}")
    
    summary.extend([
        f"Day/Night Cycle: {'Enabled' if env_config.get('dayNightCycle', False) else 'Disabled'}",
        f"Weather: {'Enabled' if env_config.get('weatherEnabled', False) else 'Disabled'}",
        f"Seasons: {'Enabled' if env_config.get('seasonEnabled', False) else 'Disabled'}"
    ])
    
    return "\n".join(summary)


def createEnvConfigFromUI(ui_values: Dict[str, Any], base_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create an environment configuration from UI values.
    
    This function combines UI values with a base configuration (either loaded from 
    a file or using the default) while prioritising the UI values.
    
    Args:
        ui_values: Dictionary of UI field values
        base_config: Base configuration to use (default if None)
        
    Returns:
        Environment configuration dictionary
    """
    # Start with default config or provided base
    env_config = base_config.copy() if base_config else getDefaultEnvConfig()
    
    # Apply UI values with priority
    ui_overrides = {}
    
    # Basic tab values
    if "numAgents" in ui_values:
        ui_overrides["numAgents"] = ui_values["numAgents"]
    
    if "gridSize" in ui_values:
        ui_overrides["gridSize"] = ui_values["gridSize"]
    
    if "rewardType" in ui_values:
        ui_overrides["rewardType"] = ui_values["rewardType"]
        
        # Handle 'global' -> 'shared' conversion
        if ui_overrides["rewardType"] == "global":
            ui_overrides["rewardType"] = "shared"
    
    if "hybridRewardMix" in ui_values and ui_values.get("rewardType") == "hybrid":
        ui_overrides["hybridRewardMix"] = ui_values["hybridRewardMix"]
    
    return mergeConfigs(env_config, ui_overrides)


def extractUIValuesFromConfig(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract common UI values from an environment configuration.
    
    This function returns a dictionary of values that can be directly used
    to populate UI fields based on a configuration.
    
    Args:
        config: Environment configuration dictionary
        
    Returns:
        Dictionary of UI field values
    """
    env_config = config.get("env_config", config)
    
    ui_values = {}
    
    # Extract common UI fields
    if "numAgents" in env_config:
        ui_values["numAgents"] = env_config["numAgents"]
    
    if "gridSize" in env_config:
        ui_values["gridSize"] = env_config["gridSize"]
    
    if "rewardType" in env_config:
        reward_type = env_config["rewardType"]
        # Handle 'shared' -> 'global' conversion for UI
        if reward_type == "shared":
            reward_type = "global"
        ui_values["rewardType"] = reward_type
    
    if "hybridRewardMix" in env_config:
        ui_values["hybridRewardMix"] = env_config["hybridRewardMix"]
    
    return ui_values


def getBasicConfigOptions() -> List[str]:
    """
    Get the list of basic configuration options that are exposed in the UI.
    
    Returns:
        List of configuration keys that are directly modifiable in the UI
    """
    return [
        "numAgents",
        "gridSize", 
        "rewardType",
        "hybridRewardMix"
    ]


def getAdvancedConfigOptions() -> List[Dict[str, Any]]:
    """
    Get a list of advanced configuration options for potential integration in UI.
    
    Returns:
        List of dictionaries with option metadata (name, type, default, description)
    """
    options = [
        {
            "name": "obstaclesEnabled",
            "type": "bool",
            "default": DEFAULT_ENV_CONFIG["obstaclesEnabled"],
            "description": "Whether obstacles are enabled in the environment"
        },
        {
            "name": "obstacleDensity", 
            "type": "float", 
            "default": DEFAULT_ENV_CONFIG["obstacleDensity"],
            "range": [0.0, 0.5],
            "description": "Percentage of grid cells that have obstacles"
        },
        {
            "name": "resourceTypes",
            "type": "list",
            "default": DEFAULT_ENV_CONFIG["resourceTypes"],
            "description": "Types of resources in the environment"
        },
        {
            "name": "resourceDensity",
            "type": "float",
            "default": DEFAULT_ENV_CONFIG["resourceDensity"],
            "range": [0.01, 0.5],
            "description": "Percentage of grid cells that have resources"
        },
        {
            "name": "resourceRegeneration",
            "type": "bool",
            "default": DEFAULT_ENV_CONFIG["resourceRegeneration"],
            "description": "Whether resources regenerate after collection"
        },
        {
            "name": "dayNightCycle",
            "type": "bool",
            "default": DEFAULT_ENV_CONFIG["dayNightCycle"],
            "description": "Enable day/night cycle"
        },
        {
            "name": "weatherEnabled",
            "type": "bool",
            "default": DEFAULT_ENV_CONFIG["weatherEnabled"],
            "description": "Enable weather effects"
        },
        {
            "name": "seasonEnabled",
            "type": "bool",
            "default": DEFAULT_ENV_CONFIG["seasonEnabled"],
            "description": "Enable seasonal changes"
        }
    ]
    
    return options 