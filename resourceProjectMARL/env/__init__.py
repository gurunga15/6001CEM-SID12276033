"""
Environment package for resource collection multi-agent reinforcement learning.

This module registers the environment with Gymnasium and provides access
to the ResourceCollectionEnv class.
"""

import os
import gymnasium as gym
from gymnasium.envs.registration import register

# Register the environment with Gymnasium
register(
    id="ResourceCollection-v0",
    entry_point="env.resourceEnv:ResourceCollectionEnv",
    max_episode_steps=10000,
)

# Import necessary modules
from .resourceEnv import ResourceCollectionEnv
from .entities import Agent, Resource, ResourceManager
from .utils import calculateGiniCoefficient, calculateJainFairnessIndex
from .config import DEFAULT_ENV_CONFIG, DAY_NIGHT_PHASES, WEATHER_EFFECTS, SEASON_EFFECTS

# Explicitly expose key classes/modules
__all__ = [
    "ResourceCollectionEnv",
    "Agent",
    "Resource",
    "ResourceManager",
    "calculateGiniCoefficient",
    "calculateJainFairnessIndex",
    "DEFAULT_ENV_CONFIG",
    "DAY_NIGHT_PHASES",
    "WEATHER_EFFECTS",
    "SEASON_EFFECTS",
] 