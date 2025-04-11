"""
Configuration constants for the ResourceCollectionEnv.
"""

from typing import Dict, Any

# Default environment configuration
DEFAULT_ENV_CONFIG = {
    # Grid and agent configuration
    "gridSize": (20, 20),  # Width, height of the grid
    "numAgents": 4,        # Number of agents in the environment
    "agentViewRadius": 5,  # How far agents can see
    "observationRadius": 5,  # Size of the local observation grid
    
    # Obstacle configuration
    "obstaclesEnabled": True,  # Whether obstacles are enabled
    "obstacleDensity": 0.1,    # Percentage of grid cells that have obstacles
    "obstacleDistribution": "random",  # 'random', 'clustered', or 'mixed'
    
    # Resource configuration
    "resourceTypes": ["food", "wood", "stone"],  # Types of resources
    "resourceValues": {     # Value of each resource type
        "food": 1, 
        "wood": 2, 
        "stone": 3
    },
    "resourceDensity": 0.1,  # Percentage of grid cells that have resources
    "resourceDistribution": "random",  # 'random', 'clustered', or 'mixed'
    "resourceRespawnProbabilities": {  # Chance to respawn each type per step
        "food": 0.02,
        "wood": 0.01,
        "stone": 0.005
    },
    "resourceRegeneration": True,  # Whether resources regenerate
    "resourceRegenerationRate": 0.05,  # Base rate of resource regeneration
    "maxInventoryCapacity": 50,  # Maximum capacity per resource type
    
    # Environment dynamics
    "dayNightCycle": True,     # Enable day/night cycle
    "dayCycleLength": 500,     # Steps in a full day
    "weatherEnabled": True,    # Enable weather effects
    "weatherTypes": ["clear", "rain", "fog", "snow", "storm"],  # Available weather types
    "weatherChangeProb": 0.01,  # Probability of weather changing per step
    "seasonEnabled": True,     # Enable seasonal changes
    "seasonTypes": ["spring", "summer", "autumn", "winter"],  # Available seasons
    "seasonChangeProb": 0.001,  # Probability of season changing per step
    "seasonLength": 2000,      # Steps in a season
    "timeLimit": 10000,        # Maximum episode length
    "maxEpisodeSteps": 10000,  # Synonym for timeLimit
    
    # Reward structure
    "rewardType": "hybrid",   # 'individual', 'shared', or 'hybrid'
    "hybridRewardMix": 0.5,   # 0.0 = fully shared, 1.0 = fully individual
    "rewardWeights": {        # Weights for different reward components
        "collect": 1.0,       # Base reward for collecting a resource
        "timeStep": -0.01,    # Small penalty for each time step
        "collision": -0.5,    # Penalty for collisions with other agents
    },
    "collisionPenalty": 0.5,  # Penalty for agent collisions
    "movementCost": 0.01,     # Cost for agent movement
    
    # Rendering
    "renderMode": None,       # None, 'human', or 'rgb_array'
}

# Day/night cycle phase definitions
DAY_NIGHT_PHASES = {
    "day": {
        "duration": 0.4,              # 40% of the cycle is day
        "visibilityFactor": 1.0,      # Full visibility during day
        "movementFailureProb": 0.0,   # No movement failures during day
    },
    "dusk": {
        "duration": 0.1,              # 10% of the cycle is dusk
        "visibilityFactor": 0.7,      # Reduced visibility during dusk
        "movementFailureProb": 0.05,  # Small chance of movement failure
    },
    "night": {
        "duration": 0.4,              # 40% of the cycle is night
        "visibilityFactor": 0.4,      # Greatly reduced visibility during night
        "movementFailureProb": 0.1,   # Higher chance of movement failure
    },
    "dawn": {
        "duration": 0.1,              # 10% of the cycle is dawn
        "visibilityFactor": 0.7,      # Reduced visibility during dawn
        "movementFailureProb": 0.05,  # Small chance of movement failure
    }
}

# Weather effect definitions
WEATHER_EFFECTS = {
    "clear": {
        "probability": 0.45,           # 45% chance of clear weather
        "visibilityFactor": 1.0,       # No visibility change
        "movementFailureProb": 0.0,    # No movement failures
        "collectionFailProb": 0.0,     # No resource collection failures
    },
    "rain": {
        "probability": 0.25,           # 25% chance of rain
        "visibilityFactor": 0.8,       # Slightly reduced visibility
        "movementFailureProb": 0.1,    # Small chance of movement failure
        "collectionFailProb": 0.1,     # Small chance of collection failure
    },
    "fog": {
        "probability": 0.15,           # 15% chance of fog
        "visibilityFactor": 0.6,       # Moderate visibility reduction
        "movementFailureProb": 0.05,   # Slight chance of movement failure
        "collectionFailProb": 0.05,    # Slight chance of collection failure
    },
    "snow": {
        "probability": 0.1,            # 10% chance of snow
        "visibilityFactor": 0.7,       # Less visibility reduction than fog but more than rain
        "movementFailureProb": 0.15,   # Higher movement failure than fog (slippery)
        "collectionFailProb": 0.1,     # Same collection failure as rain
    },
    "storm": {
        "probability": 0.05,           # 5% chance of storm
        "visibilityFactor": 0.4,       # Severe visibility reduction
        "movementFailureProb": 0.2,    # High chance of movement failure
        "collectionFailProb": 0.2,     # High chance of collection failure
    }
}

# Season effect definitions
SEASON_EFFECTS = {
    "spring": {
        "resourceRespawnMultiplier": {  # Multipliers for respawn rates
            "food": 1.5,
            "wood": 1.0,
            "stone": 0.8
        },
        "weatherTendencies": {  # Weather probabilities in this season
            "clear": 0.4,
            "rain": 0.5,
            "fog": 0.05,
            "snow": 0.0,  # No snow in spring
            "storm": 0.05
        },
        "dayNightCycle": {
            "dayDuration": 0.4,     # Default day duration
            "dawnDuration": 0.1,    # Default dawn duration
            "duskDuration": 0.1,    # Default dusk duration
            "nightDuration": 0.4    # Default night duration
        }
    },
    "summer": {
        "resourceRespawnMultiplier": {
            "food": 1.2,
            "wood": 1.2,
            "stone": 1.0
        },
        "weatherTendencies": {
            "clear": 0.65,
            "rain": 0.2,
            "fog": 0.05,
            "snow": 0.0,  # No snow in summer
            "storm": 0.1
        },
        "dayNightCycle": {
            "dayDuration": 0.6,     # Longer days in summer
            "dawnDuration": 0.1,
            "duskDuration": 0.1,
            "nightDuration": 0.2    # Shorter nights in summer
        }
    },
    "autumn": {
        "resourceRespawnMultiplier": {
            "food": 1.0,
            "wood": 1.5,
            "stone": 1.2
        },
        "weatherTendencies": {
            "clear": 0.4,
            "rain": 0.3,
            "fog": 0.15,
            "snow": 0.1,  # Occasional snow in autumn
            "storm": 0.05
        },
        "dayNightCycle": {
            "dayDuration": 0.4,     # Default day duration
            "dawnDuration": 0.1,    # Default dawn duration
            "duskDuration": 0.1,    # Default dusk duration
            "nightDuration": 0.4    # Default night duration
        }
    },
    "winter": {
        "resourceRespawnMultiplier": {
            "food": 0.5,
            "wood": 0.8,
            "stone": 1.5
        },
        "weatherTendencies": {
            "clear": 0.2,
            "rain": 0.05,
            "fog": 0.15,
            "snow": 0.4,  # High chance of snow in winter
            "storm": 0.2
        },
        "dayNightCycle": {
            "dayDuration": 0.2,     # Shorter days in winter
            "dawnDuration": 0.1,
            "duskDuration": 0.1,
            "nightDuration": 0.6    # Longer nights in winter
        }
    }
} 