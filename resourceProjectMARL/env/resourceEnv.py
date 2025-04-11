"""
Main environment class for multi-agent resource collection.
"""

from typing import Dict, List, Tuple, Optional, Union, Any, Set
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

# Use relative imports instead of absolute package names
from .entities import Agent, Resource, ResourceManager
from .utils import (
    SpatialHashGrid, manhattanDistance, euclideanDistance,
    generateRandomPositions, generateClusteredPositions,
    calculateGiniCoefficient, calculateJainFairnessIndex
)
from .config import (
    DEFAULT_ENV_CONFIG, WEATHER_EFFECTS, DAY_NIGHT_PHASES, SEASON_EFFECTS
)


class ResourceCollectionEnv(MultiAgentEnv):
    """
    Multi-agent reinforcement learning environment for resource collection.
    
    Agents navigate a grid world to collect resources while adapting to
    environmental dynamics such as day/night cycles, weather, and seasons.
    """
    
    metadata = {"render_modes": ["human", "rgb_array", None]}
    
    def __init__(self, config: dict = None):
        """
        Initialise the environment.
        
        Args:
            config: Configuration dictionary, overrides defaults
        """
        # Call parent class constructor
        super().__init__()
        
        # Load configuration with defaults
        self.config = DEFAULT_ENV_CONFIG.copy()
        if config:
            self.config.update(config)
        
        # Extract key configuration parameters
        self.gridSize = self.config["gridSize"]
        self.numAgents = self.config["numAgents"]
        self.renderMode = self.config.get("renderMode")
        self.rewardType = self.config.get("rewardType", "individual")
        self.hybridRewardMix = self.config.get("hybridRewardMix", 0.5)
        self.maxSteps = self.config.get("maxSteps", 500)
        
        # Set up the spatial grid for collision detection
        self.grid = SpatialHashGrid(cellSize=1.0, worldSize=self.gridSize)
        
        # Setup environment state variables
        self._resetEnvironmentalState()
        
        # Create agents
        self.agents = {}
        self._setupAgents()
        
        # Create resource manager
        self.resourceManager = ResourceManager(
            gridSize=self.gridSize,
            resourceTypes=self.config["resourceTypes"],
            resourceValues=self.config["resourceValues"],
            density=self.config["resourceDensity"],
            distribution=self.config["resourceDistribution"],
            grid=self.grid
        )
        
        # Set up obstacles
        self.obstacles = set()
        self.obstaclesEnabled = self.config.get("obstaclesEnabled", False)
        
        # initialise agent tracking for RLlib's MultiAgentEnv
        self.possible_agents = [f"agent_{i}" for i in range(self.numAgents)]
        
        # Track environment dynamics
        self.stepCount = 0
        self.episodeCollisions = 0
        self.episodeResourcesCollected = {rtype: 0 for rtype in self.config["resourceTypes"]}
        
        # Store last actions for record keeping
        self.last_actions = {}
        
        # Define action and observation spaces
        self._defineSpaces()
        
        # initialise renderer
        self.renderer = None
        if self.renderMode == "human":
            self._initRenderer()
    
    def _defineSpaces(self):
        """Define action and observation spaces for each agent."""
        # Action space: Discrete
        # 0: No-op, 1-4: Move (N, E, S, W), 5: Collect
        self.action_space = spaces.Discrete(6)
        
        # Define the spaces as dictionaries for each agent to match Ray API requirements
        self.action_spaces = {f"agent_{i}": self.action_space for i in range(self.numAgents)}
        
        # Check if we should use flattened observations for better compatibility
        useFlattened = self.config.get("use_flattened_obs", False)
        
        # Observation space components
        # Grid observation (agent-centric view of nearby cells)
        obsRadius = self.config["observationRadius"]
        viewSize = 2 * obsRadius + 1  # Size of the observation grid (e.g., 5x5 for radius 2)
        
        # Ensure observation size is compatible with CNN default sizes
        if viewSize not in [10, 42, 64, 84]:
            print(f"Warning: Observation size {viewSize}x{viewSize} is not in the default CNN sizes [10, 42, 64, 84].")
            print("You may need to specify custom conv_filters in your model configuration.")
        
        # Determine number of channels for grid observation
        # +1 for agents, +1 for obstacles, and one for each resource type
        numChannels = len(self.config["resourceTypes"]) + 2
        
        # Define the grid observation space - dynamically set the shape based on current configuration
        gridObs = spaces.Box(
            low=0,
            high=1,
            shape=(viewSize, viewSize, numChannels),
            dtype=np.float32
        )
        
        # Agent state: position, inventory, etc.
        agentStateSize = 2 + len(self.config["resourceTypes"])  # position (x,y) + inventory counts
        agentState = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(agentStateSize,),
            dtype=np.float32
        )
        
        # Environmental features
        envFeatureSize = 0
        if self.config["dayNightCycle"]:
            envFeatureSize += 1  # Time of day
        if self.config["weatherEnabled"]:
            # Get weather types from config or defaults
            weatherTypes = self.config.get("weatherTypes", list(WEATHER_EFFECTS.keys()))
            envFeatureSize += len(weatherTypes)  # One-hot weather
        if self.config["seasonEnabled"]:
            # Get season types from config or defaults
            seasonTypes = self.config.get("seasonTypes", list(SEASON_EFFECTS.keys()))
            envFeatureSize += len(seasonTypes)  # One-hot season
        
        envFeatures = spaces.Box(
            low=0,
            high=1,
            shape=(envFeatureSize,),
            dtype=np.float32
        )
        
        if useFlattened:
            # Calculate the total size of the flattened observation
            total_size = (
                np.prod(gridObs.shape) +  # Flattened grid
                agentStateSize +          # Agent state vector 
                envFeatureSize            # Environmental features
            )
            
            # Create a flattened observation space
            self.observation_space = spaces.Box(
                low=-np.inf,  # Conservative lower bound
                high=np.inf,  # Conservative upper bound
                shape=(int(total_size),),  # Ensure integer shape
                dtype=np.float32
            )
        else:
            # Combined observation space as Dict
            self.observation_space = spaces.Dict({
                "grid": gridObs,
                "agent_state": agentState,
                "env_features": envFeatures
            })
        
        # Define the observation spaces for each agent
        self.observation_spaces = {f"agent_{i}": self.observation_space for i in range(self.numAgents)}
    
    def _setupAgents(self):
        """Create agents and place them in the environment."""
        # Generate starting positions for agents
        startPositions = generateRandomPositions(
            count=self.numAgents,
            gridSize=self.gridSize,
            excludePositions=set()
        )
        
        # Create agents
        for i in range(self.numAgents):
            agentId = f"agent_{i}"
            position = startPositions[i]
            viewRadius = self.config["agentViewRadius"]
            
            # Create agent with given properties
            agent = Agent(
                agentId=agentId,
                position=position,
                viewRadius=viewRadius
            )
            
            # Add agent to the collection
            self.agents[agentId] = agent
            
            # Add agent to the spatial grid
            self.grid.addEntity(agentId, position)
    
    def _resetEnvironmentalState(self):
        """Reset environmental conditions."""
        # Time and day/night cycle
        self.timeOfDay = 0.0  # 0.0 to 1.0, where 0.0 is midnight
        self.dayPhase = "night"
        
        # Weather
        if self.config["weatherEnabled"]:
            # Get weather types from WEATHER_EFFECTS if not in config
            weatherTypes = self.config.get("weatherTypes", list(WEATHER_EFFECTS.keys()))
            self.currentWeather = np.random.choice(weatherTypes)
        else:
            self.currentWeather = "clear"
        
        # Season
        if self.config["seasonEnabled"]:
            # Get season types from SEASON_EFFECTS if not in config
            seasonTypes = self.config.get("seasonTypes", list(SEASON_EFFECTS.keys()))
            self.currentSeason = np.random.choice(seasonTypes)
        else:
            self.currentSeason = "summer"
    
    def seed(self, seed: Optional[int] = None):
        """
        Set the seed for the environment.
        
        Args:
            seed: The seed to use
        """
        if seed is not None:
            np.random.seed(seed)
            self.np_random = np.random.RandomState(seed)
        else:
            self.np_random = np.random.RandomState()
        return [seed]
        
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, Any]]]:
        """
        Reset the environment to an initial state.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Initial observations and info dictionaries (per agent)
        """
        # Handle seed following gymnasium style
        if seed is not None:
            self.seed(seed)
        elif "seed" in self.config and self.config["seed"] is not None:
            # Use seed from config
            self.seed(self.config["seed"])
            
        # Parse options if provided
        if options is not None:
            # Example: extract render mode from options
            if "render_mode" in options:
                self.renderMode = options["render_mode"]
        
        # Reset step counter
        self.stepCount = 0
        self.episodeCollisions = 0
        self.episodeResourcesCollected = {rtype: 0 for rtype in self.config["resourceTypes"]}
        
        # Reset environmental conditions
        self._resetEnvironmentalState()
        
        # Reset spatial grid
        self.grid = SpatialHashGrid(cellSize=1.0, worldSize=self.gridSize)
        
        # Reset agents
        self._setupAgents()
        
        # Generate obstacles
        self._generateObstacles()
        
        # Reset resources (do this after obstacles so resources don't overlap obstacles)
        self.resourceManager.reset(self.grid)
        
        # Reset last actions
        self.last_actions = {}
        
        # Get initial observations
        observations = self._getObservations()
        
        # Debug: verify observations match their spaces
        for agent_id, obs in observations.items():
            try:
                # Check if the observation is in the space
                if agent_id in self.observation_spaces:
                    if not self.observation_spaces[agent_id].contains(obs):
                        # Print detailed debug information
                        if isinstance(obs, dict):
                            for key, value in obs.items():
                                print(f"Obs key: {key}, shape: {value.shape}, dtype: {value.dtype}")
                                expected_shape = self.observation_spaces[agent_id].spaces[key].shape
                                print(f"Expected shape for {key}: {expected_shape}")
                        else:
                            print(f"Observation shape: {obs.shape}, dtype: {obs.dtype}")
                            expected_shape = self.observation_spaces[agent_id].shape
                            print(f"Expected shape: {expected_shape}")
                            
                        # Try to fix the observation if possible
                        if hasattr(self, "_fixObservation"):
                            print(f"Attempting to fix observation for {agent_id}")
                            observations[agent_id] = self._fixObservation(obs, agent_id)
                            # Verify fix worked
                            if not self.observation_spaces[agent_id].contains(observations[agent_id]):
                                print(f"Fix failed for {agent_id}. Still not matching observation space.")
                            else:
                                print(f"Successfully fixed observation for {agent_id}")
                        else:
                            print(f"No _fixObservation method available to fix mismatch")
                else:
                    print(f"Warning: agent_id {agent_id} not in observation_spaces")
            except Exception as e:
                print(f"Error checking observation for {agent_id}: {e}")
        
        # Create per-agent info dictionaries
        infos = {}
        baseInfo = self._get_info()
        for agentId in self.agents:
            infos[agentId] = baseInfo.get(agentId, {})
        
        return observations, infos
    
    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """
        Take a step in the environment with the given actions.
        
        Args:
            actions: Dictionary of actions for each agent
            
        Returns:
            Observations, rewards, terminated flags, truncated flags, and info dictionaries
        """
        # Store last actions for episode recording
        self.last_actions = actions.copy()
        
        # Validate actions (all agents must have an action)
        if not all(agent_id in actions for agent_id in self.agents):
            missing = set(self.agents) - set(actions.keys())
            raise ValueError(f"Missing actions for agents: {missing}")
        
        # Process actions: Move agents and collect resources
        for agent_id, action in actions.items():
            # Make sure agent_id is in the correct format
            # If it already starts with "agent_", use it directly
            if isinstance(agent_id, str) and agent_id.startswith("agent_"):
                agentIdStr = agent_id
                # Extract numeric id if needed for collection
                if agent_id[6:].isdigit():
                    agentIdNum = int(agent_id[6:])
                else:
                    agentIdNum = 0  # Default if not a number
            # If it's a digit string, convert to int and prefix
            elif isinstance(agent_id, str) and agent_id.isdigit():
                agentIdNum = int(agent_id)
                agentIdStr = f"agent_{agentIdNum}"
            # If it's an int, prefix with "agent_"
            elif isinstance(agent_id, int):
                agentIdNum = agent_id
                agentIdStr = f"agent_{agentIdNum}"
            else:
                # Unknown format, try using as-is
                agentIdStr = str(agent_id)
                agentIdNum = 0
            
            # Skip if agent doesn't exist
            if agentIdStr not in self.agents:
                print(f"Warning: Agent '{agentIdStr}' not found in environment. Available agents: {list(self.agents.keys())}")
                continue
            
            agent = self.agents[agentIdStr]
            
            # Get action as integer
            if isinstance(action, np.ndarray):
                action = action.item()
            
            # Process action
            if action < 4:  # Movement actions
                # Map action to direction: 0=up, 1=right, 2=down, 3=left
                if action == 0:  # Up
                    newPos = (agent.position[0], max(0, agent.position[1] - 1))
                elif action == 1:  # Right
                    newPos = (min(self.gridSize[0] - 1, agent.position[0] + 1), agent.position[1])
                elif action == 2:  # Down
                    newPos = (agent.position[0], min(self.gridSize[1] - 1, agent.position[1] + 1))
                elif action == 3:  # Left
                    newPos = (max(0, agent.position[0] - 1), agent.position[1])
                
                # Check for obstacles
                if newPos in self.obstacles:
                    # Cannot move to obstacle
                    continue
                
                # Check for collisions with other agents
                collision = False
                for other_id, other_agent in self.agents.items():
                    if other_id != agentIdStr and newPos == other_agent.position:
                        collision = True
                        agent.collisions += 1
                        other_agent.collisions += 1
                        break
                
                # If no collision, move agent
                if not collision:
                    self.grid.moveEntity(agentIdStr, agent.position, newPos)
                    agent.position = newPos
                    agent.distanceMoved += manhattanDistance(agent.position, newPos)
            
            elif action == 4:  # Collect resource
                # Try to collect resource at agent position
                self._collect_resource(agentIdNum)
        
        # Increment step counter
        self.stepCount += 1
        
        # Update day/night cycle
        if self.config["dayNightCycle"]:
            self._update_time_of_day()
        
        # Update weather
        if self.config["weatherEnabled"]:
            self._update_weather()
        
        # Update seasons
        if self.config["seasonEnabled"]:
            self._update_season()
        
        # Spawn new resources if needed
        self._spawn_resources()
        
        # Generate observations
        observations = self._getObservations()
        
        # Calculate rewards
        rewards = self._calculate_rewards(actions)
        
        # Calculate information dictionaries
        infos = self._get_info()
        
        # Check termination conditions (episode is truly finished)
        terminateds = {agent_id: False for agent_id in self.agents}
        # Add __all__ key for RLlib compatibility
        terminateds["__all__"] = False
        
        # Check truncation (episode ended due to time limit)
        isTimeLimitReached = self.stepCount >= self.maxSteps
        truncateds = {agent_id: isTimeLimitReached for agent_id in self.agents}
        # Add __all__ key for RLlib compatibility
        truncateds["__all__"] = isTimeLimitReached
        
        # Return observations, rewards, terminated flags, truncated flags, and info dictionaries
        return observations, rewards, terminateds, truncateds, infos
    
    def _collect_resource(self, agent_id: int):
        """Collect a resource at the specified agent's position."""
        agent = self.agents[f"agent_{agent_id}"]
        resourcesAtPos = self.resourceManager.getResourcesAtPosition(agent.position)
        
        if resourcesAtPos:
            resource = resourcesAtPos[0]  # Take the first resource
            
            # Attempt collection (might fail due to weather)
            if np.random.random() >= self._getWeatherEffect("collectionFailProb"):
                # Collect resource
                resource.collected = True
                
                # Update agent inventory
                if resource.resourceType not in agent.inventory:
                    agent.inventory[resource.resourceType] = 0
                
                agent.inventory[resource.resourceType] += 1
                agent.totalCollected += 1
                        
                # Update episode statistics
                self.episodeResourcesCollected[resource.resourceType] += 1

                # Remove from grid
                self.grid.removeEntity(resource.resourceId, resource.position)
        
    def _update_time_of_day(self):
        """Update the time of day and associated effects."""
        # Update time
        dayCycleLength = self.config["dayCycleLength"]
        self.timeOfDay = (self.timeOfDay + 1.0 / dayCycleLength) % 1.0
        
        # Determine current phase
        self.dayPhase = self._getDayPhase()
    
    def _getDayPhase(self) -> str:
        """Get the current phase of the day/night cycle."""
        # Get duration values from config or use defaults
        if self.config["seasonEnabled"] and self.currentSeason in SEASON_EFFECTS:
            # Get season-specific day/night cycle settings
            seasonData = SEASON_EFFECTS.get(self.currentSeason, {})
            dayNightConfig = seasonData.get("dayNightCycle", {})
        else:
            # Use default values if seasons not enabled
            dayNightConfig = {}
        
        # Get phase durations (with defaults if not specified)
        dayDuration = dayNightConfig.get("dayDuration", 0.4)
        dawnDuration = dayNightConfig.get("dawnDuration", 0.1)
        duskDuration = dayNightConfig.get("duskDuration", 0.1)
        nightDuration = dayNightConfig.get("nightDuration", 0.4)
        
        # Calculate phase boundaries
        dawnStart = 0.0
        dayStart = dawnStart + dawnDuration
        duskStart = dayStart + dayDuration
        nightStart = duskStart + duskDuration
        
        # Determine current phase
        if dawnStart <= self.timeOfDay < dayStart:
            return "dawn"
        elif dayStart <= self.timeOfDay < duskStart:
            return "day"
        elif duskStart <= self.timeOfDay < nightStart:
            return "dusk"
        else:
            return "night"
    
    def _update_weather(self):
        """Update the current weather condition."""
        # Check if weather should change
        if np.random.random() > self.config["weatherChangeProb"]:
            return  # Weather doesn't change this step
        
        # Get the available weather types from config
        weatherTypes = self.config["weatherTypes"]
        
        # Get base probabilities from WEATHER_EFFECTS
        baseProbs = {weather: WEATHER_EFFECTS[weather].get("probability", 0.0) 
                    for weather in weatherTypes if weather in WEATHER_EFFECTS}
        
        # Apply seasonal modifiers if seasons are enabled
        if self.config["seasonEnabled"] and self.currentSeason in SEASON_EFFECTS:
            seasonData = SEASON_EFFECTS[self.currentSeason]
            if "weatherTendencies" in seasonData:
                seasonTendencies = seasonData["weatherTendencies"]
                
                # Calculate adjusted probabilities by multiplying base with seasonal
                adjustedProbs = {}
                for weather in baseProbs:
                    if weather in seasonTendencies:
                        # Multiply base probability with seasonal tendency
                        adjustedProbs[weather] = baseProbs[weather] * seasonTendencies[weather]
                    else:
                        # Use a small default value if not specified for this season
                        adjustedProbs[weather] = baseProbs[weather] * 0.01
                
                # Normalise probabilities to sum to 1.0
                totalProb = sum(adjustedProbs.values())
                if totalProb > 0:
                    normalizedProbs = {w: p/totalProb for w, p in adjustedProbs.items()}
                    
                    # Extract weather types and their probabilities for np.random.choice
                    weatherChoices = list(normalizedProbs.keys())
                    probabilities = [normalizedProbs[w] for w in weatherChoices]
                    
                    # Choose new weather using normalised probabilities
                    self.currentWeather = np.random.choice(weatherChoices, p=probabilities)
                    return
        
        # Default fallback: use base probabilities directly
        weatherChoices = list(baseProbs.keys())
        probabilities = [baseProbs[w] for w in weatherChoices]
        
        # Normalise base probabilities
        totalProb = sum(probabilities)
        if totalProb > 0:
            probabilities = [p/totalProb for p in probabilities]
            self.currentWeather = np.random.choice(weatherChoices, p=probabilities)
        else:
            # If no valid probabilities, just choose randomly
            self.currentWeather = np.random.choice(weatherTypes)
    
    def _update_season(self):
        """Update the current season."""
        # Choose a new season
        seasonTypes = self.config["seasonTypes"]
        if len(seasonTypes) > 1:
            # Choose a different season than current
            otherSeasons = [s for s in seasonTypes if s != self.currentSeason]
            self.currentSeason = np.random.choice(otherSeasons)
    
    def _getWeatherEffect(self, effectName: str) -> float:
        """
        Get the value of a weather effect.
        
        Args:
            effectName: Name of the effect to retrieve
            
        Returns:
            Effect value (float)
        """
        if not self.config["weatherEnabled"]:
            return 0.0
        
        # Get base effect value
        baseEffect = WEATHER_EFFECTS.get(self.currentWeather, {}).get(effectName, 0.0)
        
        # Apply day/night modifier if applicable
        if self.config["dayNightCycle"]:
            dayNightModifier = DAY_NIGHT_PHASES.get(self.dayPhase, {}).get(effectName, 1.0)
            baseEffect *= dayNightModifier
        
        return baseEffect
    
    def _getSeasonEffect(self) -> Dict[str, float]:
        """
        Get the effect of the current season on resource availability.
        
        Returns:
            Dictionary mapping resource types to availability multipliers
        """
        if not self.config["seasonEnabled"]:
            return {rtype: 1.0 for rtype in self.config["resourceTypes"]}
        
        # Get base season effects
        seasonEffects = SEASON_EFFECTS.get(self.currentSeason, {}).get("resourceRespawnMultiplier", {})
        
        # Apply to all resource types
        return {
            rtype: seasonEffects.get(rtype, 1.0)
            for rtype in self.config["resourceTypes"]
        }
    
    def _getObservations(self) -> Dict[str, Union[Dict[str, np.ndarray], np.ndarray]]:
        """
        Get observations for all agents.
        
        Returns:
            Dictionary mapping agent IDs to observation dictionaries
        """
        observations = {}
        
        # Check if we should use flattened observations for better compatibility
        useFlattened = self.config.get("use_flattened_obs", False)
        
        for agentId, agent in self.agents.items():
            # Get grid observation (agent-centric view)
            gridObs = self._getGridObservation(agent)
            
            # Get agent state (position, inventory)
            agentState = self._getAgentState(agent)
            
            # Get environmental features
            envFeatures = self._getEnvironmentalFeatures()
            
            # Ensure all arrays are float32
            gridObs = gridObs.astype(np.float32)
            agentState = agentState.astype(np.float32)
            envFeatures = envFeatures.astype(np.float32)
            
            if useFlattened:
                # Flatten the grid observation
                flat_grid = gridObs.flatten()
                
                # Concatenate all components into a single flat vector
                flat_obs = np.concatenate([flat_grid, agentState, envFeatures])
                
                # Ensure the observation matches the expected shape
                if agentId in self.observation_spaces:
                    expected_shape = self.observation_spaces[agentId].shape
                    
                    # Reshape if needed
                    if flat_obs.shape != expected_shape:
                        # If shapes don't match but sizes are compatible, reshape
                        if np.prod(flat_obs.shape) == np.prod(expected_shape):
                            flat_obs = flat_obs.reshape(expected_shape)
                        else:
                            # If sizes are different, we need to pad or trim
                            print(f"Warning: Observation size mismatch - got {flat_obs.shape}, expected {expected_shape}")
                            
                            # Create a zero array of expected shape
                            padded_obs = np.zeros(expected_shape, dtype=np.float32)
                            
                            # Copy as much as possible
                            min_size = min(flat_obs.size, padded_obs.size)
                            padded_obs.flat[:min_size] = flat_obs.flat[:min_size]
                            flat_obs = padded_obs
                
                # Use the flat vector as the observation
                observations[agentId] = flat_obs
            else:
                # Use structured Dict observation
                structured_obs = {
                    "grid": gridObs,
                    "agent_state": agentState,
                    "env_features": envFeatures
                }
                
                # Ensure observation spaces match
                if agentId in self.observation_spaces:
                    obs_space = self.observation_spaces[agentId]
                    if isinstance(obs_space, spaces.Dict):
                        for key, space in obs_space.spaces.items():
                            if key in structured_obs:
                                value = structured_obs[key]
                                
                                # Check for shape mismatch
                                if value.shape != space.shape:
                                    print(f"Warning: Shape mismatch for {agentId}/{key}: got {value.shape}, expected {space.shape}")
                                    
                                    # Try to fix the shape
                                    if len(value.shape) == len(space.shape):
                                        # Create a zero array of expected shape and copy as much as possible
                                        fixed_value = np.zeros(space.shape, dtype=space.dtype)
                                        slice_indices = tuple(slice(0, min(s1, s2)) for s1, s2 in zip(value.shape, space.shape))
                                        fixed_value[slice_indices] = value[slice_indices]
                                        structured_obs[key] = fixed_value
                                    else:
                                        print(f"Error: Cannot fix dimension mismatch for {agentId}/{key}")
                
                observations[agentId] = structured_obs
        
        return observations
    
    def _getGridObservation(self, agent: Agent) -> np.ndarray:
        """
        Get the grid observation for an agent.
        
        Args:
            agent: Agent to get observation for
            
        Returns:
            Grid observation as numpy array
        """
        # Get observation radius and create empty grid
        obsRadius = self.config["observationRadius"]
        gridWidth = 2 * obsRadius + 1
        gridHeight = 2 * obsRadius + 1
        # Add an extra channel for obstacles
        numChannels = len(self.config["resourceTypes"]) + 2  # +1 for agents, +1 for obstacles
        
        observation = np.zeros((gridHeight, gridWidth, numChannels), dtype=np.float32)
        
        # Get agent's position
        agentX, agentY = agent.position
        
        # Calculate visibility based on time of day and weather
        baseVisibility = agent.viewRadius
        visibilityFactor = 1.0
        
        # Apply day/night cycle visibility factor
        if self.config["dayNightCycle"]:
            dayNightFactor = DAY_NIGHT_PHASES.get(self.dayPhase, {}).get("visibilityFactor", 1.0)
            visibilityFactor *= dayNightFactor
        
        # Apply weather visibility factor
        if self.config["weatherEnabled"]:
            weatherFactor = WEATHER_EFFECTS.get(self.currentWeather, {}).get("visibilityFactor", 1.0)
            visibilityFactor *= weatherFactor
        
        # Apply the combined visibility factor
        effectiveViewRadius = max(1, int(baseVisibility * visibilityFactor))
        
        # Fill in the observation grid
        for dy in range(-obsRadius, obsRadius + 1):
            for dx in range(-obsRadius, obsRadius + 1):
                worldX = agentX + dx
                worldY = agentY + dy
                
                # Check if position is within grid bounds
                if 0 <= worldX < self.gridSize[0] and 0 <= worldY < self.gridSize[1]:
                    # Check if position is within visibility radius
                    distance = manhattanDistance((agentX, agentY), (worldX, worldY))
                    
                    if distance <= effectiveViewRadius:
                        # Position is visible - fill in observations
                        gridX = dx + obsRadius
                        gridY = dy + obsRadius
                        
                        worldPos = (worldX, worldY)
                        
                        # Check for obstacles (in channel 1)
                        if worldPos in self.obstacles:
                            observation[gridY, gridX, 1] = 1.0
                        
                        # Check for agents (in channel 0)
                        agentsAtPos = [a for a in self.agents.values() if a.position == worldPos]
                        if agentsAtPos:
                            # Mark agent presence
                            observation[gridY, gridX, 0] = 1.0
                        
                        # Check for resources (starting from channel 2)
                        resources = self.resourceManager.getResourcesAtPosition(worldPos)
                        for resource in resources:
                            # Get resource type index
                            resourceTypeIndex = self.config["resourceTypes"].index(resource.resourceType)
                            
                            # Mark resource in the corresponding channel (offset by 2 for agents and obstacles)
                            observation[gridY, gridX, resourceTypeIndex + 2] = 1.0
        
        # Ensure correct dtype
        return observation.astype(np.float32)
    
    def _getAgentState(self, agent: Agent) -> np.ndarray:
        """
        Get the agent's state representation.
        
        Args:
            agent: Agent to get state for
            
        Returns:
            Agent state as numpy array
        """
        # Normalise position to [0, 1]
        normX = agent.position[0] / (self.gridSize[0] - 1)
        normY = agent.position[1] / (self.gridSize[1] - 1)
        
        # Create state with normalised position
        state = [normX, normY]
        
        # Add inventory counts
        for resourceType in self.config["resourceTypes"]:
            resourceCount = agent.inventory.get(resourceType, 0)
            state.append(resourceCount)
        
        return np.array(state, dtype=np.float32)
    
    def _getEnvironmentalFeatures(self) -> np.ndarray:
        """
        Get environmental features for observation.
        
        Returns:
            Environmental features as numpy array
        """
        features = []
        
        # Add time of day if day/night cycle is enabled
        if self.config["dayNightCycle"]:
            features.append(self.timeOfDay)
        
        # Add weather if enabled (one-hot encoding)
        if self.config["weatherEnabled"]:
            weatherTypes = self.config.get("weatherTypes", list(WEATHER_EFFECTS.keys()))
            for weatherType in weatherTypes:
                features.append(1.0 if self.currentWeather == weatherType else 0.0)
        
        # Add season if enabled (one-hot encoding)
        if self.config["seasonEnabled"]:
            seasonTypes = self.config.get("seasonTypes", list(SEASON_EFFECTS.keys()))
            for seasonType in seasonTypes:
                features.append(1.0 if self.currentSeason == seasonType else 0.0)
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_rewards(self, actions: Dict[str, int]) -> Dict[str, float]:
        """
        Calculate rewards for all agents.
            
        Returns:
            Dictionary mapping agent IDs to rewards
        """
        rewards = {}
        
        # Get reward parameters from config or use defaults
        collisionPenalty = self.config.get("collisionPenalty", 0.5)
        movementCost = self.config.get("movementCost", 0.01)
        
        # Calculate individual rewards
        individualRewards = {}
        for agentId, agent in self.agents.items():
            # Base reward - resource collection
            reward = 0.0
            
            # Check if agent collected resources this step
            for resourceType in self.config["resourceTypes"]:
                resourceValue = self.config["resourceValues"].get(resourceType, 1.0)
                resourceCollected = agent.inventory.get(resourceType, 0) - agent.previousInventory.get(resourceType, 0)
                reward += resourceCollected * resourceValue
            
            # Penalty for collisions
            collisionsDelta = agent.collisions - agent.previousCollisions
            reward -= collisionsDelta * collisionPenalty
            
            # Small penalty for movement to encourage efficiency
            if actions.get(agentId, 0) in (1, 2, 3, 4):  # Movement action
                reward -= movementCost
            
            # Update previous inventory and collisions for next step
            agent.previousInventory = agent.inventory.copy()
            agent.previousCollisions = agent.collisions
            
            # Store individual reward
            individualRewards[agentId] = reward
        
        # Calculate final rewards based on reward type
        if self.rewardType == "individual":
            # Each agent gets its own reward
            rewards = individualRewards
        
        elif self.rewardType == "shared":
            # All agents get the team's total reward
            teamReward = sum(individualRewards.values())
            rewards = {agentId: teamReward for agentId in self.agents}
        
        elif self.rewardType == "hybrid":
            # Mix of individual and shared rewards
            teamReward = sum(individualRewards.values())
            
            for agentId in self.agents:
                # Individual component + scaled team component
                rewards[agentId] = (1 - self.hybridRewardMix) * individualRewards[agentId] + \
                                  self.hybridRewardMix * teamReward
        
        else:
            # Default to individual rewards
            rewards = individualRewards
        
        return rewards
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Get info dictionary with additional environmental information.
        
        Returns:
            Info dictionary
        """
        info = {}
        
        # Add basic env info to all agent info dicts
        baseInfo = {
            "stepCount": self.stepCount,
            "totalResourcesCollected": sum(self.episodeResourcesCollected.values()),
            "totalCollisions": self.episodeCollisions,
        }
        
        # Add environmental conditions
        if self.config["dayNightCycle"]:
            baseInfo["timeOfDay"] = self.timeOfDay
            baseInfo["dayPhase"] = self.dayPhase
        
        if self.config["weatherEnabled"]:
            baseInfo["weather"] = self.currentWeather
        
        if self.config["seasonEnabled"]:
            baseInfo["season"] = self.currentSeason
        
        # Add fairness metrics
        metrics = self._calculateMetrics()
        for key, value in metrics.items():
            baseInfo[key] = value
        
        # Create info dict for each agent
        for agentId, agent in self.agents.items():
            info[agentId] = baseInfo.copy()
            
            # Add agent-specific info
            info[agentId]["position"] = agent.position
            info[agentId]["inventory"] = agent.inventory
            info[agentId]["collisions"] = agent.collisions
            info[agentId]["totalCollected"] = agent.totalCollected
        
        return info
    
    def _calculateMetrics(self) -> Dict[str, float]:
        """
        Calculate various metrics for the current state.
        
        Computes metrics for:
        - Fairness (Gini coefficient and Jain's index)
        - Resource collection efficiency
        - Collision rate
        - Resource specialisation
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Calculate Gini coefficient for resource distribution
        resourceCounts = [agent.totalCollected for agent in self.agents.values()]
        resourceCountsArray = np.array(resourceCounts)
        
        if sum(resourceCounts) > 0:
            # For fairness, we use 1 - Gini so higher values are better
            metrics["fairnessGini"] = 1.0 - calculateGiniCoefficient(resourceCountsArray)
            # Also calculate Jain's fairness index
            metrics["jainFairnessIndex"] = calculateJainFairnessIndex(resourceCountsArray)
        else:
            metrics["fairnessGini"] = 1.0  # Perfect equality if no resources collected
            metrics["jainFairnessIndex"] = 1.0
        
        # Calculate resource collection efficiency
        if self.stepCount > 0:
            metrics["resourcesPerStep"] = sum(resourceCounts) / self.stepCount
        else:
            metrics["resourcesPerStep"] = 0.0
        
        # Calculate collision rate
        if self.stepCount > 0:
            metrics["collisionsPerStep"] = self.episodeCollisions / self.stepCount
        else:
            metrics["collisionsPerStep"] = 0.0
        
        # Calculate resource specialisation if available
        if hasattr(self, "episodeResourcesCollected") and self.episodeResourcesCollected:
            # Track which agent collected each resource type
            resourceCollectionsByType = {}
            for agentId, agent in self.agents.items():
                for resType, amount in agent.inventory.items():
                    if resType not in resourceCollectionsByType:
                        resourceCollectionsByType[resType] = {}
                    resourceCollectionsByType[resType][agentId] = amount
            
            # Calculate specialisation as average Gini coefficient across resource types
            specialisationScores = []
            for resType, collections in resourceCollectionsByType.items():
                if collections:
                    values = np.array(list(collections.values()))
                    if np.sum(values) > 0:
                        specialisationScores.append(calculateGiniCoefficient(values))
            
            if specialisationScores:
                metrics["resourceSpecialisation"] = np.mean(specialisationScores)
            else:
                metrics["resourceSpecialisation"] = 0.0
        
        return metrics
    
    def render(self):
        """Render the environment."""
        if self.renderMode is None:
            return
        
        # We don't implement rendering here - it's handled by the PyQt6 visualiser
        # in renderDemo.py. This method just exists for compatibility.
        pass
    
    def _initRenderer(self):
        """initialise the renderer if needed."""
        # No-op: rendering is handled by the external PyQt6 visualiser
        pass
    
    def close(self):
        """Clean up resources used by the environment."""
        if self.renderer:
            self.renderer = None

    def get_agent_ids(self) -> Set[str]:
        """
        Return the agent IDs for this environment.
        
        Required by Ray MultiAgentEnv.
        
        Returns:
            Set of agent IDs
        """
        return set(self.agents.keys())

    def __str__(self) -> str:
        """
        Return a string representation of the environment.
        
        Returns:
            String representation
        """
        return f"ResourceCollectionEnv(numAgents={self.numAgents}, gridSize={self.gridSize})"

    def get_sub_environments(self):
        """
        Return a list of sub-environments, needed for Ray 2.42.0.
        
        For our single environment, we just return a list with self.
        
        Returns:
            List containing this environment
        """
        return [self]
        
    def get_observation_space(self, agent_id=None):
        """
        Get the observation space for the specified agent.
        
        Args:
            agent_id: Agent ID to get observation space for
        
        Returns:
            Observation space
        """
        if agent_id is not None and agent_id in self.observation_spaces:
            return self.observation_spaces[agent_id]
        return self.observation_space
        
    def get_action_space(self, agent_id=None):
        """
        Get the action space for the specified agent.
        
        Args:
            agent_id: Agent ID to get action space for
        
        Returns:
            Action space
        """
        if agent_id is not None and agent_id in self.action_spaces:
            return self.action_spaces[agent_id]
        return self.action_space
    
    def _fixObservation(self, obs: Dict[str, np.ndarray], agent_id: str) -> Dict[str, np.ndarray]:
        """
        Fix an observation to ensure it fits within the observation space.
        
        Args:
            obs: The current observation
            agent_id: The agent ID
            
        Returns:
            Fixed observation that matches the observation space
        """
        fixed_obs = {}
        obs_space = self.observation_spaces[agent_id]
        
        # Handle dict observations
        if isinstance(obs_space, spaces.Dict) and isinstance(obs, dict):
            # Fix each part of the observation
            for key, space in obs_space.spaces.items():
                if key in obs:
                    # Ensure correct dtype
                    value = obs[key].astype(space.dtype)
                
                # Ensure correct shape by padding or truncating
                if hasattr(space, "shape") and value.shape != space.shape:
                    if len(value.shape) == len(space.shape):
                        # Pad or truncate each dimension
                        padded_value = np.zeros(space.shape, dtype=space.dtype)
                        slice_indices = tuple(slice(0, min(s1, s2)) for s1, s2 in zip(value.shape, space.shape))
                        padded_value[slice_indices] = value[slice_indices]
                        value = padded_value
                    else:
                        # If dimensions don't match, recreate array with zeros
                        value = np.zeros(space.shape, dtype=space.dtype)
                
                # Ensure values are within bounds for Box spaces
                if isinstance(space, spaces.Box):
                    value = np.clip(value, space.low, space.high)
                
                fixed_obs[key] = value
            else:
                # If key is missing, use zeros with correct shape
                if isinstance(space, spaces.Box):
                    fixed_obs[key] = np.zeros(space.shape, dtype=space.dtype)
                elif isinstance(space, spaces.Discrete):
                        fixed_obs[key] = np.array(0, dtype=np.int32)
                else:
                    # For other space types, we'd need custom handling
                        fixed_obs[key] = np.zeros((1,), dtype=np.float32)
        
        # Handle flattened observations
        elif isinstance(obs_space, spaces.Box) and not isinstance(obs, dict):
            # Ensure correct dtype
            fixed_obs = obs.astype(obs_space.dtype)
            
            # Ensure correct shape
            if fixed_obs.shape != obs_space.shape:
                # Reshape if possible
                if np.prod(fixed_obs.shape) == np.prod(obs_space.shape):
                    fixed_obs = fixed_obs.reshape(obs_space.shape)
                else:
                    # If cannot reshape, create zeros and copy as much as possible
                    padded_obs = np.zeros(obs_space.shape, dtype=obs_space.dtype)
                    min_len = min(fixed_obs.size, padded_obs.size)
                    padded_obs.flat[:min_len] = fixed_obs.flat[:min_len]
                    fixed_obs = padded_obs
            
            # Clip to valid range
            fixed_obs = np.clip(fixed_obs, obs_space.low, obs_space.high)
        
        return fixed_obs 

    def _generateObstacles(self):
        """Generate obstacles on the grid."""
        if not self.obstaclesEnabled:
            return
        
        # Clear existing obstacles
        for pos in self.obstacles:
            self.grid.removeEntity(f"obstacle_{pos[0]}_{pos[1]}", pos)
        self.obstacles.clear()
        
        # Calculate number of obstacles to create
        totalCells = self.gridSize[0] * self.gridSize[1]
        numObstacles = int(totalCells * self.config["obstacleDensity"])
        
        # Exclude agent positions
        excludePositions = {agent.position for agent in self.agents.values()}
        
        # Generate positions for obstacles based on distribution
        obstacleDistribution = self.config["obstacleDistribution"]
        if obstacleDistribution == "clustered":
            numClusters = max(1, int(numObstacles / 10))  # Approx. 10 obstacles per cluster
            clusterRadius = 3
            
            obstaclePositions = generateClusteredPositions(
                count=numObstacles,
                gridSize=self.gridSize,
                numClusters=numClusters,
                clusterRadius=clusterRadius,
                excludePositions=excludePositions
            )
        elif obstacleDistribution == "mixed":
            # For mixed distribution, do 50% random and 50% clustered
            randomCount = numObstacles // 2
            clusteredCount = numObstacles - randomCount
            
            # Generate random positions
            randomPositions = generateRandomPositions(
                count=randomCount,
                gridSize=self.gridSize,
                excludePositions=excludePositions
            )
            
            # Generate clustered positions
            numClusters = max(1, int(clusteredCount / 10))
            clusterRadius = 3
            
            # Update excluded positions to include the random ones
            excludePositionsWithRandom = excludePositions.union(set(randomPositions))
            
            clusteredPositions = generateClusteredPositions(
                count=clusteredCount,
                gridSize=self.gridSize,
                numClusters=numClusters,
                clusterRadius=clusterRadius,
                excludePositions=excludePositionsWithRandom
            )
            
            # Combine both sets of positions
            obstaclePositions = randomPositions + clusteredPositions
        else:  # Default to "random"
            obstaclePositions = generateRandomPositions(
                count=numObstacles,
                gridSize=self.gridSize,
                excludePositions=excludePositions
            )
        
        # Place obstacles on the grid, ensuring no duplicates
        for pos in obstaclePositions:
            # Only add if position isn't already occupied by another obstacle
            if pos not in self.obstacles:
                obstacleId = f"obstacle_{pos[0]}_{pos[1]}"
                self.grid.addEntity(obstacleId, pos)
                self.obstacles.add(pos)
    
    def _spawn_resources(self):
        """Spawn new resources based on environmental conditions."""
        # Get seasonal effects on resource spawning
        seasonEffects = self._getSeasonEffect()
        
        # Get weather effects on spawn rate
        weatherSpawnModifier = 1.0
        if self.config["weatherEnabled"]:
            weatherSpawnModifier = self._getWeatherEffect("resourceSpawnModifier")
        
        # Get day/night effects on spawn rate
        dayNightModifier = 1.0
        if self.config["dayNightCycle"]:
            dayNightModifier = DAY_NIGHT_PHASES.get(self.dayPhase, {}).get("resourceSpawnModifier", 1.0)
        
        # Combine all modifiers
        for resourceType, baseMultiplier in seasonEffects.items():
            # Calculate final spawn probability for this resource type
            finalSpawnModifier = baseMultiplier * weatherSpawnModifier * dayNightModifier
            
            # Use resource manager to spawn resources with the calculated modifier
            self.resourceManager.spawnResources(
                resourceType=resourceType,
                spawnModifier=finalSpawnModifier,
                grid=self.grid,
                obstacles=self.obstacles,
                agentPositions=[agent.position for agent in self.agents.values()]
            ) 