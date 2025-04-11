"""
Entity classes for the ResourceCollectionEnv.
"""

from typing import Dict, List, Tuple, Set, Optional, Any
import numpy as np
from .utils import generateRandomPositions, generateClusteredPositions


class Agent:
    """
    Agent entity that can move around and collect resources.
    """
    
    def __init__(self, agentId: str, position: Tuple[int, int], viewRadius: int = 5):
        """
        Initialise an agent.
        
        Args:
            agentId: Unique identifier for the agent
            position: Initial position (x, y)
            viewRadius: How far the agent can see
        """
        self.agentId = agentId
        self.position = position
        self.viewRadius = viewRadius
        self.inventory: Dict[str, int] = {}  # Resources collected
        self.previousInventory: Dict[str, int] = {}  # Previous inventory for reward calculation
        self.totalCollected = 0
        self.collisions = 0
        self.previousCollisions = 0  # For tracking new collisions
        self.distanceMoved = 0  # Track total distance moved
    
    def move(self, newPosition: Tuple[int, int]):
        """
        Move the agent to a new position.
        
        Args:
            newPosition: Position to move to
        """
        self.position = newPosition
    
    def collectResource(self, resourceType: str, value: float):
        """
        Add a resource to the agent's inventory.
        
        Args:
            resourceType: Type of resource
            value: Value of the resource
        """
        if resourceType not in self.inventory:
            self.inventory[resourceType] = 0
        
        self.inventory[resourceType] += 1
        self.totalCollected += value
    
    def resetInventory(self):
        """Reset the agent's inventory."""
        self.inventory = {}
        # Note: We don't reset totalCollected to maintain stats across episodes


class Resource:
    """
    Resource entity that can be collected by agents.
    """
    
    def __init__(self, resourceId: str, resourceType: str, position: Tuple[int, int], value: float = 1.0):
        """
        Initialise a resource.
        
        Args:
            resourceId: Unique identifier for the resource
            resourceType: Type of resource (e.g., 'food', 'wood')
            position: Position on the grid
            value: Value of the resource when collected
        """
        self.resourceId = resourceId
        self.resourceType = resourceType
        self.position = position
        self.value = value
        self.collected = False
        self.respawnTimer = 0
    
    def collect(self):
        """Mark the resource as collected."""
        self.collected = True
    
    def respawn(self, newPosition: Tuple[int, int]):
        """
        Respawn the resource at a new position.
        
        Args:
            newPosition: Position to respawn at
        """
        self.position = newPosition
        self.collected = False
        self.respawnTimer = 0


class ResourceManager:
    """
    Manages all resources in the environment.
    """
    
    def __init__(self, gridSize: Tuple[int, int], resourceTypes: List[str], resourceValues: Dict[str, float], 
                 density: float = 0.1, distribution: str = "random", grid = None):
        """
        Initialise the resource manager.
        
        Args:
            gridSize: Size of the environment grid (width, height)
            resourceTypes: List of available resource types
            resourceValues: Dictionary mapping resource types to values
            density: Density of resources in the environment
            distribution: Distribution pattern ("random" or "clustered")
            grid: Spatial hash grid for resource tracking
        """
        self.gridSize = gridSize
        self.resourceTypes = resourceTypes
        self.resourceValues = resourceValues
        self.density = density
        self.distribution = distribution
        self.grid = grid
        self.resources: List[Resource] = []
        self.resourceIdCounter = 0
        
        # Create default respawn probabilities if needed
        self.respawnProbabilities = {rt: 0.01 for rt in resourceTypes}
        
        # Index resources by position for quick lookups
        self.positionToResources: Dict[Tuple[int, int], List[Resource]] = {}
    
    def reset(self, grid):
        """
        Reset the resource manager and generate new resources.
        
        Args:
            grid: Spatial hash grid to use for resource tracking
        """
        self.resources = []
        self.resourceIdCounter = 0
        self.positionToResources = {}
        self.grid = grid
        
        # Generate resources based on density and distribution
        self._generateResources()
    
    def _generateResources(self):
        """Generate resources based on configuration."""
        # Calculate number of resources to create
        totalCells = self.gridSize[0] * self.gridSize[1]
        numResources = int(totalCells * self.density)
        
        # Generate positions based on distribution
        if self.distribution == "clustered":
            # Clustered distribution - create clusters of resources
            numClusters = max(1, int(numResources / 10))  # Approximately a0 resources per cluster
            clusterRadius = 3
            
            positions = generateClusteredPositions(
                count=numResources, 
                gridSize=self.gridSize,
                numClusters=numClusters,
                clusterRadius=clusterRadius,
                excludePositions=[]
            )
        elif self.distribution == "mixed":
            # Mixed distribution - 50% random, 50% clustered
            randomCount = numResources // 2
            clusteredCount = numResources - randomCount
            
            # Generate random positions
            randomPositions = generateRandomPositions(
                count=randomCount,
                gridSize=self.gridSize,
                excludePositions=[]
            )
            
            # Generate clustered positions, excluding the random ones
            numClusters = max(1, int(clusteredCount / 10))
            clusterRadius = 3
            
            clusteredPositions = generateClusteredPositions(
                count=clusteredCount,
                gridSize=self.gridSize,
                numClusters=numClusters,
                clusterRadius=clusterRadius,
                excludePositions=randomPositions
            )
            
            # Combine both position sets
            positions = randomPositions + clusteredPositions
        else:
            # Default to random distribution - scatter resources randomly
            positions = generateRandomPositions(
                count=numResources,
                gridSize=self.gridSize,
                excludePositions=[]
            )
        
        # Create resources at the generated positions
        for i, pos in enumerate(positions):
            # Choose a random resource type
            resourceType = np.random.choice(self.resourceTypes)
            
            # Create the resource
            self.createResource(resourceType, pos)
    
    def createResource(self, resourceType: str, position: Tuple[int, int]) -> Resource:
        """
        Create a new resource and add it to the manager.
        
        Args:
            resourceType: Type of resource to create
            position: Position for the new resource
            
        Returns:
            The created resource
        """
        resourceId = f"resource_{self.resourceIdCounter}"
        self.resourceIdCounter += 1
        
        resource = Resource(
            resourceId, 
            resourceType, 
            position, 
            self.resourceValues.get(resourceType, 1.0)
        )
        
        self.resources.append(resource)
        
        # Add to position index
        if position not in self.positionToResources:
            self.positionToResources[position] = []
        self.positionToResources[position].append(resource)
        
        return resource
    
    def getResourcesAtPosition(self, position: Tuple[int, int]) -> List[Resource]:
        """
        Get all resources at a specific position.
        
        Args:
            position: Position to check
            
        Returns:
            List of resources at the position
        """
        return self.positionToResources.get(position, [])
    
    def removeResourceFromPosition(self, resource: Resource):
        """
        Remove a resource from the position index.
        
        Args:
            resource: Resource to remove
        """
        position = resource.position
        if position in self.positionToResources:
            if resource in self.positionToResources[position]:
                self.positionToResources[position].remove(resource)
            
            if not self.positionToResources[position]:
                del self.positionToResources[position]
    
    def updateRespawns(self, emptyPositions: List[Tuple[int, int]], rng: np.random.RandomState):
        """
        Update respawn timers and potentially respawn collected resources.
        
        Args:
            emptyPositions: List of empty positions available for respawning
            rng: Random number generator
        """
        if not emptyPositions:
            return  # No empty positions to respawn to
        
        for resource in self.resources:
            if resource.collected:
                # Check respawn probability for this resource type
                respawnProb = self.respawnProbabilities.get(resource.resourceType, 0.01)
                
                if rng.random() < respawnProb:
                    # Choose a random empty position
                    if emptyPositions:
                        newPositionIndex = rng.choice(len(emptyPositions))
                        newPosition = emptyPositions[newPositionIndex]
                        
                        # Remove this position from available positions
                        emptyPositions.pop(newPositionIndex)
                        
                        # Respawn the resource
                        self.removeResourceFromPosition(resource)
                        resource.respawn(newPosition)
                        
                        # Update position index
                        if newPosition not in self.positionToResources:
                            self.positionToResources[newPosition] = []
                        self.positionToResources[newPosition].append(resource)
    
    def regenerateResources(self, grid, rate: float = 0.01, seasonEffect: Dict[str, float] = None):
        """
        Regenerate resources that have been collected.
        
        Args:
            grid: Spatial hash grid for collision detection
            rate: Base regeneration rate
            seasonEffect: Season multipliers for regeneration rate
        """
        if seasonEffect is None:
            seasonEffect = {rt: 1.0 for rt in self.resourceTypes}
        
        # Find empty positions in the grid
        occupied = set()
        for resource in self.resources:
            if not resource.collected:
                occupied.add(resource.position)
        
        # Generate list of all grid positions
        allPositions = [(x, y) for x in range(self.gridSize[0]) for y in range(self.gridSize[1])]
        
        # Filter out occupied positions
        emptyPositions = [pos for pos in allPositions if pos not in occupied]
        
        # For each collected resource, try to regenerate
        for resource in self.resources:
            if resource.collected:
                # Apply season effect to regeneration rate
                adjustedRate = rate * seasonEffect.get(resource.resourceType, 1.0)
                
                # Attempt regeneration
                if np.random.random() < adjustedRate and emptyPositions:
                    # Choose a random empty position
                    posIndex = np.random.choice(len(emptyPositions))
                    newPos = emptyPositions.pop(posIndex)
                    
                    # Respawn resource
                    resource.respawn(newPos)
                    
                    # Update position index and grid
                    if self.grid:
                        self.grid.addEntity(resource.resourceId, newPos)
                    
                    if newPos not in self.positionToResources:
                        self.positionToResources[newPos] = []
                    self.positionToResources[newPos].append(resource)
    
    def spawnResources(self, resourceType: str, spawnModifier: float = 1.0, grid = None, obstacles: Set[Tuple[int, int]] = None, agentPositions: List[Tuple[int, int]] = None):
        """
        Spawn new resources of the specified type based on the spawn modifier.
        
        Args:
            resourceType: Type of resource to spawn
            spawnModifier: Modifier affecting spawn probability (1.0 = normal)
            grid: Spatial hash grid for collision detection
            obstacles: Set of obstacle positions to avoid
            agentPositions: List of agent positions to avoid
        """
        # Base spawn chance - adjust this based on your needs
        baseSpawnChance = 0.02
        
        # Apply modifier
        spawnChance = baseSpawnChance * spawnModifier
        
        # Only spawn if random check passes
        if np.random.random() > spawnChance:
            return
        
        # Collect positions to exclude
        excludePositions = set()
        
        # Add existing resources
        for resource in self.resources:
            if not resource.collected:
                excludePositions.add(resource.position)
        
        # Add obstacles
        if obstacles:
            excludePositions.update(obstacles)
        
        # Add agent positions
        if agentPositions:
            excludePositions.update(agentPositions)
        
        # Generate possible positions
        allPositions = [(x, y) for x in range(self.gridSize[0]) for y in range(self.gridSize[1])]
        availablePositions = [pos for pos in allPositions if pos not in excludePositions]
        
        # If no positions available, return
        if not availablePositions:
            return
        
        # Choose a random position
        position = availablePositions[np.random.choice(len(availablePositions))]
        
        # Create the resource
        resource = self.createResource(resourceType, position)
        
        # Add to grid if provided
        if grid:
            grid.addEntity(resource.resourceId, position) 