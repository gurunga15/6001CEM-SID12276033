"""
Utility functions for the ResourceCollectionEnv.
"""

from typing import Dict, List, Tuple, Set, Optional, Any
import numpy as np
import math


class SpatialHashGrid:
    """
    Spatial hash grid for efficient collision detection and nearest-neighbour queries.
    """
    
    def __init__(self, cellSize: float, worldSize: Tuple[int, int]):
        """
        Initialise the spatial hash grid.
        
        Args:
            cellSize: Size of each grid cell for hashing
            worldSize: Size of the world in (width, height)
        """
        self.cellSize = cellSize
        self.worldSize = worldSize
        
        # Dictionary mapping cell coordinates to set of entity IDs in that cell
        self.entities: Dict[Tuple[int, int], Set[str]] = {}
        
        # Dictionary mapping entity IDs to their positions
        self.entityPositions: Dict[str, Tuple[int, int]] = {}
        
        # Create grid representation for faster lookups (for rendering and observation)
        # Channels: 0 = agents, 1 = obstacles, 2+ = resource types
        self.grid = np.zeros((worldSize[0], worldSize[1], 10), dtype=np.float32)
        
    def hashPosition(self, position: Tuple[int, int]) -> Tuple[int, int]:
        """
        Convert world position to grid cell coordinates.
        
        Args:
            position: Position to hash
            
        Returns:
            Grid cell coordinates
        """
        return (int(position[0] // self.cellSize), int(position[1] // self.cellSize))
    
    def addEntity(self, entityId: str, position: Tuple[int, int], entityType: str = "default"):
        """
        Add an entity to the spatial hash.
        
        Args:
            entityId: Unique identifier for the entity
            position: Position of the entity
            entityType: Type of entity (e.g., "agent", "resource")
        """
        cellCoord = self.hashPosition(position)
        if cellCoord not in self.entities:
            self.entities[cellCoord] = set()
        self.entities[cellCoord].add(entityId)
        
        # Store entity position
        self.entityPositions[entityId] = position
    
    def removeEntity(self, entityId: str, position: Tuple[int, int]):
        """
        Remove an entity from the spatial hash.
        
        Args:
            entityId: Unique identifier for the entity
            position: Position of the entity
        """
        cellCoord = self.hashPosition(position)
        if cellCoord in self.entities and entityId in self.entities[cellCoord]:
            self.entities[cellCoord].remove(entityId)
            
            # If cell is empty, remove it
            if not self.entities[cellCoord]:
                del self.entities[cellCoord]
        
        # Remove from entity positions
        if entityId in self.entityPositions:
            del self.entityPositions[entityId]
    
    def moveEntity(self, entityId: str, oldPosition: Tuple[int, int], newPosition: Tuple[int, int]):
        """
        Move an entity to a new position in the spatial hash.
        
        Args:
            entityId: Unique identifier for the entity
            oldPosition: Previous position of the entity
            newPosition: New position of the entity
        """
        # Remove from old position
        oldCellCoord = self.hashPosition(oldPosition)
        if oldCellCoord in self.entities and entityId in self.entities[oldCellCoord]:
            self.entities[oldCellCoord].remove(entityId)
            
            # If cell is empty, remove it
            if not self.entities[oldCellCoord]:
                del self.entities[oldCellCoord]
        
        # Add to new position
        newCellCoord = self.hashPosition(newPosition)
        if newCellCoord not in self.entities:
            self.entities[newCellCoord] = set()
        self.entities[newCellCoord].add(entityId)
        
        # Update entity position
        self.entityPositions[entityId] = newPosition
    
    def getEntitiesAtPosition(self, position: Tuple[int, int], entityType: str = None) -> List[str]:
        """
        Get all entities at a specific position.
        
        Args:
            position: Position to check
            entityType: Optional filter for entity type
            
        Returns:
            List of entity IDs at the position
        """
        cellCoord = self.hashPosition(position)
        if cellCoord not in self.entities:
            return []
        
        # Get all entities in this cell
        entitiesInCell = list(self.entities[cellCoord])
        
        # Filter by type if needed (no filtering implemented yet)
        # In a more complete implementation, we would store entity types
        return entitiesInCell
    
    def clear(self):
        """Clear the spatial hash grid."""
        self.entities.clear()
        self.entityPositions.clear()
        self.grid = np.zeros((self.worldSize[0], self.worldSize[1], 10), dtype=np.float32)


def manhattanDistance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """
    Calculate the Manhattan distance between two positions.
    
    Args:
        pos1: First position
        pos2: Second position
        
    Returns:
        Manhattan distance
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def euclideanDistance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """
    Calculate the Euclidean distance between two positions.
    
    Args:
        pos1: First position
        pos2: Second position
        
    Returns:
        Euclidean distance
    """
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def generateRandomPositions(
    count: int, 
    gridSize: Tuple[int, int], 
    excludePositions: List[Tuple[int, int]] = None,
    rng: Optional[np.random.RandomState] = None
) -> List[Tuple[int, int]]:
    """
    Generate random positions on a grid.
    
    Args:
        count: Number of positions to generate
        gridSize: Size of the grid (width, height)
        excludePositions: Positions to avoid
        rng: Random number generator
        
    Returns:
        List of randomly generated positions
    """
    if rng is None:
        rng = np.random.RandomState()
    
    if excludePositions is None:
        excludePositions = []
    
    width, height = gridSize
    positions = []
    
    excludeSet = set(excludePositions)
    allCells = [(x, y) for x in range(width) for y in range(height)]
    availableCells = [cell for cell in allCells if cell not in excludeSet]
    
    if len(availableCells) < count:
        count = len(availableCells)  # Can't have more positions than available cells
    
    # Sample without replacement
    indices = rng.choice(len(availableCells), count, replace=False)
    positions = [availableCells[i] for i in indices]
    
    return positions


def generateClusteredPositions(
    count: int,
    gridSize: Tuple[int, int],
    numClusters: int,
    clusterRadius: int,
    excludePositions: List[Tuple[int, int]] = None,
    rng: Optional[np.random.RandomState] = None
) -> List[Tuple[int, int]]:
    """
    Generate positions clustered around several centers.
    
    Args:
        count: Number of positions to generate
        gridSize: Size of the grid (width, height)
        numClusters: Number of cluster centers
        clusterRadius: Radius of each cluster
        excludePositions: Positions to avoid
        rng: Random number generator
        
    Returns:
        List of clustered positions
    """
    if rng is None:
        rng = np.random.RandomState()
    
    if excludePositions is None:
        excludePositions = []
    
    width, height = gridSize
    positions = []
    excludeSet = set(excludePositions)
    
    # Generate cluster centres, ensuring they're not in excluded positions
    clusterCentres = []
    while len(clusterCentres) < numClusters:
        x = rng.integers(0, width)
        y = rng.integers(0, height)
        if (x, y) not in excludeSet and (x, y) not in clusterCentres:
            clusterCentres.append((x, y))
    
    # Distribute positions among clusters
    posPerCluster = count // numClusters
    remainingPos = count % numClusters
    
    for i, centre in enumerate(clusterCentres):
        clusterCount = posPerCluster + (1 if i < remainingPos else 0)
        
        # Try to place positions around this cluster centre
        for _ in range(clusterCount):
            attempts = 0
            while attempts < 50:  # Prevent infinite loops
                # Generate position within cluster radius
                dx = rng.integers(-clusterRadius, clusterRadius + 1)
                dy = rng.integers(-clusterRadius, clusterRadius + 1)
                
                x = centre[0] + dx
                y = centre[1] + dy
                
                # Check if position is valid
                if (0 <= x < width and 0 <= y < height and 
                    (x, y) not in excludeSet and 
                    (x, y) not in positions):
                    positions.append((x, y))
                    break
                
                attempts += 1
    
    # If we couldn't generate enough positions with clusters, fill in randomly
    if len(positions) < count:
        remainingCount = count - len(positions)
        remainingPositions = generateRandomPositions(
            remainingCount, gridSize, 
            excludePositions + positions, 
            rng
        )
        positions.extend(remainingPositions)
    
    return positions


def calculateGiniCoefficient(values: np.ndarray) -> float:
    """
    Calculate the Gini coefficient for fairness evaluation.
    
    The Gini coefficient measures inequality in a distribution.
    0 = perfect equality, 1 = perfect inequality.
    
    Args:
        values: Array of values (e.g., resources collected by each agent)
        
    Returns:
        Gini coefficient
    """
    # If all values are 0 or there's only one value, return 0 (perfect equality)
    if np.all(values == 0) or len(values) <= 1:
        return 0.0
    
    # Sort values
    sortedValues = np.sort(values)
    n = len(sortedValues)
    
    # Calculate Gini coefficient
    indices = np.arange(1, n + 1)
    return (2 * np.sum(indices * sortedValues) / (n * np.sum(sortedValues))) - (n + 1) / n


def calculateJainFairnessIndex(values: np.ndarray) -> float:
    """
    Calculate Jain's fairness index for fairness evaluation.
    
    Jain's fairness index ranges from 1/n (worst case) to 1 (best case),
    where n is the number of agents. A value of 1 means all agents received
    equal resources.
    
    Args:
        values: Array of values (e.g., resources collected by each agent)
        
    Returns:
        Jain's fairness index (1/n to 1)
    """
    # Handle edge cases
    if len(values) <= 1 or np.sum(values) == 0:
        return 1.0
    
    n = len(values)
    squaredSum = np.sum(values) ** 2
    sumOfSquares = np.sum(values ** 2)
    
    # Jain's fairness index formula
    return squaredSum / (n * sumOfSquares) if sumOfSquares > 0 else 1.0 