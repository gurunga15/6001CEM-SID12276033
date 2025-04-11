"""
PyQt6-based visualiser for the resource collection environment.
"""
import os
import sys
from typing import Dict, Any, List, Optional, Union

# Add the project root to the path
scriptDir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(scriptDir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Get the parent directory (resourceProjectMARL)
project_dir = os.path.dirname(scriptDir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# Import required modules
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsRectItem,
    QGraphicsEllipseItem, QGraphicsTextItem, QGraphicsLineItem, QVBoxLayout, 
    QHBoxLayout, QWidget, QPushButton, QLabel, QSlider
)
from PyQt6.QtGui import QBrush, QPen, QColor, QFont
from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF

# Simple colour definitions right in this file to avoid unnecessary dependencies
DEFAULT_CELL_SIZE = 30
DEFAULT_STEP_DELAY = 200  # milliseconds

# Agent colour palette
AGENT_COLOURS = [
    QColor(255, 0, 0),      # Red
    QColor(0, 0, 255),      # Blue
    QColor(0, 128, 0),      # Green
    QColor(255, 165, 0),    # Orange
    QColor(128, 0, 128),    # Purple
    QColor(0, 128, 128),    # Teal
    QColor(255, 192, 203),  # Pink
    QColor(255, 255, 0),    # Yellow
    QColor(165, 42, 42),    # Brown
    QColor(0, 255, 255),    # Cyan
]

# Resource colour palette
RESOURCE_COLOURS = {
    "food": QColor(0, 255, 0),      # Green
    "wood": QColor(139, 69, 19),    # Brown
    "stone": QColor(128, 128, 128), # Gray
    "water": QColor(0, 191, 255),   # Deep sky blue
    "gold": QColor(255, 215, 0),    # Gold
}


class ResourceCollectionVisualiser(QMainWindow):
    """
    PyQt6-based visualiser for the resource collection environment.
    """
    
    def __init__(self, env, cellSize: int = DEFAULT_CELL_SIZE):
        """
        Initialise the visualiser.
        
        Args:
            env: ResourceCollectionEnv instance
            cellSize: Size of each grid cell in pixels
        """
        super().__init__()
        
        self.env = env
        self.cellSize = cellSize
        self.gridSize = env.config.get("gridSize", (20, 20))
        self.numAgents = env.config.get("numAgents", 4)
        
        # Colour settings
        self.agentColours = AGENT_COLOURS
        self.resourceColours = RESOURCE_COLOURS
        
        # Create UI elements
        self.setupUI()
        
        # Set up timer for updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateSimulation)
        
        # Simulation state
        self.running = False
        self.stepDelay = DEFAULT_STEP_DELAY  # milliseconds
        self.stepCount = 0
        self.episodeCount = 0
        self.done = False
        self.trainer = None
        self.obs = None
        self.info = {}
        self.rewards = {}
        self.algorithm = ""
        
        # Initial state
        self.setWindowTitle("Resource Collection Environment Visualiser")
        self.updateStatusBar()
    
    def setupUI(self):
        """Set up the user interface with the graphics view."""
        # Main widget and layout
        mainWidget = QWidget()
        self.setCentralWidget(mainWidget)
        mainLayout = QVBoxLayout(mainWidget)
        
        # Create graphics view
        self.graphicsView = QGraphicsView()
        self.graphicsView.setRenderHint(0)  # No antialiasing for crisp grid
        self.graphicsView.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)
        mainLayout.addWidget(self.graphicsView)
        
        # Control panel
        controlPanel = QWidget()
        controlLayout = QHBoxLayout(controlPanel)
        
        # Step button
        self.stepButton = QPushButton("Step")
        self.stepButton.clicked.connect(self.step)
        controlLayout.addWidget(self.stepButton)
        
        # Play/Pause button
        self.playButton = QPushButton("Play")
        self.playButton.clicked.connect(self.togglePlay)
        controlLayout.addWidget(self.playButton)
        
        # Reset button
        self.resetButton = QPushButton("Reset")
        self.resetButton.clicked.connect(self.reset)
        controlLayout.addWidget(self.resetButton)
        
        # Speed slider
        speedLabel = QLabel("Speed:")
        controlLayout.addWidget(speedLabel)
        
        self.speedSlider = QSlider(Qt.Orientation.Horizontal)
        self.speedSlider.setMinimum(1)
        self.speedSlider.setMaximum(10)
        self.speedSlider.setValue(5)
        self.speedSlider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.speedSlider.setTickInterval(1)
        self.speedSlider.valueChanged.connect(self.adjustSpeed)
        controlLayout.addWidget(self.speedSlider)
        
        # Status panel
        self.statusLabel = QLabel("Step: 0 | Episode: 0")
        controlLayout.addWidget(self.statusLabel)
        
        mainLayout.addWidget(controlPanel)
        
        # Set up the graphics scene size
        width = self.gridSize[0] * self.cellSize
        height = self.gridSize[1] * self.cellSize
        self.scene.setSceneRect(0, 0, width, height)
        
        # Calculate window size (with some padding)
        self.resize(width + 50, height + 100)
    
    def reset(self):
        """Reset the environment and the visualisation."""
        self.stepCount = 0
        self.episodeCount += 1
        self.done = False
        self.obs, _ = self.env.reset()
        self.updateStatusBar()
        self.drawEnvironment()
    
    def step(self):
        """Take a single step in the environment using the trained policy."""
        if self.done:
            self.reset()
            return
        
        if self.trainer is None:
            print("No trainer loaded! Using random actions.")
            # Create random actions
            actions = {}
            for agentId in self.obs.keys():
                action_space = self.env.action_spaces[agentId]
                actions[agentId] = action_space.sample()
        else:
            # Get actions from policy
            actions = {}
            for agentId, agentObs in self.obs.items():
                action = self.trainer.compute_single_action(
                    observation=agentObs,
                    policy_id=agentId if self.algorithm != "ppo" else None,
                )
                actions[agentId] = action
        
        # Step the environment
        self.obs, self.rewards, terminateds, truncateds, self.info = self.env.step(actions)
        
        # Check if episode is done
        self.done = terminateds["__all__"] or truncateds["__all__"]
        
        # Update step count
        self.stepCount += 1
        
        # Update the visualisation
        self.drawEnvironment()
        self.updateStatusBar()
        
        # Auto-reset when done if in play mode
        if self.done and self.running:
            QTimer.singleShot(1000, self.reset)  # Reset after 1 second delay
    
    def drawEnvironment(self):
        """Draw the current state of the environment."""
        # Clear existing items
        self.scene.clear()
        
        # Get world size
        width, height = self.gridSize
        
        # Draw grid
        pen = QPen(QColor(200, 200, 200))
        for x in range(width + 1):
            line = QGraphicsLineItem(x * self.cellSize, 0, x * self.cellSize, height * self.cellSize)
            line.setPen(pen)
            self.scene.addItem(line)
        
        for y in range(height + 1):
            line = QGraphicsLineItem(0, y * self.cellSize, width * self.cellSize, y * self.cellSize)
            line.setPen(pen)
            self.scene.addItem(line)
        
        # Draw agents
        if hasattr(self.env, 'agents'):
            for i, (agentId, agent) in enumerate(self.env.agents.items()):
                colour = self.agentColours[i % len(self.agentColours)]
                self.drawAgent(agent.position, colour, agentId)
        
        # Draw resources
        if hasattr(self.env, 'resourceManager'):
            for resource in self.env.resourceManager.resources:
                if not resource.collected:
                    colour = self.resourceColours.get(resource.resourceType, QColor(100, 100, 100))
                    self.drawResource(resource.position, colour, resource.resourceType, resource.value)
        
        # Draw environmental indicators
        if hasattr(self.env, 'timeOfDay'):
            self.drawTimeOfDay(self.env.timeOfDay)
        
        if hasattr(self.env, 'currentWeather'):
            self.drawWeather(self.env.currentWeather)
        
        if hasattr(self.env, 'currentSeason'):
            self.drawSeason(self.env.currentSeason)
    
    def drawAgent(self, position, colour, agentId):
        """Draw an agent at the specified position."""
        x, y = position
        size = self.cellSize * 0.8  # Slightly smaller than cell
        offset = (self.cellSize - size) / 2
        
        ellipse = QGraphicsEllipseItem(
            x * self.cellSize + offset, 
            y * self.cellSize + offset, 
            size, size
        )
        ellipse.setBrush(QBrush(colour))
        ellipse.setPen(QPen(Qt.GlobalColor.black, 2))
        self.scene.addItem(ellipse)
        
        # Add agent ID label
        text = QGraphicsTextItem(agentId)
        text.setFont(QFont("Arial", 8))
        text.setPos(
            x * self.cellSize + self.cellSize/4,
            y * self.cellSize + self.cellSize/4
        )
        self.scene.addItem(text)
    
    def drawResource(self, position, colour, resourceType, value):
        """Draw a resource at the specified position."""
        x, y = position
        size = self.cellSize * 0.7  # Smaller than cell
        offset = (self.cellSize - size) / 2
        
        rect = QGraphicsRectItem(
            x * self.cellSize + offset, 
            y * self.cellSize + offset, 
            size, size
        )
        rect.setBrush(QBrush(colour))
        rect.setPen(QPen(Qt.GlobalColor.black, 1))
        self.scene.addItem(rect)
        
        # Add resource type label
        text = QGraphicsTextItem(resourceType[0].upper())  # First letter of type
        text.setFont(QFont("Arial", 8))
        text.setPos(
            x * self.cellSize + self.cellSize/3,
            y * self.cellSize + self.cellSize/3
        )
        self.scene.addItem(text)
    
    def drawTimeOfDay(self, timeOfDay):
        """Draw an indicator for time of day."""
        # Draw a sun/moon indicator in the top-right corner
        isDaytime = 0.25 <= timeOfDay < 0.75  # Day is between 0.25 and 0.75
        
        x = self.gridSize[0] * self.cellSize - 60
        y = 10
        size = 30
        
        if isDaytime:
            # Draw sun
            sunColour = QColor(255, 255, 0)  # Yellow
            ellipse = QGraphicsEllipseItem(x, y, size, size)
            ellipse.setBrush(QBrush(sunColour))
            ellipse.setPen(QPen(Qt.GlobalColor.black, 1))
            self.scene.addItem(ellipse)
        else:
            # Draw moon
            moonColour = QColor(200, 200, 200)  # Light gray
            ellipse = QGraphicsEllipseItem(x, y, size, size)
            ellipse.setBrush(QBrush(moonColour))
            ellipse.setPen(QPen(Qt.GlobalColor.black, 1))
            self.scene.addItem(ellipse)
        
        # Add text showing exact time
        timeText = QGraphicsTextItem(f"Time: {timeOfDay:.2f}")
        timeText.setFont(QFont("Arial", 8))
        timeText.setPos(x - 40, y + size + 5)
        self.scene.addItem(timeText)
    
    def drawWeather(self, weather):
        """Draw an indicator for current weather."""
        # Draw weather indicator in the top-left corner
        x = 10
        y = 10
        
        weatherText = QGraphicsTextItem(f"Weather: {weather}")
        weatherText.setFont(QFont("Arial", 10))
        weatherText.setPos(x, y)
        self.scene.addItem(weatherText)
    
    def drawSeason(self, season):
        """Draw an indicator for current season."""
        # Draw season indicator in the top-left corner
        x = 10
        y = 30
        
        seasonText = QGraphicsTextItem(f"Season: {season}")
        seasonText.setFont(QFont("Arial", 10))
        seasonText.setPos(x, y)
        self.scene.addItem(seasonText)
    
    def updateStatusBar(self):
        """Update the status bar with current information."""
        totalReward = sum(self.rewards.values()) if self.rewards else 0
        statusText = (
            f"Step: {self.stepCount} | Episode: {self.episodeCount} | "
            f"Algorithm: {self.algorithm} | Reward: {totalReward:.2f}"
        )
        
        if hasattr(self.env, '_calculateMetrics'):
            metrics = self.env._calculateMetrics()
            if metrics:
                fairness = metrics.get("fairnessGini", 0.0)
                statusText += f" | Fairness: {fairness:.4f}"
        
        self.statusLabel.setText(statusText)
    
    def togglePlay(self):
        """Toggle between playing and pausing the simulation."""
        self.running = not self.running
        
        if self.running:
            self.playButton.setText("Pause")
            self.timer.start(self.stepDelay)
        else:
            self.playButton.setText("Play")
            self.timer.stop()
    
    def adjustSpeed(self, value):
        """Adjust the simulation speed based on slider value."""
        # Inverse relationship: higher value = lower delay = faster simulation
        self.stepDelay = int(1000 / value)
        
        if self.running:
            self.timer.stop()
            self.timer.start(self.stepDelay)
    
    def updateSimulation(self):
        """Update the simulation when the timer fires."""
        if self.running:
            self.step()
    
    def setTrainer(self, trainer, algorithm=""):
        """Set the trainer for policy-based actions."""
        self.trainer = trainer
        self.algorithm = algorithm
    
    def loadReplay(self, replayFile):
        """
        Load a recorded episode for replay.
        
        Args:
            replayFile: Path to the pickle file containing episode recording
        """
        import pickle
        
        try:
            with open(replayFile, "rb") as f:
                self.replayData = pickle.load(f)
            
            # Set replay mode
            self.replayMode = True
            self.replayStep = 0
            self.replayLength = len(self.replayData["actions"])
            
            # Reset environment with the same config if available
            if "env_config" in self.replayData["metadata"]:
                # Backup trainer
                oldTrainer = self.trainer
                
                # Reset environment with replay config
                self.env.reset()
                
                # Restore trainer
                self.trainer = oldTrainer
            else:
                # Just reset with current config
                self.reset()
            
            # Update status
            self.episodeCount = 0
            self.stepCount = 0
            self.updateStatusBar()
            
            # Update window title
            self.setWindowTitle(f"Replay - Episode {self.replayData['metadata'].get('episode_id', 'Unknown')}")
            
            # Update UI for replay mode
            self.playButton.setText("Play Replay")
            
            # Draw initial state
            self.drawEnvironment()
            
            print(f"Loaded replay with {self.replayLength} steps")
            
            return True
        except Exception as e:
            print(f"Error loading replay: {e}")
            self.replayMode = False
            return False
    
    def replayStep(self):
        """Take a single step in replay mode."""
        if not hasattr(self, 'replayMode') or not self.replayMode:
            self.step()  # Fall back to normal step
            return
        
        if self.replayStep >= self.replayLength:
            # End of replay
            self.playButton.setText("Replay Complete")
            self.timer.stop()
            self.running = False
            return
        
        # Get data for this replay step
        actions = self.replayData["actions"][self.replayStep]
        rewards = self.replayData["rewards"][self.replayStep]
        
        # Step the environment with recorded actions
        self.obs, self.rewards, terminateds, truncateds, self.info = self.env.step(actions)
        
        # Update step count
        self.stepCount += 1
        self.replayStep += 1
        
        # Update the visualisation
        self.drawEnvironment()
        self.updateStatusBar()
        
        # Update replay progress in status
        self.statusLabel.setText(
            f"{self.statusLabel.text()} | Replay: {self.replayStep}/{self.replayLength} "
            f"({self.replayStep/self.replayLength*100:.1f}%)"
        )
        
        # Check if replay is complete
        if self.replayStep >= self.replayLength or terminateds.get("__all__", False):
            self.playButton.setText("Replay Complete")
            self.timer.stop()
            self.running = False
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Ensure the timer is stopped when the window is closed
        self.timer.stop()
        event.accept() 