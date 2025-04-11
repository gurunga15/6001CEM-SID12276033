#!/usr/bin/env python

"""
UI launcher for PPO training on the resource collection environment.
This provides a graphical interface to set training parameters and launch training.
"""

import os
import sys
import subprocess
import random
from typing import Dict, Any, Optional, List, Tuple

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox, 
    QCheckBox, QComboBox, QGroupBox, QFileDialog, QGridLayout,
    QTabWidget, QFormLayout, QSlider, QMessageBox
)
from PyQt6.QtCore import Qt, QProcess, QTimer

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import default environment configuration
from env.config import DEFAULT_ENV_CONFIG


class TrainingLauncher(QMainWindow):
    """Main window for training launcher."""
    
    def __init__(self):
        """Initialise the UI."""
        super().__init__()
        
        self.setWindowTitle("Resource Collection MARL Training")
        self.setMinimumSize(600, 700)
        
        # Create central widget and main layout
        container = QWidget()
        self.setCentralWidget(container)
        self.mainLayout = QVBoxLayout(container)
        
        # Create tab widget
        self.tabWidget = QTabWidget()
        self.mainLayout.addWidget(self.tabWidget)
        
        # Create tabs
        self.createBasicTab()
        self.createAdvancedTab()
        self.createModelTab()
        
        # Create buttons
        self.createButtons()
        
        # Status label
        self.statusLabel = QLabel("Ready")
        self.mainLayout.addWidget(self.statusLabel)
        
        # Set up process
        self.process = QProcess()
        self.process.readyReadStandardOutput.connect(self.handleOutput)
        self.process.readyReadStandardError.connect(self.handleError)
        self.process.finished.connect(self.handleFinished)
        
        # Set the window properties
        self.setWindowTitle("MARL Resource Collection Trainer")
        self.resize(800, 600)
        
        # Initialise loaded config
        self.loadedConfig = None
        
        # Validate workers after initialisation
        QTimer.singleShot(0, self.validateWorkers)
    
    def createBasicTab(self):
        """Create the basic configuration tab."""
        basicTab = QWidget()
        layout = QFormLayout(basicTab)
        
        # Output directory
        self.outputDirField = QLineEdit("results/default")
        outputDirLayout = QHBoxLayout()
        outputDirLayout.addWidget(self.outputDirField)
        outputDirButton = QPushButton("Browse...")
        outputDirButton.clicked.connect(self.browseOutputDir)
        outputDirLayout.addWidget(outputDirButton)
        layout.addRow("Output Directory:", outputDirLayout)
        
        # Iterations
        self.iterationsField = QSpinBox()
        self.iterationsField.setRange(1, 100000)
        self.iterationsField.setValue(1000)
        layout.addRow("Iterations:", self.iterationsField)
        
        # Number of agents - get default from environment config
        self.numAgentsField = QSpinBox()
        self.numAgentsField.setRange(1, 10)
        self.numAgentsField.setValue(DEFAULT_ENV_CONFIG.get("numAgents", 4))
        layout.addRow("Number of Agents:", self.numAgentsField)
        
        # Grid size - get default from environment config
        gridSizeLayout = QHBoxLayout()
        defaultGridSize = DEFAULT_ENV_CONFIG.get("gridSize", (15, 15))
        self.gridWidthField = QSpinBox()
        self.gridWidthField.setRange(5, 50)
        self.gridWidthField.setValue(defaultGridSize[0])
        self.gridHeightField = QSpinBox()
        self.gridHeightField.setRange(5, 50)
        self.gridHeightField.setValue(defaultGridSize[1])
        gridSizeLayout.addWidget(self.gridWidthField)
        gridSizeLayout.addWidget(QLabel("x"))
        gridSizeLayout.addWidget(self.gridHeightField)
        layout.addRow("Grid Size:", gridSizeLayout)
        
        # Seed
        seedLayout = QHBoxLayout()
        self.seedField = QSpinBox()
        self.seedField.setRange(0, 999999)
        self.seedField.setValue(random.randint(0, 999999))
        self.seedField.setSpecialValueText("Random")  # 0 means random
        seedLayout.addWidget(self.seedField)
        randomSeedButton = QPushButton("Random")
        randomSeedButton.clicked.connect(self.setRandomSeed)
        seedLayout.addWidget(randomSeedButton)
        layout.addRow("Seed:", seedLayout)
        
        # Reward type - get default from environment config
        self.rewardTypeCombo = QComboBox()
        self.rewardTypeCombo.addItems(["individual", "global", "hybrid"])
        defaultRewardType = DEFAULT_ENV_CONFIG.get("rewardType", "hybrid")
        # Convert "shared" to "global" for UI consistency
        if defaultRewardType == "shared":
            defaultRewardType = "global"
        self.rewardTypeCombo.setCurrentText(defaultRewardType)
        self.rewardTypeCombo.currentTextChanged.connect(self.updateHybridMixVisibility)
        layout.addRow("Reward Type:", self.rewardTypeCombo)
        
        # Hybrid mix - get default from environment config
        self.hybridMixLayout = QHBoxLayout()
        self.hybridMixField = QDoubleSpinBox()
        self.hybridMixField.setRange(0.0, 1.0)
        self.hybridMixField.setValue(DEFAULT_ENV_CONFIG.get("hybridRewardMix", 0.5))
        self.hybridMixField.setSingleStep(0.1)
        self.hybridMixLayout.addWidget(self.hybridMixField)
        self.hybridMixSlider = QSlider(Qt.Orientation.Horizontal)
        self.hybridMixSlider.setRange(0, 100)
        hybridMixPercent = int(DEFAULT_ENV_CONFIG.get("hybridRewardMix", 0.5) * 100)
        self.hybridMixSlider.setValue(hybridMixPercent)
        self.hybridMixSlider.valueChanged.connect(lambda v: self.hybridMixField.setValue(v / 100.0))
        self.hybridMixField.valueChanged.connect(lambda v: self.hybridMixSlider.setValue(int(v * 100)))
        self.hybridMixLayout.addWidget(self.hybridMixSlider)
        
        # Create the hybrid mix label and set its object name
        self.hybridMixLabel = QLabel("Hybrid Mix:")
        self.hybridMixLabel.setObjectName("hybridMixLabel")
        layout.addRow(self.hybridMixLabel, self.hybridMixLayout)
        
        # Record episodes checkbox - matches train_main.py default (recordEpisodes=True)
        self.recordEpisodesCheck = QCheckBox("Disable episode recording")
        self.recordEpisodesCheck.setChecked(False)  # Unchecked = record episodes (default)
        layout.addRow("", self.recordEpisodesCheck)
        
        # Update initial hybrid mix visibility
        self.updateHybridMixVisibility(self.rewardTypeCombo.currentText())
        
        self.tabWidget.addTab(basicTab, "Basic")
    
    def createAdvancedTab(self):
        """Create the advanced configuration tab."""
        advancedTab = QWidget()
        layout = QFormLayout(advancedTab)
        
        # Centralised critic
        self.centralisedCriticCheck = QCheckBox("Use centralised critic (MAPPO)")
        self.centralisedCriticCheck.setChecked(True)
        layout.addRow("", self.centralisedCriticCheck)
        
        # Attention mechanism
        self.attentionCheck = QCheckBox("Use attention mechanism")
        self.attentionCheck.setChecked(False)
        layout.addRow("", self.attentionCheck)
        
        # GPU usage
        self.useGpuCheck = QCheckBox("Use GPU for training (if available)")
        self.useGpuCheck.setChecked(False)  # Default to not using GPU, matching train_main.py
        layout.addRow("", self.useGpuCheck)
        
        # Entropy scheduler - set to True to match train_main.py default (useEntropyScheduler=True)
        self.entropySchedulerCheck = QCheckBox("Use entropy scheduling")
        self.entropySchedulerCheck.setChecked(True)  # True = use entropy scheduling (default)
        self.entropySchedulerCheck.stateChanged.connect(self.updateEntropyFieldsVisibility)
        layout.addRow("", self.entropySchedulerCheck)
        
        # Initial entropy
        self.initialEntropyField = QDoubleSpinBox()
        self.initialEntropyField.setRange(0.0001, 1.0)
        self.initialEntropyField.setValue(0.01)
        self.initialEntropyField.setDecimals(4)
        self.initialEntropyField.setSingleStep(0.001)
        layout.addRow("Initial Entropy:", self.initialEntropyField)
        
        # Final entropy
        self.finalEntropyField = QDoubleSpinBox()
        self.finalEntropyField.setRange(0.0001, 1.0)
        self.finalEntropyField.setValue(0.001)
        self.finalEntropyField.setDecimals(4)
        self.finalEntropyField.setSingleStep(0.001)
        layout.addRow("Final Entropy:", self.finalEntropyField)
        
        # Entropy schedule type
        self.entropyScheduleCombo = QComboBox()
        self.entropyScheduleCombo.addItems(["linear", "exponential", "step"])
        self.entropyScheduleCombo.setCurrentText("linear")
        layout.addRow("Entropy Schedule:", self.entropyScheduleCombo)
        
        # Update initial entropy fields visibility
        self.updateEntropyFieldsVisibility()
        
        # Evaluation interval
        self.evalIntervalField = QSpinBox()
        self.evalIntervalField.setRange(1, 100)
        self.evalIntervalField.setValue(5)
        layout.addRow("Evaluation Interval:", self.evalIntervalField)
        
        # Evaluation duration
        self.evalDurationField = QSpinBox()
        self.evalDurationField.setRange(1, 100)
        self.evalDurationField.setValue(10)
        layout.addRow("Evaluation Duration:", self.evalDurationField)
        
        # Evaluation parallel workers
        self.evalNumWorkersField = QSpinBox()
        self.evalNumWorkersField.setRange(1, 9)
        self.evalNumWorkersField.setValue(1)
        self.evalNumWorkersField.setToolTip("Number of parallel workers for evaluation (total workers must not exceed 10)")
        layout.addRow("Evaluation Workers:", self.evalNumWorkersField)
        
        # Training parallel workers
        self.numWorkersField = QSpinBox()
        self.numWorkersField.setRange(1, 9)
        self.numWorkersField.setValue(8)
        self.numWorkersField.valueChanged.connect(self.validateWorkers)
        self.numWorkersField.setToolTip("Number of parallel workers for training (total workers must not exceed 10)")
        layout.addRow("Training Workers:", self.numWorkersField)
        
        # Connect signals for worker validation
        self.evalNumWorkersField.valueChanged.connect(self.validateWorkers)
        
        # Ablation mode
        self.ablationModeCombo = QComboBox()
        self.ablationModeCombo.addItem("None")
        self.ablationModeCombo.addItems(["critic", "reward"])
        layout.addRow("Ablation Mode:", self.ablationModeCombo)
        
        # Configuration file
        self.configPathField = QLineEdit("")
        configPathLayout = QHBoxLayout()
        configPathLayout.addWidget(self.configPathField)
        configPathButton = QPushButton("Browse...")
        configPathButton.clicked.connect(self.browseConfigPath)
        configPathLayout.addWidget(configPathButton)
        layout.addRow("Config File (Optional):", configPathLayout)
        
        # Resume training
        self.resumeCheck = QCheckBox("Resume training from checkpoint")
        self.resumeCheck.setChecked(False)
        self.resumeCheck.stateChanged.connect(self.updateCheckpointFieldVisibility)
        layout.addRow("", self.resumeCheck)
        
        # Checkpoint path
        self.checkpointPathField = QLineEdit("")
        checkpointPathLayout = QHBoxLayout()
        checkpointPathLayout.addWidget(self.checkpointPathField)
        checkpointPathButton = QPushButton("Browse...")
        checkpointPathButton.clicked.connect(self.browseCheckpointPath)
        checkpointPathLayout.addWidget(checkpointPathButton)
        layout.addRow("Checkpoint Path:", checkpointPathLayout)
        
        # Update initial checkpoint field visibility
        self.updateCheckpointFieldVisibility()
        
        self.tabWidget.addTab(advancedTab, "Advanced")
    
    def createModelTab(self):
        """Create the model configuration tab with read-only optimal parameters."""
        modelTab = QWidget()
        layout = QVBoxLayout(modelTab)
        
        # Add information label
        infoLabel = QLabel("The model uses optimal PPO parameters that have been fine-tuned for this environment. "
                         "These values are shown for reference only.")
        infoLabel.setWordWrap(True)
        layout.addWidget(infoLabel)
        
        # PPO Parameters Group
        ppoGroup = QGroupBox("PPO Parameters (Reference Only)")
        ppoLayout = QFormLayout(ppoGroup)
        
        # Learning rate
        ppoLayout.addRow("Learning Rate:", QLabel("3e-4"))
        
        # Gamma (discount factor)
        ppoLayout.addRow("Gamma (Discount):", QLabel("0.99"))
        
        # Lambda (GAE)
        ppoLayout.addRow("Lambda (GAE):", QLabel("0.95"))
        
        # KL Coefficient
        ppoLayout.addRow("KL Coefficient:", QLabel("0.2"))
        
        # Clip parameter
        ppoLayout.addRow("Clip Parameter:", QLabel("0.2"))
        
        # VF Clip parameter
        ppoLayout.addRow("VF Clip Parameter:", QLabel("10.0"))
        
        # SGD iterations
        ppoLayout.addRow("SGD Iterations:", QLabel("10"))
        
        # SGD minibatch size
        ppoLayout.addRow("SGD Minibatch Size:", QLabel("128"))
        
        # Train batch size
        ppoLayout.addRow("Train Batch Size:", QLabel("4000"))
        
        layout.addWidget(ppoGroup)
        
        # Neural network parameters
        nnGroup = QGroupBox("Neural Network Parameters (Reference Only)")
        nnLayout = QFormLayout(nnGroup)
        
        # Hidden layers
        nnLayout.addRow("Hidden Layers:", QLabel("256, 256, 128"))
        
        # Activation function
        nnLayout.addRow("Activation Function:", QLabel("ReLU"))
        
        layout.addWidget(nnGroup)
        
        # Add a stretcher at the end to prevent content from expanding too much
        layout.addStretch(1)
        
        self.tabWidget.addTab(modelTab, "Model")
    
    def createButtons(self):
        """Create the action buttons."""
        buttonLayout = QHBoxLayout()
        
        # Launch button
        self.launchButton = QPushButton("Launch Training")
        self.launchButton.clicked.connect(self.launchTraining)
        buttonLayout.addWidget(self.launchButton)
        
        # Evaluate button
        self.evaluateButton = QPushButton("Evaluate Model")
        self.evaluateButton.clicked.connect(self.launchEvaluation)
        buttonLayout.addWidget(self.evaluateButton)
        
        # Visualisation button
        self.visualisationButton = QPushButton("Run Visualisations")
        self.visualisationButton.clicked.connect(self.runVisualisation)
        buttonLayout.addWidget(self.visualisationButton)
        
        # Cancel button
        self.cancelButton = QPushButton("Stop")
        self.cancelButton.clicked.connect(self.stopProcess)
        self.cancelButton.setEnabled(False)
        buttonLayout.addWidget(self.cancelButton)
        
        self.mainLayout.addLayout(buttonLayout)
    
    def updateHybridMixVisibility(self, reward_type):
        """Show hybrid mix widgets only when reward type is hybrid."""
        is_hybrid = reward_type == "hybrid"
        self.hybridMixLabel.setVisible(is_hybrid)
        for i in range(self.hybridMixLayout.count()):
            self.hybridMixLayout.itemAt(i).widget().setVisible(is_hybrid)
    
    def updateEntropyFieldsVisibility(self):
        """Show entropy fields only when entropy scheduler is enabled."""
        is_enabled = self.entropySchedulerCheck.isChecked()
        self.initialEntropyField.setEnabled(is_enabled)
        self.finalEntropyField.setEnabled(is_enabled)
        self.entropyScheduleCombo.setEnabled(is_enabled)
    
    def updateCheckpointFieldVisibility(self):
        """Show checkpoint path field only when resume is enabled."""
        is_enabled = self.resumeCheck.isChecked()
        self.checkpointPathField.setEnabled(is_enabled)
        browse_button = self.findChild(QPushButton, "Browse...", Qt.FindChildOption.FindChildrenRecursively)
        if browse_button:
            browse_button.setEnabled(is_enabled)
    
    def browseOutputDir(self):
        """Open file dialog to select output directory."""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self.outputDirField.setText(folder)
    
    def browseConfigPath(self):
        """Open file dialog to select configuration file."""
        file, _ = QFileDialog.getOpenFileName(self, "Select Config File", "", "Python Files (*.py);;All Files (*)")
        if file:
            self.configPathField.setText(file)
            # Try to load and apply the config
            self.loadConfigFile(file)
    
    def loadConfigFile(self, configPath):
        """Load configuration from file and update UI fields."""
        try:
            # Import the loadConfig function
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from train.common import loadConfig
            
            # Load the configuration
            config = loadConfig(configPath)
            print(f"Successfully loaded configuration from {configPath}")
            
            # Store the loaded config for later use
            self.loadedConfig = config
            
            # Update UI fields with values from config
            success = self.updateUIFromConfig(config)
            
            # Show success message
            if success:
                self.statusLabel.setText(f"Loaded configuration from {os.path.basename(configPath)}")
                
                # Ask if the user wants to keep the loaded configuration values
                reply = QMessageBox.question(
                    self, 
                    "Use Loaded Configuration?",
                    "The configuration has been loaded and UI fields have been updated.\n\n"
                    "Do you want to use these settings for resuming training?\n\n"
                    "Clicking 'Yes' will use the loaded values. Clicking 'No' will use your manual UI settings.",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )
                
                if reply == QMessageBox.StandardButton.No:
                    self.loadedConfig = None
                    self.statusLabel.setText("Using UI settings instead of loaded configuration")
                    return False
                    
                return True
            else:
                self.statusLabel.setText("Config loaded but could not fully update UI")
                return False
        except Exception as e:
            print(f"Error loading configuration file: {e}")
            self.loadedConfig = None
            QMessageBox.warning(self, "Configuration Error", 
                f"Failed to load configuration file: {str(e)}\n\nCurrent UI settings will be used.")
            self.statusLabel.setText("Failed to load configuration")
            return False
    
    def browseCheckpointPath(self):
        """Open file dialog to select checkpoint file."""
        # Open file dialog to select the checkpoint file
        file, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Checkpoint File", 
            "results/", 
            "Checkpoint Files (*.json *.pkl *)"
        )
        if file:
            self.checkpointPathField.setText(file)
            
            # Try to identify and offer to load the corresponding config file
            try:
                # Get the directory of the checkpoint
                checkpointDir = os.path.dirname(file)
                
                # Navigate up two levels if this is inside a checkpoints directory
                if os.path.basename(checkpointDir) == "checkpoints" or os.path.basename(os.path.dirname(checkpointDir)).startswith("checkpoint_"):
                    configDir = os.path.dirname(os.path.dirname(checkpointDir))
                else:
                    configDir = os.path.dirname(checkpointDir)
                
                # Look for config.py (or config.pkl for backward compatibility)
                configFile = os.path.join(configDir, "config.py")
                if not os.path.exists(configFile):
                    configFile = os.path.join(configDir, "config.pkl")
                
                # If the config file exists and not already loaded, suggest loading it
                if os.path.exists(configFile) and not self.configPathField.text():
                    reply = QMessageBox.question(
                        self, 
                        "Load Config File",
                        f"Do you want to also load the configuration file from this training run?\n\n{configFile}",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.Yes
                    )
                    
                    if reply == QMessageBox.StandardButton.Yes:
                        self.configPathField.setText(configFile)
                        self.loadConfigFile(configFile)
            except Exception as e:
                # If any error occurs during config detection, just ignore it
                print(f"Warning: Error detecting config file: {e}")
    
    def setRandomSeed(self):
        """Set a random seed value."""
        self.seedField.setValue(random.randint(1, 999999))
    
    def updateUIFromConfig(self, config):
        """Update UI fields with values from loaded config. Returns True if successful."""
        try:
            updated_count = 0
            
            # Environment configuration
            env_config = config.get("env_config", {})
            
            # Update basic tab
            if "numAgents" in env_config:
                self.numAgentsField.setValue(env_config["numAgents"])
                updated_count += 1
                
            if "gridSize" in env_config and isinstance(env_config["gridSize"], (list, tuple)) and len(env_config["gridSize"]) == 2:
                self.gridWidthField.setValue(env_config["gridSize"][0])
                self.gridHeightField.setValue(env_config["gridSize"][1])
                updated_count += 1
                
            if "rewardType" in env_config:
                reward_type = env_config["rewardType"]
                # Convert "shared" to "global" for UI consistency
                if reward_type == "shared":
                    reward_type = "global"
                    
                index = self.rewardTypeCombo.findText(reward_type)
                if index >= 0:
                    self.rewardTypeCombo.setCurrentIndex(index)
                    updated_count += 1
                    
            if "hybridRewardMix" in env_config:
                self.hybridMixField.setValue(env_config["hybridRewardMix"])
                updated_count += 1
            
            # Update advanced tab
            use_centralised_critic = True  # Default
            if "multiagent" in config and "policies" in config["multiagent"]:
                # If only one policy and it's shared, centralised critic is used
                use_centralised_critic = len(config["multiagent"]["policies"]) == 1
            self.centralisedCriticCheck.setChecked(use_centralised_critic)
            updated_count += 1
            
            # Check for attention mechanism
            use_attention = False
            if "model" in config and "custom_model_config" in config["model"]:
                use_attention = config["model"]["custom_model_config"].get("use_attention", False)
            self.attentionCheck.setChecked(use_attention)
            updated_count += 1
            
            # Entropy scheduler
            if "entropy_coeff" in config or "entropy_coeff_schedule" in config:
                # If entropy_coeff_schedule exists, scheduler is enabled
                use_scheduler = "entropy_coeff_schedule" in config
                self.entropySchedulerCheck.setChecked(use_scheduler)
                
                # Initial entropy - first value in schedule or fixed value
                if use_scheduler and isinstance(config.get("entropy_coeff_schedule"), (list, tuple)) and len(config["entropy_coeff_schedule"]) > 0:
                    # Schedule is a list of [time_fraction, value] pairs
                    initial_entropy = config["entropy_coeff_schedule"][0][1]
                    self.initialEntropyField.setValue(initial_entropy)
                    
                    # Final entropy - last value in schedule
                    final_entropy = config["entropy_coeff_schedule"][-1][1]
                    self.finalEntropyField.setValue(final_entropy)
                else:
                    # Use the fixed coefficient or defaults
                    self.initialEntropyField.setValue(config.get("initial_entropy", 0.01))
                    self.finalEntropyField.setValue(config.get("final_entropy", 0.001))
                
                # Schedule type
                schedule_type = config.get("entropy_schedule", "linear")
                index = self.entropyScheduleCombo.findText(schedule_type)
                if index >= 0:
                    self.entropyScheduleCombo.setCurrentIndex(index)
                    
                updated_count += 1
            
            # Evaluation parameters
            self.evalIntervalField.setValue(config.get("evaluation_interval", 5))
            self.evalDurationField.setValue(config.get("evaluation_duration", 10))
            self.evalNumWorkersField.setValue(config.get("evaluation_num_workers", 1))
            updated_count += 1
            
            # Training workers
            self.numWorkersField.setValue(config.get("num_workers", 8))
            updated_count += 1
            
            print(f"UI updated from config file: {updated_count} settings applied")
            
            return updated_count > 0
            
        except Exception as e:
            print(f"Error updating UI from config: {e}")
            # Continue with current UI settings if there's an error
            return False
    
    def buildCommandArgs(self):
        """Build command line arguments from UI values."""
        args = []
        
        # If resuming with a checkpoint and config, prioritise those values 
        resuming = self.resumeCheck.isChecked() and self.checkpointPathField.text() and os.path.exists(self.checkpointPathField.text())
        use_config_file = self.configPathField.text() and os.path.exists(self.configPathField.text())
        
        # Always include output directory
        if self.outputDirField.text():
            args.extend(["--output-dir", self.outputDirField.text()])
        
        # If resuming with a config, add minimal overrides
        if resuming and use_config_file:
            # First add config file
            args.extend(["--config", self.configPathField.text()])
            
            # Add resume flag and checkpoint path
            args.append("--resume")
            args.extend(["--checkpoint", self.checkpointPathField.text()])
            
            # Only add iterations (critical parameter that might need changing)
            args.extend(["--iterations", str(self.iterationsField.value())])
            
            # Add seed if explicitly set (not random)
            if self.seedField.value() > 0:
                args.extend(["--seed", str(self.seedField.value())])
                
            # Add GPU flag if explicitly checked
            if self.useGpuCheck.isChecked():
                args.append("--use-gpu")
                
            # Add no-record-episodes if checked
            if self.recordEpisodesCheck.isChecked():
                args.append("--no-record-episodes")
                
            # Return early - use loaded config for everything else
            return args
        
        # Using config file without resuming, or not using config file
        if use_config_file:
            args.extend(["--config", self.configPathField.text()])
        
        # Standard behaviour when not resuming with config
        args.extend(["--iterations", str(self.iterationsField.value())])
        args.extend(["--num-agents", str(self.numAgentsField.value())])
        args.extend(["--grid-size", str(self.gridWidthField.value()), str(self.gridHeightField.value())])
        
        if self.seedField.value() > 0:  # 0 means random
            args.extend(["--seed", str(self.seedField.value())])
        
        args.extend(["--reward-type", self.rewardTypeCombo.currentText()])
        
        if self.rewardTypeCombo.currentText() == "hybrid":
            args.extend(["--hybrid-mix", str(self.hybridMixField.value())])
        
        if self.recordEpisodesCheck.isChecked():
            args.append("--no-record-episodes")
        
        # Advanced tab
        if not self.centralisedCriticCheck.isChecked():
            args.append("--no-centralised-critic")
        
        if self.attentionCheck.isChecked():
            args.append("--use-attention")
            
        # Add GPU flag only when checked
        if self.useGpuCheck.isChecked():
            args.append("--use-gpu")
        
        if self.entropySchedulerCheck.isChecked():
            args.append("--entropy-scheduler")
            args.extend(["--initial-entropy", str(self.initialEntropyField.value())])
            args.extend(["--final-entropy", str(self.finalEntropyField.value())])
            args.extend(["--entropy-schedule", self.entropyScheduleCombo.currentText()])
        
        # Add evaluation parameters if different from defaults
        if self.evalIntervalField.value() != 5:  # Default in train_main.py
            args.extend(["--evaluation-interval", str(self.evalIntervalField.value())])
            
        if self.evalDurationField.value() != 10:  # Default in train_main.py
            args.extend(["--evaluation-duration", str(self.evalDurationField.value())])
            
        if self.evalNumWorkersField.value() != 1:  # Default in train_main.py
            args.extend(["--evaluation-num-workers", str(self.evalNumWorkersField.value())])
        
        if self.ablationModeCombo.currentText() != "None":
            args.extend(["--ablation", self.ablationModeCombo.currentText()])
        
        if resuming:
            args.append("--resume")
            args.extend(["--checkpoint", self.checkpointPathField.text()])
        
        # Training workers
        if self.numWorkersField.value() != 8:  # Default in train_main.py
            args.extend(["--num-workers", str(self.numWorkersField.value())])
        
        return args
    
    def launchTraining(self):
        """Launch the training process with the current settings."""
        cmd = [sys.executable, "train_main.py"]
        cmd.extend(self.buildCommandArgs())
        
        # Debug - print command
        print("Running:", " ".join(cmd))
        
        # Update UI
        self.statusLabel.setText("Training in progress...")
        self.launchButton.setEnabled(False)
        self.evaluateButton.setEnabled(False)
        self.cancelButton.setEnabled(True)
        
        # Start process
        self.process.setWorkingDirectory(os.path.dirname(os.path.abspath(__file__)))
        self.process.start(cmd[0], cmd[1:])
    
    def launchEvaluation(self):
        """Launch evaluation for an existing checkpoint."""
        # Get checkpoint
        checkpoint, _ = QFileDialog.getOpenFileName(self, "Select Checkpoint", "", "Checkpoint Files (*)")
        if not checkpoint:
            return
        
        # Launch evaluation
        cmd = [sys.executable, "eval/evaluatePolicy.py"]
        cmd.extend(["--checkpoint", checkpoint])
        cmd.extend(["--episodes", "10"])
        cmd.append("--render")  # Enable rendering for evaluation
        
        # Debug - print command
        print("Running:", " ".join(cmd))
        
        # Update UI
        self.statusLabel.setText("Evaluation in progress...")
        self.launchButton.setEnabled(False)
        self.evaluateButton.setEnabled(False)
        self.cancelButton.setEnabled(True)
        
        # Start process
        self.process.setWorkingDirectory(os.path.dirname(os.path.abspath(__file__)))
        self.process.start(cmd[0], cmd[1:])
    
    def stopProcess(self):
        """Stop the current process."""
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()
            self.statusLabel.setText("Process stopped")
        
        self.launchButton.setEnabled(True)
        self.evaluateButton.setEnabled(True)
        self.visualisationButton.setEnabled(True)
        self.cancelButton.setEnabled(False)
    
    def handleOutput(self):
        """Handle standard output from the process."""
        data = self.process.readAllStandardOutput().data().decode()
        print(data, end="", flush=True)
        
        # Parse progress from log if possible
        lines = data.splitlines()
        for line in lines:
            if "Iteration" in line and "reward=" in line:
                self.statusLabel.setText(line)
    
    def handleError(self):
        """Handle standard error from the process."""
        data = self.process.readAllStandardError().data().decode()
        print("ERROR:", data, end="", flush=True)
    
    def handleFinished(self, exitCode, exitStatus):
        """Handle process completion."""
        print(f"\nProcess finished with exit code {exitCode}")
        
        if exitCode == 0:
            if "Generating visualisations" in self.statusLabel.text():
                self.statusLabel.setText("Visualisations generated successfully")
            else:
                self.statusLabel.setText("Process completed successfully")
        else:
            self.statusLabel.setText(f"Process failed with exit code {exitCode}")
        
        self.launchButton.setEnabled(True)
        self.evaluateButton.setEnabled(True)
        self.visualisationButton.setEnabled(True)
        self.cancelButton.setEnabled(False)
    
    def runVisualisation(self):
        """Run visualisations for training results."""
        # First try to get a specific metrics.csv file
        metrics_file, _ = QFileDialog.getOpenFileName(
            self, 
            "Select metrics.csv File", 
            "results/", 
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not metrics_file:
            # If cancelled, try selecting a directory
            results_dir = QFileDialog.getExistingDirectory(
                self, 
                "Select Results Directory Containing metrics.csv", 
                "results/"
            )
            
            if not results_dir:
                return
            
            # Use the directory
            cmd = [sys.executable, "eval/plotMetrics.py", results_dir]
        else:
            # Use the specific file
            cmd = [sys.executable, "eval/plotMetrics.py", metrics_file]
        
        # Debug - print command
        print("Running:", " ".join(cmd))
        
        # Update UI
        self.statusLabel.setText("Generating visualisations...")
        self.launchButton.setEnabled(False)
        self.evaluateButton.setEnabled(False)
        self.visualisationButton.setEnabled(False)
        self.cancelButton.setEnabled(True)
        
        # Start process
        self.process.setWorkingDirectory(os.path.dirname(os.path.abspath(__file__)))
        self.process.start(cmd[0], cmd[1:])

    def validateWorkers(self):
        """
        Validate that the total number of workers doesn't exceed 10.
        This includes both training workers and evaluation workers.
        """
        # Calculate total workers
        total_workers = self.numWorkersField.value() + self.evalNumWorkersField.value()
        
        # Maximum allowed workers
        max_workers = 10
        
        # If total exceeds maximum, adjust evaluation workers first
        if total_workers > max_workers:
            if self.sender() == self.numWorkersField:
                # If numWorkers was changed, adjust evalNumWorkers
                self.evalNumWorkersField.setValue(max(1, max_workers - self.numWorkersField.value()))
            else:
                # If evalNumWorkers was changed or initial call, adjust num_workers
                self.evalNumWorkersField.setValue(max(1, max_workers - self.numWorkersField.value()))
            
            # Show a temporary tooltip to inform the user
            sender = self.sender()
            if sender:
                original_tooltip = sender.toolTip()
                sender.setToolTip(f"Total workers limited to {max_workers}")
                QTimer.singleShot(3000, lambda: sender.setToolTip(original_tooltip))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrainingLauncher()
    window.show()
    sys.exit(app.exec()) 