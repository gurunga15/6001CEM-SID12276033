# Multi-Agent Reinforcement Learning for Resource Collection

This project implements a multi-agent reinforcement learning environment for resource collection tasks. The environment features multiple agents navigating a grid world to collect various resources, with environmental dynamics like day/night cycles, weather, and seasons.

## Features

- **Dynamic Environment**: Configurable grid world with day/night cycles, weather changes, and seasonal effects
- **Multiple Resources**: Different resource types with varying values and regeneration rates
- **Multi-Agent Learning**: Support for PPO algorithm with different reward structures
- **Centralised Training with Decentralised Execution**: Implementation with centralised critics
- **Metrics Tracking**: Comprehensive metrics including fairness indices, resource specialisation, and efficiency
- **Episode Recording**: System for recording and replaying agent behaviour

## Installation

### Prerequisites

- Python 3.8 or newer
- PyQt6 (for GUI visualisation)
- Ray/RLlib (for scalability reinforcement learning)
- PyTorch (for AI backend)
- Gymnasium (framework)
- Matplot/Seaborn/Tensorflow (for data visualisation)

### With pip

```bash
# Navigate to the project directory
cd ./resourceProjectMARL

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
resourceProjectMARL/
├── algorithms/                # Algorithm implementations
│   ├── callbacks.py           # Metrics tracking and episode recording callbacks
│   ├── exploration.py         # Exploration strategies (entropy scheduling)
│   ├── models.py              # Neural network models for RL
│   ├── ppoPolicies.py         # PPO policy implementation
│   └── __init__.py
├── env/                       # Environment package
│   ├── config.py              # Environment configuration defaults
│   ├── entities.py            # Agent and Resource class definitions
│   ├── resourceEnv.py         # Core environment class (ResourceCollectionEnv)
│   ├── utils.py               # Utility functions for environment
│   └── __init__.py
├── eval/                      # Evaluation tools
│   ├── evaluatePolicy.py      # Tools for evaluating trained policies
│   ├── plotMetrics.py         # Metrics plotting and visualisation
│   ├── replayEpisode.py       # Replay recorded episodes
│   ├── renderDemo.py          # Demonstration script
│   ├── runVisualisation.py    # Run visualisations for analysis
│   └── __init__.py
├── gui/                       # GUI components
│   ├── viewer.py              # PyQt6 GUI visualiser (currently buggy)
│   └── __init__.py
├── train/                     # Training utilities
│   ├── ablation.py            # Run ablation studies
│   ├── common.py              # Common training functions
│   └── __init__.py
├── utils/                     # Utility functions
│   └── config_utils.py        # Configuration utilities
├── results/                   # Saved results directory
├── run_gui.py                 # GUI launcher script
├── train_main.py              # Main training script
├── ui_launcher.py             # Training UI launcher
├── CHANGES.md                 # Change log
├── requirements.txt           # Dependencies
└── __init__.py
```

## Usage

### Training Agents

The primary way to train agents is using the main training script or the UI launcher:

```bash
# Command-line training
python train_main.py --num-agents 4 --reward-type hybrid --iterations 100

# OR use the UI launcher
python ui_launcher.py
```

The UI launcher provides a working graphical interface for only configuriing training parameters, run agent training and then auto plotting or selecting to plot.

#### Training Parameters

- `--num-agents`: Number of agents (default: 4)
- `--grid-size`: Size of grid as width height (default: 20 20)
- `--reward-type`: Reward structure: individual, global, or hybrid (default: hybrid)
- `--hybrid-mix`: For hybrid rewards, proportion of team vs individual reward (default: 0.5)
- `--iterations`: Number of training iterations (default: 1000)
- `--record-episodes`: Whether to record episodes for replay (default: True)

### Evaluating Trained Policies (Only works while starting a new training session; automatic)

After training, you can evaluate policies using:

```bash
# Evaluate a trained policy
python -m eval.evaluatePolicy --checkpoint PATH_TO_CHECKPOINT
```

### Replaying Recorded Episodes (Unavailable - config issues)

To replay recorded episodes:

```bash
# Replay a recorded episode
python -m eval.replayEpisode --replay-file PATH_TO_REPLAY_FILE
```

### Visualising Training Results

Use UI launcher to generate plots from training results or:

```bash
# Plot training metrics
python -m eval.plotMetrics --experiment-dir PATH_TO_RESULTS
```

## Known Issues and Limitations

### Working Features:
- Core environment functionality: agents, resources, day/night cycle, weather, seasons
- PPO training with different reward structures (individual, global, hybrid)
- Metrics tracking and recording (fairness, efficiency, specialisation)
- Episode recording and replay functionality
- Command-line training and UI launcher

### Current Issues:
- **GUI Visualisation**: The PyQt6 GUI (run_gui.py) struggling to work with initial user agent/env configurations for agent training and will crash or display incorrectly.
- **Replay Visualisation**: While episodes are correctly recorded to pickle files, the visualisation of replays has not been finalised.
- **Resource Placement**: In some configurations, resources may spawn in unreachable locations or overlap with obstacles.
- **Memory Usage**: With long episodes or many agents, memory usage can become excessive, especially with episode recording enabled.

## Environment Configuration

The environment has many configurable parameters:

- `gridSize`: Size of the grid world (width, height)
- `numAgents`: Number of agents in the environment
- `resourceTypes`: Types of resources available (e.g., "food", "wood", "stone")
- `resourceValues`: Value of each resource type
- `dayNightCycle`: Enable/disable day/night cycle
- `weatherEnabled`: Enable/disable weather effects
- `seasonEnabled`: Enable/disable seasonal changes
- `rewardType`: Type of reward structure ("individual", "shared", or "hybrid")
- `hybridRewardMix`: When using hybrid rewards, the balance between individual and team rewards (0-1)

See `env/config.py` for the full list of configuration options.

## Troubleshooting

- If you encounter errors with Ray/RLlib, ensure you have compatible versions (see requirements.txt).
- For memory issues, reduce the number of agents or disable episode recording.
- If training seems unstable, try adjusting the learning rate or entropy coefficient.
- The GUI is currently buggy; use terminal-based training and evaluation instead.

## Future Work

- Fix and finalise GUI visualisation
- Implement improved resource placement algorithms
- Add support for more MARL algorithms (QMIX, MADDPG)
- Optimise memory usage for episode recording
- Rework replay system with better visualisation tools

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Ray RLlib for the reinforcement learning framework
- PyTorch for neural network implementations
- Gymnasium for the environment interface 