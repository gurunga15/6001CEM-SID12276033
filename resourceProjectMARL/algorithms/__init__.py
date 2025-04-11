"""
Algorithms package for multi-agent reinforcement learning.

This package contains PPO (Proximal Policy Optimization) implementation
optimised for multi-agent environments with RLlib and PyTorch.
"""

# Add parent directory to path for proper imports
import os
import sys

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import policy implementation
from .ppoPolicies import createDefaultPPOPolicy

# Import Ray
import ray

# Import callbacks
from .callbacks import (
    MetricsCallback,
    TrainingProgressCallback,
    EpisodeRecorder
)

# Import custom models
from .models import ResourceCNNModel

# Explicitly expose key classes/modules
__all__ = [
    "createDefaultPPOPolicy",
    "ResourceCNNModel",
    "MetricsCallback",
    "TrainingProgressCallback",
    "EpisodeRecorder"
] 