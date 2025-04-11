"""
Exploration schedulers and utilities for PPO in resource collection environments.

This module provides utilities for controlling exploration during training,
particularly through dynamic adjustment of the entropy coefficient.

Note: This module is designed specifically for PPO's entropy-based exploration
and is not intended for epsilon-greedy based exploration used in other methods.
"""

import os
import sys
import numpy as np
from typing import Dict, Any, Optional, Callable, List, Tuple

# Import Ray components carefully to handle version differences
try:
    from ray.rllib.agents.callbacks import DefaultCallbacks
except ImportError:
    # For newer Ray versions
    from ray.rllib.algorithms.callbacks import DefaultCallbacks


class EntropyScheduler:
    """
    Scheduler for dynamically adjusting the entropy coefficient during training.
    
    This allows for higher exploration early in training and more exploitation later.
    
    Note: This is specific to PPO's entropy-based exploration and is not compatible
    with epsilon-greedy exploration methods used in other algorithms.
    """
    
    def __init__(
        self,
        initial_entropy: float = 0.01,
        final_entropy: float = 0.001,
        schedule_type: str = "linear",
        total_iterations: int = 1000,
        warmup_iterations: int = 0
    ):
        """
        Initialise the entropy scheduler.
        
        Args:
            initial_entropy: Starting entropy coefficient
            final_entropy: Final entropy coefficient
            schedule_type: Type of scheduling ("linear", "exponential", "step")
            total_iterations: Total training iterations
            warmup_iterations: Number of iterations to maintain initial entropy before scheduling
        """
        self.initial_entropy = initial_entropy
        self.final_entropy = final_entropy
        self.schedule_type = schedule_type
        self.total_iterations = total_iterations
        self.warmup_iterations = warmup_iterations
    
    def get_entropy_coefficient(self, iteration: int) -> float:
        """
        Get the entropy coefficient for the current iteration.
        
        Args:
            iteration: Current training iteration
            
        Returns:
            Scheduled entropy coefficient
        """
        # During warmup, use initial entropy
        if iteration < self.warmup_iterations:
            return self.initial_entropy
        
        # Calculate progress (0.0 to 1.0) after warmup period
        adjusted_iteration = iteration - self.warmup_iterations
        adjusted_total = self.total_iterations - self.warmup_iterations
        
        if adjusted_total <= 0:  # Avoid division by zero
            return self.final_entropy
            
        progress = min(1.0, max(0.0, adjusted_iteration / adjusted_total))
        
        # Apply different schedule types
        if self.schedule_type == "linear":
            # Linear decay
            return self.initial_entropy + (self.final_entropy - self.initial_entropy) * progress
        
        elif self.schedule_type == "exponential":
            # Exponential decay
            if self.initial_entropy == 0:
                return self.final_entropy
            return self.initial_entropy * (self.final_entropy / self.initial_entropy) ** progress
        
        elif self.schedule_type == "step":
            # Step function decay
            steps = 4  # Number of steps
            step_idx = min(int(progress * steps), steps - 1)
            step_values = np.linspace(self.initial_entropy, self.final_entropy, steps)
            return step_values[step_idx]
        
        else:
            # Default to linear decay
            return self.initial_entropy + (self.final_entropy - self.initial_entropy) * progress


class ExplorationSchedulerCallback(DefaultCallbacks):
    """
    Callback to adjust exploration parameters during training.
    
    This callback can be used to dynamically adjust entropy coefficient and other
    exploration parameters based on training progress.
    
    Note: This is specific to PPO's entropy-based exploration and is not compatible
    with epsilon-greedy exploration methods used in other algorithms.
    """
    
    def __init__(self, entropy_scheduler: Optional[EntropyScheduler] = None, verbose: bool = True):
        """
    Initialise the exploration scheduler callback.
        
        Args:
            entropy_scheduler: Entropy scheduler instance (created with default settings if None)
            verbose: Whether to print detailed logs about entropy changes
        """
        super().__init__()
        self.entropy_scheduler = entropy_scheduler or EntropyScheduler()
        self.verbose = verbose
    
    def on_train_result(self, algorithm, result, **kwargs):
        """
        Adjust exploration parameters based on training progress.
        
        Args:
            algorithm: Training algorithm (PPO)
            result: Training result dictionary
            **kwargs: Additional keyword arguments
        """
        # Get current iteration
        iteration = result.get("training_iteration", 0)
        
        # Get updated entropy coefficient
        entropy_coeff = self.entropy_scheduler.get_entropy_coefficient(iteration)
        
        # Get all policy IDs
        try:
            # For newer Ray versions
            policies = list(algorithm.policies.keys())
        except:
            # For older Ray versions, try different approaches
            try:
                policies = list(algorithm.config["multiagent"]["policies"].keys())
            except:
                # Fallback to worker-based access
                local_worker = algorithm.workers.local_worker()
                if hasattr(local_worker, "policy_map"):
                    policies = list(local_worker.policy_map.keys())
                else:
                    policies = []
        
        # Update entropy coefficient for all policies
        updated = False
        for policy_id in policies:
            try:
                # Get the policy
                policy = algorithm.get_policy(policy_id)
                
                # Try multiple ways to update entropy coefficient for different Ray versions
                if hasattr(policy, "entropy_coeff"):
                    policy.entropy_coeff = entropy_coeff
                    # Also update in config for persistence
                    if hasattr(policy, "config") and isinstance(policy.config, dict):
                        policy.config["entropy_coeff"] = entropy_coeff
                    updated = True
                elif hasattr(policy, "_entropy_coeff"):
                    policy._entropy_coeff = entropy_coeff
                    # Also update in config for persistence
                    if hasattr(policy, "config") and isinstance(policy.config, dict):
                        policy.config["entropy_coeff"] = entropy_coeff
                    updated = True
                # Direct config update (may be needed for newer Ray versions)
                elif hasattr(policy, "config") and isinstance(policy.config, dict):
                    policy.config["entropy_coeff"] = entropy_coeff
                    updated = True
                    
            except Exception as e:
                if self.verbose:
                    print(f"[EntropyScheduler] Error updating entropy for policy {policy_id}: {e}")
        
        # Try updating global config too (for newer Ray versions)
        try:
            if hasattr(algorithm, "config") and isinstance(algorithm.config, dict):
                algorithm.config["entropy_coeff"] = entropy_coeff
        except Exception:
            pass
        
        # Log updated entropy value
        if self.verbose:
            print(f"[EntropyScheduler] Iteration {iteration}: entropy_coeff set to {entropy_coeff:.6f}")
        
        # Add to metrics regardless of verbose setting
        result["entropy_coeff"] = entropy_coeff
        
        # Ensure custom_metrics dict exists
        if "custom_metrics" not in result:
            result["custom_metrics"] = {}
        result["custom_metrics"]["entropy_coeff"] = entropy_coeff
        
        if not updated:
            print(f"[EntropyScheduler] Warning: Could not update entropy coefficient for any policy.")


def create_combined_callback(
    entropy_scheduler: Optional[EntropyScheduler] = None,
    callback_classes: List[type] = None
) -> type:
    """
    Create a combined callback class that includes exploration scheduling.
    
    This function creates a new callback class that combines the exploration scheduler
    with other callback classes through multiple inheritance.
    
    Args:
        entropy_scheduler: Entropy scheduler to use
        callback_classes: List of callback classes to combine with exploration scheduler
        
    Returns:
        Combined callback class
    """
    if callback_classes is None:
        callback_classes = []
    
    class CombinedCallback(*callback_classes, ExplorationSchedulerCallback):
        def __init__(self, *args, **kwargs):
            # initialise the exploration scheduler
            ExplorationSchedulerCallback.__init__(self, entropy_scheduler)
            
            # initialise other parent classes
            for cls in callback_classes:
                if cls != ExplorationSchedulerCallback:
                    # Try to initialise parent class
                    try:
                        cls.__init__(self, *args, **kwargs)
                    except Exception as e:
                        print(f"Error initializing {cls.__name__}: {e}")
        
        def on_train_result(self, algorithm, result, **kwargs):
            # Call parent implementations
            ExplorationSchedulerCallback.on_train_result(self, algorithm, result, **kwargs)
            
            for cls in callback_classes:
                if cls != ExplorationSchedulerCallback and hasattr(cls, "on_train_result"):
                    # Call parent on_train_result if it exists
                    try:
                        cls.on_train_result(self, algorithm, result, **kwargs)
                    except Exception as e:
                        print(f"Error in {cls.__name__}.on_train_result: {e}")
    
    return CombinedCallback


def add_entropy_scheduling_to_config(
    config: Dict[str, Any],
    initial_entropy: float = 0.01,
    final_entropy: float = 0.001,
    schedule_type: str = "linear",
    total_iterations: int = 1000,
    warmup_iterations: int = 0
) -> Dict[str, Any]:
    """
    Add entropy scheduling to a PPO configuration.
    
    Args:
        config: PPO configuration dictionary
        initial_entropy: Initial entropy coefficient
        final_entropy: Final entropy coefficient
        schedule_type: Type of scheduling ("linear", "exponential", "step")
        total_iterations: Total training iterations
        warmup_iterations: Number of iterations to maintain initial entropy before scheduling
        
    Returns:
        Updated configuration dictionary
    """
    # Create entropy scheduler
    entropy_scheduler = EntropyScheduler(
        initial_entropy=initial_entropy,
        final_entropy=final_entropy,
        schedule_type=schedule_type,
        total_iterations=total_iterations,
        warmup_iterations=warmup_iterations
    )
    
    # Set the initial entropy coefficient in the config
    config["entropy_coeff"] = initial_entropy
    
    # Create a callback for entropy scheduling
    entropy_callback = ExplorationSchedulerCallback(entropy_scheduler)
    
    # Add callback to the config
    if "callbacks_class" in config:
        # For newer Ray versions
        old_callback_cls = config["callbacks_class"]
        combined_callback = create_combined_callback(
            entropy_scheduler=entropy_scheduler,
            callback_classes=[old_callback_cls]
        )
        config["callbacks_class"] = combined_callback
    elif "callbacks" in config:
        # For older Ray versions
        old_callbacks = config["callbacks"]
        if isinstance(old_callbacks, type):
            # If it's a class, combine them
            combined_callback = create_combined_callback(
                entropy_scheduler=entropy_scheduler,
                callback_classes=[old_callbacks]
            )
            config["callbacks"] = combined_callback
        else:
            # If it's an instance, just replace with entropy callback
            # This is a limitation - can't easily combine with an instance
            print("Warning: Replacing existing callback instance with entropy scheduler")
            config["callbacks"] = entropy_callback
    else:
        # No existing callbacks, add entropy callback
        try:
            # Try new API first
            config["callbacks_class"] = ExplorationSchedulerCallback
        except:
            # Fall back to old API
            config["callbacks"] = entropy_callback
    
    return config 