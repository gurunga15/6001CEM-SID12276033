#!/usr/bin/env python

"""
Main training script for PPO on the resource collection environment.

This script provides a single unified interface for training PPO with various enhancements,
including entropy scheduling and ablation studies.
"""

import os
import sys
import argparse
import time
import ray
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import pprint

# Add the project root to the path
scriptDir = os.path.dirname(os.path.abspath(__file__))
if scriptDir not in sys.path:
    sys.path.insert(0, scriptDir)

# For Gymnasium compatibility with Ray
import gymnasium as gym
sys.modules["gym"] = gym  # Monkey-patch to keep RLlib happy with Gymnasium

# Import components
from env.resourceEnv import ResourceCollectionEnv
from algorithms.ppoPolicies import createDefaultPPOPolicy
from algorithms.callbacks import (
    MetricsCallback, 
    TrainingProgressCallback, 
    EpisodeRecorder,
    DefaultCallbacks
)
from algorithms.exploration import EntropyScheduler, ExplorationSchedulerCallback
from train.common import setupOutputDir, saveConfig, loadConfig, visualiseTrainingProgress
from env.utils import calculateGiniCoefficient, calculateJainFairnessIndex

# Check Ray version
print(f"Using Ray version: {ray.__version__}")


def train_main(
    # Core training parameters
    configPath: str = None,
    outputDir: str = None,
    iterations: int = 1000,
    resume: bool = False,
    checkpointPath: str = None,
    numAgents: int = 4,
    seed: int = None,
    
    # Environment parameters
    gridSize: tuple = (15, 15),
    rewardType: str = "hybrid",
    hybridRewardMix: float = 0.5,
    
    # PPO configuration
    useCentralisedCritic: bool = True,
    useAttention: bool = False,
    useGpu: bool = False,
    numWorkers: int = 8,
    
    # Entropy scheduling
    useEntropyScheduler: bool = True,
    initialEntropy: float = 0.01,
    finalEntropy: float = 0.001,
    entropyScheduleType: str = "linear",
    
    # Ablation mode
    ablationMode: str = None,
    
    # Additional parameters
    evaluationInterval: int = 5,
    evaluationDuration: int = 10,
    evaluationNumWorkers: int = 1,
    recordEpisodes: bool = True
):
    """
    Train PPO on the resource collection environment with various enhancements.
    
    Args:
        configPath: Path to configuration file (optional)
        outputDir: Directory to save results and checkpoints
        iterations: Number of training iterations
        resume: Whether to resume training from a checkpoint
        checkpointPath: Path to the checkpoint to resume from
        numAgents: Number of agents in the environment
        seed: Random seed for reproducibility
        
        gridSize: Size of the environment grid as (width, height)
        rewardType: Type of reward structure ("individual", "global", or "hybrid")
        hybridRewardMix: Mix ratio for hybrid rewards (0.0 = global, 1.0 = individual)
        
        useCentralisedCritic: Whether to use a centralised critic (MAPPO)
        useAttention: Whether to use attention mechanism in centralised critic
        useGpu: Whether to use GPU for training (enabled with --use-gpu flag)
        numWorkers: Number of parallel workers for training
        
        useEntropyScheduler: Whether to use entropy scheduling
        initialEntropy: Initial entropy coefficient
        finalEntropy: Final entropy coefficient
        entropyScheduleType: Type of entropy scheduling ("linear", "exponential", "step")
        
        ablationMode: Run in ablation mode ("critic", "reward", "policy", or None)
        
        evaluationInterval: How often to run evaluation (iterations)
        evaluationDuration: How many episodes to run for evaluation
        evaluationNumWorkers: Number of workers for evaluation
        recordEpisodes: Whether to record episodes for replay
    """
    # Initialise Ray
    if not ray.is_initialized():
        ray.init(num_cpus=os.cpu_count() or 4, include_dashboard=True)  # Enable dashboard for monitoring
    
    # Set default seed if none provided
    if seed is None:
        seed = int(time.time()) % 100000
        print(f"No seed provided, using generated seed: {seed}")
    
    # Check for ablation mode
    if ablationMode:
        run_ablation_study(
            ablationType=ablationMode,
            baseOutputDir=outputDir,
            iterations=iterations,
            numAgents=numAgents,
            gridSize=gridSize,
            seed=seed,
            useEntropyScheduler=useEntropyScheduler,
            initialEntropy=initialEntropy,
            finalEntropy=finalEntropy,
            entropyScheduleType=entropyScheduleType,
            useGpu=useGpu,
            numWorkers=numWorkers,
            evaluationNumWorkers=evaluationNumWorkers,
            configPath=configPath
        )
        return
    
    # Set up output directory
    if outputDir is None:
        outputDir = setupOutputDir("ppo", checkpointPath if resume else None)
    
    # Load configuration from file if provided
    config = {}
    if configPath and os.path.exists(configPath):
        try:
            config = loadConfig(configPath)
            print(f"Successfully loaded configuration from {configPath}")
        except Exception as e:
            print(f"Error loading configuration file: {e}")
            print("Proceeding with default configuration")
    
    # Create environment configuration
    env_config = config.get("env_config", {}).copy()
    
    # When resuming, trust the config values more than command-line arguments
    # for environment parameters that should be consistent between runs
    if not resume:
        # Only update these if we're not resuming (new training run)
        env_config.update({
            "numAgents": numAgents,
            "gridSize": gridSize,
            "renderMode": None,  # No rendering during training
            "use_flattened_obs": True,  # Use flattened observations for better compatibility
            "rewardType": rewardType,
            "hybridRewardMix": hybridRewardMix
        })
    else:
        # When resuming, only update fields that don't affect the environment structure
        # but print a warning if there are mismatches
        for param_name, cmd_value, config_value in [
            ("numAgents", numAgents, env_config.get("numAgents")),
            ("gridSize", gridSize, env_config.get("gridSize")),
            ("rewardType", rewardType, env_config.get("rewardType")),
            ("hybridRewardMix", hybridRewardMix, env_config.get("hybridRewardMix"))
        ]:
            if cmd_value != config_value and config_value is not None:
                print(f"Warning: Command-line {param_name}={cmd_value} differs from checkpoint config {param_name}={config_value}")
                print(f"Using config value {config_value} for consistency when resuming training")
        
        # Always set these regardless
        env_config.update({
            "renderMode": None,  # No rendering during training
            "use_flattened_obs": True,  # Use flattened observations for better compatibility
        })
    
    # Add seed to environment config if provided
    if seed is not None:
        env_config["seed"] = seed
    
    env_config["output_dir"] = outputDir
    # Create general config for PPO
    config_overrides = {
        "env_config": env_config,
        "num_sgd_iter": config.get("num_sgd_iter", 10),
        "sgd_minibatch_size": config.get("sgd_minibatch_size", 128),
        "train_batch_size": config.get("train_batch_size", 4000),
        "lr": config.get("lr", 3e-4),
        "gamma": config.get("gamma", 0.99),
        "lambda_": config.get("lambda_", 0.95),
        "kl_coeff": config.get("kl_coeff", 0.2),
        "clip_param": config.get("clip_param", 0.2),
        "vf_clip_param": config.get("vf_clip_param", 10.0),
        "entropy_coeff": initialEntropy if useEntropyScheduler else config.get("entropy_coeff", 0.01),
        "num_workers": numWorkers,
        "framework": "torch",
        "seed": seed,  # Set random seed if not made for reproducibility
        
        # Evaluation configuration
        "evaluation_duration": evaluationDuration,
        "evaluation_duration_unit": "episodes",
        "evaluation_interval": evaluationInterval,
        "evaluation_num_workers": evaluationNumWorkers,
        "evaluation_config": {
            "explore": False,  # Disable exploration during evaluation
            "env_config": env_config.copy()  # Use same environment config for evaluation
        },
        
        # Model configuration
        "model": {
            "fcnet_hiddens": [256, 256, 128],
            "fcnet_activation": "relu",
        },
    }
    
    # Create PPO policy
    ppoPolicy = createDefaultPPOPolicy(
        useCentralisedCritic=useCentralisedCritic,
        numAgents=numAgents,
        config=config_overrides,
        rewardType=rewardType,
        hybridRewardMix=hybridRewardMix,
        useAttention=useAttention,
        useGpu=useGpu
    )
    
    # Add TensorBoard logging
    tensorboard_dir = os.path.join(outputDir, "tensorboard")
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # Configure TensorBoard logger
    ppoPolicy.config["logger_config"] = {
        "logdir": tensorboard_dir,
        "type": "ray.tune.logger.TBXLogger"
    }
    
    # Create entropy scheduler if enabled
    entropy_scheduler = None
    if useEntropyScheduler:
        entropy_scheduler = EntropyScheduler(
            initial_entropy=initialEntropy,
            final_entropy=finalEntropy,
            schedule_type=entropyScheduleType,
            total_iterations=iterations,
            warmup_iterations=int(iterations * 0.1)  # 10% warmup period
        )
    
    # Create comprehensive callback for metrics tracking
    class UnifiedTrainingCallback(DefaultCallbacks):
        """
        Unified callback for metrics tracking, training progress and coordination.
        
        This callback consolidates multiple functions into a single implementation:
        1. Tracks training metrics such as rewards and episode lengths
        2. Calculates coordination metrics (specialisation, overlap, task division)
        3. Logs metrics to CSV file for later analysis and visualisation
        4. Handles entropy scheduling for exploration
        
        Using a unified callback ensures consistent calculation of metrics and avoids
        redundancy in the codebase. All metrics are stored in the metrics.csv file,
        which is used by visualisation tools to create performance graphs.
        """
        
        def __init__(self):
            super().__init__()
            # For metrics tracking
            self.episode_data = {}
            self.coordination_metrics = {}
            self.reset_coordination_metrics()
            
            # For entropy scheduling
            if useEntropyScheduler:
                self.entropy_scheduler = entropy_scheduler
            
            # For CSV logging
            self.csv_file = os.path.join(outputDir, "metrics.csv")
            with open(self.csv_file, "w") as f:
                header = "iteration,episode_reward_mean,episode_reward_min,episode_reward_max," + \
                        "episode_len_mean,fairness_gini,jain_fairness_index," + \
                        "resource_specialisation,agent_overlap,task_division,policy_divergence"
                
                if useEntropyScheduler:
                    header += ",entropy_coeff"
                
                f.write(header + "\n")
        
        def reset_coordination_metrics(self):
            """Reset the coordination metrics for a new episode."""
            self.coordination_metrics = {
                "overlapping_agents": 0,
                "total_positions": 0,
                "resource_collections": {},
                "agent_actions": {}
            }
        
        def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs):
            """Called at the start of each episode."""
            # Reset episode-specific data
            episode.user_data["fair_distribution"] = {}
            episode.user_data["resource_collected"] = {}
            episode.user_data["env_conditions"] = {}
            episode.user_data["agentPositions"] = {}
            episode.user_data["collisions"] = 0
            
            # Reset coordination metrics
            self.reset_coordination_metrics()
        
        def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs):
            """Called at each step of an episode."""
            try:
                # Extract the resource collection environment
                env = base_env.get_sub_environments()[env_index]
                
                # Track agent positions for overlap calculation
                positions = {}
                for agent_id, agent in env.agents.items():
                    pos = agent.position
                    if pos not in positions:
                        positions[pos] = []
                    positions[pos].append(agent_id)
                    
                    # Track for action diversity
                    if agent_id not in self.coordination_metrics["agent_actions"]:
                        self.coordination_metrics["agent_actions"][agent_id] = {}
                
                # Calculate position overlaps
                self.coordination_metrics["total_positions"] += 1
                for pos, agents in positions.items():
                    if len(agents) > 1:
                        self.coordination_metrics["overlapping_agents"] += 1
                
                # Track resource collections
                for agent_id, agent in env.agents.items():
                    # Initialise resource_collected dict for this agent if needed
                    if "resource_collected" not in episode.user_data:
                        episode.user_data["resource_collected"] = {}
                    if agent_id not in episode.user_data["resource_collected"]:
                        episode.user_data["resource_collected"][agent_id] = {}
                    
                    for res_type, amount in agent.inventory.items():
                        prev = episode.user_data["resource_collected"].get(agent_id, {}).get(res_type, 0)
                        if amount > prev:
                            # Resource collected this step
                            if res_type not in self.coordination_metrics["resource_collections"]:
                                self.coordination_metrics["resource_collections"][res_type] = {}
                            if agent_id not in self.coordination_metrics["resource_collections"][res_type]:
                                self.coordination_metrics["resource_collections"][res_type][agent_id] = 0
                            
                            self.coordination_metrics["resource_collections"][res_type][agent_id] += 1
                    
                    # Update resource tracking
                    episode.user_data["resource_collected"][agent_id] = agent.inventory.copy()
            except Exception as e:
                print(f"Warning: Error tracking metrics in on_episode_step: {e}")
        
        def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
            """
            Called at the end of an episode to calculate and record metrics.
            
            This method calculates key coordination metrics:
            1. Agent Overlap: Measures how often agents occupy the same grid cell,
               which indicates inefficient exploration (higher is worse)
            2. Resource Specialisation: Uses Gini coefficient to measure how specialised
               agents are in collecting specific resource types (higher means more specialised)
            3. Task Division: Measures how well agents divide tasks among themselves
               (higher means more balanced workload)
            4. Policy Divergence: Measures how different agent policies have become
               (higher means more diverse behaviour)
            
            These metrics are recorded in episode.custom_metrics and later aggregated
            in the metrics.csv file for analysis and visualisation.
            
            Args:
                worker: RolloutWorker instance
                base_env: Base environment
                policies: Dict of policies
                episode: Episode instance containing data
                env_index: Environment index
                kwargs: Additional arguments
            """
            # Extract the environment
            env = base_env.get_sub_environments()[env_index]
            
            # Record metrics from environment if available
            if hasattr(env, "_calculateMetrics") and callable(getattr(env, "_calculateMetrics")):
                metrics = env._calculateMetrics()
                for k, v in metrics.items():
                    episode.custom_metrics[k] = v
            
            # Calculate coordination metrics
            # 1. Agent overlap (how often agents occupy the same cell)
            if self.coordination_metrics["total_positions"] > 0:
                overlap_ratio = self.coordination_metrics["overlapping_agents"] / self.coordination_metrics["total_positions"]
                episode.custom_metrics["agentOverlap"] = overlap_ratio
            else:
                episode.custom_metrics["agentOverlap"] = 0.0
            
            # 2. Resource specialization (Gini coefficient of resource collection by type)
            specialization_scores = []
            for res_type, collections in self.coordination_metrics["resource_collections"].items():
                if collections:
                    values = np.array(list(collections.values()))
                    if np.sum(values) > 0:
                        specialization_scores.append(calculateGiniCoefficient(values))
            
            if specialization_scores:
                episode.custom_metrics["resourceSpecialisation"] = np.mean(specialization_scores)
            else:
                episode.custom_metrics["resourceSpecialisation"] = 0.0
            
            # 3. Task division (how well agents divide tasks)
            # For now, use a placeholder based on resource specialization
            episode.custom_metrics["taskDivision"] = episode.custom_metrics.get("resourceSpecialisation", 0.0)
            
            # 4. Policy divergence (for multi-policy setups)
            # Placeholder for now - will implement proper metric later
            episode.custom_metrics["policyDivergence"] = 0.0
        
        def on_train_result(self, algorithm, result, **kwargs):
            """Log detailed metrics and update entropy if needed."""
            # Store result for visualisation
            iteration = result.get("training_iteration", 0)
            print(f"Training iteration: {iteration}")
            # Update entropy coefficient if using scheduler
            if useEntropyScheduler:
                # Get current iteration
                
                # Get updated entropy coefficient
                entropy_coeff = self.entropy_scheduler.get_entropy_coefficient(iteration)
                
                # Update policies
                try:
                    # Check if algorithm is the actual algorithm object with policies
                    if hasattr(algorithm, "policies") and algorithm.policies:
                        for policy_id in algorithm.policies.keys():
                            policy = algorithm.get_policy(policy_id)
                            if hasattr(policy, "entropy_coeff"):
                                policy.entropy_coeff = entropy_coeff
                                if hasattr(policy, "config") and isinstance(policy.config, dict):
                                    policy.config["entropy_coeff"] = entropy_coeff
                except (AttributeError, TypeError):
                    # If algorithm doesn't have policies or get_policy, log a warning
                    print("Warning: Could not update entropy coefficient - algorithm parameter is not the expected type")
                
                # Add to metrics
                result["entropy_coeff"] = entropy_coeff
                if "custom_metrics" not in result:
                    result["custom_metrics"] = {}
                result["custom_metrics"]["entropy_coeff"] = entropy_coeff
                
                print(f"[EntropyScheduler] Updated entropy to {entropy_coeff:.6f}")
            
            # Store algorithm info in global vars - safely access workers
            try:
                if hasattr(algorithm, "workers") and callable(getattr(algorithm, "workers")) and callable(getattr(algorithm.workers(), "local_worker", None)):
                    algorithm.workers().local_worker().global_vars["algorithm"] = "ppo"
                    algorithm.workers().local_worker().global_vars["training_iteration"] = iteration
            except (AttributeError, TypeError):
                # Skip if algorithm doesn't have the expected attributes/methods
                pass
            
            # Extract key metrics - prefer env_runners section which has proper aggregation
            env_runners_data = result.get("env_runners", {})
            
            # Extract key metrics with fallbacks to older locations
            episode_len_mean = env_runners_data.get("episode_len_mean", result.get("episode_len_mean", 0))
            episode_reward_mean = env_runners_data.get("episode_reward_mean", result.get("episode_reward_mean", 0))
            episode_reward_min = env_runners_data.get("episode_reward_min", result.get("episode_reward_min", 0))
            episode_reward_max = env_runners_data.get("episode_reward_max", result.get("episode_reward_max", 0))
            
            # If metrics are still 0, check alternative locations for episode stats
            if episode_len_mean == 0 and "episodes_this_iter" in env_runners_data:
                print(f"Using alternative metric sources. Found {env_runners_data['episodes_this_iter']} episodes.")
            
            # Extract custom metrics - prefer env_runners section
            custom_metrics = env_runners_data.get("custom_metrics", result.get("custom_metrics", {}))
            
            # Extract standard metrics
            fairness_gini = custom_metrics.get("fairnessGini_mean", 0)
            jain_index = custom_metrics.get("jainFairnessIndex_mean", 0)
            
            # Extract coordination metrics
            resource_specialisation = custom_metrics.get("resourceSpecialisation_mean", 0)
            agent_overlap = custom_metrics.get("agentOverlap_mean", 0)
            task_division = custom_metrics.get("taskDivision_mean", 0)
            policy_divergence = custom_metrics.get("policyDivergence_mean", 0)
            
            # Write to CSV
            with open(self.csv_file, "a") as f:
                row = f"{iteration},{episode_reward_mean},{episode_reward_min},{episode_reward_max},{episode_len_mean}," + \
                     f"{fairness_gini},{jain_index},{resource_specialisation},{agent_overlap}," + \
                     f"{task_division},{policy_divergence}"
                
                if useEntropyScheduler:
                    entropy_coeff = result.get("entropy_coeff", 0)
                    row += f",{entropy_coeff}"
                
                f.write(row + "\n")
            
            # Print concise progress report with key metrics
            print(f"Iteration {iteration}: reward={episode_reward_mean:.2f} (min={episode_reward_min:.2f}, max={episode_reward_max:.2f}), length={episode_len_mean:.1f}")
            print(f"Coordination: specialisation={resource_specialisation:.3f}, overlap={agent_overlap:.3f}, task_div={task_division:.3f}")
    
    # Add episode recorder if enabled
    if recordEpisodes:
        class CompleteTrainingCallback(UnifiedTrainingCallback, EpisodeRecorder):
            """
            Extended callback that adds episode recording functionality.
            
            This class combines the metrics tracking and coordination metrics from
            UnifiedTrainingCallback with the episode recording capabilities from
            EpisodeRecorder. It saves episode data to replay files, which can be
            later visualised using the replay tools.
            
            Episode recording is optional and can be enabled/disabled through the
            recordEpisodes parameter to control disk usage during training.
            """
            
            def __init__(self):
                UnifiedTrainingCallback.__init__(self)
                EpisodeRecorder.__init__(self, 
                    output_dir=os.path.join(outputDir, "replays"),
                    max_episodes=10  # Keep at most 10 episodes
                )
                # Flag to prevent infinite recursion
                self._in_callback = False
            
            def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs):
                """Called at the start of each episode."""
                if self._in_callback:
                    return
                    
                self._in_callback = True
                try:
                    # First call UnifiedTrainingCallback's method directly using super()
                    UnifiedTrainingCallback.on_episode_start(self, worker=worker, base_env=base_env, 
                                                        policies=policies, episode=episode, 
                                                        env_index=env_index, **kwargs)
                    
                    # Then call EpisodeRecorder's method directly
                    EpisodeRecorder.on_episode_start(self, worker=worker, base_env=base_env, 
                                                policies=policies, episode=episode, 
                                                env_index=env_index, **kwargs)
                except Exception as e:
                    print(f"Warning: Error in CompleteTrainingCallback.on_episode_start: {e}")
                finally:
                    self._in_callback = False
            
            def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs):
                """Called at each step of an episode."""
                if self._in_callback:
                    return
                    
                self._in_callback = True
                try:
                    # Call both parent methods directly
                    UnifiedTrainingCallback.on_episode_step(self, worker=worker, base_env=base_env, 
                                                        episode=episode, env_index=env_index, **kwargs)
                    
                    EpisodeRecorder.on_episode_step(self, worker=worker, base_env=base_env, 
                                                episode=episode, env_index=env_index, **kwargs)
                except Exception as e:
                    print(f"Warning: Error in CompleteTrainingCallback.on_episode_step: {e}")
                finally:
                    self._in_callback = False
            
            def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
                """Called at the end of an episode."""
                if self._in_callback:
                    return
                    
                self._in_callback = True
                try:
                    # Call both parent methods directly
                    UnifiedTrainingCallback.on_episode_end(self, worker=worker, base_env=base_env, 
                                                       policies=policies, episode=episode, 
                                                       env_index=env_index, **kwargs)
                    
                    EpisodeRecorder.on_episode_end(self, worker=worker, base_env=base_env, 
                                               policies=policies, episode=episode, 
                                               env_index=env_index, **kwargs)
                except Exception as e:
                    print(f"Warning: Error in CompleteTrainingCallback.on_episode_end: {e}")
                finally:
                    self._in_callback = False
            
            def on_postprocess_trajectory(self, *, worker, episode, agent_id, policy_id, policies, postprocessed_batch, original_batches, **kwargs):
                """Called after processing a trajectory."""
                if self._in_callback:
                    return
                    
                self._in_callback = True
                try:
                    # Only call EpisodeRecorder since UnifiedTrainingCallback doesn't implement this
                    EpisodeRecorder.on_postprocess_trajectory(
                        self, worker=worker, episode=episode, agent_id=agent_id,
                        policy_id=policy_id, policies=policies,
                        postprocessed_batch=postprocessed_batch,
                        original_batches=original_batches, **kwargs
                    )
                except Exception as e:
                    print(f"Warning: Error in CompleteTrainingCallback.on_postprocess_trajectory: {e}")
                finally:
                    self._in_callback = False
                
            def on_train_result(self, *, algorithm, result, **kwargs):
                """Called after each training iteration."""
                if self._in_callback:
                    return
                    
                self._in_callback = True
                try:
                    # Call both parent implementations
                    UnifiedTrainingCallback.on_train_result(self, algorithm=algorithm, result=result, **kwargs)
                    
                    EpisodeRecorder.on_train_result(self, algorithm=algorithm, result=result, **kwargs)
                except Exception as e:
                    print(f"Warning: Error in CompleteTrainingCallback.on_train_result: {e}")
                finally:
                    self._in_callback = False
        
        # Use the complete callback with episode recording
        callback_class = CompleteTrainingCallback
    else:
        # Use the unified callback without episode recording
        callback_class = UnifiedTrainingCallback
    
    # Use appropriate callback configuration for Ray version
    ppoPolicy.config["callbacks_class"] = callback_class
    
    # Get the final configuration
    finalConfig = ppoPolicy.getConfig()
    
    # Save configuration
    try:
        configPath = saveConfig(finalConfig, outputDir)
        print(f"Configuration saved to {configPath}")
    except Exception as e:
        print(f"Warning: Could not save configuration: {e}")
        print("Training will continue, but configuration may not be properly saved")
    
    # Create trainer
    trainer = ppoPolicy.getTrainer()
    
    # Resume from checkpoint if specified
    if resume and checkpointPath:
        print(f"Resuming training from checkpoint: {checkpointPath}")
        try:
            trainer.restore(checkpointPath)
            print("Successfully restored checkpoint!")
        except Exception as e:
            print(f"Error restoring checkpoint: {e}")
            
            # Check if the file exists
            if not os.path.exists(checkpointPath):
                print(f"Checkpoint file {checkpointPath} does not exist.")
                sys.exit(1)
                
            # If this is a reference file (JSON or Python), try to find the actual checkpoint it references
            if checkpointPath.endswith(".json") or checkpointPath.endswith(".py"):
                try:
                    checkpoint_data = {}
                    
                    # Handle different file types
                    if checkpointPath.endswith(".json"):
                        # Try to read the JSON to find the checkpoint path
                        with open(checkpointPath, "r") as f:
                            checkpoint_data = json.load(f)
                    elif checkpointPath.endswith(".py"):
                        # Try to load the Python module
                        import importlib.util
                        module_name = os.path.splitext(os.path.basename(checkpointPath))[0]
                        spec = importlib.util.spec_from_file_location(module_name, checkpointPath)
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            sys.modules[module_name] = module
                            spec.loader.exec_module(module)
                            
                            # Check for checkpoint data
                            if hasattr(module, "CHECKPOINT_DATA"):
                                checkpoint_data = module.CHECKPOINT_DATA
                            elif hasattr(module, "getCheckpointData"):
                                checkpoint_data = module.getCheckpointData()
                    
                    # Check if it contains a checkpoint_path field
                    if "checkpoint_path" in checkpoint_data:
                        actualCheckpointPath = checkpoint_data["checkpoint_path"]
                        print(f"Reference file points to checkpoint at: {actualCheckpointPath}")
                        
                        # Try to restore using the actual path
                        try:
                            trainer.restore(actualCheckpointPath)
                            print("Successfully restored using referenced checkpoint!")
                        except Exception as inner_e:
                            print(f"Error restoring referenced checkpoint: {inner_e}")
                            sys.exit(1)
                    else:
                        print("Reference file does not contain a valid checkpoint_path field.")
                        sys.exit(1)
                except Exception as ref_e:
                    print(f"Error parsing checkpoint reference file: {ref_e}")
                    sys.exit(1)
            else:
                # Not a reference file and restoration failed
                print("Could not restore checkpoint. Exiting.")
                sys.exit(1)
        except Exception as e:
            print(f"Error restoring checkpoint: {e}")
            
    # Training loop
    print(f"Starting training for {iterations} iterations...")
    print(f"Results will be saved to: {outputDir}")
    print(f"TensorBoard logs available at: {tensorboard_dir}")
    print(f"Run 'tensorboard --logdir={tensorboard_dir}' to view live training metrics")
    
    checkpoint_freq = 100  # Save checkpoints every 100 iterations 
    try:
        for i in range(iterations):
            # Set the current iteration in global vars for callbacks to access
            try:
                # New Ray API (2.0+)
                if hasattr(trainer, "workers") and callable(getattr(trainer, "workers")):
                    trainer.workers().local_worker().global_vars["training_iteration"] = i
                # Even newer Ray API (2.4+) where workers might be a property
                elif hasattr(trainer, "get_policy") and callable(getattr(trainer, "get_policy")):
                    # Access global_vars through policy if available
                    policy = trainer.get_policy()
                    if hasattr(policy, "global_vars"):
                        policy.global_vars["training_iteration"] = i
                # Fallback option
                else:
                    # Just store in the trainer itself as a fallback
                    if not hasattr(trainer, "global_vars"):
                        trainer.global_vars = {}
                    trainer.global_vars["training_iteration"] = i
                    print(f"Using trainer.global_vars fallback for iteration {i}")
            except Exception as e:
                print(f"Warning: Could not set training_iteration in global_vars: {e}")
                print("This won't affect training, just some logging/metrics")
            
            # Train one iteration
            result = trainer.train()
            
            # Save checkpoint periodically
            if (i + 1) % checkpoint_freq == 0 or (i + 1) == iterations:
                checkpoint = trainer.save(os.path.join(outputDir, "checkpoints"))
                print(f"Checkpoint saved at: {checkpoint}")
        
        # Training completed
        print(f"Training completed. Final checkpoint saved.")
        
        # Plot training progress
        visualiseTrainingProgress(outputDir)
    
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving final checkpoint...")
        trainer.save(os.path.join(outputDir, "checkpoints", "interrupted"))
    
    finally:
        # Cleanup
        trainer.stop()


def run_ablation_study(
    ablationType: str,
    baseOutputDir: str = None,
    iterations: int = 200,
    numAgents: int = 4,
    gridSize: tuple = (15, 15),
    seed: int = None,
    useEntropyScheduler: bool = True,
    initialEntropy: float = 0.01,
    finalEntropy: float = 0.001,
    entropyScheduleType: str = "linear",
    useGpu: bool = False,
    numWorkers: int = 8,
    evaluationNumWorkers: int = 1,
    configPath: str = None
):
    """
    Run an ablation study comparing different PPO configurations.
    
    Args:
        ablationType: Type of ablation to perform ("critic", "reward", "policy")
        baseOutputDir: Base directory to save results
        iterations: Number of training iterations for each variant
        numAgents: Number of agents in the environment
        gridSize: Grid size for the environment
        seed: Random seed for reproducibility
        useEntropyScheduler: Whether to use entropy scheduling
        initialEntropy: Initial entropy coefficient (if using scheduler)
        finalEntropy: Final entropy coefficient (if using scheduler)
        entropyScheduleType: Type of entropy scheduling (if using scheduler)
        useGpu: Whether to use GPU for training
        numWorkers: Number of parallel workers for training
        evaluationNumWorkers: Number of parallel workers for evaluation
        configPath: Path to configuration file for the ablation study
    """
    # Create output directory
    if baseOutputDir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        baseOutputDir = f"results/ablation_{ablationType}_{timestamp}"
    
    os.makedirs(baseOutputDir, exist_ok=True)
    print(f"Saving ablation results to: {baseOutputDir}")
    
    # Define variants based on ablation type
    variants = []
    
    if ablationType == "critic":
        variants = [
            ("centralised_critic", True, False, "hybrid", 0.5),
            ("decentralised_critic", False, False, "hybrid", 0.5)
        ]
    elif ablationType == "reward":
        variants = [
            ("individual_reward", True, False, "individual", None),
            ("global_reward", True, False, "global", None),
            ("hybrid_reward_25", True, False, "hybrid", 0.25),
            ("hybrid_reward_50", True, False, "hybrid", 0.5),
            ("hybrid_reward_75", True, False, "hybrid", 0.75),
        ]
    else:
        print(f"Unknown ablation type: {ablationType}")
        return
    
    # Save ablation configuration
    ablation_config = {
        "ablation_type": ablationType,
        "num_agents": numAgents,
        "grid_size": gridSize,
        "iterations": iterations,
        "seed": seed,
        "variants": [v[0] for v in variants],
        "use_entropy_scheduler": useEntropyScheduler,
        "entropy_scheduler_config": {
            "initial_entropy": initialEntropy,
            "final_entropy": finalEntropy,
            "schedule_type": entropyScheduleType
        } if useEntropyScheduler else None,
        "use_gpu": useGpu,
        "evaluation_num_workers": evaluationNumWorkers
    }
    
    # Save as Python file only
    with open(os.path.join(baseOutputDir, "ablation_config.py"), "w") as f:
        f.write("# Auto-generated ablation configuration file\n")
        f.write("# Do not edit manually\n\n")
        f.write("CONFIG = ")
        f.write(pprint.pformat(ablation_config, indent=4, width=100))
        f.write("\n\n# Export the config for easy import\ndef getConfig():\n    return CONFIG\n")
    
    # Run each variant
    for variant_info in variants:
        # Unpack variant info
        variant_name = variant_info[0]
        use_centralised_critic = variant_info[1]
        use_attention = variant_info[2]
        reward_type = variant_info[3]
        hybrid_mix = variant_info[4]
        
        # Create output directory for this variant
        variant_dir = os.path.join(baseOutputDir, variant_name)
        os.makedirs(variant_dir, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"Running variant: {variant_name}")
        print(f"Centralised critic: {use_centralised_critic}")
        print(f"Attention mechanism: {use_attention}")
        print(f"Reward type: {reward_type}" + (f", mix: {hybrid_mix}" if hybrid_mix is not None else ""))
        print(f"Entropy scheduler: {useEntropyScheduler}")
        print(f"{'='*80}\n")
        
        # Run training for this variant
        train_main(
            outputDir=variant_dir,
            iterations=iterations,
            numAgents=numAgents,
            seed=seed,
            gridSize=gridSize,
            rewardType=reward_type,
            hybridRewardMix=hybrid_mix if hybrid_mix is not None else 0.5,
            useCentralisedCritic=use_centralised_critic,
            useAttention=use_attention,
            useGpu=useGpu,
            useEntropyScheduler=useEntropyScheduler,
            initialEntropy=initialEntropy,
            finalEntropy=finalEntropy,
            entropyScheduleType=entropyScheduleType,
            recordEpisodes=True,  # No episode recording for ablation studies
            evaluationNumWorkers=evaluationNumWorkers,
            numWorkers=numWorkers,
            configPath=configPath
        )
    
    print(f"\nAblation study completed. Results saved to {baseOutputDir}")
    print("To compare results, use the visualisation tools in the GUI or analysis scripts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on the resource collection environment")
    
    # Core training parameters
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--output-dir", type=str, help="Directory to save results and checkpoints")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    
    # Environment configuration
    parser.add_argument("--num-agents", type=int, default=4, help="Number of agents in the environment")
    parser.add_argument("--grid-size", type=int, nargs=2, default=[15, 15], help="Grid size (width height)")
    parser.add_argument("--reward-type", type=str, choices=["individual", "global", "hybrid"], default="hybrid",
                       help="Type of reward structure")
    parser.add_argument("--hybrid-mix", type=float, default=0.5, 
                       help="Mix ratio for hybrid rewards (0.0 = global, 1.0 = individual)")
    
    # PPO configuration
    parser.add_argument("--no-centralised-critic", action="store_true", help="Disable centralised critic (use standard PPO)")
    parser.add_argument("--use-attention", action="store_true", help="Use attention mechanism in centralised critic")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for training if available")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of parallel workers for training")
    
    # Entropy scheduling
    parser.add_argument("--entropy-scheduler", action="store_true", help="Use entropy scheduling")
    parser.add_argument("--initial-entropy", type=float, default=0.01, help="Initial entropy coefficient")
    parser.add_argument("--final-entropy", type=float, default=0.001, help="Final entropy coefficient")
    parser.add_argument("--entropy-schedule", type=str, choices=["linear", "exponential", "step"], default="linear",
                       help="Type of entropy scheduling")
    
    # Evaluation parameters
    parser.add_argument("--evaluation-interval", type=int, default=5, help="How often to run evaluation (iterations)")
    parser.add_argument("--evaluation-duration", type=int, default=10, help="Number of episodes to run during evaluation")
    parser.add_argument("--evaluation-num-workers", type=int, default=1, help="Number of parallel workers for evaluation")
    
    # Ablation mode
    parser.add_argument("--ablation", type=str, choices=["critic", "reward"], 
                       help="Run in ablation mode comparing different configurations")
    
    # Additional options
    parser.add_argument("--no-record-episodes", action="store_true", help="Enable episode recording")
    
    args = parser.parse_args()
    
    # Convert grid size to tuple
    grid_size = tuple(args.grid_size)
    
    train_main(
        configPath=args.config,
        outputDir=args.output_dir,
        iterations=args.iterations,
        resume=args.resume,
        checkpointPath=args.checkpoint,
        numAgents=args.num_agents,
        seed=args.seed,
        
        gridSize=grid_size,
        rewardType=args.reward_type,
        hybridRewardMix=args.hybrid_mix,
        
        useCentralisedCritic=not args.no_centralised_critic,
        useAttention=args.use_attention,
        useGpu=args.use_gpu,
        
        useEntropyScheduler=args.entropy_scheduler,
        initialEntropy=args.initial_entropy,
        finalEntropy=args.final_entropy,
        entropyScheduleType=args.entropy_schedule,
        
        ablationMode=args.ablation,
        
        evaluationInterval=args.evaluation_interval,
        evaluationDuration=args.evaluation_duration,
        evaluationNumWorkers=args.evaluation_num_workers,
        recordEpisodes=not args.no_record_episodes,
        numWorkers=args.num_workers
    ) 