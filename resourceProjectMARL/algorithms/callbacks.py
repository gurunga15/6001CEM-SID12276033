"""
Custom RLlib callbacks for tracking metrics in multi-agent reinforcement learning.

This module provides callbacks that track and log various performance metrics during 
training and evaluation of multi-agent reinforcement learning algorithms.

Key metrics tracked:
- Resource collection efficiency (resources per timestep)
- Fairness in resource distribution (using multiple fairness indices)
- Collision rates (safety metric)
- Resource specialisation (how agents specialise in collecting different resources)
- Environment adaptation (performance in different conditions)
"""

import os
import time
import numpy as np
from typing import Dict, Any, Optional, List, Set, Tuple
from datetime import datetime
import random
import pickle

import ray
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import PolicyID, AgentID

# Import utility functions for fairness calculation
try:
    # First try relative import
    from ..env.utils import calculateGiniCoefficient, calculateJainFairnessIndex
except ImportError:
    # Then try direct import if env is in sys.path
    try:
        from env.utils import calculateGiniCoefficient, calculateJainFairnessIndex
    except ImportError:
        # Fallback to absolute import with parent directory
        import sys
        import os
        
        # Add parent directory to path if needed
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
            
        # Now try import with modified path
        from env.utils import calculateGiniCoefficient, calculateJainFairnessIndex


class MetricsCallback(DefaultCallbacks):
    """
    Custom callback for tracking various metrics during training.
    
    This callback extends RLlib's DefaultCallbacks to track custom metrics
    such as fairness, coordination efficiency, and environment-specific data.
    
    Key metrics include:
    - Resource collection (total and per-step)
    - Fairness indices (Gini coefficient and Jain's index)
    - Collision rates
    - Resource specialisation
    - Environmental condition performance
    """
    
    def on_environment_created(self, *, algorithm=None, env=None, env_context=None, **kwargs):
        """
        Called when a new environment is created.
        
        Required for Ray 2.42.0 compatibility.
        
        Args:
            algorithm: RLlib algorithm instance (may be None in some contexts)
            env: The created environment (may be None in some contexts)
            env_context: Environment context
            kwargs: Additional arguments
        """
        # No-op, just needed for Ray 2.42.0 compatibility
        pass
    
    def on_episode_start(
        self, 
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: EpisodeV2,
        env_index: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Called at the start of each episode.
        
        Initialises episode-specific metrics trackers in episode.user_data.
        
        Args:
            worker: RolloutWorker instance
            base_env: BaseEnv instance
            policies: Dict of policies
            episode: Episode instance
            env_index: Environment index
            kwargs: Additional arguments
        """
        # Initialise episode-specific variables in episode.user_data
        episode.user_data["resourcesCollected"] = {}
        episode.user_data["resourcesByType"] = {}  # Track resources by type for each agent
        episode.user_data["resourceTypeCollected"] = {}  # Track total by resource type
        episode.user_data["collisions"] = 0
        episode.user_data["coordinationEvents"] = 0
        episode.user_data["fairnessHistory"] = []
        episode.user_data["jainFairnessHistory"] = []
        episode.user_data["environmentalConditions"] = {}
        
        # Initialise per-agent tracking
        env = base_env.get_sub_environments()[env_index]
        if hasattr(env, "agents"):
            episode.user_data["agentTracking"] = {
                agent_id: {
                    "resources": 0,
                    "resourcesByType": {},
                    "collisions": 0,
                    "distance": 0
                } for agent_id in env.agents
            }
        
        # Mark start time for calculating episodes per second
        episode.user_data["startTime"] = time.time()
    
    def on_episode_step(
        self, 
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: EpisodeV2,
        env_index: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Called at each step of an episode.
        
        Tracks resource collection, agent positions, and environmental conditions.
        
        Args:
            worker: RolloutWorker instance
            base_env: Base environment
            episode: Episode instance containing data
            env_index: Environment index
            kwargs: Additional arguments
        """
        # Extract the resource collection environment
        env = base_env.get_sub_environments()[env_index]
        
        # Track resource collection by agent
        for agentId, agent in env.agents.items():
            # Initialise agent data structures if first time seeing this agent
            if agentId not in episode.user_data["resourcesCollected"]:
                episode.user_data["resourcesCollected"][agentId] = 0
                episode.user_data["resourcesByType"][agentId] = {}
                
            # Track total collected
            totalCollected = agent.totalCollected
            previousCollected = episode.user_data["resourcesCollected"].get(agentId, 0)
            
            if totalCollected > previousCollected:
                episode.user_data["resourcesCollected"][agentId] = totalCollected
                
                # Track resources collected by type
                for resourceType, amount in agent.inventory.items():
                    # Update for this agent
                    previousAmount = episode.user_data["resourcesByType"].get(agentId, {}).get(resourceType, 0)
                    
                    if amount > previousAmount:
                        # Resources collected this step
                        if agentId not in episode.user_data["resourcesByType"]:
                            episode.user_data["resourcesByType"][agentId] = {}
                        
                        episode.user_data["resourcesByType"][agentId][resourceType] = amount
                        
                        # Update total collection for this resource type
                        if resourceType not in episode.user_data["resourceTypeCollected"]:
                            episode.user_data["resourceTypeCollected"][resourceType] = 0
                        
                        episode.user_data["resourceTypeCollected"][resourceType] += (amount - previousAmount)
                        
                        # Track resource in custom metrics for visualisation
                        metricKey = f"collected_{resourceType}"
                        if metricKey not in episode.custom_metrics:
                            episode.custom_metrics[metricKey] = 0
                        episode.custom_metrics[metricKey] += (amount - previousAmount)
        
        # Calculate current fair distribution
        resourcesByAgent = {agentId: data for agentId, data in episode.user_data["resourcesCollected"].items()}
        
        if resourcesByAgent and sum(resourcesByAgent.values()) > 0:
            # Track resources by agent considering the reward type
            # Get reward type and mix from environment (same as in _calculate_rewards)
            resourceValues = np.array(list(resourcesByAgent.values()))
            
            # Calculate fairness metrics that match the environment's reward calculation
            rewardType = getattr(env, "rewardType", "individual")
            hybridRewardMix = getattr(env, "hybridRewardMix", 0.5)
            
            # Calculate effective rewards that match the environment's _calculate_rewards logic
            if rewardType == "shared":
                # All agents get the team's total reward
                teamReward = sum(resourceValues)
                effectiveRewards = np.full_like(resourceValues, teamReward)
            elif rewardType == "hybrid":
                # Mix of individual and shared rewards
                teamReward = sum(resourceValues)
                effectiveRewards = (1 - hybridRewardMix) * resourceValues + hybridRewardMix * teamReward
            else:  # individual
                effectiveRewards = resourceValues
            
            # Calculate metrics based on effective rewards
            fairnessGini = 1.0 - calculateGiniCoefficient(effectiveRewards)
            jainFairnessIndex = calculateJainFairnessIndex(effectiveRewards)
            
            # Store fairness values
            episode.user_data["fairnessHistory"].append(fairnessGini)
            episode.user_data["jainFairnessHistory"].append(jainFairnessIndex)
            
            # Update current fair distribution
            episode.user_data["fair_distribution"] = dict(zip(resourcesByAgent.keys(), effectiveRewards))
        
        # Track collisions
        collisions = sum(agent.collisions for agent in env.agents.values())
        episode.user_data["collisions"] = collisions
        
        # Track environmental conditions
        if hasattr(env, "currentWeather"):
            episode.user_data["environmentalConditions"]["weather"] = env.currentWeather
        
        if hasattr(env, "currentSeason"):
            episode.user_data["environmentalConditions"]["season"] = env.currentSeason
        
        if hasattr(env, "dayPhase"):
            episode.user_data["environmentalConditions"]["dayPhase"] = env.dayPhase
        
        # Record agent positions for overlap calculation
        for agentId, agent in env.agents.items():
            if "agentPositions" not in episode.user_data:
                episode.user_data["agentPositions"] = {}
            
            if agentId not in episode.user_data["agentPositions"]:
                episode.user_data["agentPositions"][agentId] = []
            
            episode.user_data["agentPositions"][agentId].append(agent.position)
    
    def on_episode_end(
        self, 
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: EpisodeV2,
        env_index: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Called at the end of an episode.
        
        Calculates final metrics for the episode, including:
        - Resource collection efficiency
        - Fairness (Gini coefficient)
        - Specialisation
        - Collisions
        
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
        
        # Calculate episode duration
        episodeDuration = time.time() - episode.user_data["startTime"]
        
        # Get the environment
        env = base_env.get_sub_environments()[env_index]
        
        # Efficiency and collection metrics
        totalResources = sum(episode.user_data["resourcesCollected"].values()) if episode.user_data["resourcesCollected"] else 0
        resourcesPerStep = totalResources / max(1, episode.length)
        
        # Get reward type and mix from environment
        rewardType = getattr(env, "rewardType", "individual")
        hybridRewardMix = getattr(env, "hybridRewardMix", 0.5)
        
        # Get fairness metrics
        if episode.user_data["resourcesCollected"]:
            resourceValues = np.array(list(episode.user_data["resourcesCollected"].values()))
            
            # Calculate effective rewards based on reward type (same as in environment)
            if rewardType == "shared":
                # All agents get the team's total reward
                teamReward = sum(resourceValues)
                effectiveRewards = np.full_like(resourceValues, teamReward)
            elif rewardType == "hybrid":
                # Mix of individual and shared rewards
                teamReward = sum(resourceValues)
                effectiveRewards = (1 - hybridRewardMix) * resourceValues + hybridRewardMix * teamReward
            else:  # individual
                effectiveRewards = resourceValues
            
            # Use environment metrics if available
            if hasattr(env, "_calculateMetrics"):
                metrics = env._calculateMetrics()
                fairnessGini = metrics.get("fairnessGini", 1.0)
                jainFairnessIndex = metrics.get("jainFairnessIndex", 1.0)
            else:
                # Calculate metrics ourselves using effective rewards
                fairnessGini = 1.0 - calculateGiniCoefficient(effectiveRewards) if np.sum(effectiveRewards) > 0 else 1.0
                jainFairnessIndex = calculateJainFairnessIndex(effectiveRewards)
            
            # Coefficient of variation (standard deviation / mean)
            if np.mean(effectiveRewards) > 0:
                resourceCV = np.std(effectiveRewards) / np.mean(effectiveRewards)
            else:
                resourceCV = 0.0
        else:
            # Default values if no resources collected
            fairnessGini = 1.0  # Perfect fairness (all agents got 0)
            jainFairnessIndex = 1.0
            resourceCV = 0.0
        
        # Use average fairness if we've been tracking it
        if episode.user_data["fairnessHistory"]:
            fairnessGini = np.mean(episode.user_data["fairnessHistory"])
        
        if episode.user_data["jainFairnessHistory"]:
            jainFairnessIndex = np.mean(episode.user_data["jainFairnessHistory"])
        
        # Calculate resource specialisation metric
        resourceSpecialisation = 0.0
        if episode.user_data["resourcesByType"]:
            # Calculate specialisation by checking which agent collected most of each resource type
            specialisationScores = []
            for agentId, resourceTypes in episode.user_data["resourcesByType"].items():
                if resourceTypes:  # Skip if agent didn't collect any resources
                    # Create array of agent's resource collections by type
                    agentSpecialisation = []
                    for resType, amount in resourceTypes.items():
                        # Calculate what percentage of this resource type this agent collected
                        totalOfType = episode.user_data["resourceTypeCollected"].get(resType, 0)
                        if totalOfType > 0:
                            agentSpecialisation.append(amount / totalOfType)
                        else:
                            agentSpecialisation.append(0.0)
                    
                    # Calculate Gini coefficient for this agent's specialisation
                    if agentSpecialisation and np.sum(agentSpecialisation) > 0:
                        specialisationScores.append(calculateGiniCoefficient(np.array(agentSpecialisation)))
            
            if specialisationScores:
                resourceSpecialisation = np.mean(specialisationScores)
        
        # Collision rate
        collisionsPerStep = episode.user_data["collisions"] / max(1, episode.length)
        
        # Environment-specific performance
        envConditions = episode.user_data["environmentalConditions"]
        for condition, value in envConditions.items():
            episode.custom_metrics[f"condition_{condition}"] = value
            # Tag reward with environment condition
            episode.custom_metrics[f"reward_by_{condition}_{value}"] = episode.total_reward
        
        # Record custom metrics (will be automatically aggregated across episodes)
        episode.custom_metrics["totalResourcesCollected"] = totalResources
        episode.custom_metrics["resourcesPerStep"] = resourcesPerStep
        episode.custom_metrics["fairnessGini"] = fairnessGini
        episode.custom_metrics["jainFairnessIndex"] = jainFairnessIndex
        episode.custom_metrics["resourceDistributionCV"] = resourceCV
        episode.custom_metrics["resourceSpecialisation"] = resourceSpecialisation
        episode.custom_metrics["collisionsPerStep"] = collisionsPerStep
        episode.custom_metrics["episodeDuration"] = episodeDuration
    
    def on_train_result(
        self, 
        *,
        algorithm,
        result: dict,
        **kwargs
    ) -> None:
        """
        Called at the end of Algorithm.train().
        
        Aggregates metrics across episodes and computes summary statistics.
        
        Args:
            algorithm: Algorithm instance
            result: Training result dict
            kwargs: Additional arguments
        """
        # Extract custom metrics if available
        if "custom_metrics" in result:
            metrics = result["custom_metrics"]
            
            # Compute fairness averages across episodes if available
            if "fairnessGini_mean" in metrics:
                result["custom_metrics"]["fairness_summary"] = metrics["fairnessGini_mean"]
            
            if "jainFairnessIndex_mean" in metrics:
                result["custom_metrics"]["jain_fairness_summary"] = metrics["jainFairnessIndex_mean"]
            
            # Compute collection efficiency
            if "resourcesPerStep_mean" in metrics:
                result["custom_metrics"]["collection_efficiency"] = metrics["resourcesPerStep_mean"]
            
            # Compute specialisation summary
            if "resourceSpecialisation_mean" in metrics:
                result["custom_metrics"]["specialisation_summary"] = metrics["resourceSpecialisation_mean"]
            
            # Compute environmental performance summaries
            # Extract all condition metrics
            condition_metrics = {k: v for k, v in metrics.items() if k.startswith("reward_by_")}
            for condition, value in condition_metrics.items():
                # Add to summary for easier tracking
                result["custom_metrics"][f"{condition}_summary"] = value
        
        # Log core metrics to make them more visible in console output
        result["meanEpisodeLength"] = result["episode_len_mean"]
        result["meanEpisodeReward"] = result["episode_reward_mean"]
        
        # Add fairness metrics to top-level for easier reference
        if "custom_metrics" in result and "fairnessGini_mean" in result["custom_metrics"]:
            result["meanFairness"] = result["custom_metrics"]["fairnessGini_mean"]
        
        if "custom_metrics" in result and "jainFairnessIndex_mean" in result["custom_metrics"]:
            result["meanJainFairness"] = result["custom_metrics"]["jainFairnessIndex_mean"]


class TrainingProgressCallback(DefaultCallbacks):
    """
    Callback to track and log training progress and timing information.
    
    Provides real-time feedback during training, including:
    - Time elapsed and estimated time remaining
    - Current performance metrics (rewards, episode length)
    - Key custom metrics (fairness, collisions, specialisation)
    
    This helps monitor training progress and detect issues early.
    """
    
    def __init__(self, log_interval: int = 10):
        """
        Initialise the progress tracking callback.
        
        Args:
            log_interval: How often to log detailed progress (iterations)
        """
        super().__init__()
        self.log_interval = log_interval
        self.lastLogTime = time.time()
        self.trainingStartTime = None
        self.iterationTimes = []
    
    def on_environment_created(self, *, algorithm=None, env=None, env_context=None, **kwargs):
        """
        Called when a new environment is created.
        
        Required for Ray 2.42.0 compatibility.
        
        Args:
            algorithm: RLlib algorithm instance (may be None in some contexts)
            env: The created environment (may be None in some contexts)
            env_context: Environment context
            kwargs: Additional arguments
        """
        # No-op, just needed for Ray 2.42.0 compatibility
        pass
    
    def on_algorithm_init(self, *, algorithm, **kwargs) -> None:
        """
        Called when the algorithm is initialised.
        
        Marks the start time of training and logs a message.
        """
        self.trainingStartTime = time.time()
        print(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def on_train_result(self, *, algorithm, result: dict, **kwargs) -> None:
        """
        Called at the end of Algorithm.train().
        
        Logs progress information and estimates remaining time.
        
        Args:
            algorithm: Algorithm instance
            result: Training result dict
            kwargs: Additional arguments
        """
        # Record the time for this iteration
        currentTime = time.time()
        iterationTime = currentTime - self.lastLogTime
        self.iterationTimes.append(iterationTime)
        self.lastLogTime = currentTime
        
        # Current training iteration
        itr = result["training_iteration"]
        
        # Calculate timing statistics
        totalTrainingTime = currentTime - self.trainingStartTime
        averageIterationTime = np.mean(self.iterationTimes[-10:])  # Average of last 10
        estimatedTimeRemaining = averageIterationTime * (algorithm.config.training_iterations - itr)
        
        # Add timing info to results
        result["time_this_iter_s"] = iterationTime
        result["time_total_s"] = totalTrainingTime
        result["time_remaining_s"] = estimatedTimeRemaining
        
        # Log detailed progress periodically
        if itr % self.log_interval == 0:
            # Format times as readable strings
            totalTimeStr = self._formatTime(totalTrainingTime)
            remainingTimeStr = self._formatTime(estimatedTimeRemaining)
            
            # Create detailed progress message
            progressMsg = (
                f"\n===== Training Progress: Iteration {itr}/{algorithm.config.training_iterations} ====="
                f"\nTime elapsed: {totalTimeStr}, Estimated time remaining: {remainingTimeStr}"
                f"\nMean episode reward: {result['episode_reward_mean']:.2f} "
                f"(min: {result['episode_reward_min']:.2f}, max: {result['episode_reward_max']:.2f})"
                f"\nMean episode length: {result['episode_len_mean']:.1f} steps"
            )
            
            # Add custom metrics if available
            if "custom_metrics" in result:
                metrics = result["custom_metrics"]
                
                # Add fairness metrics if available
                if "fairnessGini_mean" in metrics:
                    fairness = metrics["fairnessGini_mean"]
                    progressMsg += f"\nFairness (Gini index): {fairness:.4f} (higher is better)"
                
                if "jainFairnessIndex_mean" in metrics:
                    jainFairness = metrics["jainFairnessIndex_mean"]
                    progressMsg += f"\nFairness (Jain's index): {jainFairness:.4f} (higher is better)"
                
                # Add collision metrics if available
                if "collisionsPerStep_mean" in metrics:
                    collisions = metrics["collisionsPerStep_mean"]
                    progressMsg += f"\nCollisions per step: {collisions:.4f}"
                
                # Add resource collection metrics if available
                if "resourcesPerStep_mean" in metrics:
                    resourcesPerStep = metrics["resourcesPerStep_mean"]
                    progressMsg += f"\nResource collection rate: {resourcesPerStep:.4f} resources/step"
                    
                if "totalResourcesCollected_mean" in metrics:
                    totalResources = metrics["totalResourcesCollected_mean"]
                    progressMsg += f"\nTotal resources collected: {totalResources:.1f}"
                
                # Add specialisation metric if available
                if "resourceSpecialisation_mean" in metrics:
                    specialisation = metrics["resourceSpecialisation_mean"]
                    progressMsg += f"\nResource specialisation: {specialisation:.4f}"
                    
                # Add environment condition performance if available
                envConditionMetrics = [k for k in metrics.keys() if k.startswith("reward_by_") and k.endswith("_mean")]
                if envConditionMetrics and len(envConditionMetrics) <= 4:  # Only show if not too many
                    progressMsg += "\nPerformance by environment condition:"
                    for metricKey in envConditionMetrics:
                        # Extract condition name from metric key
                        conditionName = metricKey.replace("reward_by_", "").replace("_mean", "")
                        progressMsg += f"\n  - {conditionName}: {metrics[metricKey]:.2f}"
            
            # Print the progress message
            print(progressMsg)
    
    def on_algorithm_restore(self, *, algorithm, **kwargs) -> None:
        """
        Called when the algorithm is restored from a checkpoint.
        
        Logs a message and resets timing.
        """
        print(f"\nTraining resumed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.lastLogTime = time.time()
    
    def _formatTime(self, seconds: float) -> str:
        """
        Format seconds as a readable time string.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string (HH:MM:SS)
        """
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}" 


class EpisodeRecorder(DefaultCallbacks):
    """Record episodes for later playback."""
    
    def __init__(self, output_dir: str = "./results/replays", max_episodes: int = 10, replay_on_episode_end: bool = False):
        """
        Initialise the episode recorder.
        
        Args:
            output_dir: Directory to save replay files
            max_episodes: Maximum number of episodes to record
            replay_on_episode_end: Whether to automatically replay episodes after they end
        """
        super().__init__()
        self.output_dir = output_dir
        self.max_episodes = max_episodes
        self.replay_on_episode_end = replay_on_episode_end
        self.recording = False
        self.current_recording = None
        self.recorded_episodes = []
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        """Start recording a new episode."""
        if len(self.recorded_episodes) >= self.max_episodes:
            self.recording = False
            episode.user_data["recording"] = False
            return
            
        # Choose whether to record this episode (randomly)
        # Use 33% probability (1/3 chance) of recording an episode
        if random.random() < 0.33:
            self.recording = True
            
            # Ensure episode.user_data exists
            if not hasattr(episode, "user_data"):
                episode.user_data = {}
                
            # Initialise metadata
            self.current_recording = {
                "observations": [{"__initial__": True}],  # Mark first step as initial
                "actions": [],
                "rewards": [],
                "infos": [],
                "metadata": {
                    "episode_id": episode.episode_id,
                    "training_iteration": 0,  # This will be updated later if available
                    "timestamp": time.time(),
                    "algorithm": "ppo",  # Default, can be updated
                    "episode_reward": 0.0,  # Will be updated at episode end
                    "episode_length": 0,  # Will be updated at episode end
                    "metrics": {}  # Will be filled with additional metrics
                }
            }
            
            # Get a list of agents - safely handle if get_agents method doesn't exist
            agent_ids = []
            try:
                # Try episode's get_agents method first
                agent_ids = episode.get_agents()
            except AttributeError:
                # Try to get agents from environment
                try:
                    env = base_env.get_sub_environments()[env_index]
                    if hasattr(env, "agents"):
                        agent_ids = list(env.agents.keys())
                    elif hasattr(env, "get_agent_ids"):
                        agent_ids = env.get_agent_ids()
                except (IndexError, AttributeError):
                    # Fallback to empty list if we can't find any agents
                    print("Warning: Could not find agent IDs for episode recording")
            
            # Normalize agent IDs for consistent tracking 
            normalized_agent_ids = []
            for agent_id in agent_ids:
                if isinstance(agent_id, str) and agent_id.startswith("agent_"):
                    # Already in the right format
                    normalized_agent_ids.append(agent_id)
                elif isinstance(agent_id, int) or (isinstance(agent_id, str) and agent_id.isdigit()):
                    # Convert to string with prefix
                    numeric_id = int(agent_id) if isinstance(agent_id, str) else agent_id
                    normalized_agent_ids.append(f"agent_{numeric_id}")
                else:
                    # Use as-is if can't be normalized
                    normalized_agent_ids.append(str(agent_id))
                    
            # Initialise user_data in episode for tracking consistent agent data
            episode.user_data["record_actions"] = {agent_id: [] for agent_id in normalized_agent_ids}
            episode.user_data["recording"] = True
            
            # Store normalized agent IDs for future reference
            episode.user_data["normalized_agent_ids"] = normalized_agent_ids
        else:
            self.recording = False
            episode.user_data["recording"] = False
            
    def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs):
        """Record each step of a recorded episode."""
        if not self.recording:
            return
            
        # Get observations, actions, rewards
        observations = {}
        actions = {}
        rewards = {}
        infos = {}
        
        # Get the actual environment instance
        try:
            env = base_env.get_sub_environments()[env_index]
        except:
            env = None
        
        # Get list of agents - safely handle both string and integer agent IDs
        try:
            agent_ids = episode.get_agents()
        except:
            if env and hasattr(env, "agents"):
                agent_ids = list(env.agents.keys())
            else:
                # Fallback - try to infer from the last_obs or similar
                if hasattr(episode, "last_raw_obs") and isinstance(episode.last_raw_obs, dict):
                    agent_ids = list(episode.last_raw_obs.keys())
                elif hasattr(episode, "last_obs_for") and isinstance(episode.last_obs_for, dict):
                    agent_ids = list(episode.last_obs_for.keys())
                else:
                    # Last resort - can't find agent IDs
                    return
        
        for agent_id in agent_ids:
            # Use EpisodeV2-compatible methods
            try:
                # Observations - use episode.observations dictionary or last_raw_obs
                if hasattr(episode, "observations") and agent_id in episode.observations:
                    observations[agent_id] = episode.observations[agent_id]
                elif hasattr(episode, "last_raw_obs") and agent_id in episode.last_raw_obs:
                    observations[agent_id] = episode.last_raw_obs[agent_id]
                elif hasattr(episode, "last_observation_for"):
                    try:
                        # Try to get observation using the method
                        last_obs = episode.last_observation_for(agent_id)
                        if last_obs is not None:
                            observations[agent_id] = last_obs
                    except:
                        pass
                
                # Try different approaches to get the actions, prioritising env.last_actions
                if env and hasattr(env, "last_actions") and env.last_actions:
                    # The env.last_actions might have differently formatted keys
                    # Try direct access, string format, and integer format
                    if agent_id in env.last_actions:
                        actions[agent_id] = env.last_actions[agent_id]
                    elif isinstance(agent_id, str) and agent_id.startswith("agent_") and agent_id[6:].isdigit():
                        # Try accessing with the numeric part
                        num_id = int(agent_id[6:])
                        if num_id in env.last_actions:
                            actions[agent_id] = env.last_actions[num_id]
                    elif isinstance(agent_id, int):
                        # Try with string format
                        str_id = f"agent_{agent_id}"
                        if str_id in env.last_actions:
                            actions[agent_id] = env.last_actions[str_id]
                            
                    # Store action in user_data if found
                    if agent_id in actions and "record_actions" in episode.user_data:
                        if agent_id not in episode.user_data["record_actions"]:
                            episode.user_data["record_actions"][agent_id] = []
                        episode.user_data["record_actions"][agent_id].append(actions[agent_id])
                
                # Try fallbacks if environment approach fails
                if agent_id not in actions:
                    if hasattr(episode, "actions") and agent_id in episode.actions:
                        actions[agent_id] = episode.actions[agent_id]
                    elif hasattr(episode, "last_action") and isinstance(episode.last_action, dict) and agent_id in episode.last_action:
                        actions[agent_id] = episode.last_action[agent_id]
                    elif hasattr(episode, "get_last_action") and callable(episode.get_last_action):
                        try:
                            last_action = episode.get_last_action(agent_id)
                            if last_action is not None:
                                actions[agent_id] = last_action
                        except:
                            pass
                
                # Rewards - try multiple approaches
                if hasattr(episode, "last_rewards") and agent_id in episode.last_rewards:
                    rewards[agent_id] = episode.last_rewards[agent_id]
                elif hasattr(episode, "last_reward_for") and callable(episode.last_reward_for):
                    try:
                        last_reward = episode.last_reward_for(agent_id)
                        if last_reward is not None:
                            rewards[agent_id] = last_reward
                    except:
                        pass
                
                # Infos - try multiple approaches
                if hasattr(episode, "last_infos") and agent_id in episode.last_infos:
                    infos[agent_id] = episode.last_infos[agent_id]
                elif hasattr(episode, "last_info_for") and callable(episode.last_info_for):
                    try:
                        last_info = episode.last_info_for(agent_id)
                        if last_info is not None:
                            infos[agent_id] = last_info
                        else:
                            infos[agent_id] = {}
                    except:
                        infos[agent_id] = {}
                else:
                    infos[agent_id] = {}
            except Exception as e:
                # Silent error - don't flood the logs
                continue
                
        # Record step if we have any data
        if observations or actions or rewards or infos:
            self.current_recording["observations"].append(observations)
            self.current_recording["actions"].append(actions)
            self.current_recording["rewards"].append(rewards)
            self.current_recording["infos"].append(infos)
    
    def on_postprocess_trajectory(self, *, worker, episode, agent_id, policy_id, policies, postprocessed_batch, original_batches, **kwargs):
        """
        Process trajectory data after RLlib has processed it.
        This is a more reliable way to capture actions and rewards.
        """
        # Skip if not recording or if user_data not initialised
        if not hasattr(episode, "user_data") or not episode.user_data.get("recording", False):
            return
            
        # Normalize agent_id to match our standard format
        normalized_agent_id = agent_id
        if isinstance(agent_id, int) or (isinstance(agent_id, str) and agent_id.isdigit()):
            numeric_id = int(agent_id) if isinstance(agent_id, str) else agent_id
            normalized_agent_id = f"agent_{numeric_id}"
        elif isinstance(agent_id, str) and not agent_id.startswith("agent_"):
            normalized_agent_id = f"agent_{agent_id}"
            
        # Skip if recordActions is not initialised for this agent
        if "recordActions" not in episode.user_data:
            episode.user_data["recordActions"] = {}
            
        # Initialise recordActions for this agent if not already done
        if normalized_agent_id not in episode.user_data["recordActions"]:
            episode.user_data["recordActions"][normalized_agent_id] = []
            
        # Skip if agent_id not in original_batches
        if agent_id not in original_batches:
            return
            
        sample_batch = original_batches[agent_id]
        
        # Get actions from the sample batch - handle different possible formats
        actions = []
        if isinstance(sample_batch, tuple):
            # If it's a tuple, check if the first element has an actions field
            if len(sample_batch) > 0:
                if hasattr(sample_batch[0], 'actions'):
                    actions = sample_batch[0].actions
                elif isinstance(sample_batch[0], dict) and 'actions' in sample_batch[0]:
                    actions = sample_batch[0]['actions']
                # Try second element if first doesn't have actions
                elif len(sample_batch) > 1:
                    if hasattr(sample_batch[1], 'actions'):
                        actions = sample_batch[1].actions
                    elif isinstance(sample_batch[1], dict) and 'actions' in sample_batch[1]:
                        actions = sample_batch[1]['actions']
        else:
            # Handle dictionary-like objects with get method
            try:
                actions = sample_batch.get("actions", [])
            except (AttributeError, TypeError):
                # If not a dict-like object, try direct attribute access
                if hasattr(sample_batch, "actions"):
                    actions = sample_batch.actions
                else:
                    # If we can't find actions, return
                    return
        
        # Skip if actions is empty or None
        if not actions:
            return
        
        # Convert tensor to list if needed
        try:
            if hasattr(actions, "tolist") and callable(getattr(actions, "tolist")):
                actions = actions.tolist()
            elif hasattr(actions, "numpy") and callable(getattr(actions, "numpy")):
                actions = actions.numpy().tolist()
            # Handle possible conversion failures
            if not isinstance(actions, list):
                actions = list(actions)
        except Exception as e:
            if hasattr(actions, "__len__"):
                # Try to create a list by iterating
                try:
                    actions = [a for a in actions]
                except:
                    # Last resort - empty list
                    actions = []
            else:
                actions = []
        
        # Skip if actions list is empty
        if not actions:
            return
            
        # Store actions in user_data for consistent access using the normalized agent ID
        episode.user_data["recordActions"][normalized_agent_id].extend(actions)
        
        # If we're still recording this episode (may have started in a previous call), update actions
        if not self.recording or self.current_recording is None:
            return
            
        # Update actions in the current recording based on sample batch data
        # Check if we need to add more steps to match the batch size
        batch_size = len(actions)
        current_actions_len = len(self.current_recording["actions"])
        
        # Make sure our recording has enough steps
        while current_actions_len < batch_size:
            self.current_recording["actions"].append({})
            self.current_recording["rewards"].append({})
            current_actions_len += 1
        
        # Extract rewards based on sample_batch format
        rewards = []
        if isinstance(sample_batch, tuple):
            if len(sample_batch) > 0:
                if hasattr(sample_batch[0], 'rewards'):
                    rewards = sample_batch[0].rewards
                elif isinstance(sample_batch[0], dict) and 'rewards' in sample_batch[0]:
                    rewards = sample_batch[0]['rewards']
        else:
            # Try to get rewards from dict-like object
            try:
                rewards = sample_batch.get("rewards", [])
            except (AttributeError, TypeError):
                # If not a dict-like object, try direct attribute access
                if hasattr(sample_batch, "rewards"):
                    rewards = sample_batch.rewards
                    
        # Convert rewards to list if needed
        try:
            if hasattr(rewards, "tolist") and callable(getattr(rewards, "tolist")):
                rewards = rewards.tolist()
            elif hasattr(rewards, "numpy") and callable(getattr(rewards, "numpy")):
                rewards = rewards.numpy().tolist()
        except:
            # If conversion fails, use empty list
            rewards = []
            
        # Update actions and rewards in the recording using the normalized agent ID
        for i in range(min(batch_size, len(self.current_recording["actions"]))):
            # Add/update this agent's action in step i
            self.current_recording["actions"][i][normalized_agent_id] = actions[i]
            
            # Add/update this agent's reward in step i
            if rewards and i < len(rewards):
                self.current_recording["rewards"][i][normalized_agent_id] = rewards[i]
    
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        """Finish recording an episode."""
        if not self.recording:
            return
        
        # Update metadata
        self.current_recording["metadata"]["episode_reward"] = episode.total_reward
        self.current_recording["metadata"]["episode_length"] = episode.length
        
        # Add environment configuration data if available
        try:
            if env_index is not None and hasattr(base_env, "get_sub_environments"):
                env = base_env.get_sub_environments()[env_index]
                if hasattr(env, "config"):
                    self.current_recording["metadata"]["env_config"] = env.config
        except Exception as e:
            print(f"Warning: Could not save environment config: {e}")
        
        # Ensure all actions are properly captured by using the recordActions from user_data
        if "recordActions" in episode.user_data and episode.user_data["recordActions"]:
            # Check if there are any actions recorded at all
            if any(len(acts) > 0 for acts in episode.user_data["recordActions"].values()):
                try:
                    # Create new properly formatted actions array based on user_data
                    actions = []
                    # Filter out empty action lists first
                    valid_action_lists = {k: v for k, v in episode.user_data["recordActions"].items() if v}
                    
                    if valid_action_lists:
                        max_steps = max(len(acts) for acts in valid_action_lists.values())
                        
                        # Initialise with empty dictionaries
                        for _ in range(max_steps):
                            actions.append({})
                            
                        # Fill in actual actions
                        for agent_id, agent_actions in valid_action_lists.items():
                            for step, action in enumerate(agent_actions):
                                if step < len(actions):
                                    actions[step][agent_id] = action
                        
                        # Replace actions in the recording
                        if len(actions) > 0:
                            self.current_recording["actions"] = actions
                except Exception as e:
                    print(f"Warning: Error reformatting actions: {e}")
        
        # Get info about environment structure for replay
        try:
            if env_index is not None and hasattr(base_env, "get_sub_environments"):
                env = base_env.get_sub_environments()[env_index]
                
                # Get environment config
                if hasattr(env, "config"):
                    config = env.config.copy() if isinstance(env.config, dict) else {}
                    self.current_recording["metadata"]["env_config"] = config
                
                # Include agent IDs in metadata
                if hasattr(env, "agents") and env.agents:
                    agent_ids = list(env.agents.keys())
                    self.current_recording["metadata"]["agent_ids"] = agent_ids
                
                # Get grid size
                if hasattr(env, "gridSize"):
                    self.current_recording["metadata"]["grid_size"] = env.gridSize
        except Exception as e:
            print(f"Warning: Error getting environment structure: {e}")
        
        # Calculate and store additional metrics
        if hasattr(base_env, "get_sub_environments") and env_index is not None:
            try:
                env = base_env.get_sub_environments()[env_index]
                if hasattr(env, "get_metrics"):
                    metrics = env.get_metrics()
                    self.current_recording["metadata"]["metrics"] = metrics
                elif hasattr(env, "_calculateMetrics"):
                    metrics = env._calculateMetrics()
                    self.current_recording["metadata"]["metrics"] = metrics
            except (IndexError, AttributeError) as e:
                print(f"Warning: Error getting metrics: {e}")
        
        # Store this recording
        self.recorded_episodes.append(self.current_recording)
        
        # Save to file
        self._save_recording(self.current_recording)
        
        # Reset current recording
        self.current_recording = None
        self.recording = False
        
        # Replay if enabled
        if self.replay_on_episode_end:
            # This would need to be implemented separately
            pass
    
    def on_train_result(self, *, algorithm, result, **kwargs):
        """Update metadata with training iteration information."""
        # Update all recordings with the current training iteration
        for recording in self.recorded_episodes:
            recording["metadata"]["training_iteration"] = result["training_iteration"]
            
    def _save_recording(self, recording):
        """Save a recording to a file."""
        if not recording:
            print("Warning: Attempted to save empty recording")
            return
            
        # Create filename with safe error handling
        try:
            metadata = recording.get("metadata", {})
            reward = metadata.get("episode_reward", 0)
            episode_id = metadata.get("episode_id", "unknown")
            length = metadata.get("episode_length", 0)
            iteration = metadata.get("training_iteration", 0)
            
            filename = f"replay_iter{iteration}_len{length}_reward{int(reward)}.pkl"
            filepath = os.path.join(self.output_dir, filename)
            
            # Make sure output directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Save to file
            with open(filepath, "wb") as f:
                pickle.dump(recording, f)
            
            print(f"Saved replay to {filepath}")
        except Exception as e:
            print(f"Error saving replay file: {e}")
            # Try fallback name if filename creation failed
            try:
                fallback_path = os.path.join(self.output_dir, f"replay_emergency_{int(time.time())}.pkl")
                with open(fallback_path, "wb") as f:
                    pickle.dump(recording, f)
                print(f"Saved fallback replay to {fallback_path}")
            except Exception as e2:
                print(f"Fatal error: Could not save replay at all: {e2}")
    
    def get_recorded_episodes(self):
        """Get all recorded episodes."""
        return self.recorded_episodes

