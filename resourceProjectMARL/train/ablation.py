"""
Ablation study utilities for PPO in the resource collection environment.

This module provides tools to run systematic ablation studies comparing different
PPO configurations including critic types and reward structures.
"""

import os
import sys
import copy
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import pprint

# Add the project root to the path
scriptDir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(scriptDir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import policies and common utilities
from algorithms.ppoPolicies import createDefaultPPOPolicy
from .common import setupOutputDir, saveConfig, loadConfig


def createAblationConfigs(
    base_config: Dict[str, Any],
    ablation_type: str,
    numAgents: int = 4
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Create configuration variants for ablation studies.
    
    Args:
        base_config: Base configuration for the environment and algorithm
        ablation_type: Type of ablation to perform ("critic", "reward", or "all")
        numAgents: Number of agents in the environment
        
    Returns:
        List of (name, config) tuples for each ablation variant
    """
    ablation_configs = []
    
    # Create a deep copy of the base config to avoid modifying the original
    base_config = copy.deepcopy(base_config)
    
    # Make sure env_config exists
    if "env_config" not in base_config:
        base_config["env_config"] = {}
    
    # Centralised vs Decentralised critic ablation
    if ablation_type == "critic" or ablation_type == "all":
        # Test with centralised critic (MAPPO)
        centralised_config = copy.deepcopy(base_config)
        if "model" not in centralised_config:
            centralised_config["model"] = {}
        centralised_config["model"]["custom_model_config"] = {
            "use_centralised_critic": True,
            "num_agents": numAgents
        }
        ablation_configs.append(("centralised_critic", centralised_config))
        
        # Test with decentralised critic (standard PPO)
        decentralised_config = copy.deepcopy(base_config)
        if "model" not in decentralised_config:
            decentralised_config["model"] = {}
        decentralised_config["model"]["custom_model_config"] = {
            "use_centralised_critic": False,
            "num_agents": numAgents
        }
        ablation_configs.append(("decentralised_critic", decentralised_config))
    
    # Reward structure ablation
    if ablation_type == "reward" or ablation_type == "all":
        # Test different reward types
        for reward_type in ["individual", "global", "hybrid"]:
            # For hybrid, test different mix ratios
            if reward_type == "hybrid":
                for mix in [0.25, 0.5, 0.75]:
                    reward_config = copy.deepcopy(base_config)
                    reward_config["env_config"]["rewardType"] = reward_type
                    reward_config["env_config"]["hybridRewardMix"] = mix
                    
                    # Update model config if needed
                    if "model" in reward_config and "custom_model_config" in reward_config["model"]:
                        reward_config["model"]["custom_model_config"]["reward_type"] = reward_type
                        reward_config["model"]["custom_model_config"]["hybrid_reward_mix"] = mix
                    
                    name = f"{reward_type}_reward_mix{mix}"
                    ablation_configs.append((name, reward_config))
            else:
                # Individual or global rewards
                reward_config = copy.deepcopy(base_config)
                reward_config["env_config"]["rewardType"] = reward_type
                
                # Update model config if needed
                if "model" in reward_config and "custom_model_config" in reward_config["model"]:
                    reward_config["model"]["custom_model_config"]["reward_type"] = reward_type
                
                ablation_configs.append((f"{reward_type}_reward", reward_config))
    
    return ablation_configs


def runAblationStudy(
    base_config: Dict[str, Any],
    ablation_type: str,
    numAgents: int = 4,
    numIterations: int = 200,
    outputDir: Optional[str] = None,
    useRayTune: bool = False
):
    """
    Run ablation study with different configurations.
    
    Args:
        base_config: Base configuration for the environment and algorithm
        ablation_type: Type of ablation to perform ("critic", "reward", "policy", or "all")
        numAgents: Number of agents in the environment
        numIterations: Number of training iterations for each configuration
        outputDir: Directory to save results (created if not provided)
        useRayTune: Whether to use Ray Tune for hyperparameter optimisation
    """
    import ray
    from ray.tune.logger import pretty_print
    
    # Create output directory
    if outputDir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outputDir = f"results/ablation_{ablation_type}_{timestamp}"
    
    os.makedirs(outputDir, exist_ok=True)
    print(f"Saving ablation results to: {outputDir}")
    
    # Save base configuration
    with open(os.path.join(outputDir, "base_config.py"), "w") as f:
        f.write("# Auto-generated base configuration file\n")
        f.write("# Do not edit manually\n\n")
        f.write("CONFIG = ")
        f.write(pprint.pformat(base_config, indent=4, width=100))
        f.write("\n\n# Export the config for easy import\ndef getConfig():\n    return CONFIG\n")
    
    # initialise Ray if not already initialised
    if not ray.is_initialized():
        ray.init(num_cpus=os.cpu_count() or 4, include_dashboard=True)
    
    # Generate ablation configurations
    ablation_configs = createAblationConfigs(base_config, ablation_type, numAgents)
    
    if useRayTune:
        # Implement Ray Tune version here if needed
        pass
    else:
        # Run manually for each configuration
        results = {}
        
        for name, config in ablation_configs:
            print(f"\n{'='*80}")
            print(f"Running ablation variant: {name}")
            print(f"{'='*80}\n")
            
            # Create subdirectory for this variant
            variant_dir = os.path.join(outputDir, name)
            os.makedirs(variant_dir, exist_ok=True)
            
            # Save configuration
            with open(os.path.join(variant_dir, "config.py"), "w") as f:
                f.write("# Auto-generated variant configuration file\n")
                f.write("# Do not edit manually\n\n")
                f.write("CONFIG = ")
                f.write(pprint.pformat(config, indent=4, width=100))
                f.write("\n\n# Export the config for easy import\ndef getConfig():\n    return CONFIG\n")
            
            # Create policy and trainer
            policy = createDefaultPPOPolicy(
                useCentralisedCritic="model" in config and 
                               "custom_model_config" in config["model"] and
                               config["model"]["custom_model_config"].get("use_centralised_critic", True),
                numAgents=numAgents,
                config=config,
                rewardType=config["env_config"].get("rewardType", "hybrid"),
                hybridRewardMix=config["env_config"].get("hybridRewardMix", 0.5)
            )
            
            trainer = policy.getTrainer()
            
            # Training loop
            training_results = []
            try:
                for i in range(numIterations):
                    result = trainer.train()
                    training_results.append(result)
                    
                    # Log progress
                    if i % 10 == 0 or i == numIterations - 1:
                        print(f"\nIteration {i+1}/{numIterations}")
                        metrics = [
                            "episode_reward_mean", 
                            "episode_len_mean",
                            "episodes_this_iter"
                        ]
                        row = {k: result.get(k, 0) for k in metrics}
                        print(pretty_print(row))
                        
                        # Check for custom metrics
                        if "custom_metrics" in result:
                            custom_metrics = {
                                f"custom_{k}": v 
                                for k, v in result["custom_metrics"].items()
                                if "_mean" in k  # Include only mean values
                            }
                            if custom_metrics:
                                print(pretty_print(custom_metrics))
                    
                    # Save checkpoint periodically
                    if (i+1) % 50 == 0 or i == numIterations - 1:
                        checkpoint_path = trainer.save(variant_dir)
                        print(f"Saved checkpoint to {checkpoint_path}")
                
                # Save final results
                results[name] = result
                
                # Note: metrics.csv is automatically created by the UnifiedTrainingCallback
                # in train_main.py. No need to explicitly save progress here.
                
            except Exception as e:
                print(f"Error during training for {name}: {e}")
            finally:
                # Clean up trainer
                trainer.stop()
        
        # Compare results
        compareAblationResults(results, ablation_configs, outputDir)


def saveTrainingProgress(results: List[Dict[str, Any]], output_file: str):
    """
    Save training progress to CSV file.
    
    This function is a simplified version of the metrics tracking in UnifiedTrainingCallback.
    It extracts key metrics from training results and saves them to a CSV file.
    
    Note: The standard approach is to use the metrics.csv file created by UnifiedTrainingCallback,
    but this function provides a fallback for cases where that callback isn't used.
    
    Args:
        results: List of training result dictionaries
        output_file: Path to output CSV file, should be "metrics.csv" for consistency
    """
    # Extract key metrics from results
    data = []
    for i, result in enumerate(results):
        row = {
            "iteration": i,
            "episode_reward_mean": result.get("episode_reward_mean", 0),
            "episode_reward_min": result.get("episode_reward_min", 0),
            "episode_reward_max": result.get("episode_reward_max", 0),
            "episode_len_mean": result.get("episode_len_mean", 0),
            "episodes_this_iter": result.get("episodes_this_iter", 0),
            "episodes_total": result.get("episodes_total", 0),
        }
        
        # Add custom metrics
        if "custom_metrics" in result:
            for k, v in result["custom_metrics"].items():
                if "_mean" in k:  # Only include mean values
                    row[k] = v
        
        data.append(row)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Saved training progress to {output_file}")


def compareAblationResults(
    results: Dict[str, Dict[str, Any]],
    ablation_configs: List[Tuple[str, Dict[str, Any]]],
    outputDir: str
):
    """
    Compare and visualise results from different ablation variants.
    
    Args:
        results: Results dictionary, keyed by variant name
        ablation_configs: List of (name, config) tuples
        outputDir: Directory to save comparison results
    """
    # Create comparison directory
    comparison_dir = os.path.join(outputDir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Create summary of final metrics
    summary_data = []
    for name, _ in ablation_configs:
        if name in results:
            result = results[name]
            row = {
                "variant": name,
                "episode_reward_mean": result.get("episode_reward_mean", 0),
                "episode_len_mean": result.get("episode_len_mean", 0),
                "episodes_total": result.get("episodes_total", 0),
            }
            
            # Add custom metrics
            if "custom_metrics" in result:
                for k, v in result["custom_metrics"].items():
                    if "_mean" in k:  # Only include mean values
                        row[k] = v
            
            summary_data.append(row)
    
    # Save summary to CSV
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(comparison_dir, "summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved summary comparison to {summary_file}")
    
    # Create comparison plots
    try:
        createComparisonPlots(ablation_configs, outputDir, comparison_dir)
    except Exception as e:
        print(f"Error creating comparison plots: {e}")


def createComparisonPlots(
    ablation_configs: List[Tuple[str, Dict[str, Any]]],
    outputDir: str,
    comparison_dir: str
):
    """
    Create comparison plots for different ablation variants.
    
    Args:
        ablation_configs: List of (name, config) tuples
        outputDir: Base output directory
        comparison_dir: Directory to save comparison plots
    """
    # Load metrics data for each variant
    # generated by the UnifiedTrainingCallback in train_main.py
    progress_data = {}
    for name, _ in ablation_configs:
        metrics_file = os.path.join(outputDir, name, "metrics.csv")
        if os.path.exists(metrics_file):
            try:
                df = pd.read_csv(metrics_file)
                progress_data[name] = df
            except Exception as e:
                print(f"Error loading metrics data for {name}: {e}")
    
    if not progress_data:
        print("No metrics data found for plotting")
        return
    
    # Create reward comparison plot
    plt.figure(figsize=(12, 8))
    for name, df in progress_data.items():
        if "episode_reward_mean" in df.columns:
            plt.plot(df["iteration"], df["episode_reward_mean"], label=name)
    
    plt.title("Episode Reward Comparison")
    plt.xlabel("Training Iteration")
    plt.ylabel("Mean Episode Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    reward_plot_file = os.path.join(comparison_dir, "reward_comparison.png")
    plt.savefig(reward_plot_file)
    plt.close()
    
    # Create episode length comparison plot
    plt.figure(figsize=(12, 8))
    for name, df in progress_data.items():
        if "episode_len_mean" in df.columns:
            plt.plot(df["iteration"], df["episode_len_mean"], label=name)
    
    plt.title("Episode Length Comparison")
    plt.xlabel("Training Iteration")
    plt.ylabel("Mean Episode Length")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    length_plot_file = os.path.join(comparison_dir, "length_comparison.png")
    plt.savefig(length_plot_file)
    plt.close()
    
    # Create custom metrics comparison plots
    # Find all custom metrics across all progress data
    custom_metrics = set()
    for df in progress_data.values():
        for col in df.columns:
            if col.endswith("_mean") and col not in ["episode_reward_mean", "episode_len_mean"]:
                custom_metrics.add(col)
    
    # Create a plot for each custom metric
    for metric in custom_metrics:
        plt.figure(figsize=(12, 8))
        for name, df in progress_data.items():
            if metric in df.columns:
                plt.plot(df["iteration"], df[metric], label=name)
        
        # Clean up metric name for title
        metric_name = metric.replace("_mean", "").replace("_", " ").title()
        plt.title(f"{metric_name} Comparison")
        plt.xlabel("Training Iteration")
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        metric_file = os.path.join(comparison_dir, f"{metric.replace('_mean', '')}_comparison.png")
        plt.savefig(metric_file)
        plt.close()
    
    print(f"Saved comparison plots to {comparison_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ablation studies for PPO")
    parser.add_argument("--ablation-type", type=str, required=True,
                       choices=["critic", "reward", "policy", "all"],
                       help="Type of ablation to perform")
    parser.add_argument("--config", type=str, help="Path to base configuration file")
    parser.add_argument("--output-dir", type=str, help="Directory to save results")
    parser.add_argument("--iterations", type=int, default=200,
                       help="Number of training iterations for each variant")
    parser.add_argument("--num-agents", type=int, default=4,
                       help="Number of agents in the environment")
    
    args = parser.parse_args()
    
    # Load base configuration
    if args.config and os.path.exists(args.config):
        base_config = loadConfig(args.config)
    else:
        # Default configuration
        base_config = {
            "env": "ResourceCollectionEnv",
            "env_config": {
                "numAgents": args.num_agents,
                "gridSize": (15, 15),
                "renderMode": None,
                "use_flattened_obs": True,
                "rewardType": "hybrid",
                "hybridRewardMix": 0.5
            },
            "framework": "torch",
            "train_batch_size": 4000,
            "sgd_minibatch_size": 128,
            "num_sgd_iter": 10,
            "lr": 3e-4,
            "gamma": 0.99,
            "lambda_": 0.95,
            "kl_coeff": 0.2,
            "clip_param": 0.2,
            "vf_clip_param": 10.0,
            "entropy_coeff": 0.01,
            "num_workers": 8
        }
    
    # Run ablation study
    runAblationStudy(
        base_config=base_config,
        ablation_type=args.ablation_type,
        numAgents=args.num_agents,
        numIterations=args.iterations,
        outputDir=args.output_dir
    ) 