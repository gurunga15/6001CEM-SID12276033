"""
Common utilities for training MARL algorithms.
"""
import os
import pprint
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import shutil

import matplotlib.pyplot as plt
import seaborn as sns


def setupOutputDir(algorithmName: str, checkpointDir: Optional[str] = None) -> str:
    """
    Set up an output directory for training results.
    
    Args:
        algorithmName: Name of the algorithm
        checkpointDir: Directory of existing checkpoint (for resuming)
        
    Returns:
        Path to the output directory
    """
    if checkpointDir:
        # If resuming from a checkpoint, use the existing directory
        return os.path.dirname(os.path.dirname(checkpointDir))
    
    # Create a new directory with version number
    base_dir = f"{algorithmName}_training_version"
    version = 1
    
    # Find the next available version number
    while os.path.exists(os.path.join("results", f"{base_dir}_{version}")):
        version += 1
    
    # Create the output directory
    outputDir = os.path.join("results", f"{base_dir}_{version}")
    os.makedirs(outputDir, exist_ok=True)
    os.makedirs(os.path.join(outputDir, "checkpoints"), exist_ok=True)
    
    # Create visualisations directory instead of logs
    os.makedirs(os.path.join(outputDir, "visualisations"), exist_ok=True)
    
    return outputDir


def saveConfig(config: Dict[str, Any], outputDir: str) -> str:
    """
    Save the configuration to a Python file.
    
    Args:
        config: Configuration dictionary
        outputDir: Directory to save the configuration
        
    Returns:
        Path to the saved configuration file
    """
    configPath = os.path.join(outputDir, "config.py")
    
    # Create a clean copy of the config for saving
    configCopy = config.copy()
    
    # Remove any non-serializable objects
    def cleanConfig(cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively clean the config dictionary."""
        cleaned = {}
        for key, value in cfg.items():
            if isinstance(value, dict):
                cleaned[key] = cleanConfig(value)
            elif isinstance(value, (str, int, float, bool, list, tuple, type(None))):
                cleaned[key] = value
            elif hasattr(value, "name") and hasattr(value, "value"):
                # Handle enum objects properly
                cleaned[key] = value.value
            elif str(value).startswith("<") and ":" in str(value):
                # Handle objects with special repr format like <EnumType: 'value'>
                # Extract the value part
                parts = str(value).split(":", 1)
                if len(parts) == 2 and "'" in parts[1]:
                    value_part = parts[1].strip()
                    # Extract the part between quotes
                    import re
                    match = re.search(r"'([^']*)'", value_part)
                    if match:
                        cleaned[key] = match.group(1)
                    else:
                        cleaned[key] = value_part
                else:
                    cleaned[key] = str(value)
            else:
                # Convert other non-serializable objects to string representation
                cleaned[key] = str(value)
        return cleaned
    
    cleanedConfig = cleanConfig(configCopy)
    
    # Save as Python file
    with open(configPath, "w") as f:
        f.write("# Auto-generated configuration file\n")
        f.write("# Do not edit manually\n\n")
        f.write("CONFIG = ")
        f.write(pprint.pformat(cleanedConfig, indent=4, width=100))
        f.write("\n\n# Export the config for easy import\ndef getConfig():\n    return CONFIG\n")
    
    return configPath


def loadConfig(configPath: str) -> Dict[str, Any]:
    """
    Load configuration from a Python file.
    
    Args:
        configPath: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    # Handle case where path has no extension (try .py first, then .pkl)
    if not os.path.exists(configPath) and not os.path.splitext(configPath)[1]:
        py_path = configPath + ".py"
        pkl_path = configPath + ".pkl"
        
        if os.path.exists(py_path):
            configPath = py_path
        elif os.path.exists(pkl_path):
            configPath = pkl_path
    
    # Handle backward compatibility with .pkl files
    if configPath.endswith('.pkl'):
        import pickle
        with open(configPath, "rb") as f:
            return pickle.load(f)
    
    # Load from Python file
    if not os.path.exists(configPath):
        raise FileNotFoundError(f"Config file not found: {configPath}")
    
    # Create a temporary module name based on the file path
    import importlib.util
    import sys
    
    # Use the filename without extension as the module name
    module_name = os.path.splitext(os.path.basename(configPath))[0]
    
    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, configPath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config from {configPath}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    # Return the CONFIG variable from the module
    if hasattr(module, 'CONFIG'):
        return module.CONFIG
    elif hasattr(module, 'getConfig'):
        return module.getConfig()
    else:
        raise ValueError(f"No CONFIG variable or getConfig() function found in {configPath}")


def loadTrainingProgress(outputDir: str) -> pd.DataFrame:
    """
    Load training progress data from the output directory.
    
    Args:
        outputDir: Directory containing training results
        
    Returns:
        DataFrame with training progress data
    """
    metricsPath = os.path.join(outputDir, "metrics.csv")
    
    # Check if metrics.csv exists
    if os.path.exists(metricsPath):
        return pd.read_csv(metricsPath)
    
    raise FileNotFoundError(f"Metrics data not found in {outputDir}")


def visualiseTrainingProgress(outputDir: str) -> None:
    """
    Visualise training progress from an output directory.
    
    Args:
        outputDir: Directory containing training results
    """
    # Look for metrics.csv
    metricsPath = os.path.join(outputDir, "metrics.csv")
    if not os.path.exists(metricsPath):
        print(f"No metrics data found in {outputDir}")
        return
    
    print(f"Found metrics file at {metricsPath}")
    # Plot training curves using metrics.csv
    plot_training_curves(metricsPath, outputDir)
    
    print(f"Training visualisations saved to {outputDir}/visualisations")


def plot_training_curves(metricsPath: str, outputDir: str = None,
                      metrics: List[str] = None, figsize=(12, 10),
                      moving_average: int = 10) -> None:
    """
    Plot training curves from a metrics CSV file.
    
    Args:
        metricsPath: Path to metrics CSV file
        outputDir: Directory to save the plots (default: path directory)
        metrics: List of metrics to plot (default: standard metrics)
        figsize: Figure size for the plot
        moving_average: Window size for moving average
    """
    # Load metrics data
    try:
        print(f"Loading metrics data from {metricsPath}")
        df = pd.read_csv(metricsPath)
        print(f"Successfully loaded CSV with {len(df)} rows and columns: {', '.join(df.columns)}")
    except Exception as e:
        print(f"Error loading metrics file {metricsPath}: {e}")
        return
    
    # If metrics not specified, use standard metrics
    if metrics is None:
        # Check if we're using metrics.csv format
        if "iteration" in df.columns:
            # Rename 'iteration' to 'training_iteration' for consistency
            df["training_iteration"] = df["iteration"]
            metrics = [
                "episode_reward_mean",
                "episode_reward_max",
                "episode_reward_min",
                "episode_len_mean",
                "fairness_gini",
                "jain_fairness_index",
                "resource_specialisation",
                "agent_overlap",
                "task_division",
                "policy_divergence"
            ]
            # Add entropy if it exists
            if "entropy_coeff" in df.columns:
                metrics.append("entropy_coeff")
    
    # Create output directory if not specified
    if outputDir is None:
        outputDir = os.path.dirname(metricsPath)
    
    # Create visualisations directory
    vis_dir = os.path.join(outputDir, "visualisations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Calculate moving averages
    for metric in metrics:
        if metric in df.columns:
            col_name = f"{metric}_ma{moving_average}"
            df[col_name] = df[metric].rolling(window=moving_average).mean()
    
    # Create a multi-panel figure
    num_plots = len([m for m in metrics if m in df.columns])
    if num_plots == 0:
        print(f"No valid metrics found in the dataframe. Available columns: {', '.join(df.columns)}")
        return
        
    rows = int(np.ceil(num_plots / 2))
    fig, axes = plt.subplots(rows, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Plot each metric
    plot_count = 0
    for i, metric in enumerate(metrics):
        if metric in df.columns:
            ax = axes[plot_count]
            plot_count += 1
            
            # Plot raw data
            ax.plot(df["training_iteration"], df[metric], alpha=0.3, label=f"{metric}")
            
            # Plot moving average if available
            col_name = f"{metric}_ma{moving_average}"
            if col_name in df.columns:
                ax.plot(df["training_iteration"], df[col_name], linewidth=2, 
                       label=f"{moving_average}-ep MA")
            
            # Add labels and legend
            ax.set_xlabel("Training Iteration")
            ax.set_ylabel(metric.replace("_", " ").replace("/", " ").title())
            ax.set_title(f"{metric.replace('_', ' ').replace('/', ' ').title()}")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
    
    # Remove any empty subplots
    for i in range(plot_count, len(axes)):
        fig.delaxes(axes[i])
    
    # Adjust layout and save
    plt.tight_layout()
    figPath = os.path.join(vis_dir, "training_progress.png")
    plt.savefig(figPath)
    print(f"Saved visualisation to {figPath}")
    plt.close(fig)
    
    # Create individual plots for each metric for better detail
    for metric in metrics:
        if metric in df.columns:
            plt.figure(figsize=(10, 6))
            
            # Plot raw data
            plt.plot(df["training_iteration"], df[metric], alpha=0.3, label=f"{metric}")
            
            # Plot moving average
            col_name = f"{metric}_ma{moving_average}"
            if col_name in df.columns:
                plt.plot(df["training_iteration"], df[col_name], linewidth=2, 
                        label=f"{moving_average}-ep MA")
            
            # Add labels and legend
            plt.xlabel("Training Iteration")
            plt.ylabel(metric.replace("_", " ").replace("/", " ").title())
            plt.title(f"{metric.replace('_', ' ').replace('/', ' ').title()}")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Save individual plot
            singleFigPath = os.path.join(vis_dir, f"{metric.replace('/', '_')}_plot.png")
            plt.savefig(singleFigPath)
            plt.close()
    
    print(f"Created {len([m for m in metrics if m in df.columns])} individual metric plots in {vis_dir}")


def saveTrainingMetadata(
    outputDir: str,
    algorithmName: str,
    startTime: datetime,
    endTime: Optional[datetime] = None,
    additionalInfo: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save metadata about the training run.
    
    Args:
        outputDir: Directory containing training results
        algorithmName: Name of the algorithm
        startTime: Training start time
        endTime: Training end time (if available)
        additionalInfo: Additional information to save
    """
    # Create metadata dictionary
    metadata = {
        "algorithm": algorithmName,
        "start_time": startTime.isoformat(),
        "end_time": endTime.isoformat() if endTime else None,
        "duration": str(endTime - startTime) if endTime else None,
    }
    
    # Add additional info if provided
    if additionalInfo:
        metadata.update(additionalInfo)
    
    # Save metadata as Python file
    metadataPath = os.path.join(outputDir, "metadata.py")
    with open(metadataPath, "w") as f:
        f.write("# Auto-generated metadata file\n")
        f.write("# Do not edit manually\n\n")
        f.write("METADATA = ")
        f.write(pprint.pformat(metadata, indent=4, width=100))
        f.write("\n\n# Export the metadata for easy import\ndef getMetadata():\n    return METADATA\n")


def copyExperimentalCode(outputDir: str, sourceDir: str = ".") -> None:
    """
    Copy the code used for the experiment to the output directory.
    
    Args:
        outputDir: Directory to save the code
        sourceDir: Source directory containing the code
    """
    # Create code directory
    codeDir = os.path.join(outputDir, "code")
    os.makedirs(codeDir, exist_ok=True)
    
    # Copy Python files
    for root, _, files in os.walk(sourceDir):
        for file in files:
            if file.endswith(".py"):
                # Skip pycache and other temporary files
                if "__pycache__" in root or "/.venv/" in root:
                    continue
                
                # Get relative path
                relPath = os.path.relpath(os.path.join(root, file), sourceDir)
                destPath = os.path.join(codeDir, relPath)
                
                # Create parent directories if needed
                os.makedirs(os.path.dirname(destPath), exist_ok=True)
                
                # Copy file
                shutil.copy2(os.path.join(root, file), destPath)
    
    print(f"Experimental code copied to {codeDir}") 