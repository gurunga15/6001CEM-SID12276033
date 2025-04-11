"""
Script for plotting and visualising training metrics.
"""
import os
import sys
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import argparse

# Add the project root to the path
scriptDir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(scriptDir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Get the parent directory (resourceProjectMARL)
project_dir = os.path.dirname(scriptDir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# Import from train module
from train.common import plot_training_curves


def find_metrics_file(path: str) -> Optional[str]:
    """
    Find the metrics.csv file either directly or in a directory.
    
    Args:
        path: Path to either a metrics.csv file or a directory containing one
        
    Returns:
        Path to the metrics.csv file, or None if not found
    """
    # Check if the path is directly to a metrics.csv file
    if os.path.isfile(path) and path.endswith("metrics.csv"):
        print(f"Using metrics file: {path}")
        return path
    
    # Check if it's a path that exists but isn't named exactly metrics.csv
    # This handles files selected via UI that might have absolute paths
    if os.path.isfile(path) and path.lower().endswith(".csv"):
        filename = os.path.basename(path)
        if "metrics" in filename.lower():
            print(f"Using metrics file: {path}")
            return path
    
    # Check if it's a directory containing metrics.csv
    if os.path.isdir(path):
        metrics_file = os.path.join(path, "metrics.csv")
        if os.path.exists(metrics_file):
            print(f"Found metrics.csv in directory: {metrics_file}")
            return metrics_file
    
    print(f"No metrics.csv file found at {path}")
    return None


def plot_experiment(metrics_file: str, output_dir: Optional[str] = None, 
                   metrics: Optional[List[str]] = None) -> None:
    """
    Plot basic training metrics.
    
    Args:
        metrics_file: Path to the metrics.csv file
        output_dir: Directory to save the plots (defaults to metrics_file directory)
        metrics: List of metrics to plot (defaults to standard set)
    """
    # Set default output directory
    if not output_dir:
        output_dir = os.path.dirname(metrics_file)
    
    # Create visualisations directory
    vis_dir = os.path.join(output_dir, "visualisations")
    os.makedirs(vis_dir, exist_ok=True)
    
    plot_training_curves(metrics_file, output_dir, metrics)


def plot_resource_distribution(metrics_file: str, output_dir: Optional[str] = None, df: Optional[pd.DataFrame] = None) -> None:
    """
    Plot resource collection distribution over time.
    
    Args:
        metrics_file: Path to the metrics.csv file
        output_dir: Directory to save the plots (defaults to metrics_file directory)
        df: Optional DataFrame if already loaded (for efficiency)
    """
    # Set default output directory
    if output_dir is None:
        output_dir = os.path.dirname(metrics_file)
    
    # Create visualisations directory
    vis_dir = os.path.join(output_dir, "visualisations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Load the data if not provided
    if df is None:
        df = pd.read_csv(metrics_file)
    
    # Find resource collection metrics
    resource_metrics = [col for col in df.columns if col.startswith("custom_metrics/collected_")]
    
    if not resource_metrics:
        print("No resource collection metrics found in the metrics file")
        return
    
    # Set up the style
    sns.set(style="darkgrid")
    
    # Plot resource distribution over time
    plt.figure(figsize=(12, 8))
    
    for metric in resource_metrics:
        resource_type = metric.split("_")[-1]
        plt.plot(df["iteration"], df[metric], label=resource_type.capitalize())
    
    plt.xlabel("Training Iteration")
    plt.ylabel("Resources Collected")
    plt.title("Resource Collection Distribution Over Time")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    plot_path = os.path.join(vis_dir, "resource_distribution.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Resource distribution plot saved to {plot_path}")
    
    # Create a stacked area chart
    plt.figure(figsize=(12, 8))
    
    data = {metric.split("_")[-1].capitalize(): df[metric].values for metric in resource_metrics}
    data["Iteration"] = df["iteration"].values
    
    stacked_df = pd.DataFrame(data)
    stacked_df = stacked_df.set_index("Iteration")
    
    stacked_df.plot.area(figsize=(12, 8), alpha=0.7)
    
    plt.xlabel("Training Iteration")
    plt.ylabel("Resources Collected")
    plt.title("Cumulative Resource Collection Over Time")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    plot_path = os.path.join(vis_dir, "resource_stacked_area.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Stacked area plot saved to {plot_path}")
    
    # Create a pie chart of final resource distribution
    if len(df) > 0:
        plt.figure(figsize=(10, 10))
        
        # Get the last row
        last_row = df.iloc[-1]
        
        # Extract resource values
        labels = [metric.split("_")[-1].capitalize() for metric in resource_metrics]
        values = [last_row[metric] for metric in resource_metrics]
        
        # Create pie chart
        plt.pie(values, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title("Final Resource Distribution")
        
        # Save the plot
        plot_path = os.path.join(vis_dir, "resource_pie_chart.png")
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Pie chart saved to {plot_path}")


def plot_specialisation_metrics(metrics_file: str, output_dir: Optional[str] = None, df: Optional[pd.DataFrame] = None) -> None:
    """
    Plot resource specialisation metrics over time.
    
    Specialisation measures how agents focus on particular resource types rather than
    collecting all resources equally. Higher values indicate more specialisation.
    
    Args:
        metrics_file: Path to the metrics.csv file
        output_dir: Directory to save the plots (defaults to metrics_file directory)
        df: Optional DataFrame if already loaded (for efficiency)
    """
    # Set default output directory
    if output_dir is None:
        output_dir = os.path.dirname(metrics_file)
    
    # Create visualisations directory
    vis_dir = os.path.join(output_dir, "visualisations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Load the data if not provided
    if df is None:
        df = pd.read_csv(metrics_file)
    
    # Find specialisation metrics
    specialisation_metrics = [col for col in df.columns if "specialisation" in col.lower() or "specialization" in col.lower()]
    
    if not specialisation_metrics:
        print("No specialisation metrics found in the metrics file")
        return
    
    # Set up the style
    sns.set(style="darkgrid")
    
    # Plot specialisation metrics over time
    plt.figure(figsize=(12, 8))
    
    for metric in specialisation_metrics:
        metric_name = metric.split("/")[-1].replace("_", " ").title()
        plt.plot(df["iteration"], df[metric], label=metric_name)
    
    plt.xlabel("Training Iteration")
    plt.ylabel("Specialisation Value")
    plt.title("Resource Specialisation Over Time")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    plot_path = os.path.join(vis_dir, "specialisation_metrics.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Specialisation metrics plot saved to {plot_path}")
    
    # If we have resource collection data and specialisation data, create a correlation plot
    resource_metrics = [col for col in df.columns if "resourcesPerStep" in col]
    if resource_metrics and specialisation_metrics:
        plt.figure(figsize=(10, 8))
        
        # Use last metric from each group
        resource_metric = resource_metrics[-1]
        specialisation_metric = specialisation_metrics[-1]
        
        # Create scatter plot
        try:
            sns.scatterplot(x=df[resource_metric], 
                          y=df[specialisation_metric], 
                          hue=df["iteration"],
                          palette="viridis", 
                          size=df["iteration"],
                          sizes=(20, 200),
                          alpha=0.6)
                          
            # Add labels and title
            plt.xlabel("Resources Per Step")
            plt.ylabel("Resource Specialisation")
            plt.title("Efficiency vs. Specialisation Correlation")
            
            # Only add colorbar if scatter plot was created successfully and enough data points
            if len(df) > 2:
                plt.colorbar(label="Training Iteration")
        except Exception as e:
            print(f"Warning: Could not create scatter plot: {e}")
        
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add trend line
        if len(df) > 2:
            try:
                sns.regplot(x=df[resource_metric], y=df[specialisation_metric], 
                           scatter=False, ci=None, line_kws={"color": "red", "linestyle": "--"})
            except Exception as e:
                print(f"Warning: Could not add trend line: {e}")
        
        # Save the plot
        plot_path = os.path.join(vis_dir, "efficiency_vs_specialisation.png")
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Efficiency vs. specialisation plot saved to {plot_path}")


def plot_fairness_metrics(metrics_file: str, output_dir: Optional[str] = None, df: Optional[pd.DataFrame] = None) -> None:
    """
    Plot fairness metrics over time.
    
    Args:
        metrics_file: Path to the metrics.csv file
        output_dir: Directory to save the plots (defaults to metrics_file directory)
        df: Optional DataFrame if already loaded (for efficiency)
    """
    # Set default output directory
    if output_dir is None:
        output_dir = os.path.dirname(metrics_file)
    
    # Create visualisations directory
    vis_dir = os.path.join(output_dir, "visualisations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Load the data if not provided
    if df is None:
        df = pd.read_csv(metrics_file)
    
    # Find fairness metrics
    fairness_metrics = [col for col in df.columns if "fairness" in col.lower()]
    
    if not fairness_metrics:
        print("No fairness metrics found in the metrics file")
        return
    
    # Set up the style
    sns.set(style="darkgrid")
    
    # Plot fairness metrics over time
    plt.figure(figsize=(12, 8))
    
    for metric in fairness_metrics:
        metric_name = metric.split("/")[-1].replace("_", " ").title()
        plt.plot(df["iteration"], df[metric], label=metric_name)
    
    plt.xlabel("Training Iteration")
    plt.ylabel("Fairness Value")
    plt.title("Fairness Metrics Over Time")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    plot_path = os.path.join(vis_dir, "fairness_metrics.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Fairness metrics plot saved to {plot_path}")
    
    # Create separate plots for different fairness metrics
    gini_metrics = [col for col in fairness_metrics if "gini" in col.lower()]
    jain_metrics = [col for col in fairness_metrics if "jain" in col.lower()]
    
    if gini_metrics:
        plt.figure(figsize=(12, 8))
        for metric in gini_metrics:
            metric_name = metric.split("/")[-1].replace("_", " ").title()
            plt.plot(df["iteration"], df[metric], label=metric_name)
        
        plt.xlabel("Training Iteration")
        plt.ylabel("Gini Fairness Value")
        plt.title("Gini Fairness Index Over Time (Higher is More Fair)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the plot
        plot_path = os.path.join(vis_dir, "gini_fairness.png")
        plt.savefig(plot_path)
        plt.close()
    
    if jain_metrics:
        plt.figure(figsize=(12, 8))
        for metric in jain_metrics:
            metric_name = metric.split("/")[-1].replace("_", " ").title()
            plt.plot(df["iteration"], df[metric], label=metric_name)
        
        plt.xlabel("Training Iteration")
        plt.ylabel("Jain's Fairness Value")
        plt.title("Jain's Fairness Index Over Time (Higher is More Fair)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the plot
        plot_path = os.path.join(vis_dir, "jain_fairness.png")
        plt.savefig(plot_path)
        plt.close()
    
    # Create a scatter plot of fairness vs. performance
    fairness_vs_reward = False
    
    # Try Gini fairness first
    if "fairness_gini" in df.columns and "episode_reward_mean" in df.columns:
        fairness_metric = "fairness_gini"
        fairness_name = "Gini Fairness Index"
        fairness_vs_reward = True
    # Try Jain's fairness if Gini not available
    elif "jain_fairness_index" in df.columns and "episode_reward_mean" in df.columns:
        fairness_metric = "jain_fairness_index"
        fairness_name = "Jain's Fairness Index"
        fairness_vs_reward = True
    
    if fairness_vs_reward:
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot
        try:
            scatter = sns.scatterplot(x=df[fairness_metric], 
                       y=df["episode_reward_mean"], 
                       hue=df["iteration"],
                       palette="viridis", 
                       size=df["iteration"],
                       sizes=(20, 200),
                       alpha=0.6)
            
            # Add labels and title
            plt.xlabel(fairness_name)
            plt.ylabel("Mean Episode Reward")
            plt.title("Fairness vs. Performance Trade-off")
            
            # Only add colorbar if scatter plot was created successfully and enough data points
            if len(df) > 2:
                plt.colorbar(label="Training Iteration")
        except Exception as e:
            print(f"Warning: Could not create scatter plot: {e}")
        
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add trend line
        if len(df) > 2:
            try:
                sns.regplot(x=df[fairness_metric], y=df["episode_reward_mean"], 
                          scatter=False, ci=None, line_kws={"color": "red", "linestyle": "--"})
            except Exception as e:
                print(f"Warning: Could not add trend line: {e}")
        
        # Save the plot
        plot_path = os.path.join(vis_dir, "fairness_vs_performance.png")
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Fairness vs. performance plot saved to {plot_path}")


def compare_experiments(experiment_dirs: Dict[str, str], output_dir: str,
                       metrics: Optional[List[str]] = None, 
                       df_dict: Optional[Dict[str, pd.DataFrame]] = None) -> None:
    """
    Compare multiple experiments by plotting their metrics together.
    
    Args:
        experiment_dirs: Dictionary mapping experiment names to directories
        output_dir: Directory to save the comparison plots
        metrics: List of metrics to compare (defaults to standard set)
        df_dict: Optional dictionary of pre-loaded DataFrames
    """
    # Create output directory and visualisations subdirectory
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "visualisations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Load data from all metrics files if not provided
    data = df_dict or {}
    if not df_dict:
        # Find metrics files
        metrics_files = {}
        for exp_name, exp_dir in experiment_dirs.items():
            metrics_file = find_metrics_file(exp_dir)
            if metrics_file:
                metrics_files[exp_name] = metrics_file
            else:
                print(f"No metrics file found for experiment '{exp_name}' in {exp_dir}")
        
        if not metrics_files:
            print("No metrics files found for any experiment")
            return
        
        # Use parallel loading for efficiency
        def load_experiment_data(name, file_path):
            try:
                df = pd.read_csv(file_path)
                # Ensure iteration column exists
                if "iteration" not in df.columns and "training_iteration" in df.columns:
                    df["iteration"] = df["training_iteration"]
                return name, df
            except Exception as e:
                print(f"Error loading {name} metrics file {file_path}: {e}")
                return name, None
        
        with ThreadPoolExecutor(max_workers=min(4, len(metrics_files))) as executor:
            futures = [executor.submit(load_experiment_data, name, path) 
                      for name, path in metrics_files.items()]
            for future in futures:
                name, df = future.result()
                if df is not None:
                    data[name] = df
    
    if not data:
        print("No valid metrics files to compare")
        return
    
    # Determine common metrics if not specified
    if metrics is None:
        # Get common metrics across all dataframes
        common_cols = set.intersection(*[set(df.columns) for df in data.values()])
        
        # Standard metrics
        standard_metrics = [
            "episode_reward_mean",
            "episode_reward_max",
            "episode_reward_min",
            "episode_len_mean",
        ]
        
        # Filter to metrics that are common
        metrics = [m for m in standard_metrics if m in common_cols]
        
        # Add common custom metrics
        custom_metrics = [col for col in common_cols if col.startswith("custom_metrics/") or 
                         col in ["fairness_gini", "jain_fairness_index", "resource_specialisation"]]
        metrics.extend(custom_metrics)
    
    # Create a separate plot for each metric
    for metric in metrics:
        # Check if this metric exists in any dataset
        if not any(metric in df.columns for df in data.values()):
            print(f"Metric {metric} not found in any dataset")
            continue
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot each experiment
        for name, df in data.items():
            if metric in df.columns:
                # Plot raw data with low alpha
                plt.plot(df["iteration"], df[metric], alpha=0.2)
                
                # Calculate and plot moving average
                ma = df[metric].rolling(window=10).mean()
                plt.plot(df["iteration"], ma, linewidth=2, label=f"{name}")
        
        # Add labels and legend
        plt.xlabel("Training Iteration")
        plt.ylabel(metric.replace("_", " ").replace("/", " ").title())
        plt.title(f"Comparison of {metric.replace('_', ' ').replace('/', ' ').title()}")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the plot
        plotName = f"compare_{metric}.png"
        outputPath = os.path.join(vis_dir, plotName)
        plt.savefig(outputPath)
        plt.close()
        
        print(f"Comparison visualisation saved to {outputPath}")


def generate_all_visualisations(path: str, output_dir: Optional[str] = None) -> None:
    """
    Generate all visualisations for a metrics file or directory.
    
    Args:
        path: Path to metrics.csv file or directory containing metrics.csv
        output_dir: Directory to save visualisations (defaults to same directory as metrics.csv)
    """
    # Find the metrics file
    metrics_file = find_metrics_file(path)
    if not metrics_file:
        print(f"No metrics.csv file found at {path}")
        return
    
    # Set default output directory
    if output_dir is None:
        output_dir = os.path.dirname(metrics_file)
    
    # Create visualisations directory
    vis_dir = os.path.join(output_dir, "visualisations")
    os.makedirs(vis_dir, exist_ok=True)
    
    print(f"Generating all visualisations from {metrics_file}")
    print(f"Saving visualisations to {vis_dir}")
    
    # Load metrics data once
    try:
        df = pd.read_csv(metrics_file)
        
        # Ensure we have an iteration column
        if "iteration" not in df.columns and "training_iteration" in df.columns:
            df["iteration"] = df["training_iteration"]
        
        # Make sure iteration column exists
        if "iteration" not in df.columns:
            # Try to create it from index if small dataset
            print("Warning: No 'iteration' column found in metrics file, creating one from index")
            df["iteration"] = df.index
            
    except Exception as e:
        print(f"Error loading metrics file: {e}")
        return
        
    # Check if we have enough data points
    if len(df) < 2:
        print(f"Warning: Only {len(df)} data points found in metrics file. Some visualisations may be limited.")
    
    success_count = 0
    
    # Generate all plots with error handling
    try:
        plot_experiment(metrics_file, output_dir)
        success_count += 1
    except Exception as e:
        print(f"Error generating basic training plots: {e}")
    
    try:
        if len(df) >= 1:
            plot_resource_distribution(metrics_file, output_dir, df)
            success_count += 1
    except Exception as e:
        print(f"Error generating resource distribution plots: {e}")
    
    try:
        if len(df) >= 1:
            plot_fairness_metrics(metrics_file, output_dir, df)
            success_count += 1
    except Exception as e:
        print(f"Error generating fairness metrics plots: {e}")
    
    try:
        if len(df) >= 1:
            plot_specialisation_metrics(metrics_file, output_dir, df)
            success_count += 1
    except Exception as e:
        print(f"Error generating specialisation metrics plots: {e}")
    
    if success_count > 0:
        print(f"Successfully generated {success_count} types of visualisations in {vis_dir}")
    else:
        print("No visualisations could be generated. Check the errors above for details.")


if __name__ == "__main__":
    # If run directly, use command line arguments or prompt for a path
    parser = argparse.ArgumentParser(description="Generate visualisations from metrics file")
    parser.add_argument("path", nargs="?", help="Path to metrics.csv file or directory containing it")
    
    args = parser.parse_args()
    
    if args.path:
        # Use the provided path
        path = args.path
    else:
        # Prompt for a path
        path = input("Enter path to metrics.csv file or directory containing it: ")
    
    generate_all_visualisations(path) 