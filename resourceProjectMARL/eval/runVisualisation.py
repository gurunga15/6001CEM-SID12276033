#!/usr/bin/env python3
"""
Script for running visualisations from the UI.

This is a simple wrapper around plotMetrics.py that integrates with the UI launcher
and makes it easy to visualise training results.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the project root to the path
scriptDir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(scriptDir))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# Add the parent directory (resourceProjectMARL)
parent_dir = os.path.dirname(scriptDir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import visualisation functions
from eval.plotMetrics import generate_all_visualisations


def run_visualisation(results_dir=None):
    """
    Run visualisations for training results.
    
    Args:
        results_dir: Directory containing training results
    """
    if results_dir is None or not os.path.exists(results_dir):
        # If no directory provided, look for results directories
        if os.path.exists(os.path.join(parent_dir, "results")):
            results_base = os.path.join(parent_dir, "results")
            results_dirs = [os.path.join(results_base, d) for d in os.listdir(results_base) 
                           if os.path.isdir(os.path.join(results_base, d))]
            
            # Sort by modification time (newest first)
            results_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            if results_dirs:
                # Use the most recent results directory
                results_dir = results_dirs[0]
                print(f"Using most recent results directory: {results_dir}")
            else:
                print("No results directories found.")
                return
        else:
            print("No results directory specified and none found.")
            return
    
    # Check if the directory contains a metrics.csv file
    metrics_file = os.path.join(results_dir, "metrics.csv")
    if not os.path.exists(metrics_file):
        print(f"No metrics.csv file found in {results_dir}")
        return
    
    # Generate visualisations
    print(f"Generating visualisations for {results_dir}")
    generate_all_visualisations(results_dir)
    
    # Open the visualisations directory in explorer
    vis_dir = os.path.join(results_dir, "visualisations")
    if os.path.exists(vis_dir):
        try:
            os.startfile(vis_dir)
            print(f"Visualisations saved to {vis_dir}")
        except:
            print(f"Visualisations saved to {vis_dir} (could not open directory automatically)")
    else:
        print(f"Visualisations could not be generated successfully.")


def main():
    """Main function for the visualisation runner."""
    parser = argparse.ArgumentParser(description="Run visualisations for training results")
    parser.add_argument("--results-dir", type=str, help="Directory containing training results (default: most recent)")
    
    args = parser.parse_args()
    
    run_visualisation(args.results_dir)


if __name__ == "__main__":
    main() 