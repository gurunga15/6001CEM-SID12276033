#!/usr/bin/env python3
"""
Script for replaying recorded episodes from training.
"""
import os
import sys
import argparse
import pickle
import numpy as np
import glob
from typing import Dict, List, Any, Optional, Tuple

# Add project root to path
scriptDir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(scriptDir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from env.resourceEnv import ResourceCollectionEnv


def find_replay_files(replay_dir: str) -> List[str]:
    """
    Find all replay files in the given directory.
    
    Args:
        replay_dir: Directory to search for replay files
        
    Returns:
        List of replay file paths
    """
    # Look for replay_iter*.pkl files (current format)
    replay_files = glob.glob(os.path.join(replay_dir, "replay_iter*.pkl"))
    
    # If none found, also check legacy format (episode_*.pkl)
    if not replay_files:
        replay_files = glob.glob(os.path.join(replay_dir, "episode_*.pkl"))
    
    # Sort by iteration number if possible
    def extract_iteration(filename):
        try:
            # Extract just the filename without path
            basename = os.path.basename(filename)
            # Try to extract iteration number for different formats
            if basename.startswith("replay_iter"):
                return int(basename.replace("replay_iter", "").replace(".pkl", ""))
            elif basename.startswith("episode_"):
                return int(basename.replace("episode_", "").replace(".pkl", ""))
            else:
                return 0
        except:
            return 0
    
    replay_files.sort(key=extract_iteration)
    return replay_files


def list_available_replays(replay_dir: str) -> None:
    """
    List all available replay files in the given directory with details.
    
    Args:
        replay_dir: Directory containing replay files
    """
    replay_files = find_replay_files(replay_dir)
    
    if not replay_files:
        print(f"No replay files found in {replay_dir}")
        return
    
    print(f"\nFound {len(replay_files)} replay files in {replay_dir}:")
    print("\n{:<4} {:<30} {:<15} {:<10} {:<10}".format(
        "No.", "Filename", "Iteration", "Episode ID", "Reward"))
    print("-" * 70)
    
    for i, replay_file in enumerate(replay_files):
        # Extract just the filename
        basename = os.path.basename(replay_file)
        
        # Try to load metadata without loading full replay
        try:
            with open(replay_file, "rb") as f:
                replay_data = pickle.load(f)
            
            # Extract metadata
            metadata = replay_data.get("metadata", {})
            iteration = metadata.get("training_iteration", "Unknown")
            episode_id = metadata.get("episode_id", "Unknown")
            
            # Try to get final reward
            rewards = replay_data.get("rewards", [])
            if rewards:
                final_reward = sum(sum(r.values()) for r in rewards if r)
            else:
                final_reward = "Unknown"
            
            print("{:<4} {:<30} {:<15} {:<10} {:<10}".format(
                i+1, basename, iteration, episode_id, final_reward))
            
        except Exception as e:
            print("{:<4} {:<30} {:<15}".format(
                i+1, basename, f"Error: {str(e)[:30]}..."))
    
    print("\n")


def select_replay_file(replay_dir: str) -> Optional[str]:
    """
    Allow the user to select a replay file from the available options.
    
    Args:
        replay_dir: Directory containing replay files
        
    Returns:
        Selected replay file path or None if none selected
    """
    replay_files = find_replay_files(replay_dir)
    
    if not replay_files:
        print(f"No replay files found in {replay_dir}")
        return None
    
    list_available_replays(replay_dir)
    
    while True:
        try:
            selection = input("Enter the number of the replay file to visualise (or 'q' to quit): ")
            
            if selection.lower() in ['q', 'quit', 'exit']:
                return None
            
            idx = int(selection) - 1
            if 0 <= idx < len(replay_files):
                return replay_files[idx]
            else:
                print(f"Please enter a number between 1 and {len(replay_files)}")
        except ValueError:
            print("Please enter a valid number or 'q' to quit")


def load_replay_data(replay_file: str) -> Dict[str, Any]:
    """
    Load replay data from a file.
    
    Args:
        replay_file: Path to the replay file
        
    Returns:
        Dictionary with replay data
    """
    try:
        with open(replay_file, "rb") as f:
            replay_data = pickle.load(f)
        return replay_data
    except Exception as e:
        print(f"Error loading replay file: {e}")
        return {}


def create_env_from_replay(replay_data: Dict[str, Any]) -> Tuple[ResourceCollectionEnv, Dict[str, Any]]:
    """
    Create an environment instance from replay data.
    
    Args:
        replay_data: Replay data dictionary
        
    Returns:
        Tuple of (environment instance, copy of environment config)
    """
    metadata = replay_data.get("metadata", {})
    env_config = metadata.get("env_config", {})
    
    # Make a deep copy of the config to avoid modifying the original
    import copy
    env_config_copy = copy.deepcopy(env_config)
    
    # Force render mode to be 'human'
    if "renderMode" in env_config_copy:
        env_config_copy["renderMode"] = "human"
    else:
        env_config_copy["renderMode"] = "human"
    
    # Create the environment
    env = ResourceCollectionEnv(env_config_copy)
    
    return env, env_config_copy


def replay_episode(replay_file: str, render_mode: str = "human", step_delay: float = 0.5) -> None:
    """
    Replay a recorded episode from a file.
    
    Args:
        replay_file: Path to the replay file
        render_mode: Rendering mode ('human' for PyQt GUI, 'console' for terminal)
        step_delay: Delay between steps in seconds (for better visualisation)
    """
    import time
    
    # Load replay data
    replay_data = load_replay_data(replay_file)
    if not replay_data:
        return
    
    # Extract replay data components
    observations = replay_data.get("observations", [])
    actions = replay_data.get("actions", [])
    rewards = replay_data.get("rewards", [])
    infos = replay_data.get("infos", [])
    metadata = replay_data.get("metadata", {})
    
    # Print replay metadata
    print("\n=== Replay Information ===")
    print(f"Episode ID: {metadata.get('episode_id', 'Unknown')}")
    print(f"Training Iteration: {metadata.get('training_iteration', 'Unknown')}")
    print(f"Algorithm: {metadata.get('algorithm', 'Unknown')}")
    print(f"Steps: {len(actions)}")
    if rewards:
        total_reward = sum(sum(r.values()) for r in rewards if r and r.values())
        print(f"Total Reward: {total_reward:.2f}")
    print("=========================\n")
    
    # Create environment
    env, env_config = create_env_from_replay(replay_data)
    
    # Reset environment
    env.reset()
    
    print(f"Starting replay of episode with {len(actions)} steps...")
    
    # Check if we need to get agent IDs from infos instead of actions
    if infos and len(infos) > 0 and all(len(action) == 0 for action in actions[:10]):
        # Get agent IDs from first info entry
        first_info = infos[0]
        agent_ids = list(first_info.keys())
        print(f"Using agent IDs from infos: {agent_ids}")
        using_empty_actions = True
    else:
        using_empty_actions = False
    
    # Step through replay
    for step_idx in range(len(actions)):
        # Get step data
        if step_idx < len(actions):
            step_actions = actions[step_idx]
            
            # Handle empty action dictionaries by providing random actions
            if using_empty_actions or not step_actions:
                # Generate a random action for each agent (0-4 range for typical actions)
                # Use agent IDs from infos if available
                if step_idx < len(infos) and infos[step_idx]:
                    agent_ids = list(infos[step_idx].keys())
                    step_actions = {agent_id: np.random.randint(0, 5) for agent_id in agent_ids 
                                   if agent_id != "__all__"}
                else:
                    # If no infos available, use default agent naming pattern
                    numAgents = env_config.get("numAgents", 4)
                    step_actions = {f"agent_{i}": np.random.randint(0, 5) for i in range(numAgents)}
        else:
            step_actions = {}
        
        # Step the environment
        try:
            _, step_rewards, _, _, _ = env.step(step_actions)
            
            # Render the current state
            env.render()
            
            # Print step information
            print(f"\nStep {step_idx + 1}/{len(actions)}")
            print(f"Actions: {step_actions}")
            print(f"Rewards: {step_rewards}")
            
            # Extract metrics if available in infos
            if step_idx < len(infos) and infos[step_idx]:
                step_info = infos[step_idx]
                if any("fairness" in key for key in step_info.get(list(step_info.keys())[0], {})):
                    # Extract from first agent's info
                    first_agent = list(step_info.keys())[0]
                    agent_info = step_info[first_agent]
                    fairness = agent_info.get("fairnessGini", "N/A")
                    print(f"Fairness: {fairness}")
        except Exception as e:
            print(f"Error in step {step_idx}: {e}")
            print("Continuing with next step...")
            continue
        
        # Delay between steps
        if step_delay > 0:
            time.sleep(step_delay)
    
    print("\nReplay complete!")
    
    # Keep the window open if using human render
    if render_mode == "human":
        print("Press Ctrl+C to exit...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    
    # Close environment
    env.close()


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Replay a recorded episode")
    parser.add_argument("--replay-dir", type=str, help="Directory containing replay files")
    parser.add_argument("--replay-file", type=str, help="Specific replay file to replay")
    parser.add_argument("--list", action="store_true", help="List available replay files")
    parser.add_argument("--render-mode", type=str, default="human", 
                       choices=["human", "console"], help="Rendering mode")
    parser.add_argument("--step-delay", type=float, default=0.5,
                       help="Delay between steps (seconds)")
    
    args = parser.parse_args()
    
    # If no replay directory specified, look in default locations
    if not args.replay_dir and not args.replay_file:
        # Try to find results directory
        default_dirs = [
            os.path.join(project_dir, "results"),
            os.path.join(os.getcwd(), "results"),
        ]
        
        for default_dir in default_dirs:
            if os.path.exists(default_dir):
                # Look for replay directories
                replay_dirs = glob.glob(os.path.join(default_dir, "*/replays"))
                if replay_dirs:
                    args.replay_dir = replay_dirs[-1]  # Use most recent
                    print(f"Using replay directory: {args.replay_dir}")
                    break
    
    # List mode - just list available replays and exit
    if args.list and args.replay_dir:
        list_available_replays(args.replay_dir)
        return
    
    # If specific replay file is given, use it
    if args.replay_file:
        replay_file = args.replay_file
    # Otherwise, prompt user to select one
    elif args.replay_dir:
        replay_file = select_replay_file(args.replay_dir)
        if not replay_file:
            print("No replay file selected. Exiting.")
            return
    else:
        print("No replay directory or file specified. Use --replay-dir or --replay-file.")
        return
    
    # Replay the selected episode
    replay_episode(replay_file, args.render_mode, args.step_delay)


if __name__ == "__main__":
    main() 