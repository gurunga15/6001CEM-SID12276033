# Project Summary - Multi-Agent Reinforcement Learning for Resource Collection

## Project Overview

This project implemented a multi-agent reinforcement learning (MARL) environment for resource collection tasks, where multiple agents navigate a grid-based world to collect resources while adapting to environmental dynamics such as day/night cycles, weather, and seasons. The system was built on Ray/RLlib and Gymnasium, with a focus on exploring different reward structures (individual, global, and hybrid) and tracking key metrics like fairness, efficiency, and coordination.

## Key Accomplishments

1. **Complete Environment Implementation**:
   - Created a configurable grid-based environment with multiple resource types
   - Implemented day/night cycles, weather systems, seasons, and obstacles
   - Designed a flexible reward structure with individual, global, and hybrid options
   - Built a robust observation and action space for agent learning

2. **Training Framework**:
   - Established PPO training with centralised critic capabilities
   - Created a comprehensive metrics tracking system via callbacks
   - Implemented episode recording for later replay and analysis
   - Built a user-friendly UI launcher for training configuration

3. **Analysis Tools**:
   - Developed evaluation tools for trained policies
   - Created metrics visualisation capabilities
   - Implemented replay mechanisms for recorded episodes
   - Built data collection systems for experimental analysis

## Challenges and Issues

### Technical Challenges

1. **Recursion Bugs in Callbacks**:
   - The callback system initially suffered from infinite recursion issues when recording episodes
   - Problem: `CompleteTrainingCallback` would recursively call itself through parent class methods
   - Solution: Implemented a recursion prevention mechanism using a flag to track callback execution status

2. **Observation Recording Problems**:
   - Episode recording initially failed to properly capture observations
   - Issue: Observations were empty dictionaries in recorded episodes
   - Resolution: Identified that the system was correctly storing actions, rewards, and info, but not the full observations (by design)
   - Adaptation: Modified the system to rely on info dictionaries for replay rather than observations

3. **GUI Integration Issues**:
   - PyQt6 GUI visualisation proved challenging to integrate with RLlib
   - Problem: Conflicts between UI thread and RLlib execution caused crashes
   - Partial solution: Created separate launch mechanisms, but full integration remains problematic
   - Current status: GUI works for basic demos but is unreliable for replay visualisation

### Research and Design Challenges

1. **Reward Structure Design**:
   - Balancing individual and team rewards proved challenging
   - Initial designs led to inefficient agent behaviour due to poor incentive alignment
   - Solution: Implemented hybrid reward system with configurable mixing parameter
   - Outcome: This allowed experimentation with different reward allocations to balance efficiency and fairness

2. **Environment Parameter Tuning**:
   - Finding appropriate values for environmental effects (day/night visibility, weather impacts)
   - Challenge: Too strong effects made learning unstable; too weak effects had no impact
   - Resolution: Conducted ablation studies to find appropriate parameter ranges
   - Result: Created balanced environmental effects that influenced agent behaviour without preventing learning

3. **Fairness Metrics Implementation**:
   - Defining appropriate fairness measures for multi-agent systems
   - Issue: Standard economic measures (Gini) didn't fully capture desired behaviour
   - Solution: Implemented multiple fairness metrics (Gini, Jain's Index) and resource specialisation measures
   - Outcome: More comprehensive evaluation framework for multi-agent coordination

## Changes and Adaptations

### Major Design Changes

1. **Algorithm Focus**:
   - Initial plan included MADDPG, QMIX, and PPO implementations
   - Change: Focused primarily on PPO with different reward structures
   - Reason: RLlib incompatability PPO proved more stable and easier to extend with centralised critics
   - Impact: More comprehensive exploration of reward structures rather than algorithm comparisons

2. **Observation Structure**:
   - Original design used complex Dict spaces with grid observations
   - Change: Added support for flattened observations for better compatibility with standard networks
   - Reason: RLlib handled flattened observations more efficiently
   - Result: Improved training stability and performance

3. **Callback Architecture**:
   - Initial design had separate callbacks for different functions
   - Change: Created a unified callback hierarchy with inheritance
   - Reason: Better code organisation and reduced duplication
   - Impact: More maintainable system but introduced recursion bugs (subsequently fixed)

### Codebase Improvements

1. **Code Organisation**:
   - Enhanced modularity by properly separating environment, algorithms, and utilities
   - Improved error handling and logging
   - Removed redundant code and simplified complex sections

## Settled Approaches

1. **Training Methodology**:
   - Settled on PPO with centralised critics as the primary algorithm
   - Established hybrid rewards as the most flexible approach
   - Standardised on episode-based training with fixed iteration counts

2. **Environment Design**:
   - Finalised the grid-based approach with configurable parameters
   - Established a standard set of environmental features (day/night, weather, seasons)
   - Settled on the resource and obstacle generation mechanisms

3. **Metrics and Evaluation**:
   - Established core metrics: reward, fairness (Gini and Jain's), efficiency, specialisation
   - Standardised the callback-based metrics collection approach
   - Finalised the episode recording format for replays

4. **Project Structure**:
   - Organised the codebase into clear modules (env, algorithms, train, eval, gui)
   - Established configuration management through config files and utilities
   - Created a standard approach for saving and loading results

## Main Issues

1. **GUI Visualisation**:
   - The PyQt6 GUI remains problematic with agent/env config
   - Future work needed to stabilise the visualisation system

2. **Algorithm Diversity**:
   - Current implementation focuses primarily on PPO
   - Future work could expand to more MARL algorithms

3. **Resource Generation**:
   - Resources can sometimes spawn in problematic locations
   - Improved placement algorithms needed

## Reflection

1. **Technical Lessons**:
   - Multi-agent environments require careful reward design for desired behaviour
   - Callback systems need safeguards against recursion
   - GUI integration should be loosely coupled from core training

2. **Research Insights**:
   - Different reward structures significantly impact agent coordination patterns
   - Environmental dynamics can enhance policy robustness if properly balanced
   - Fairness, efficiency, and specialisation often involve trade-offs
   - Hybrid reward structures offer the best balance for multi-agent coordination

3. **Development Practices**:
   - Importance of comprehensive metrics tracking from the beginning
   - Value of modular design for flexible experimentation
   - Benefits of unified configuration management
   - Need for thorough testing of callback and recording systems
