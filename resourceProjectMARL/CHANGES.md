# Recent Changes to Resource Collection Environment

This document summarises the last enhancements made to the Multi-Agent Resource Collection environment for improved metrics tracking and analysis.

## Enhanced Fairness Metrics

1. **Added Jain's Fairness Index**
   - Implemented in `env/utils.py` as `calculateJainFairnessIndex()`
   - Provides an alternative fairness measure (range 1/n to 1)
   - Complements the existing Gini coefficient for more robust fairness analysis

2. **Improved Fairness Calculation**
   - Modified the Gini coefficient to be reported as "fairness" (1 - Gini)
   - Higher values now consistently indicate greater fairness across all metrics
   - Added proper documentation for all fairness metrics

## Enhanced Coordination Tracking

1. **Agent Overlap Monitoring**
   - Added tracking of agent position overlaps (multiple agents in same cell)
   - Implemented in `algorithms/callbacks.py` in the coordination metrics
   - Helps assess how well agents distribute spatially

2. **Resource Specialisation**
   - Added metrics to track if agents specialise in collecting specific resource types
   - Calculated using Gini coefficient across agent resource collection counts
   - Higher values indicate stronger specialisation (agents focus on different resources)

3. **Task Division**
   - Added metrics for overall task allocation efficiency
   - Combines specialisation and spatial distribution data
   - Useful for evaluating emergent coordination behaviour

## Environment Enhancements and Bug Fixes

1. **Improved Weather System**
   - Added snow weather type for more diverse environmental conditions
   - Balanced weather effects on visibility and movement
   - Standardised weather effect parameters for consistency
   - Updated visibility calculations for all weather types
   - Added season-appropriate weather tendencies (more snow in winter, more rain in spring)
   - Enhanced weather changing probability based on current season
   - Implemented proper probability normalization for weather tendencies

2. **Enhanced Day/Night Cycle System**
   - Fixed day/night phase durations to match configuration (40% day, 40% night, 10% dawn/dusk)
   - Switched from discrete visionRadiusModifier to multiplicative visibilityFactor for smoother effects
   - Added season-specific day/night cycle configurations (longer days in summer, longer nights in winter)
   - Improved visibility calculation by properly combining weather and day/night effects

3. **Added Obstacle System**
   - Added obstacles to the grid environment for more challenging navigation
   - Implemented obstacle generation with configurable density and distribution
   - Added detection and handling of obstacle collisions
   - Updated observation space to include obstacle channels
   - Added support for mixed obstacle distribution (combining random and clustered patterns)

4. **Enhanced Resource Distribution**
   - Added support for mixed resource distribution (combining random and clustered patterns)
   - Improved resource generation to avoid obstacle positions
   - Enhanced seasonal effects on resource availability
   - Fixed resource respawning to honor environmental constraints

5. **API Compatibility Fixes**
   - Fixed action_space usage to use brackets instead of parentheses
   - Updated environment interfaces to follow gymnasium standards
   - Improved compatibility with Ray's MultiAgentEnv API
   - Fixed inconsistencies in environment method naming

6. **Code Quality Improvements**
   - Removed redundant imports to improve readability and reduce loading time
   - Fixed resource collection logic for more consistent behaviour
   - Improved error handling and reporting for debugging
   - Enhanced agent movement and collision detection

7. **Performance Optimisations**
   - Streamlined observation generation for better performance
   - Improved resource and obstacle management for efficiency
   - Reduced computational overhead in environment stepping function

## Training Improvements

1. **Unified Training Interface**
   - Created a single entry point (`train_main.py`) for all training options
   - Consolidated configuration options for better discoverability
   - Improved error handling and validation of training parameters

2. **Enhanced Exploration Control**
   - Added entropy coefficient scheduling for better exploration control
   - Implemented linear, exponential, and step-based schedules
   - Added proper warmup period for more stable initial learning

3. **Visualisation Capabilities**
   - Added TensorBoard integration for real-time training visualisation
   - Improved CSV logging with all relevant metrics
   - Added visualisation capabilities for entropy scheduling

4. **Documentation and Examples**
   - Added extensive documentation for all components
   - Provided example configurations for different training scenarios
   - Included detailed API documentation for custom usage

These changes significantly enhance the analytical capabilities of the codebase, enabling more robust evaluation of agent performance, fairness, and emergent behaviours for the dissertation research. The codebase reorganization makes the system easier to understand, maintain, and extend while ensuring reproducible experiments. 