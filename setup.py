#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="resourceProjectMARL",
    version="0.3.0",
    email="gurunga15@coventry.ac.uk",
    description="Multi-Agent Reinforcement Learning for Resource Collection Environment",
    packages=find_packages(),
install_requires = [
    "numpy>=1.26.0",
    "gymnasium>=1.0.0",           # Newer gymnasium interface supported by RLlib
    "ray[rllib]==2.44.1",         # Latest stable Ray + RLlib
    "torch>=2.2.0",               # Torch 2.2.x, matches with cu118/cu121
    "pandas>=2.6.0",              # Updated for performance fixes
    "matplotlib>=3.8.0",          # Visual support for training metrics
    "seaborn>=0.13.0",            # Newer statistical visualisation tools
    "PyQt6>=6.8.0",               # Matches latest PyQt with better Qt 6.6+ support
    "tensorboard>=2.14.0",        # Compatible with latest PyTorch and RLlib logging
],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 