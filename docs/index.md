# SPARTAN Documentation

Welcome to the SPARTAN framework documentation.

## Contents

- [Getting Started](getting_started.md)
- [API Reference](api_reference.md)
- [Algorithms](algorithms.md)
- [Examples](examples.md)

## Overview

SPARTAN (Secure Privacy-Adaptive Reasoning with Test-time Attack Neutralization) is a unified framework for detecting and defending against mechanistic privacy attacks in Reasoning Large Language Models.

## Quick Links

- [GitHub Repository](https://github.com/hmshujaatzaheer/spartan-framework)
- [PyPI Package](https://pypi.org/project/spartan-framework/)
- [Issue Tracker](https://github.com/hmshujaatzaheer/spartan-framework/issues)

## Core Components

### MPLQ (Mechanistic Privacy Leakage Quantification)
Detects privacy leakage through TTC mechanisms using:
- PRM score distribution analysis
- Vote concentration metrics
- MCTS value network deviation

### RAAS (Reasoning-Aware Adaptive Sanitization)
Applies adaptive defense based on detected risk:
- Feature-selective PRM noise injection
- Vote distribution flattening
- MCTS value perturbation

### RPPO (Reasoning-Privacy Pareto Optimization)
Optimizes defense parameters for best privacy-utility tradeoff:
- UCB bandit for arm selection
- Multi-objective reward function
- Pareto front tracking
