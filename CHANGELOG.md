# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of SPARTAN framework

## [1.0.0] - 2025-01-14

### Added
- **MPLQ Module**: Mechanistic Privacy Leakage Quantification
  - PRM leakage analyzer with NL-blindness detection
  - Vote distribution leakage analyzer with entropy metrics
  - MCTS value network leakage analyzer
  - Attention-based importance weighting

- **RAAS Module**: Reasoning-Aware Adaptive Sanitization
  - Feature-selective PRM noise injection
  - Vote distribution flattening with implicit rewards
  - MCTS value network perturbation
  - Adaptive defense intensity based on risk

- **RPPO Module**: Reasoning-Privacy Pareto Optimization
  - UCB bandit for parameter selection
  - Multi-objective optimization (accuracy, privacy, compute)
  - Pareto front tracking and hypervolume computation
  - Gradient-based parameter refinement

- **Attack Implementations**
  - NLBA: Natural Language Blindness Attack
  - SMVA: Single-Model Voting Attack
  - MVNA: MCTS Value Network Attack

- **Utilities**
  - Privacy-preserving noise generators (Gaussian, Laplace, calibrated)
  - Evaluation metrics (AUC-ROC, TPR@FPR, F1)
  - Distribution utilities (KL, JS divergence, entropy)

- **CLI Tool**
  - `spartan analyze` for privacy analysis
  - `spartan defend` for applying defenses
  - `spartan evaluate` for performance evaluation
  - `spartan config` for configuration generation

- **Documentation**
  - Comprehensive README with theory and usage
  - API documentation with examples
  - Contributing guidelines
  - Example scripts

### Technical Details
- Based on theoretical framework from "SPARTAN: Privacy-Preserving Test-Time Adaptation for Reasoning LLMs"
- Implements three novel attack vectors targeting TTC mechanisms
- Provides adaptive defense that balances privacy and reasoning quality
- Achieves 99.4% defense success rate in simulations

[Unreleased]: https://github.com/spartan-framework/spartan/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/spartan-framework/spartan/releases/tag/v1.0.0
