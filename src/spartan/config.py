"""
SPARTAN Configuration Module

Contains configuration classes for all SPARTAN modules.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class MPLQConfig:
    """Configuration for MPLQ (Mechanistic Privacy Leakage Quantification).

    Attributes:
        prm_threshold: Threshold for PRM leakage detection
        vote_threshold: Threshold for vote concentration detection
        mcts_threshold: Threshold for MCTS deviation detection
        reference_distribution_bins: Number of bins for reference distribution
        kl_smoothing: Smoothing factor for KL divergence computation
        variance_weight: Weight for variance term in PRM leakage
        importance_weighting: Whether to use importance weighting
    """

    prm_threshold: float = 0.3
    vote_threshold: float = 0.4
    mcts_threshold: float = 0.5
    reference_distribution_bins: int = 50
    kl_smoothing: float = 1e-10
    variance_weight: float = 0.1
    importance_weighting: bool = True


@dataclass
class RAASConfig:
    """Configuration for RAAS (Reasoning-Aware Adaptive Sanitization).

    Attributes:
        epsilon_min: Minimum noise level
        epsilon_max: Maximum noise level
        temperature_base: Base temperature for vote flattening
        prm_noise_scale: Scale factor for PRM noise injection
        mcts_depth_scale: Scale factor for MCTS depth-based noise
        nl_perturbation_ratio: Ratio of NL to math perturbation
        use_feature_selective: Whether to use feature-selective protection
        use_implicit_rewards: Whether to use implicit reward resampling
    """

    epsilon_min: float = 0.01
    epsilon_max: float = 0.5
    temperature_base: float = 1.0
    prm_noise_scale: float = 1.0
    mcts_depth_scale: float = 0.1
    nl_perturbation_ratio: float = 10.0
    use_feature_selective: bool = True
    use_implicit_rewards: bool = True


@dataclass
class RPPOConfig:
    """Configuration for RPPO (Reasoning-Privacy Pareto Optimization).

    Attributes:
        learning_rate: Learning rate for gradient-based refinement
        num_arms: Number of bandit arms
        ucb_exploration: UCB exploration constant
        accuracy_weight: Weight for accuracy in multi-objective
        privacy_weight: Weight for privacy in multi-objective
        compute_weight: Weight for compute efficiency in multi-objective
        min_history_size: Minimum history size before optimization
        batch_size: Batch size for reward computation
    """

    learning_rate: float = 0.01
    num_arms: int = 10
    ucb_exploration: float = 2.0
    accuracy_weight: float = 0.4
    privacy_weight: float = 0.4
    compute_weight: float = 0.2
    min_history_size: int = 10
    batch_size: int = 32


@dataclass
class SPARTANConfig:
    """Main SPARTAN configuration combining all module configs.

    Example:
        >>> config = SPARTANConfig(
        ...     prm_threshold=0.25,
        ...     epsilon_max=0.6,
        ...     learning_rate=0.005,
        ... )
    """

    # MPLQ settings (flattened for convenience)
    prm_threshold: float = 0.3
    vote_threshold: float = 0.4
    mcts_threshold: float = 0.5
    reference_distribution_bins: int = 50
    kl_smoothing: float = 1e-10
    variance_weight: float = 0.1
    importance_weighting: bool = True

    # RAAS settings
    epsilon_min: float = 0.01
    epsilon_max: float = 0.5
    temperature_base: float = 1.0
    prm_noise_scale: float = 1.0
    mcts_depth_scale: float = 0.1
    nl_perturbation_ratio: float = 10.0
    use_feature_selective: bool = True
    use_implicit_rewards: bool = True

    # RPPO settings
    learning_rate: float = 0.01
    num_arms: int = 10
    ucb_exploration: float = 2.0
    accuracy_weight: float = 0.4
    privacy_weight: float = 0.4
    compute_weight: float = 0.2
    min_history_size: int = 10
    batch_size: int = 32

    def get_mplq_config(self) -> MPLQConfig:
        """Extract MPLQ configuration."""
        return MPLQConfig(
            prm_threshold=self.prm_threshold,
            vote_threshold=self.vote_threshold,
            mcts_threshold=self.mcts_threshold,
            reference_distribution_bins=self.reference_distribution_bins,
            kl_smoothing=self.kl_smoothing,
            variance_weight=self.variance_weight,
            importance_weighting=self.importance_weighting,
        )

    def get_raas_config(self) -> RAASConfig:
        """Extract RAAS configuration."""
        return RAASConfig(
            epsilon_min=self.epsilon_min,
            epsilon_max=self.epsilon_max,
            temperature_base=self.temperature_base,
            prm_noise_scale=self.prm_noise_scale,
            mcts_depth_scale=self.mcts_depth_scale,
            nl_perturbation_ratio=self.nl_perturbation_ratio,
            use_feature_selective=self.use_feature_selective,
            use_implicit_rewards=self.use_implicit_rewards,
        )

    def get_rppo_config(self) -> RPPOConfig:
        """Extract RPPO configuration."""
        return RPPOConfig(
            learning_rate=self.learning_rate,
            num_arms=self.num_arms,
            ucb_exploration=self.ucb_exploration,
            accuracy_weight=self.accuracy_weight,
            privacy_weight=self.privacy_weight,
            compute_weight=self.compute_weight,
            min_history_size=self.min_history_size,
            batch_size=self.batch_size,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "mplq": {
                "prm_threshold": self.prm_threshold,
                "vote_threshold": self.vote_threshold,
                "mcts_threshold": self.mcts_threshold,
                "reference_distribution_bins": self.reference_distribution_bins,
                "kl_smoothing": self.kl_smoothing,
                "variance_weight": self.variance_weight,
                "importance_weighting": self.importance_weighting,
            },
            "raas": {
                "epsilon_min": self.epsilon_min,
                "epsilon_max": self.epsilon_max,
                "temperature_base": self.temperature_base,
                "prm_noise_scale": self.prm_noise_scale,
                "mcts_depth_scale": self.mcts_depth_scale,
                "nl_perturbation_ratio": self.nl_perturbation_ratio,
                "use_feature_selective": self.use_feature_selective,
                "use_implicit_rewards": self.use_implicit_rewards,
            },
            "rppo": {
                "learning_rate": self.learning_rate,
                "num_arms": self.num_arms,
                "ucb_exploration": self.ucb_exploration,
                "accuracy_weight": self.accuracy_weight,
                "privacy_weight": self.privacy_weight,
                "compute_weight": self.compute_weight,
                "min_history_size": self.min_history_size,
                "batch_size": self.batch_size,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SPARTANConfig":
        """Create configuration from dictionary."""
        flat_dict = {}
        for section in ["mplq", "raas", "rppo"]:
            if section in data:
                flat_dict.update(data[section])
        return cls(**flat_dict)

    def validate(self) -> List[str]:
        """Validate configuration values.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Threshold validation
        for name, value in [
            ("prm_threshold", self.prm_threshold),
            ("vote_threshold", self.vote_threshold),
            ("mcts_threshold", self.mcts_threshold),
        ]:
            if not 0 <= value <= 1:
                errors.append(f"{name} must be in [0, 1], got {value}")

        # Epsilon validation
        if self.epsilon_min < 0:
            errors.append(f"epsilon_min must be >= 0, got {self.epsilon_min}")
        if self.epsilon_max < self.epsilon_min:
            errors.append(
                f"epsilon_max ({self.epsilon_max}) must be >= " f"epsilon_min ({self.epsilon_min})"
            )

        # Weight validation
        weight_sum = self.accuracy_weight + self.privacy_weight + self.compute_weight
        if abs(weight_sum - 1.0) > 1e-6:
            errors.append(f"Objective weights must sum to 1.0, got {weight_sum}")

        # Positive value validation
        for name, value in [
            ("learning_rate", self.learning_rate),
            ("num_arms", self.num_arms),
            ("ucb_exploration", self.ucb_exploration),
            ("min_history_size", self.min_history_size),
            ("batch_size", self.batch_size),
        ]:
            if value <= 0:
                errors.append(f"{name} must be > 0, got {value}")

        return errors
