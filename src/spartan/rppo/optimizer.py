"""
RPPO Core Optimizer

Implements Algorithm 3: Reasoning-Privacy Pareto Optimization (RPPO)
from the SPARTAN paper.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spartan.config import RPPOConfig, SPARTANConfig
from spartan.rppo.bandit import UCBBandit
from spartan.rppo.pareto import ParetoFront


@dataclass
class RPPOResult:
    """Result from RPPO optimization.

    Attributes:
        params: Optimized parameter dictionary
        reward: Total reward achieved
        accuracy_reward: Accuracy component
        privacy_reward: Privacy component
        compute_reward: Compute efficiency component
        arm_index: Selected bandit arm index
        is_pareto_optimal: Whether params are on Pareto front
    """

    params: Dict[str, Any]
    reward: float
    accuracy_reward: float
    privacy_reward: float
    compute_reward: float
    arm_index: int
    is_pareto_optimal: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "params": self.params,
            "reward": self.reward,
            "accuracy_reward": self.accuracy_reward,
            "privacy_reward": self.privacy_reward,
            "compute_reward": self.compute_reward,
            "arm_index": self.arm_index,
            "is_pareto_optimal": self.is_pareto_optimal,
        }


class RPPO:
    """Reasoning-Privacy Pareto Optimization.

    Implements online multi-objective optimization using UCB bandit
    with optional gradient-based refinement. Balances:
    - Accuracy (reasoning quality)
    - Privacy (attack resistance)
    - Efficiency (computational cost)

    Example:
        >>> rppo = RPPO()
        >>> rppo.update({"risk_score": 0.3, "accuracy": 0.9, "compute": 0.5})
        >>> result = rppo.get_optimal_params()
        >>> print(f"Best params: {result.params}")
    """

    def __init__(
        self,
        config: Optional[SPARTANConfig] = None,
    ):
        """Initialize RPPO optimizer.

        Args:
            config: SPARTAN configuration
        """
        self.config = config or SPARTANConfig()
        self.rppo_config = self.config.get_rppo_config()

        # Objective weights
        self._omega = np.array(
            [
                self.rppo_config.accuracy_weight,
                self.rppo_config.privacy_weight,
                self.rppo_config.compute_weight,
            ]
        )

        # Initialize bandit arms with parameter configurations
        self._arms = self._initialize_arms(self.rppo_config.num_arms)

        # UCB bandit
        self._bandit = UCBBandit(
            num_arms=len(self._arms),
            exploration_constant=self.rppo_config.ucb_exploration,
        )

        # Pareto front tracking
        self._pareto_front = ParetoFront()

        # History
        self._history: List[Dict[str, Any]] = []
        self._episode = 0

        # Gradient-based refinement
        self._learning_rate = self.rppo_config.learning_rate
        self._current_params = self._arms[0].copy()

    def _initialize_arms(self, num_arms: int) -> List[Dict[str, Any]]:
        """Initialize bandit arms with diverse parameter configurations.

        Args:
            num_arms: Number of arms to create

        Returns:
            List of parameter configurations
        """
        arms = []

        # Generate diverse configurations
        epsilon_mins = np.linspace(0.001, 0.05, num_arms)
        epsilon_maxs = np.linspace(0.2, 0.8, num_arms)

        for i in range(num_arms):
            arm = {
                "epsilon_min": float(epsilon_mins[i]),
                "epsilon_max": float(epsilon_maxs[i]),
                "alpha": 0.3 + 0.1 * (i % 3),  # PRM weight
                "beta": 0.35 + 0.1 * ((i + 1) % 3),  # Vote weight
                "gamma": 0.35 - 0.1 * ((i + 2) % 3),  # MCTS weight
                "thresholds": {
                    "prm": 0.2 + 0.1 * (i / num_arms),
                    "vote": 0.3 + 0.1 * (i / num_arms),
                    "mcts": 0.4 + 0.1 * (i / num_arms),
                },
            }

            # Normalize weights
            total = arm["alpha"] + arm["beta"] + arm["gamma"]
            arm["alpha"] /= total
            arm["beta"] /= total
            arm["gamma"] /= total

            arms.append(arm)

        return arms

    def update(
        self,
        observation: Dict[str, Any],
    ) -> None:
        """Update optimizer with new observation.

        Args:
            observation: Dictionary containing:
                - risk_score: Privacy risk (lower is better)
                - accuracy (optional): Task accuracy
                - compute (optional): Normalized compute usage
                - defense_intensity (optional): Defense epsilon used
        """
        self._history.append(observation)
        self._episode += 1

        # Compute multi-objective reward
        reward_components = self._compute_reward(observation)
        total_reward = self._scalarize_reward(reward_components)

        # Update bandit with reward
        current_arm = self._bandit.get_current_arm()
        self._bandit.update(current_arm, total_reward)

        # Update Pareto front
        self._pareto_front.add_point(
            point=reward_components,
            params=self._current_params.copy(),
        )

        # Select next arm
        if self._episode >= self.rppo_config.min_history_size:
            next_arm = self._bandit.select_arm()
            self._current_params = self._arms[next_arm].copy()

            # Optional gradient refinement
            if self._episode % 10 == 0:
                self._gradient_refinement(observation)

    def _compute_reward(
        self,
        observation: Dict[str, Any],
    ) -> np.ndarray:
        """Compute multi-objective reward components.

        Args:
            observation: Observation dictionary

        Returns:
            Array of [accuracy_reward, privacy_reward, compute_reward]
        """
        # Accuracy reward (default 0.8 if not provided)
        accuracy = observation.get("accuracy", 0.8)
        r_acc = float(accuracy)

        # Privacy reward (1 - risk_score)
        risk_score = observation.get("risk_score", 0.5)
        r_priv = 1.0 - float(risk_score)

        # Compute efficiency reward (1 - normalized_compute)
        compute = observation.get("compute", 0.3)
        r_comp = 1.0 - float(compute)

        return np.array([r_acc, r_priv, r_comp])

    def _scalarize_reward(
        self,
        reward_components: np.ndarray,
    ) -> float:
        """Scalarize multi-objective reward.

        R = ω₁R_acc + ω₂R_priv + ω₃R_comp

        Args:
            reward_components: [accuracy, privacy, compute] rewards

        Returns:
            Scalarized reward
        """
        return float(np.dot(self._omega, reward_components))

    def _gradient_refinement(
        self,
        observation: Dict[str, Any],
    ) -> None:
        """Apply gradient-based parameter refinement.

        Uses finite differences to estimate gradients and update params.

        Args:
            observation: Recent observation for gradient estimation
        """
        # Estimate gradient using recent history
        if len(self._history) < 5:
            return

        recent = self._history[-5:]

        # Compute gradient for epsilon_max
        risk_scores = [obs.get("risk_score", 0.5) for obs in recent]
        defense_intensities = [obs.get("defense_intensity", 0.1) for obs in recent]

        if len(set(defense_intensities)) > 1:
            # Finite difference approximation
            grad_epsilon = np.corrcoef(defense_intensities, risk_scores)[0, 1]

            if not np.isnan(grad_epsilon):
                # Update epsilon_max in direction that reduces risk
                delta = -self._learning_rate * grad_epsilon * 0.1
                new_epsilon_max = self._current_params["epsilon_max"] + delta
                self._current_params["epsilon_max"] = float(np.clip(new_epsilon_max, 0.1, 0.9))

    def get_optimal_params(self) -> Optional[RPPOResult]:
        """Get current optimal parameters.

        Returns:
            RPPOResult with optimized parameters, or None if insufficient data
        """
        if self._episode < self.rppo_config.min_history_size:
            return None

        # Get best arm from bandit
        best_arm_idx = self._bandit.get_best_arm()
        best_params = self._arms[best_arm_idx].copy()

        # Merge with gradient-refined params
        best_params.update(
            {k: v for k, v in self._current_params.items() if k in ["epsilon_min", "epsilon_max"]}
        )

        # Compute current rewards
        if len(self._history) > 0:
            recent_obs = self._history[-1]
            reward_components = self._compute_reward(recent_obs)
            total_reward = self._scalarize_reward(reward_components)
        else:
            reward_components = np.array([0.8, 0.5, 0.7])
            total_reward = self._scalarize_reward(reward_components)

        # Check if on Pareto front
        is_pareto = self._pareto_front.is_pareto_optimal(reward_components)

        return RPPOResult(
            params=best_params,
            reward=total_reward,
            accuracy_reward=float(reward_components[0]),
            privacy_reward=float(reward_components[1]),
            compute_reward=float(reward_components[2]),
            arm_index=best_arm_idx,
            is_pareto_optimal=is_pareto,
        )

    def get_pareto_front(self) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """Get current Pareto front.

        Returns:
            List of (reward_point, params) tuples on Pareto front
        """
        return self._pareto_front.get_front()

    def reset(self) -> None:
        """Reset optimizer state."""
        self._history = []
        self._episode = 0
        self._bandit.reset()
        self._pareto_front = ParetoFront()
        self._current_params = self._arms[0].copy()

    def set_objective_weights(
        self,
        accuracy_weight: float,
        privacy_weight: float,
        compute_weight: float,
    ) -> None:
        """Update objective weights.

        Args:
            accuracy_weight: Weight for accuracy
            privacy_weight: Weight for privacy
            compute_weight: Weight for compute efficiency
        """
        total = accuracy_weight + privacy_weight + compute_weight
        self._omega = np.array(
            [
                accuracy_weight / total,
                privacy_weight / total,
                compute_weight / total,
            ]
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get optimizer statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            "total_episodes": self._episode,
            "num_arms": len(self._arms),
            "current_arm": self._bandit.get_current_arm(),
            "best_arm": self._bandit.get_best_arm() if self._episode > 0 else None,
            "arm_statistics": self._bandit.get_arm_stats(),
            "pareto_front_size": len(self._pareto_front.get_front()),
            "objective_weights": self._omega.tolist(),
        }
