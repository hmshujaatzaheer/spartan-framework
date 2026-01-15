"""
RAAS Core Sanitizer

Implements Algorithm 2: Reasoning-Aware Adaptive Sanitization (RAAS)
from the SPARTAN paper.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spartan.config import RAASConfig, SPARTANConfig
from spartan.mplq import MPLQResult
from spartan.raas.mcts_defense import MCTSDefense
from spartan.raas.prm_defense import PRMDefense
from spartan.raas.vote_defense import VoteDefense


@dataclass
class RAASResult:
    """Result from RAAS sanitization.

    Attributes:
        sanitized_output: The sanitized output string
        original_output: The original output
        defense_applied: Whether any defense was applied
        epsilon_used: Defense intensity used
        prm_defense_applied: Whether PRM defense was applied
        vote_defense_applied: Whether vote defense was applied
        mcts_defense_applied: Whether MCTS defense was applied
        details: Detailed defense information
    """

    sanitized_output: str
    original_output: str
    defense_applied: bool
    epsilon_used: float
    prm_defense_applied: bool = False
    vote_defense_applied: bool = False
    mcts_defense_applied: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "sanitized_output": self.sanitized_output,
            "original_output": self.original_output,
            "defense_applied": self.defense_applied,
            "epsilon_used": self.epsilon_used,
            "prm_defense_applied": self.prm_defense_applied,
            "vote_defense_applied": self.vote_defense_applied,
            "mcts_defense_applied": self.mcts_defense_applied,
            "details": self.details,
        }


class RAAS:
    """Reasoning-Aware Adaptive Sanitization.

    Implements adaptive defense mechanisms that scale with detected privacy
    risk. Key design principles:

    1. Importance-Aware Defense: High-importance samples receive stronger protection
    2. Feature-Selective Protection: Noise applied only to sensitive components
    3. NL-Blindness Exploitation: Safely perturb explanatory text
    4. Implicit Reward Alignment: Maintain reasoning quality under defense

    Example:
        >>> raas = RAAS()
        >>> result = raas.sanitize(
        ...     output="The answer is 4",
        ...     risk_analysis=mplq_result,
        ...     reasoning_steps=["Step 1...", "Step 2..."],
        ... )
        >>> print(result.sanitized_output)
    """

    def __init__(
        self,
        config: Optional[SPARTANConfig] = None,
    ):
        """Initialize RAAS sanitizer.

        Args:
            config: SPARTAN configuration
        """
        self.config = config or SPARTANConfig()
        self.raas_config = self.config.get_raas_config()

        # Initialize defense modules
        self.prm_defense = PRMDefense(
            noise_scale=self.raas_config.prm_noise_scale,
            nl_perturbation_ratio=self.raas_config.nl_perturbation_ratio,
            use_feature_selective=self.raas_config.use_feature_selective,
        )
        self.vote_defense = VoteDefense(
            temperature_base=self.raas_config.temperature_base,
            use_implicit_rewards=self.raas_config.use_implicit_rewards,
        )
        self.mcts_defense = MCTSDefense(
            depth_scale=self.raas_config.mcts_depth_scale,
        )

        # Defense parameters (can be updated by RPPO)
        self._epsilon_min = self.raas_config.epsilon_min
        self._epsilon_max = self.raas_config.epsilon_max
        self._thresholds = {
            "prm": self.config.prm_threshold,
            "vote": self.config.vote_threshold,
            "mcts": self.config.mcts_threshold,
        }

    def sanitize(
        self,
        output: str,
        risk_analysis: MPLQResult,
        reasoning_steps: Optional[List[str]] = None,
        vote_distribution: Optional[List[float]] = None,
        mcts_tree: Optional[Dict[str, Any]] = None,
    ) -> RAASResult:
        """Apply adaptive sanitization based on risk analysis.

        Implements Algorithm 2 from the SPARTAN paper.

        Args:
            output: Original output to sanitize
            risk_analysis: MPLQ risk analysis result
            reasoning_steps: Optional reasoning steps for PRM defense
            vote_distribution: Optional vote distribution for vote defense
            mcts_tree: Optional MCTS tree for MCTS defense

        Returns:
            RAASResult containing sanitized output and details
        """
        details: Dict[str, Any] = {}

        # Compute defense intensity based on risk and importance
        # ε = ε_min + (ε_max - ε_min) * L * ψ
        epsilon = self._compute_epsilon(
            risk_score=risk_analysis.total_risk,
            importance_weight=risk_analysis.importance_weight,
        )
        details["computed_epsilon"] = epsilon

        sanitized_output = output
        prm_applied = False
        vote_applied = False
        mcts_applied = False

        # PRM Defense: Feature-selective noise injection
        if risk_analysis.prm_leakage > self._thresholds["prm"] and reasoning_steps is not None:
            prm_result = self.prm_defense.apply(
                reasoning_steps=reasoning_steps,
                epsilon=epsilon,
                prm_leakage=risk_analysis.prm_leakage,
                threshold=self._thresholds["prm"],
            )
            details["prm_defense"] = prm_result
            prm_applied = True
            # Update output if reasoning trace is modified
            if prm_result.get("modified_trace"):
                sanitized_output = self._update_output_with_trace(
                    output, prm_result["modified_trace"]
                )

        # Vote Defense: Distribution flattening with implicit rewards
        if risk_analysis.vote_leakage > self._thresholds["vote"] and vote_distribution is not None:
            vote_result = self.vote_defense.apply(
                vote_distribution=vote_distribution,
                epsilon=epsilon,
                vote_leakage=risk_analysis.vote_leakage,
                threshold=self._thresholds["vote"],
            )
            details["vote_defense"] = vote_result
            vote_applied = True
            # Resample output based on flattened distribution
            if vote_result.get("resampled_output"):
                sanitized_output = vote_result["resampled_output"]

        # MCTS Defense: Value network perturbation
        if risk_analysis.mcts_leakage > self._thresholds["mcts"] and mcts_tree is not None:
            mcts_result = self.mcts_defense.apply(
                mcts_tree=mcts_tree,
                epsilon=epsilon,
                mcts_leakage=risk_analysis.mcts_leakage,
                threshold=self._thresholds["mcts"],
            )
            details["mcts_defense"] = mcts_result
            mcts_applied = True
            # Update output from re-run search
            if mcts_result.get("rerun_output"):
                sanitized_output = mcts_result["rerun_output"]

        defense_applied = prm_applied or vote_applied or mcts_applied

        return RAASResult(
            sanitized_output=sanitized_output,
            original_output=output,
            defense_applied=defense_applied,
            epsilon_used=epsilon,
            prm_defense_applied=prm_applied,
            vote_defense_applied=vote_applied,
            mcts_defense_applied=mcts_applied,
            details=details,
        )

    def _compute_epsilon(
        self,
        risk_score: float,
        importance_weight: float,
    ) -> float:
        """Compute defense intensity epsilon.

        ε = ε_min + (ε_max - ε_min) * L * ψ

        Args:
            risk_score: Total privacy risk (L)
            importance_weight: Sample importance (ψ)

        Returns:
            Defense intensity epsilon
        """
        epsilon = (
            self._epsilon_min
            + (self._epsilon_max - self._epsilon_min) * risk_score * importance_weight
        )
        return float(np.clip(epsilon, self._epsilon_min, self._epsilon_max))

    def _update_output_with_trace(
        self,
        output: str,
        modified_trace: List[str],
    ) -> str:
        """Update output incorporating modified reasoning trace.

        Args:
            output: Original output
            modified_trace: Modified reasoning steps

        Returns:
            Updated output string
        """
        # In practice, this would reformat the output with the new trace
        # For now, we return the original output as the final answer
        # but the trace modification is recorded
        return output

    def update_params(
        self,
        params: Dict[str, Any],
    ) -> None:
        """Update defense parameters from RPPO optimization.

        Args:
            params: Dictionary of parameter updates
        """
        if "epsilon_min" in params:
            self._epsilon_min = params["epsilon_min"]
        if "epsilon_max" in params:
            self._epsilon_max = params["epsilon_max"]
        if "thresholds" in params:
            self._thresholds.update(params["thresholds"])

        # Update sub-module parameters
        if "prm_noise_scale" in params:
            self.prm_defense.noise_scale = params["prm_noise_scale"]
        if "temperature_base" in params:
            self.vote_defense.temperature_base = params["temperature_base"]
        if "mcts_depth_scale" in params:
            self.mcts_defense.depth_scale = params["mcts_depth_scale"]

    def get_params(self) -> Dict[str, Any]:
        """Get current defense parameters."""
        return {
            "epsilon_min": self._epsilon_min,
            "epsilon_max": self._epsilon_max,
            "thresholds": self._thresholds.copy(),
            "prm_noise_scale": self.prm_defense.noise_scale,
            "temperature_base": self.vote_defense.temperature_base,
            "mcts_depth_scale": self.mcts_defense.depth_scale,
        }
