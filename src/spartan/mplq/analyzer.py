"""
MPLQ Core Analyzer

Implements Algorithm 1: Mechanistic Privacy Leakage Quantification (MPLQ)
from the SPARTAN paper.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spartan.config import MPLQConfig, SPARTANConfig
from spartan.mplq.mcts_leakage import MCTSLeakageAnalyzer
from spartan.mplq.prm_leakage import PRMLeakageAnalyzer
from spartan.mplq.vote_leakage import VoteLeakageAnalyzer


@dataclass
class MPLQResult:
    """Result from MPLQ privacy leakage analysis.

    Attributes:
        total_risk: Aggregated privacy risk score (0-1)
        prm_leakage: PRM-specific leakage score
        vote_leakage: Voting distribution leakage score
        mcts_leakage: MCTS value network leakage score
        importance_weight: Sample importance weight
        component_weights: Learned weights (alpha, beta, gamma)
        details: Detailed analysis information
    """

    total_risk: float
    prm_leakage: float
    vote_leakage: float
    mcts_leakage: float
    importance_weight: float
    component_weights: Tuple[float, float, float]
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "total_risk": self.total_risk,
            "prm_leakage": self.prm_leakage,
            "vote_leakage": self.vote_leakage,
            "mcts_leakage": self.mcts_leakage,
            "importance_weight": self.importance_weight,
            "component_weights": {
                "alpha": self.component_weights[0],
                "beta": self.component_weights[1],
                "gamma": self.component_weights[2],
            },
            "details": self.details,
        }

    def exceeds_threshold(
        self,
        prm_threshold: float = 0.3,
        vote_threshold: float = 0.4,
        mcts_threshold: float = 0.5,
    ) -> Dict[str, bool]:
        """Check which components exceed their thresholds."""
        return {
            "prm": self.prm_leakage > prm_threshold,
            "vote": self.vote_leakage > vote_threshold,
            "mcts": self.mcts_leakage > mcts_threshold,
        }


class MPLQ:
    """Mechanistic Privacy Leakage Quantification.

    Implements the MPLQ algorithm for detecting privacy leakage through
    reasoning LLM test-time compute mechanisms.

    The algorithm operates in four phases:
    1. PRM Leakage Analysis - KL divergence from reference distribution
    2. Voting Distribution Analysis - Entropy-based concentration detection
    3. MCTS Value Network Analysis - Deviation from baseline values
    4. Aggregate Risk Score - Importance-weighted combination

    Example:
        >>> mplq = MPLQ()
        >>> result = mplq.analyze(
        ...     query="What is 2+2?",
        ...     reasoning_steps=["Step 1: ...", "Step 2: ..."],
        ...     prm_scores=[0.9, 0.95],
        ...     vote_distribution=[0.8, 0.1, 0.1],
        ... )
        >>> print(f"Risk: {result.total_risk:.4f}")
    """

    def __init__(
        self,
        config: Optional[SPARTANConfig] = None,
        reference_distribution: Optional[np.ndarray] = None,
    ):
        """Initialize MPLQ analyzer.

        Args:
            config: SPARTAN configuration
            reference_distribution: Pre-computed reference distribution for PRM scores
        """
        self.config = config or SPARTANConfig()
        self.mplq_config = self.config.get_mplq_config()

        # Initialize sub-analyzers
        self.prm_analyzer = PRMLeakageAnalyzer(
            reference_distribution=reference_distribution,
            kl_smoothing=self.mplq_config.kl_smoothing,
            variance_weight=self.mplq_config.variance_weight,
        )
        self.vote_analyzer = VoteLeakageAnalyzer()
        self.mcts_analyzer = MCTSLeakageAnalyzer()

        # Learned weights (can be updated by RPPO)
        self._alpha = 0.4  # PRM weight
        self._beta = 0.35  # Vote weight
        self._gamma = 0.25  # MCTS weight

    def analyze(
        self,
        query: str,
        reasoning_steps: Optional[List[str]] = None,
        prm_scores: Optional[List[float]] = None,
        vote_distribution: Optional[List[float]] = None,
        mcts_values: Optional[List[float]] = None,
        mcts_tree: Optional[Dict[str, Any]] = None,
    ) -> MPLQResult:
        """Analyze privacy leakage for a query.

        Implements Algorithm 1 from the SPARTAN paper.

        Args:
            query: The input query
            reasoning_steps: List of reasoning step strings
            prm_scores: PRM scores for each reasoning step
            vote_distribution: Vote distribution from self-consistency
            mcts_values: Value scores from MCTS nodes
            mcts_tree: Full MCTS tree structure (optional)

        Returns:
            MPLQResult containing leakage scores and analysis
        """
        details: Dict[str, Any] = {"query_length": len(query)}

        # Phase 1: PRM Leakage Analysis
        prm_leakage = 0.0
        if prm_scores is not None and len(prm_scores) > 0:
            prm_result = self.prm_analyzer.analyze(
                prm_scores=prm_scores,
                reasoning_steps=reasoning_steps,
            )
            prm_leakage = prm_result["leakage_score"]
            details["prm"] = prm_result

        # Phase 2: Voting Distribution Analysis
        vote_leakage = 0.0
        if vote_distribution is not None and len(vote_distribution) > 0:
            vote_result = self.vote_analyzer.analyze(
                vote_distribution=vote_distribution,
            )
            vote_leakage = vote_result["leakage_score"]
            details["vote"] = vote_result

        # Phase 3: MCTS Value Network Analysis
        mcts_leakage = 0.0
        if mcts_values is not None and len(mcts_values) > 0:
            mcts_result = self.mcts_analyzer.analyze(
                mcts_values=mcts_values,
                mcts_tree=mcts_tree,
            )
            mcts_leakage = mcts_result["leakage_score"]
            details["mcts"] = mcts_result

        # Phase 4: Compute importance weight
        importance_weight = self._compute_importance_weight(
            query=query,
            reasoning_steps=reasoning_steps,
            prm_scores=prm_scores,
        )
        details["importance_weight_raw"] = importance_weight

        # Phase 5: Aggregate risk score with importance weighting
        weighted_sum = (
            self._alpha * prm_leakage + self._beta * vote_leakage + self._gamma * mcts_leakage
        )

        # Apply sigmoid with importance weighting
        # L_total = σ(ψ * (α*L_prm + β*L_vote + γ*L_mcts))
        if self.mplq_config.importance_weighting:
            total_risk = self._sigmoid(importance_weight * weighted_sum)
        else:
            total_risk = self._sigmoid(weighted_sum)

        return MPLQResult(
            total_risk=total_risk,
            prm_leakage=prm_leakage,
            vote_leakage=vote_leakage,
            mcts_leakage=mcts_leakage,
            importance_weight=importance_weight,
            component_weights=(self._alpha, self._beta, self._gamma),
            details=details,
        )

    def _compute_importance_weight(
        self,
        query: str,
        reasoning_steps: Optional[List[str]] = None,
        prm_scores: Optional[List[float]] = None,
    ) -> float:
        """Compute sample importance weight.

        Following Wen et al., high-importance samples face higher attack success rates.
        We estimate importance based on:
        - Query complexity (length, special tokens)
        - Reasoning depth (number of steps)
        - PRM score patterns (high confidence may indicate memorization)

        Args:
            query: Input query
            reasoning_steps: Reasoning steps if available
            prm_scores: PRM scores if available

        Returns:
            Importance weight in range [0.5, 2.0]
        """
        importance = 1.0

        # Factor 1: Query complexity
        query_complexity = min(len(query) / 500, 1.0)  # Normalize by 500 chars
        importance += 0.2 * query_complexity

        # Factor 2: Reasoning depth
        if reasoning_steps is not None:
            depth_factor = min(len(reasoning_steps) / 10, 1.0)  # Normalize by 10 steps
            importance += 0.3 * depth_factor

        # Factor 3: PRM score patterns
        if prm_scores is not None and len(prm_scores) > 0:
            scores = np.array(prm_scores)
            # High mean and low variance suggests memorization
            mean_score = np.mean(scores)
            variance = np.var(scores)

            if mean_score > 0.9 and variance < 0.01:
                importance += 0.5  # High confidence, likely memorized
            elif mean_score > 0.8:
                importance += 0.2

        # Clamp to [0.5, 2.0]
        return np.clip(importance, 0.5, 2.0)

    def _sigmoid(self, x: float) -> float:
        """Compute sigmoid function with numerical stability."""
        if x >= 0:
            return 1.0 / (1.0 + np.exp(-x))
        else:
            exp_x = np.exp(x)
            return exp_x / (1.0 + exp_x)

    def update_weights(
        self,
        alpha: float,
        beta: float,
        gamma: float,
    ) -> None:
        """Update component weights from RPPO optimization.

        Args:
            alpha: Weight for PRM leakage
            beta: Weight for vote leakage
            gamma: Weight for MCTS leakage
        """
        # Normalize weights to sum to 1
        total = alpha + beta + gamma
        if total > 0:
            self._alpha = alpha / total
            self._beta = beta / total
            self._gamma = gamma / total

    def get_weights(self) -> Tuple[float, float, float]:
        """Get current component weights."""
        return (self._alpha, self._beta, self._gamma)

    def set_reference_distribution(
        self,
        distribution: np.ndarray,
    ) -> None:
        """Set reference distribution for PRM analysis.

        Args:
            distribution: Reference distribution from non-member data
        """
        self.prm_analyzer.set_reference_distribution(distribution)
