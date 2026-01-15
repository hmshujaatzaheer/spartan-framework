"""
Core SPARTAN Framework Implementation

Integrates MPLQ (attack detection), RAAS (adaptive defense), and RPPO (optimization)
into a unified privacy protection system for reasoning LLMs.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np

from spartan.config import SPARTANConfig
from spartan.models.base import BaseReasoningLLM
from spartan.mplq import MPLQ, MPLQResult
from spartan.raas import RAAS, RAASResult
from spartan.rppo import RPPO, RPPOResult


@dataclass
class SPARTANResult:
    """Result from SPARTAN processing pipeline.

    Attributes:
        output: The sanitized output string
        original_output: The original unsanitized output
        risk_score: Overall privacy risk score (0-1)
        risk_analysis: Detailed MPLQ analysis result
        defense_result: RAAS defense application result
        optimization_result: RPPO optimization result (if available)
        defense_applied: Whether defense was applied
        metadata: Additional metadata about the processing
    """

    output: str
    original_output: str
    risk_score: float
    risk_analysis: MPLQResult
    defense_result: RAASResult
    optimization_result: Optional[RPPOResult] = None
    defense_applied: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "output": self.output,
            "original_output": self.original_output,
            "risk_score": self.risk_score,
            "risk_analysis": self.risk_analysis.to_dict(),
            "defense_result": self.defense_result.to_dict(),
            "optimization_result": (
                self.optimization_result.to_dict() if self.optimization_result else None
            ),
            "defense_applied": self.defense_applied,
            "metadata": self.metadata,
        }


class SPARTAN:
    """SPARTAN: Secure Privacy-Adaptive Reasoning with Test-time Attack Neutralization.

    Main class that orchestrates the three core modules:
    - MPLQ: Mechanistic Privacy Leakage Quantification
    - RAAS: Reasoning-Aware Adaptive Sanitization
    - RPPO: Reasoning-Privacy Pareto Optimization

    Example:
        >>> from spartan import SPARTAN
        >>> from spartan.models import MockReasoningLLM
        >>>
        >>> llm = MockReasoningLLM()
        >>> spartan = SPARTAN(llm)
        >>> result = spartan.process("What is 2+2?")
        >>> print(result.output)
    """

    def __init__(
        self,
        model: BaseReasoningLLM,
        config: Optional[SPARTANConfig] = None,
        enable_optimization: bool = True,
    ):
        """Initialize SPARTAN framework.

        Args:
            model: The reasoning LLM to protect
            config: Configuration for SPARTAN modules
            enable_optimization: Whether to enable RPPO optimization
        """
        self.model = model
        self.config = config or SPARTANConfig()
        self.enable_optimization = enable_optimization

        # Initialize modules
        self.mplq = MPLQ(config=self.config)
        self.raas = RAAS(config=self.config)
        self.rppo = RPPO(config=self.config) if enable_optimization else None

        # Historical data for optimization
        self._history: List[Dict[str, Any]] = []

    def process(
        self,
        query: str,
        num_samples: int = 5,
        return_trace: bool = False,
    ) -> SPARTANResult:
        """Process a query through the SPARTAN pipeline.

        Args:
            query: The input query to process
            num_samples: Number of samples for self-consistency voting
            return_trace: Whether to include reasoning trace in metadata

        Returns:
            SPARTANResult containing the sanitized output and analysis
        """
        # Step 1: Generate reasoning output from LLM
        llm_output = self.model.generate(query, num_samples=num_samples)

        # Step 2: Run MPLQ analysis
        risk_analysis = self.mplq.analyze(
            query=query,
            reasoning_steps=llm_output.reasoning_steps,
            prm_scores=llm_output.prm_scores,
            vote_distribution=llm_output.vote_distribution,
            mcts_values=llm_output.mcts_values,
        )

        # Step 3: Get optimization parameters if enabled
        optimization_result = None
        if self.rppo is not None and len(self._history) > 0:
            optimization_result = self.rppo.get_optimal_params()
            # Update RAAS with optimized parameters
            if optimization_result is not None:
                self.raas.update_params(optimization_result.params)

        # Step 4: Apply RAAS defense based on risk
        defense_result = self.raas.sanitize(
            output=llm_output.output,
            risk_analysis=risk_analysis,
            reasoning_steps=llm_output.reasoning_steps,
            vote_distribution=llm_output.vote_distribution,
            mcts_tree=llm_output.mcts_tree,
        )

        # Step 5: Record to history for optimization
        history_entry = {
            "query": query,
            "risk_score": risk_analysis.total_risk,
            "defense_intensity": defense_result.epsilon_used,
            "original_output": llm_output.output,
            "sanitized_output": defense_result.sanitized_output,
        }
        self._history.append(history_entry)

        # Update RPPO with new data
        if self.rppo is not None:
            self.rppo.update(history_entry)

        # Build metadata
        metadata = {
            "num_samples": num_samples,
            "model_name": self.model.name,
        }
        if return_trace:
            metadata["reasoning_trace"] = llm_output.reasoning_steps

        return SPARTANResult(
            output=defense_result.sanitized_output,
            original_output=llm_output.output,
            risk_score=risk_analysis.total_risk,
            risk_analysis=risk_analysis,
            defense_result=defense_result,
            optimization_result=optimization_result,
            defense_applied=defense_result.defense_applied,
            metadata=metadata,
        )

    def batch_process(
        self,
        queries: List[str],
        num_samples: int = 5,
    ) -> List[SPARTANResult]:
        """Process multiple queries through SPARTAN.

        Args:
            queries: List of input queries
            num_samples: Number of samples for self-consistency voting

        Returns:
            List of SPARTANResult for each query
        """
        return [self.process(query, num_samples=num_samples) for query in queries]

    def evaluate_defense(
        self,
        test_queries: List[str],
        ground_truth_labels: List[int],
    ) -> Dict[str, float]:
        """Evaluate defense effectiveness.

        Args:
            test_queries: Queries to evaluate on
            ground_truth_labels: 1 for member, 0 for non-member

        Returns:
            Dictionary of evaluation metrics
        """
        results = self.batch_process(test_queries)

        risk_scores = [r.risk_score for r in results]

        # Compute metrics
        from spartan.utils.metrics import compute_auc_roc, compute_tpr_at_fpr

        auc_roc = compute_auc_roc(ground_truth_labels, risk_scores)
        tpr_at_fpr_01 = compute_tpr_at_fpr(ground_truth_labels, risk_scores, fpr_threshold=0.01)

        # Accuracy retention (mock - would need actual labels)
        avg_risk_reduction = (
            np.mean([1 - r.risk_score for r in results if r.defense_applied])
            if any(r.defense_applied for r in results)
            else 0.0
        )

        return {
            "auc_roc": auc_roc,
            "tpr_at_fpr_0.01": tpr_at_fpr_01,
            "avg_risk_reduction": avg_risk_reduction,
            "defense_rate": sum(r.defense_applied for r in results) / len(results),
        }

    def get_config(self) -> SPARTANConfig:
        """Get current configuration."""
        return self.config

    def set_config(self, config: SPARTANConfig) -> None:
        """Update configuration.

        Args:
            config: New configuration to apply
        """
        self.config = config
        self.mplq = MPLQ(config=config)
        self.raas = RAAS(config=config)
        if self.rppo is not None:
            self.rppo = RPPO(config=config)

    def reset_history(self) -> None:
        """Clear optimization history."""
        self._history = []
        if self.rppo is not None:
            self.rppo.reset()

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Get processing history."""
        return self._history.copy()
