"""
Natural Language Blindness Attack (NLBA)

Exploits PRM's natural language blindness to infer membership.
Based on Ma et al.'s finding that PRMs ignore explanatory text.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from spartan.attacks.base import AttackResult, BaseAttack


class NLBAAttack(BaseAttack):
    """Natural Language Blindness Attack targeting PRMs.

    PRMs exhibit "natural language blindness" - ignoring explanatory
    text while focusing on mathematical expressions. This attack
    exploits this vulnerability by:

    1. Generating queries with varying NL content
    2. Observing PRM score patterns
    3. Detecting memorization through score anomalies

    Higher scores on high-NL content indicate memorization
    (the model "knows" the answer despite uninformative context).

    Example:
        >>> attack = NLBAAttack()
        >>> result = attack.execute(
        ...     target_model=model,
        ...     query="Solve: x + 2 = 5",
        ...     prm_scores=[0.95, 0.92, 0.98],
        ... )
    """

    def __init__(
        self,
        threshold: float = 0.6,
        nl_ratio_threshold: float = 0.7,
        score_variance_weight: float = 0.3,
    ):
        """Initialize NLBA attack.

        Args:
            threshold: Decision threshold for membership
            nl_ratio_threshold: NL content ratio threshold
            score_variance_weight: Weight for score variance in signal
        """
        super().__init__(threshold=threshold)
        self.nl_ratio_threshold = nl_ratio_threshold
        self.score_variance_weight = score_variance_weight

    def execute(
        self,
        target_model: Any,
        query: str,
        prm_scores: Optional[List[float]] = None,
        reasoning_steps: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AttackResult:
        """Execute NLBA attack.

        Args:
            target_model: Model to attack
            query: Query used for generation
            prm_scores: PRM scores for reasoning steps
            reasoning_steps: Text of reasoning steps
            **kwargs: Additional arguments

        Returns:
            AttackResult with membership prediction
        """
        # Get PRM scores if not provided
        if prm_scores is None and hasattr(target_model, "generate"):
            output = target_model.generate(query)
            prm_scores = getattr(output, "prm_scores", None)
            reasoning_steps = getattr(output, "reasoning_steps", None)

        if prm_scores is None:
            # Cannot execute without PRM scores
            return AttackResult(
                success_score=0.5,
                membership_prediction=False,
                confidence=0.0,
                details={"error": "no_prm_scores"},
            )

        # Compute attack signal
        signal, leakage_signals = self._compute_nlba_signal(
            prm_scores=prm_scores,
            reasoning_steps=reasoning_steps,
        )

        # Make prediction
        prediction, confidence = self.predict_membership(signal)

        return AttackResult(
            success_score=signal,
            membership_prediction=prediction,
            confidence=confidence,
            leakage_signals=leakage_signals,
            details={
                "attack_type": "NLBA",
                "num_steps": len(prm_scores),
                "threshold": self.threshold,
            },
        )

    def compute_attack_signal(
        self,
        prm_scores: List[float],
        reasoning_steps: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> float:
        """Compute NLBA attack signal.

        Args:
            prm_scores: PRM scores
            reasoning_steps: Reasoning step text
            **kwargs: Additional arguments

        Returns:
            Attack signal value
        """
        signal, _ = self._compute_nlba_signal(prm_scores, reasoning_steps)
        return signal

    def _compute_nlba_signal(
        self,
        prm_scores: List[float],
        reasoning_steps: Optional[List[str]] = None,
    ) -> tuple:
        """Compute NLBA signal with leakage details.

        Args:
            prm_scores: PRM scores for steps
            reasoning_steps: Step text

        Returns:
            Tuple of (signal, leakage_signals)
        """
        scores = np.array(prm_scores)
        leakage_signals: Dict[str, Any] = {}

        # Signal 1: Score mean (high = potential memorization)
        score_mean = float(np.mean(scores))
        leakage_signals["score_mean"] = score_mean

        # Signal 2: Score variance (low = consistent memorization)
        score_var = float(np.var(scores))
        leakage_signals["score_variance"] = score_var

        # Signal 3: NL-blindness correlation
        nl_correlation = 0.0
        if reasoning_steps is not None and len(reasoning_steps) == len(scores):
            nl_ratios = self._compute_nl_ratios(reasoning_steps)
            leakage_signals["nl_ratios"] = nl_ratios

            if len(nl_ratios) > 1:
                corr = np.corrcoef(nl_ratios, scores)[0, 1]
                if not np.isnan(corr):
                    nl_correlation = float(corr)

            leakage_signals["nl_correlation"] = nl_correlation

        # Signal 4: High score on high-NL steps
        high_nl_high_score = 0.0
        if reasoning_steps is not None:
            nl_ratios = self._compute_nl_ratios(reasoning_steps)
            high_nl_mask = np.array(nl_ratios) > self.nl_ratio_threshold

            if np.any(high_nl_mask):
                high_nl_scores = scores[high_nl_mask]
                high_nl_high_score = float(np.mean(high_nl_scores > 0.8))

            leakage_signals["high_nl_high_score_ratio"] = high_nl_high_score

        # Combine signals into attack score
        # High mean + low variance + positive NL correlation = memorization
        signal = (
            0.4 * score_mean
            + 0.2 * (1 - min(score_var * 10, 1))  # Low variance
            + 0.2 * (0.5 + nl_correlation / 2)  # Correlation [-1,1] -> [0,1]
            + 0.2 * high_nl_high_score
        )

        signal = float(np.clip(signal, 0, 1))
        leakage_signals["combined_signal"] = signal

        return signal, leakage_signals

    def _compute_nl_ratios(self, steps: List[str]) -> List[float]:
        """Compute NL content ratio for each step.

        Args:
            steps: Reasoning step text

        Returns:
            List of NL ratios
        """
        ratios = []
        for step in steps:
            if len(step) == 0:
                ratios.append(0.0)
                continue

            # Count alphabetic characters (NL indicator)
            alpha_count = sum(c.isalpha() or c.isspace() for c in step)
            ratio = alpha_count / len(step)
            ratios.append(ratio)

        return ratios


# Last updated: 2026-01-15
