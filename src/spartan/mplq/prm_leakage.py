"""
PRM Leakage Analyzer

Detects privacy leakage through Process Reward Model (PRM) score distributions.
Implements equation (1) from the SPARTAN paper:

L_prm(x) = D_KL(P_M(r|x) || P_ref(r)) + λ · Var({r_i})
"""

from typing import Any, Dict, List, Optional

import numpy as np


class PRMLeakageAnalyzer:
    """Analyzer for PRM-based privacy leakage.

    PRMs provide step-level verification scores during reasoning. These scores
    may leak information about training data distribution through confidence
    patterns on memorized vs. novel reasoning steps.

    Key insight from Ma et al.: PRMs exhibit "natural language blindness",
    ignoring explanatory text while focusing on mathematical expressions.
    This vulnerability is exploitable for both attack and defense.

    Example:
        >>> analyzer = PRMLeakageAnalyzer()
        >>> result = analyzer.analyze(
        ...     prm_scores=[0.95, 0.92, 0.98],
        ...     reasoning_steps=["Step 1...", "Step 2...", "Step 3..."]
        ... )
        >>> print(f"Leakage: {result['leakage_score']:.4f}")
    """

    def __init__(
        self,
        reference_distribution: Optional[np.ndarray] = None,
        kl_smoothing: float = 1e-10,
        variance_weight: float = 0.1,
        num_bins: int = 50,
    ):
        """Initialize PRM leakage analyzer.

        Args:
            reference_distribution: Pre-computed reference distribution
            kl_smoothing: Smoothing factor for KL divergence
            variance_weight: Weight for variance term (λ in paper)
            num_bins: Number of bins for histogram computation
        """
        self.kl_smoothing = kl_smoothing
        self.variance_weight = variance_weight
        self.num_bins = num_bins

        # Initialize reference distribution
        if reference_distribution is not None:
            self._reference_distribution = reference_distribution
        else:
            # Default: uniform distribution (no prior information)
            self._reference_distribution = np.ones(num_bins) / num_bins

        # Bin edges for histogram
        self._bin_edges = np.linspace(0, 1, num_bins + 1)

    def analyze(
        self,
        prm_scores: List[float],
        reasoning_steps: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Analyze PRM scores for privacy leakage.

        Args:
            prm_scores: List of PRM scores for reasoning steps
            reasoning_steps: Optional reasoning step text for NL blindness analysis

        Returns:
            Dictionary containing leakage score and analysis details
        """
        if len(prm_scores) == 0:
            return {
                "leakage_score": 0.0,
                "kl_divergence": 0.0,
                "variance": 0.0,
                "num_steps": 0,
            }

        scores = np.array(prm_scores)

        # Compute observed distribution
        observed_dist = self._compute_distribution(scores)

        # Compute KL divergence from reference
        kl_div = self._kl_divergence(observed_dist, self._reference_distribution)

        # Compute variance of scores
        variance = float(np.var(scores))

        # Combine into leakage score: L_prm = KL + λ * Var
        leakage_score = kl_div + self.variance_weight * variance

        # Normalize to [0, 1] using sigmoid-like transform
        leakage_score = self._normalize_score(leakage_score)

        # Additional analysis
        result = {
            "leakage_score": leakage_score,
            "kl_divergence": kl_div,
            "variance": variance,
            "num_steps": len(scores),
            "mean_score": float(np.mean(scores)),
            "max_score": float(np.max(scores)),
            "min_score": float(np.min(scores)),
        }

        # NL blindness detection
        if reasoning_steps is not None:
            nl_analysis = self._analyze_nl_blindness(
                prm_scores=scores,
                reasoning_steps=reasoning_steps,
            )
            result["nl_blindness"] = nl_analysis

        return result

    def _compute_distribution(self, scores: np.ndarray) -> np.ndarray:
        """Compute histogram distribution of scores.

        Args:
            scores: Array of PRM scores

        Returns:
            Normalized distribution over bins
        """
        # Clip scores to [0, 1]
        scores = np.clip(scores, 0, 1)

        # Compute histogram
        hist, _ = np.histogram(scores, bins=self._bin_edges)

        # Normalize to probability distribution
        total = hist.sum()
        if total > 0:
            dist = hist / total
        else:
            dist = np.ones(self.num_bins) / self.num_bins

        # Add smoothing to avoid zeros
        dist = dist + self.kl_smoothing
        dist = dist / dist.sum()

        return dist

    def _kl_divergence(
        self,
        p: np.ndarray,
        q: np.ndarray,
    ) -> float:
        """Compute KL divergence D_KL(P || Q).

        Args:
            p: Observed distribution
            q: Reference distribution

        Returns:
            KL divergence value
        """
        # Add smoothing to reference
        q = q + self.kl_smoothing
        q = q / q.sum()

        # Compute KL divergence
        # D_KL(P || Q) = Σ p(x) * log(p(x) / q(x))
        kl = np.sum(p * np.log(p / q))

        return float(kl)

    def _normalize_score(self, score: float) -> float:
        """Normalize score to [0, 1] range.

        Uses a scaled tanh function for soft normalization.

        Args:
            score: Raw leakage score

        Returns:
            Normalized score in [0, 1]
        """
        # Scale factor based on typical KL divergence ranges
        scale = 2.0
        normalized = (np.tanh(score / scale) + 1) / 2
        return float(normalized)

    def _analyze_nl_blindness(
        self,
        prm_scores: np.ndarray,
        reasoning_steps: List[str],
    ) -> Dict[str, Any]:
        """Analyze NL blindness vulnerability.

        Ma et al. found that PRMs largely ignore natural language explanatory
        text while focusing on mathematical expressions. This creates
        exploitable patterns.

        Args:
            prm_scores: PRM scores for steps
            reasoning_steps: Text of reasoning steps

        Returns:
            NL blindness analysis results
        """
        if len(prm_scores) != len(reasoning_steps):
            return {"error": "Mismatched lengths"}

        # Estimate NL content ratio for each step
        nl_ratios = []
        for step in reasoning_steps:
            # Simple heuristic: ratio of alphabetic characters
            alpha_count = sum(c.isalpha() for c in step)
            total_count = len(step) if len(step) > 0 else 1
            nl_ratios.append(alpha_count / total_count)

        nl_ratios = np.array(nl_ratios)

        # Correlate NL ratio with PRM scores
        if len(nl_ratios) > 1:
            correlation = float(np.corrcoef(nl_ratios, prm_scores)[0, 1])
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0

        # High negative correlation suggests NL blindness
        # (more NL content → lower scores, ignoring explanations)
        nl_blindness_indicator = max(0, -correlation)

        return {
            "nl_ratios": nl_ratios.tolist(),
            "correlation": correlation,
            "nl_blindness_indicator": nl_blindness_indicator,
            "mean_nl_ratio": float(np.mean(nl_ratios)),
        }

    def set_reference_distribution(
        self,
        distribution: np.ndarray,
    ) -> None:
        """Set reference distribution from non-member data.

        Args:
            distribution: Reference distribution array
        """
        if len(distribution) != self.num_bins:
            # Resample to match bins
            distribution = np.interp(
                np.linspace(0, 1, self.num_bins),
                np.linspace(0, 1, len(distribution)),
                distribution,
            )

        # Normalize
        distribution = distribution + self.kl_smoothing
        self._reference_distribution = distribution / distribution.sum()

    def get_reference_distribution(self) -> np.ndarray:
        """Get current reference distribution."""
        return self._reference_distribution.copy()
