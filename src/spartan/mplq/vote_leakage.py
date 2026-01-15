"""
Vote Leakage Analyzer

Detects privacy leakage through self-consistency voting distributions.
Implements equation (2) from the SPARTAN paper:

L_vote(x) = 1 - H(v) / log(K) = 1 + Σ v_k * log(v_k) / log(K)
"""

from typing import Any, Dict, List, Optional

import numpy as np


class VoteLeakageAnalyzer:
    """Analyzer for voting distribution-based privacy leakage.

    Self-consistency voting creates observable vote distributions that may
    reveal model internals and training data characteristics. Low entropy
    (concentrated) distributions indicate potential memorization.

    Example:
        >>> analyzer = VoteLeakageAnalyzer()
        >>> result = analyzer.analyze(vote_distribution=[0.9, 0.05, 0.05])
        >>> print(f"Leakage: {result['leakage_score']:.4f}")  # High leakage
    """

    def __init__(
        self,
        min_entropy_threshold: float = 0.1,
        smoothing: float = 1e-10,
    ):
        """Initialize vote leakage analyzer.

        Args:
            min_entropy_threshold: Minimum entropy threshold for detection
            smoothing: Smoothing factor for entropy computation
        """
        self.min_entropy_threshold = min_entropy_threshold
        self.smoothing = smoothing

    def analyze(
        self,
        vote_distribution: List[float],
        raw_votes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Analyze vote distribution for privacy leakage.

        Args:
            vote_distribution: Normalized vote distribution
            raw_votes: Optional raw vote strings for additional analysis

        Returns:
            Dictionary containing leakage score and analysis details
        """
        if len(vote_distribution) == 0:
            return {
                "leakage_score": 0.0,
                "entropy": 0.0,
                "normalized_entropy": 0.0,
                "num_candidates": 0,
            }

        votes = np.array(vote_distribution)

        # Normalize if not already
        vote_sum = votes.sum()
        if vote_sum > 0:
            votes = votes / vote_sum
        else:
            votes = np.ones_like(votes) / len(votes)

        # Add smoothing
        votes = votes + self.smoothing
        votes = votes / votes.sum()

        K = len(votes)

        # Compute entropy: H(v) = -Σ v_k * log(v_k)
        entropy = self._compute_entropy(votes)

        # Maximum entropy for K candidates
        max_entropy = np.log(K) if K > 1 else 1.0

        # Normalized entropy in [0, 1]
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        # Leakage score: L_vote = 1 - H(v) / log(K)
        # High concentration (low entropy) → high leakage
        leakage_score = 1.0 - normalized_entropy

        # Clamp to [0, 1]
        leakage_score = float(np.clip(leakage_score, 0, 1))

        result = {
            "leakage_score": leakage_score,
            "entropy": float(entropy),
            "max_entropy": float(max_entropy),
            "normalized_entropy": float(normalized_entropy),
            "num_candidates": K,
            "vote_distribution": votes.tolist(),
            "max_vote_share": float(np.max(votes)),
            "concentration_ratio": self._compute_concentration_ratio(votes),
        }

        # Additional pattern analysis
        patterns = self._analyze_vote_patterns(votes)
        result["patterns"] = patterns

        # Raw vote analysis if available
        if raw_votes is not None:
            raw_analysis = self._analyze_raw_votes(raw_votes)
            result["raw_vote_analysis"] = raw_analysis

        return result

    def _compute_entropy(self, distribution: np.ndarray) -> float:
        """Compute Shannon entropy of distribution.

        H(v) = -Σ v_k * log(v_k)

        Args:
            distribution: Probability distribution

        Returns:
            Entropy value
        """
        # Mask zero values to avoid log(0)
        mask = distribution > 0
        entropy = -np.sum(distribution[mask] * np.log(distribution[mask]))
        return float(entropy)

    def _compute_concentration_ratio(self, distribution: np.ndarray) -> float:
        """Compute vote concentration ratio.

        Ratio of top vote share to second highest.
        High ratio indicates strong consensus (potential memorization).

        Args:
            distribution: Vote distribution

        Returns:
            Concentration ratio
        """
        if len(distribution) < 2:
            return 1.0

        sorted_votes = np.sort(distribution)[::-1]
        top = sorted_votes[0]
        second = sorted_votes[1]

        if second > 0:
            ratio = top / second
        else:
            ratio = float("inf")

        # Cap at reasonable value
        return float(min(ratio, 100.0))

    def _analyze_vote_patterns(self, distribution: np.ndarray) -> Dict[str, Any]:
        """Analyze voting patterns for suspicious behavior.

        Args:
            distribution: Vote distribution

        Returns:
            Pattern analysis results
        """
        K = len(distribution)

        # Pattern 1: Single dominant answer
        max_share = np.max(distribution)
        is_dominant = max_share > 0.8

        # Pattern 2: Uniform distribution
        expected_uniform = 1.0 / K
        uniformity = 1.0 - np.std(distribution) / expected_uniform if expected_uniform > 0 else 0.0
        is_uniform = uniformity > 0.9

        # Pattern 3: Two-way split
        sorted_dist = np.sort(distribution)[::-1]
        if K >= 2:
            two_way_share = sorted_dist[0] + sorted_dist[1]
            is_two_way = two_way_share > 0.9 and sorted_dist[1] > 0.1
        else:
            is_two_way = False

        # Pattern 4: Long tail (many small votes)
        if K >= 3:
            tail_share = 1.0 - sorted_dist[0] - sorted_dist[1]
            has_long_tail = tail_share > 0.1 and sorted_dist[-1] < 0.05
        else:
            has_long_tail = False

        return {
            "is_dominant": is_dominant,
            "is_uniform": is_uniform,
            "is_two_way": is_two_way,
            "has_long_tail": has_long_tail,
            "uniformity_score": float(uniformity),
            "gini_coefficient": self._compute_gini(distribution),
        }

    def _compute_gini(self, distribution: np.ndarray) -> float:
        """Compute Gini coefficient of vote distribution.

        Measures inequality: 0 = perfect equality, 1 = maximum inequality

        Args:
            distribution: Vote distribution

        Returns:
            Gini coefficient
        """
        sorted_dist = np.sort(distribution)
        n = len(sorted_dist)
        if n == 0:
            return 0.0

        # Gini formula
        cumsum = np.cumsum(sorted_dist)
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_dist)) - (n + 1) * cumsum[-1]) / (
            n * cumsum[-1]
        )

        return float(np.clip(gini, 0, 1))

    def _analyze_raw_votes(self, raw_votes: List[str]) -> Dict[str, Any]:
        """Analyze raw vote strings for patterns.

        Args:
            raw_votes: List of vote response strings

        Returns:
            Raw vote analysis
        """
        if len(raw_votes) == 0:
            return {"num_votes": 0}

        # Count unique answers
        unique_answers = len(set(raw_votes))

        # Average answer length
        avg_length = np.mean([len(v) for v in raw_votes])

        # Length variance
        length_variance = np.var([len(v) for v in raw_votes])

        # Check for exact duplicates
        vote_counts: Dict[str, int] = {}
        for v in raw_votes:
            vote_counts[v] = vote_counts.get(v, 0) + 1

        most_common_count = max(vote_counts.values())
        exact_duplicate_ratio = most_common_count / len(raw_votes)

        return {
            "num_votes": len(raw_votes),
            "unique_answers": unique_answers,
            "diversity_ratio": unique_answers / len(raw_votes),
            "avg_length": float(avg_length),
            "length_variance": float(length_variance),
            "exact_duplicate_ratio": exact_duplicate_ratio,
        }


# Last updated: 2026-01-15
