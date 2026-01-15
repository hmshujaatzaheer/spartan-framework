"""
Single-Model Voting Attack (SMVA)

Exploits self-consistency voting patterns to infer membership.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from spartan.attacks.base import BaseAttack, AttackResult


class SMVAAttack(BaseAttack):
    """Single-Model Voting Attack targeting self-consistency.
    
    Self-consistency voting creates observable vote distributions.
    Members tend to produce more concentrated (lower entropy)
    distributions due to memorization.
    
    Attack methodology:
    1. Sample multiple responses from the model
    2. Compute vote distribution entropy
    3. Low entropy indicates potential membership
    
    Example:
        >>> attack = SMVAAttack()
        >>> result = attack.execute(
        ...     target_model=model,
        ...     query="What is 2+2?",
        ...     num_samples=10,
        ... )
    """
    
    def __init__(
        self,
        threshold: float = 0.55,
        num_samples: int = 10,
        entropy_weight: float = 0.5,
    ):
        """Initialize SMVA attack.
        
        Args:
            threshold: Decision threshold
            num_samples: Number of samples for voting
            entropy_weight: Weight for entropy in signal
        """
        super().__init__(threshold=threshold)
        self.num_samples = num_samples
        self.entropy_weight = entropy_weight
    
    def execute(
        self,
        target_model: Any,
        query: str,
        vote_distribution: Optional[List[float]] = None,
        num_samples: Optional[int] = None,
        **kwargs: Any,
    ) -> AttackResult:
        """Execute SMVA attack.
        
        Args:
            target_model: Model to attack
            query: Query for generation
            vote_distribution: Pre-computed vote distribution
            num_samples: Override number of samples
            **kwargs: Additional arguments
            
        Returns:
            AttackResult with membership prediction
        """
        samples = num_samples or self.num_samples
        
        # Get vote distribution if not provided
        if vote_distribution is None and hasattr(target_model, 'generate'):
            output = target_model.generate(query, num_samples=samples)
            vote_distribution = getattr(output, 'vote_distribution', None)
        
        if vote_distribution is None:
            return AttackResult(
                success_score=0.5,
                membership_prediction=False,
                confidence=0.0,
                details={"error": "no_vote_distribution"},
            )
        
        # Compute attack signal
        signal, leakage_signals = self._compute_smva_signal(vote_distribution)
        
        # Make prediction
        prediction, confidence = self.predict_membership(signal)
        
        return AttackResult(
            success_score=signal,
            membership_prediction=prediction,
            confidence=confidence,
            leakage_signals=leakage_signals,
            details={
                "attack_type": "SMVA",
                "num_candidates": len(vote_distribution),
                "num_samples": samples,
                "threshold": self.threshold,
            },
        )
    
    def compute_attack_signal(
        self,
        vote_distribution: List[float],
        **kwargs: Any,
    ) -> float:
        """Compute SMVA attack signal.
        
        Args:
            vote_distribution: Vote distribution
            **kwargs: Additional arguments
            
        Returns:
            Attack signal value
        """
        signal, _ = self._compute_smva_signal(vote_distribution)
        return signal
    
    def _compute_smva_signal(
        self,
        vote_distribution: List[float],
    ) -> tuple:
        """Compute SMVA signal with details.
        
        Args:
            vote_distribution: Vote distribution
            
        Returns:
            Tuple of (signal, leakage_signals)
        """
        votes = np.array(vote_distribution)
        leakage_signals: Dict[str, Any] = {}
        
        # Normalize
        vote_sum = votes.sum()
        if vote_sum > 0:
            votes = votes / vote_sum
        else:
            votes = np.ones_like(votes) / len(votes)
        
        K = len(votes)
        
        # Signal 1: Entropy (lower = more memorized)
        entropy = self._compute_entropy(votes)
        max_entropy = np.log(K) if K > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        leakage_signals["entropy"] = float(entropy)
        leakage_signals["max_entropy"] = float(max_entropy)
        leakage_signals["normalized_entropy"] = float(normalized_entropy)
        
        # Signal 2: Concentration (max vote share)
        max_share = float(np.max(votes))
        leakage_signals["max_share"] = max_share
        
        # Signal 3: Top-2 share
        sorted_votes = np.sort(votes)[::-1]
        top2_share = float(sorted_votes[0] + sorted_votes[1]) if K >= 2 else max_share
        leakage_signals["top2_share"] = top2_share
        
        # Signal 4: Gini coefficient (inequality)
        gini = self._compute_gini(votes)
        leakage_signals["gini"] = gini
        
        # Combine signals
        # Low entropy + high concentration + high gini = memorization
        signal = (
            self.entropy_weight * (1 - normalized_entropy) +
            (1 - self.entropy_weight) * 0.5 * max_share +
            (1 - self.entropy_weight) * 0.5 * gini
        )
        
        signal = float(np.clip(signal, 0, 1))
        leakage_signals["combined_signal"] = signal
        
        return signal, leakage_signals
    
    def _compute_entropy(self, distribution: np.ndarray) -> float:
        """Compute Shannon entropy."""
        mask = distribution > 0
        if not np.any(mask):
            return 0.0
        entropy = -np.sum(distribution[mask] * np.log(distribution[mask]))
        return float(entropy)
    
    def _compute_gini(self, distribution: np.ndarray) -> float:
        """Compute Gini coefficient."""
        sorted_dist = np.sort(distribution)
        n = len(sorted_dist)
        if n == 0 or sorted_dist.sum() == 0:
            return 0.0
        
        cumsum = np.cumsum(sorted_dist)
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_dist)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])
        
        return float(np.clip(gini, 0, 1))
