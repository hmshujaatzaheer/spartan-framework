"""
Base Attack Class

Abstract base class for all SPARTAN attack implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class AttackResult:
    """Result from executing an attack.
    
    Attributes:
        success_score: Attack success probability (0-1)
        membership_prediction: Predicted membership (True = member)
        confidence: Confidence in prediction
        leakage_signals: Dictionary of extracted leakage signals
        details: Additional attack-specific details
    """
    success_score: float
    membership_prediction: bool
    confidence: float
    leakage_signals: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success_score": self.success_score,
            "membership_prediction": self.membership_prediction,
            "confidence": self.confidence,
            "leakage_signals": self.leakage_signals,
            "details": self.details,
        }


class BaseAttack(ABC):
    """Abstract base class for privacy attacks.
    
    All attack implementations should inherit from this class
    and implement the required methods.
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        num_shadow_models: int = 0,
    ):
        """Initialize attack.
        
        Args:
            threshold: Decision threshold for membership
            num_shadow_models: Number of shadow models (if applicable)
        """
        self.threshold = threshold
        self.num_shadow_models = num_shadow_models
        self._attack_name = self.__class__.__name__
    
    @abstractmethod
    def execute(
        self,
        target_model: Any,
        query: str,
        **kwargs: Any,
    ) -> AttackResult:
        """Execute the attack.
        
        Args:
            target_model: Model to attack
            query: Query to use for attack
            **kwargs: Additional attack-specific arguments
            
        Returns:
            AttackResult containing attack outcome
        """
        pass
    
    @abstractmethod
    def compute_attack_signal(
        self,
        **kwargs: Any,
    ) -> float:
        """Compute the attack signal/score.
        
        Args:
            **kwargs: Signal-specific inputs
            
        Returns:
            Attack signal value
        """
        pass
    
    def predict_membership(
        self,
        signal: float,
    ) -> tuple:
        """Predict membership based on attack signal.
        
        Args:
            signal: Computed attack signal
            
        Returns:
            Tuple of (prediction, confidence)
        """
        prediction = signal > self.threshold
        confidence = abs(signal - self.threshold) / max(self.threshold, 1 - self.threshold)
        confidence = float(np.clip(confidence, 0, 1))
        
        return prediction, confidence
    
    def evaluate(
        self,
        results: List[AttackResult],
        ground_truth: List[bool],
    ) -> Dict[str, float]:
        """Evaluate attack performance.
        
        Args:
            results: List of attack results
            ground_truth: True membership labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if len(results) != len(ground_truth):
            raise ValueError("Results and ground truth length mismatch")
        
        predictions = [r.membership_prediction for r in results]
        scores = [r.success_score for r in results]
        
        # Compute metrics
        from spartan.utils.metrics import (
            compute_auc_roc,
            compute_accuracy,
            compute_tpr_at_fpr,
        )
        
        auc = compute_auc_roc(
            [int(g) for g in ground_truth],
            scores,
        )
        accuracy = compute_accuracy(
            [int(g) for g in ground_truth],
            [int(p) for p in predictions],
        )
        tpr_at_fpr = compute_tpr_at_fpr(
            [int(g) for g in ground_truth],
            scores,
            fpr_threshold=0.01,
        )
        
        return {
            "auc_roc": auc,
            "accuracy": accuracy,
            "tpr_at_fpr_0.01": tpr_at_fpr,
            "num_samples": len(results),
        }
    
    @property
    def name(self) -> str:
        """Get attack name."""
        return self._attack_name
