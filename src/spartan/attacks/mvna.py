"""
MCTS Value Network Attack (MVNA)

Exploits MCTS value network patterns to infer membership.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from spartan.attacks.base import BaseAttack, AttackResult


class MVNAAttack(BaseAttack):
    """MCTS Value Network Attack.
    
    MCTS value networks may encode memorized reasoning trajectories.
    This attack detects membership by:
    
    1. Analyzing value network outputs during search
    2. Comparing against expected baseline distributions
    3. High-confidence values indicate memorization
    
    Example:
        >>> attack = MVNAAttack()
        >>> result = attack.execute(
        ...     target_model=model,
        ...     query="Solve the equation",
        ...     mcts_values=[0.9, 0.85, 0.95],
        ... )
    """
    
    def __init__(
        self,
        threshold: float = 0.55,
        baseline_mean: float = 0.5,
        baseline_std: float = 0.15,
    ):
        """Initialize MVNA attack.
        
        Args:
            threshold: Decision threshold
            baseline_mean: Expected mean for non-members
            baseline_std: Expected std for non-members
        """
        super().__init__(threshold=threshold)
        self.baseline_mean = baseline_mean
        self.baseline_std = baseline_std
    
    def execute(
        self,
        target_model: Any,
        query: str,
        mcts_values: Optional[List[float]] = None,
        mcts_tree: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AttackResult:
        """Execute MVNA attack.
        
        Args:
            target_model: Model to attack
            query: Query for generation
            mcts_values: Value network outputs
            mcts_tree: Full MCTS tree structure
            **kwargs: Additional arguments
            
        Returns:
            AttackResult with membership prediction
        """
        # Get MCTS values if not provided
        if mcts_values is None and hasattr(target_model, 'generate'):
            output = target_model.generate(query)
            mcts_values = getattr(output, 'mcts_values', None)
            mcts_tree = getattr(output, 'mcts_tree', None)
        
        if mcts_values is None:
            return AttackResult(
                success_score=0.5,
                membership_prediction=False,
                confidence=0.0,
                details={"error": "no_mcts_values"},
            )
        
        # Compute attack signal
        signal, leakage_signals = self._compute_mvna_signal(
            mcts_values=mcts_values,
            mcts_tree=mcts_tree,
        )
        
        # Make prediction
        prediction, confidence = self.predict_membership(signal)
        
        return AttackResult(
            success_score=signal,
            membership_prediction=prediction,
            confidence=confidence,
            leakage_signals=leakage_signals,
            details={
                "attack_type": "MVNA",
                "num_nodes": len(mcts_values),
                "threshold": self.threshold,
            },
        )
    
    def compute_attack_signal(
        self,
        mcts_values: List[float],
        mcts_tree: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> float:
        """Compute MVNA attack signal.
        
        Args:
            mcts_values: Value network outputs
            mcts_tree: Full tree structure
            **kwargs: Additional arguments
            
        Returns:
            Attack signal value
        """
        signal, _ = self._compute_mvna_signal(mcts_values, mcts_tree)
        return signal
    
    def _compute_mvna_signal(
        self,
        mcts_values: List[float],
        mcts_tree: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """Compute MVNA signal with details.
        
        Args:
            mcts_values: Value scores
            mcts_tree: Tree structure
            
        Returns:
            Tuple of (signal, leakage_signals)
        """
        values = np.array(mcts_values)
        leakage_signals: Dict[str, Any] = {}
        
        # Signal 1: Mean deviation from baseline
        mean_value = float(np.mean(values))
        mean_deviation = abs(mean_value - self.baseline_mean)
        leakage_signals["mean_value"] = mean_value
        leakage_signals["mean_deviation"] = mean_deviation
        
        # Signal 2: Maximum value (high = memorized)
        max_value = float(np.max(values))
        leakage_signals["max_value"] = max_value
        
        # Signal 3: Std comparison
        value_std = float(np.std(values))
        std_ratio = value_std / self.baseline_std if self.baseline_std > 0 else 1.0
        leakage_signals["value_std"] = value_std
        leakage_signals["std_ratio"] = std_ratio
        
        # Signal 4: High-value node ratio
        high_value_ratio = float(np.mean(values > 0.8))
        leakage_signals["high_value_ratio"] = high_value_ratio
        
        # Signal 5: Z-score of max value
        z_score = (max_value - self.baseline_mean) / self.baseline_std if self.baseline_std > 0 else 0
        z_score = float(np.clip(z_score, -5, 5))
        leakage_signals["max_z_score"] = z_score
        
        # Signal 6: Depth-value correlation (if tree available)
        depth_correlation = 0.0
        if mcts_tree is not None and "depths" in mcts_tree:
            depths = np.array(mcts_tree["depths"])
            if len(depths) == len(values) and len(depths) > 1:
                corr = np.corrcoef(depths, values)[0, 1]
                if not np.isnan(corr):
                    depth_correlation = float(corr)
        leakage_signals["depth_correlation"] = depth_correlation
        
        # Combine signals into attack score
        # High mean + high max + abnormal std + high ratio = memorization
        signal = (
            0.25 * min(mean_deviation * 2, 1) +
            0.25 * max_value +
            0.2 * high_value_ratio +
            0.15 * (1 - min(abs(std_ratio - 1), 1)) +  # Close to baseline std
            0.15 * (0.5 + z_score / 10)  # Z-score contribution
        )
        
        signal = float(np.clip(signal, 0, 1))
        leakage_signals["combined_signal"] = signal
        
        return signal, leakage_signals
