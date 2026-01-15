"""
MCTS Leakage Analyzer

Detects privacy leakage through MCTS value network outputs.
Implements equation (3) from the SPARTAN paper:

L_mcts(x) = max_{n ∈ T} |V_θ(n) - V_baseline(n)|
"""

from typing import Any, Dict, List, Optional

import numpy as np


class MCTSLeakageAnalyzer:
    """Analyzer for MCTS value network-based privacy leakage.

    MCTS tree search mechanisms employ learned value functions whose
    outputs may encode memorized reasoning trajectories from training.
    Anomalous value network outputs compared to baseline indicate
    memorized trajectories.

    Example:
        >>> analyzer = MCTSLeakageAnalyzer()
        >>> result = analyzer.analyze(mcts_values=[0.9, 0.95, 0.85, 0.92])
        >>> print(f"Leakage: {result['leakage_score']:.4f}")
    """

    def __init__(
        self,
        baseline_mean: float = 0.5,
        baseline_std: float = 0.15,
        deviation_threshold: float = 0.3,
    ):
        """Initialize MCTS leakage analyzer.

        Args:
            baseline_mean: Expected mean value for non-memorized data
            baseline_std: Expected standard deviation for baseline
            deviation_threshold: Threshold for significant deviation
        """
        self.baseline_mean = baseline_mean
        self.baseline_std = baseline_std
        self.deviation_threshold = deviation_threshold

        # Baseline distribution parameters (can be learned)
        self._baseline_params = {
            "mean": baseline_mean,
            "std": baseline_std,
        }

    def analyze(
        self,
        mcts_values: List[float],
        mcts_tree: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Analyze MCTS values for privacy leakage.

        Args:
            mcts_values: Value scores from MCTS nodes
            mcts_tree: Optional full tree structure for depth analysis

        Returns:
            Dictionary containing leakage score and analysis details
        """
        if len(mcts_values) == 0:
            return {
                "leakage_score": 0.0,
                "max_deviation": 0.0,
                "num_nodes": 0,
            }

        values = np.array(mcts_values)

        # Compute baseline values (expected for non-member data)
        baseline_values = self._compute_baseline_values(len(values), mcts_tree)

        # Compute deviations: |V_θ(n) - V_baseline(n)|
        deviations = np.abs(values - baseline_values)

        # Maximum deviation as leakage score
        max_deviation = float(np.max(deviations))

        # Normalize to [0, 1] using scaled sigmoid
        leakage_score = self._normalize_deviation(max_deviation)

        result = {
            "leakage_score": leakage_score,
            "max_deviation": max_deviation,
            "mean_deviation": float(np.mean(deviations)),
            "num_nodes": len(values),
            "value_mean": float(np.mean(values)),
            "value_std": float(np.std(values)),
            "baseline_mean": float(np.mean(baseline_values)),
            "baseline_std": float(np.std(baseline_values)),
        }

        # Detailed node analysis
        node_analysis = self._analyze_nodes(values, deviations, mcts_tree)
        result["node_analysis"] = node_analysis

        # Trajectory analysis
        if mcts_tree is not None:
            trajectory_analysis = self._analyze_trajectories(mcts_tree, values)
            result["trajectory_analysis"] = trajectory_analysis

        return result

    def _compute_baseline_values(
        self,
        num_nodes: int,
        mcts_tree: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Compute expected baseline values for nodes.

        For nodes at different depths, baseline expectations may differ.
        Without tree structure, use uniform baseline.

        Args:
            num_nodes: Number of nodes
            mcts_tree: Optional tree structure

        Returns:
            Array of baseline values
        """
        if mcts_tree is not None and "depths" in mcts_tree:
            # Depth-adjusted baseline
            depths = np.array(mcts_tree["depths"])
            # Values typically decrease with depth (more uncertainty)
            baseline = self.baseline_mean - 0.05 * depths
            return np.clip(baseline, 0, 1)
        else:
            # Uniform baseline
            return np.full(num_nodes, self.baseline_mean)

    def _normalize_deviation(self, deviation: float) -> float:
        """Normalize deviation to [0, 1] range.

        Uses scaled sigmoid centered at threshold.

        Args:
            deviation: Raw deviation value

        Returns:
            Normalized score in [0, 1]
        """
        # Scale factor determines sensitivity
        scale = self.deviation_threshold

        # Sigmoid centered at threshold
        normalized = 1.0 / (1.0 + np.exp(-(deviation - scale) / (scale / 3)))

        return float(np.clip(normalized, 0, 1))

    def _analyze_nodes(
        self,
        values: np.ndarray,
        deviations: np.ndarray,
        mcts_tree: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Analyze individual node characteristics.

        Args:
            values: Node values
            deviations: Deviation from baseline
            mcts_tree: Optional tree structure

        Returns:
            Node-level analysis
        """
        # Find high-deviation nodes
        high_dev_mask = deviations > self.deviation_threshold
        num_high_deviation = int(np.sum(high_dev_mask))

        # Value distribution analysis
        high_value_mask = values > 0.8
        num_high_value = int(np.sum(high_value_mask))

        # Extremity analysis
        extreme_high = int(np.sum(values > 0.95))
        extreme_low = int(np.sum(values < 0.05))

        result = {
            "num_high_deviation": num_high_deviation,
            "high_deviation_ratio": num_high_deviation / len(values) if len(values) > 0 else 0,
            "num_high_value": num_high_value,
            "high_value_ratio": num_high_value / len(values) if len(values) > 0 else 0,
            "extreme_high_count": extreme_high,
            "extreme_low_count": extreme_low,
            "value_range": float(np.max(values) - np.min(values)),
        }

        # Depth-wise analysis if tree available
        if mcts_tree is not None and "depths" in mcts_tree:
            depths = np.array(mcts_tree["depths"])
            unique_depths = np.unique(depths)

            depth_stats = {}
            for d in unique_depths:
                mask = depths == d
                depth_stats[int(d)] = {
                    "mean_value": float(np.mean(values[mask])),
                    "mean_deviation": float(np.mean(deviations[mask])),
                    "count": int(np.sum(mask)),
                }
            result["depth_stats"] = depth_stats

        return result

    def _analyze_trajectories(
        self,
        mcts_tree: Dict[str, Any],
        values: np.ndarray,
    ) -> Dict[str, Any]:
        """Analyze trajectory patterns in MCTS tree.

        Args:
            mcts_tree: Tree structure
            values: Node values

        Returns:
            Trajectory analysis results
        """
        result: Dict[str, Any] = {"available": True}

        # Check for trajectory information
        if "trajectories" not in mcts_tree:
            return {"available": False}

        trajectories = mcts_tree["trajectories"]

        # Trajectory value statistics
        trajectory_values = []
        for traj in trajectories:
            if isinstance(traj, list) and len(traj) > 0:
                # Get values along trajectory
                traj_indices = [
                    t.get("node_idx", i) if isinstance(t, dict) else i for i, t in enumerate(traj)
                ]
                valid_indices = [idx for idx in traj_indices if idx < len(values)]
                if valid_indices:
                    traj_vals = values[valid_indices]
                    trajectory_values.append(
                        {
                            "mean": float(np.mean(traj_vals)),
                            "max": float(np.max(traj_vals)),
                            "min": float(np.min(traj_vals)),
                            "length": len(valid_indices),
                        }
                    )

        if trajectory_values:
            result["num_trajectories"] = len(trajectory_values)
            result["mean_trajectory_value"] = float(
                np.mean([tv["mean"] for tv in trajectory_values])
            )
            result["max_trajectory_value"] = float(max(tv["max"] for tv in trajectory_values))
            result["avg_trajectory_length"] = float(
                np.mean([tv["length"] for tv in trajectory_values])
            )

        return result

    def set_baseline_params(
        self,
        mean: float,
        std: float,
    ) -> None:
        """Update baseline distribution parameters.

        Args:
            mean: New baseline mean
            std: New baseline std
        """
        self.baseline_mean = mean
        self.baseline_std = std
        self._baseline_params = {"mean": mean, "std": std}

    def get_baseline_params(self) -> Dict[str, float]:
        """Get current baseline parameters."""
        return self._baseline_params.copy()


# Last updated: 2026-01-15
