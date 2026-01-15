"""
MCTS Defense Module

Implements value network perturbation for MCTS-based privacy protection.
"""

from typing import Any, Dict, List, Optional

import numpy as np


class MCTSDefense:
    """MCTS value network perturbation defense.

    Adds depth-scaled noise to value network outputs in MCTS trees,
    obscuring memorized trajectory patterns while preserving search quality.

    Example:
        >>> defense = MCTSDefense()
        >>> result = defense.apply(
        ...     mcts_tree={"values": [0.9, 0.8, 0.7], "depths": [0, 1, 2]},
        ...     epsilon=0.3,
        ...     mcts_leakage=0.6,
        ...     threshold=0.5,
        ... )
    """

    def __init__(
        self,
        depth_scale: float = 0.1,
        max_perturbation: float = 0.3,
        rerun_search: bool = True,
    ):
        """Initialize MCTS defense.

        Args:
            depth_scale: Scale factor for depth-based noise
            max_perturbation: Maximum perturbation magnitude
            rerun_search: Whether to simulate search re-run
        """
        self.depth_scale = depth_scale
        self.max_perturbation = max_perturbation
        self.rerun_search = rerun_search

    def apply(
        self,
        mcts_tree: Dict[str, Any],
        epsilon: float,
        mcts_leakage: float,
        threshold: float,
    ) -> Dict[str, Any]:
        """Apply MCTS defense to search tree.

        Args:
            mcts_tree: MCTS tree structure with values and depths
            epsilon: Defense intensity
            mcts_leakage: Detected MCTS leakage score
            threshold: Leakage threshold

        Returns:
            Defense result with perturbed values
        """
        if "values" not in mcts_tree:
            return {
                "applied": False,
                "error": "no_values_in_tree",
            }

        values = np.array(mcts_tree["values"])
        depths = np.array(mcts_tree.get("depths", np.zeros(len(values))))

        original_values = values.copy()

        # Compute perturbations: V'(n) = V(n) + N(0, ε * depth(n))
        perturbed_values, perturbation_info = self._perturb_values(
            values=values,
            depths=depths,
            epsilon=epsilon,
        )

        result = {
            "applied": True,
            "original_values": original_values.tolist(),
            "perturbed_values": perturbed_values.tolist(),
            "perturbation_info": perturbation_info,
            "epsilon_used": epsilon,
            "num_nodes": len(values),
        }

        # Simulate search re-run with perturbed values
        if self.rerun_search:
            rerun_result = self._simulate_rerun(
                mcts_tree=mcts_tree,
                perturbed_values=perturbed_values,
            )
            result["rerun_result"] = rerun_result
            if rerun_result.get("output"):
                result["rerun_output"] = rerun_result["output"]

        return result

    def _perturb_values(
        self,
        values: np.ndarray,
        depths: np.ndarray,
        epsilon: float,
    ) -> tuple:
        """Apply depth-scaled noise to value network outputs.

        V'(n) = V(n) + N(0, ε * depth_scale * depth(n))

        Args:
            values: Original value scores
            depths: Node depths in tree
            epsilon: Defense intensity

        Returns:
            Tuple of (perturbed_values, perturbation_info)
        """
        # Compute noise standard deviation for each node
        noise_std = epsilon * self.depth_scale * (depths + 1)

        # Cap noise to max_perturbation
        noise_std = np.clip(noise_std, 0, self.max_perturbation)

        # Sample noise
        noise = np.random.normal(0, noise_std)

        # Apply perturbation
        perturbed = values + noise

        # Clip to valid range [0, 1]
        perturbed = np.clip(perturbed, 0, 1)

        perturbation_info = {
            "noise_std_per_node": noise_std.tolist(),
            "actual_noise": noise.tolist(),
            "mean_perturbation": float(np.mean(np.abs(noise))),
            "max_perturbation": float(np.max(np.abs(noise))),
            "values_clipped": int(np.sum((perturbed != values + noise))),
        }

        return perturbed, perturbation_info

    def _simulate_rerun(
        self,
        mcts_tree: Dict[str, Any],
        perturbed_values: np.ndarray,
    ) -> Dict[str, Any]:
        """Simulate MCTS re-run with perturbed values.

        In practice, this would actually re-run the search.
        Here we simulate the effect of perturbed values on selection.

        Args:
            mcts_tree: Original tree structure
            perturbed_values: Perturbed value scores

        Returns:
            Re-run result
        """
        result: Dict[str, Any] = {"simulated": True}

        # Find best trajectory under perturbed values
        # This is a simplified simulation

        if "trajectories" in mcts_tree and len(mcts_tree["trajectories"]) > 0:
            # Evaluate trajectories with perturbed values
            trajectories = mcts_tree["trajectories"]
            trajectory_scores = []

            for traj in trajectories:
                if isinstance(traj, dict) and "node_indices" in traj:
                    indices = traj["node_indices"]
                elif isinstance(traj, list):
                    indices = list(range(len(traj)))
                else:
                    continue

                valid_indices = [i for i in indices if i < len(perturbed_values)]
                if valid_indices:
                    score = float(np.mean(perturbed_values[valid_indices]))
                    trajectory_scores.append(score)
                else:
                    trajectory_scores.append(0.0)

            if trajectory_scores:
                best_traj_idx = int(np.argmax(trajectory_scores))
                result["best_trajectory_index"] = best_traj_idx
                result["trajectory_scores"] = trajectory_scores

                # Get output from best trajectory if available
                if "outputs" in mcts_tree and best_traj_idx < len(mcts_tree["outputs"]):
                    result["output"] = mcts_tree["outputs"][best_traj_idx]

        # If no trajectories, select node with highest perturbed value
        if "output" not in result:
            best_node_idx = int(np.argmax(perturbed_values))
            result["best_node_index"] = best_node_idx
            result["best_node_value"] = float(perturbed_values[best_node_idx])

            # Get output from best node if available
            if "node_outputs" in mcts_tree and best_node_idx < len(mcts_tree["node_outputs"]):
                result["output"] = mcts_tree["node_outputs"][best_node_idx]

        return result

    def perturb_single_value(
        self,
        value: float,
        depth: int,
        epsilon: float,
    ) -> float:
        """Perturb a single value score.

        Args:
            value: Original value
            depth: Node depth
            epsilon: Defense intensity

        Returns:
            Perturbed value
        """
        noise_std = epsilon * self.depth_scale * (depth + 1)
        noise_std = min(noise_std, self.max_perturbation)
        noise = np.random.normal(0, noise_std)
        perturbed = float(np.clip(value + noise, 0, 1))
        return perturbed


# Last updated: 2026-01-15
