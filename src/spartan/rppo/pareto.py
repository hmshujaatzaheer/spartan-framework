"""
Pareto Front Utilities

Tools for tracking and analyzing Pareto-optimal configurations.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ParetoFront:
    """Pareto front tracker for multi-objective optimization.

    Maintains a set of Pareto-optimal points where no point
    dominates another in all objectives.

    Example:
        >>> pf = ParetoFront()
        >>> pf.add_point([0.8, 0.7, 0.6], {"config": "A"})
        >>> pf.add_point([0.9, 0.6, 0.5], {"config": "B"})
        >>> front = pf.get_front()
    """

    def __init__(self):
        """Initialize Pareto front."""
        self._points: List[np.ndarray] = []
        self._params: List[Dict[str, Any]] = []

    def add_point(
        self,
        point: np.ndarray,
        params: Dict[str, Any],
    ) -> bool:
        """Add a point to the Pareto front if not dominated.

        Args:
            point: Objective values (higher is better)
            params: Associated parameters

        Returns:
            True if point was added to front
        """
        point = np.array(point)

        # Check if dominated by existing points
        for existing in self._points:
            if self._dominates(existing, point):
                return False

        # Remove points dominated by new point
        non_dominated = []
        non_dominated_params = []

        for i, existing in enumerate(self._points):
            if not self._dominates(point, existing):
                non_dominated.append(existing)
                non_dominated_params.append(self._params[i])

        # Add new point
        non_dominated.append(point)
        non_dominated_params.append(params)

        self._points = non_dominated
        self._params = non_dominated_params

        return True

    def _dominates(
        self,
        point_a: np.ndarray,
        point_b: np.ndarray,
    ) -> bool:
        """Check if point_a dominates point_b.

        A dominates B if A is >= B in all objectives and > B in at least one.

        Args:
            point_a: First point
            point_b: Second point

        Returns:
            True if point_a dominates point_b
        """
        return np.all(point_a >= point_b) and np.any(point_a > point_b)

    def is_pareto_optimal(
        self,
        point: np.ndarray,
    ) -> bool:
        """Check if a point would be Pareto-optimal.

        Args:
            point: Point to check

        Returns:
            True if point is not dominated by any existing point
        """
        point = np.array(point)

        for existing in self._points:
            if self._dominates(existing, point):
                return False

        return True

    def get_front(self) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """Get current Pareto front.

        Returns:
            List of (point, params) tuples
        """
        return list(zip(self._points, self._params))

    def get_points(self) -> List[np.ndarray]:
        """Get Pareto-optimal points only."""
        return self._points.copy()

    def get_params(self) -> List[Dict[str, Any]]:
        """Get params for Pareto-optimal points."""
        return self._params.copy()

    def size(self) -> int:
        """Get number of points on Pareto front."""
        return len(self._points)

    def clear(self) -> None:
        """Clear the Pareto front."""
        self._points = []
        self._params = []

    def get_hypervolume(
        self,
        reference_point: Optional[np.ndarray] = None,
    ) -> float:
        """Compute hypervolume indicator of Pareto front.

        Hypervolume measures the "size" of the objective space
        dominated by the Pareto front.

        Args:
            reference_point: Reference point (origin if None)

        Returns:
            Hypervolume value
        """
        if len(self._points) == 0:
            return 0.0

        points = np.array(self._points)

        if reference_point is None:
            reference_point = np.zeros(points.shape[1])

        # Simple hypervolume computation for low dimensions
        # For high dimensions, use approximate methods
        if points.shape[1] <= 3:
            return self._compute_hypervolume_exact(points, reference_point)
        else:
            return self._compute_hypervolume_monte_carlo(points, reference_point)

    def _compute_hypervolume_exact(
        self,
        points: np.ndarray,
        reference: np.ndarray,
    ) -> float:
        """Exact hypervolume computation for 2D/3D.

        Args:
            points: Pareto points
            reference: Reference point

        Returns:
            Hypervolume
        """
        if points.shape[1] == 1:
            # 1D: just the max point minus reference
            return float(np.max(points[:, 0]) - reference[0])

        if points.shape[1] == 2:
            # 2D: sort by first objective and sweep
            sorted_idx = np.argsort(points[:, 0])[::-1]
            sorted_points = points[sorted_idx]

            hv = 0.0
            prev_x = reference[0]

            for point in sorted_points:
                width = point[0] - prev_x
                height = point[1] - reference[1]
                hv += width * height
                prev_x = point[0]

            return float(hv)

        # 3D: more complex, use inclusion-exclusion
        # Simplified approximation
        return self._compute_hypervolume_monte_carlo(points, reference)

    def _compute_hypervolume_monte_carlo(
        self,
        points: np.ndarray,
        reference: np.ndarray,
        num_samples: int = 10000,
    ) -> float:
        """Monte Carlo hypervolume approximation.

        Args:
            points: Pareto points
            reference: Reference point
            num_samples: Number of random samples

        Returns:
            Approximate hypervolume
        """
        # Find bounding box
        max_point = np.max(points, axis=0)

        # Sample random points in bounding box
        samples = np.random.uniform(reference, max_point, size=(num_samples, len(reference)))

        # Count samples dominated by at least one Pareto point
        dominated = np.zeros(num_samples, dtype=bool)

        for point in points:
            dominated |= np.all(samples <= point, axis=1)

        # Estimate hypervolume
        box_volume = float(np.prod(max_point - reference))
        hv = box_volume * np.mean(dominated)

        return hv

    def get_closest_to_ideal(
        self,
        ideal_point: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Get Pareto point closest to ideal.

        Args:
            ideal_point: Ideal objective values (default: all 1s)

        Returns:
            Closest (point, params) tuple, or None if empty
        """
        if len(self._points) == 0:
            return None

        points = np.array(self._points)

        if ideal_point is None:
            ideal_point = np.ones(points.shape[1])

        # Compute distances to ideal
        distances = np.linalg.norm(points - ideal_point, axis=1)
        closest_idx = int(np.argmin(distances))

        return (self._points[closest_idx], self._params[closest_idx])
