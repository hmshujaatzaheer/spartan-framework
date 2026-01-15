"""
UCB Bandit Implementation

Upper Confidence Bound multi-armed bandit for RPPO arm selection.
"""

from typing import Any, Dict, List, Optional

import numpy as np


class UCBBandit:
    """Upper Confidence Bound (UCB) multi-armed bandit.

    Implements UCB1 algorithm for balancing exploration vs exploitation
    in parameter configuration selection.

    UCB(a) = R̄_a + c * sqrt(log(t) / N_a)

    Example:
        >>> bandit = UCBBandit(num_arms=10, exploration_constant=2.0)
        >>> arm = bandit.select_arm()
        >>> bandit.update(arm, reward=0.8)
    """

    def __init__(
        self,
        num_arms: int,
        exploration_constant: float = 2.0,
    ):
        """Initialize UCB bandit.

        Args:
            num_arms: Number of arms
            exploration_constant: UCB exploration parameter (c)
        """
        self.num_arms = num_arms
        self.exploration_constant = exploration_constant

        # Arm statistics
        self._counts = np.zeros(num_arms)
        self._rewards = np.zeros(num_arms)
        self._total_pulls = 0
        self._current_arm = 0

        # History
        self._reward_history: List[List[float]] = [[] for _ in range(num_arms)]

    def select_arm(self) -> int:
        """Select arm using UCB1 algorithm.

        Returns:
            Selected arm index
        """
        # Initial exploration: try each arm once
        for arm in range(self.num_arms):
            if self._counts[arm] == 0:
                self._current_arm = arm
                return arm

        # UCB1 selection
        ucb_values = self._compute_ucb_values()
        selected_arm = int(np.argmax(ucb_values))

        self._current_arm = selected_arm
        return selected_arm

    def _compute_ucb_values(self) -> np.ndarray:
        """Compute UCB values for all arms.

        UCB(a) = R̄_a + c * sqrt(log(t) / N_a)

        Returns:
            Array of UCB values
        """
        ucb_values = np.zeros(self.num_arms)

        for arm in range(self.num_arms):
            if self._counts[arm] == 0:
                ucb_values[arm] = float("inf")
            else:
                # Average reward
                avg_reward = self._rewards[arm] / self._counts[arm]

                # Exploration bonus
                exploration = self.exploration_constant * np.sqrt(
                    np.log(self._total_pulls + 1) / self._counts[arm]
                )

                ucb_values[arm] = avg_reward + exploration

        return ucb_values

    def update(
        self,
        arm: int,
        reward: float,
    ) -> None:
        """Update arm statistics with observed reward.

        Args:
            arm: Arm index
            reward: Observed reward
        """
        if not 0 <= arm < self.num_arms:
            raise ValueError(f"Invalid arm index: {arm}")

        self._counts[arm] += 1
        self._rewards[arm] += reward
        self._total_pulls += 1
        self._reward_history[arm].append(reward)

    def get_current_arm(self) -> int:
        """Get currently selected arm."""
        return self._current_arm

    def get_best_arm(self) -> int:
        """Get arm with highest average reward.

        Returns:
            Best arm index
        """
        avg_rewards = np.zeros(self.num_arms)

        for arm in range(self.num_arms):
            if self._counts[arm] > 0:
                avg_rewards[arm] = self._rewards[arm] / self._counts[arm]
            else:
                avg_rewards[arm] = 0

        return int(np.argmax(avg_rewards))

    def get_arm_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all arms.

        Returns:
            List of arm statistics
        """
        stats = []

        for arm in range(self.num_arms):
            count = int(self._counts[arm])
            avg_reward = float(self._rewards[arm] / count) if count > 0 else 0.0

            arm_stat = {
                "arm_index": arm,
                "count": count,
                "total_reward": float(self._rewards[arm]),
                "average_reward": avg_reward,
            }

            if self._reward_history[arm]:
                arm_stat["reward_std"] = float(np.std(self._reward_history[arm]))
                arm_stat["min_reward"] = float(min(self._reward_history[arm]))
                arm_stat["max_reward"] = float(max(self._reward_history[arm]))

            stats.append(arm_stat)

        return stats

    def reset(self) -> None:
        """Reset bandit state."""
        self._counts = np.zeros(self.num_arms)
        self._rewards = np.zeros(self.num_arms)
        self._total_pulls = 0
        self._current_arm = 0
        self._reward_history = [[] for _ in range(self.num_arms)]

    def get_exploration_rate(self) -> float:
        """Get current exploration rate.

        Returns:
            Fraction of pulls that were exploration vs exploitation
        """
        if self._total_pulls == 0:
            return 1.0

        # Estimate based on UCB bonus magnitude
        ucb_values = self._compute_ucb_values()

        if len(ucb_values[ucb_values < float("inf")]) == 0:
            return 1.0

        valid_values = ucb_values[ucb_values < float("inf")]
        avg_rewards = self._rewards / np.maximum(self._counts, 1)

        exploration_bonus = np.mean(valid_values - avg_rewards[self._counts > 0])

        # Higher bonus = more exploration
        return float(np.clip(exploration_bonus / 2, 0, 1))


# Last updated: 2026-01-15
