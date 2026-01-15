"""
Vote Defense Module

Implements vote distribution flattening with implicit reward resampling
for self-consistency voting privacy protection.
"""

from typing import Any, Dict, List, Optional

import numpy as np


class VoteDefense:
    """Vote distribution flattening defense.

    Applies temperature scaling to flatten concentrated vote distributions,
    reducing privacy leakage from memorization patterns. Uses implicit
    reward-based resampling inspired by PRIME to maintain reasoning quality.

    Example:
        >>> defense = VoteDefense()
        >>> result = defense.apply(
        ...     vote_distribution=[0.9, 0.05, 0.05],
        ...     epsilon=0.3,
        ...     vote_leakage=0.7,
        ...     threshold=0.4,
        ... )
    """

    def __init__(
        self,
        temperature_base: float = 1.0,
        use_implicit_rewards: bool = True,
        min_temperature: float = 1.0,
        max_temperature: float = 5.0,
    ):
        """Initialize vote defense.

        Args:
            temperature_base: Base temperature for softmax
            use_implicit_rewards: Whether to use implicit reward resampling
            min_temperature: Minimum temperature value
            max_temperature: Maximum temperature value
        """
        self.temperature_base = temperature_base
        self.use_implicit_rewards = use_implicit_rewards
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature

    def apply(
        self,
        vote_distribution: List[float],
        epsilon: float,
        vote_leakage: float,
        threshold: float,
        candidate_outputs: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Apply vote defense to distribution.

        Args:
            vote_distribution: Original vote distribution
            epsilon: Defense intensity
            vote_leakage: Detected vote leakage score
            threshold: Leakage threshold
            candidate_outputs: Optional candidate output strings

        Returns:
            Defense result with flattened distribution
        """
        if len(vote_distribution) == 0:
            return {
                "applied": False,
                "original_distribution": [],
                "flattened_distribution": [],
            }

        votes = np.array(vote_distribution)

        # Normalize input
        vote_sum = votes.sum()
        if vote_sum > 0:
            votes = votes / vote_sum
        else:
            votes = np.ones_like(votes) / len(votes)

        original_entropy = self._compute_entropy(votes)

        # Compute temperature based on leakage and epsilon
        # T = 1 + ε * (L_vote - τ)
        temperature = self._compute_temperature(
            epsilon=epsilon,
            vote_leakage=vote_leakage,
            threshold=threshold,
        )

        # Apply temperature-scaled softmax flattening
        # v' = softmax(v / T)
        flattened = self._temperature_flatten(votes, temperature)

        flattened_entropy = self._compute_entropy(flattened)

        result = {
            "applied": True,
            "original_distribution": votes.tolist(),
            "flattened_distribution": flattened.tolist(),
            "temperature_used": temperature,
            "original_entropy": original_entropy,
            "flattened_entropy": flattened_entropy,
            "entropy_increase": flattened_entropy - original_entropy,
            "epsilon_used": epsilon,
        }

        # Apply implicit reward resampling if enabled
        if self.use_implicit_rewards and candidate_outputs is not None:
            resampled_output, resample_info = self._implicit_reward_resample(
                distribution=flattened,
                candidates=candidate_outputs,
            )
            result["resampled_output"] = resampled_output
            result["resample_info"] = resample_info

        return result

    def _compute_temperature(
        self,
        epsilon: float,
        vote_leakage: float,
        threshold: float,
    ) -> float:
        """Compute temperature for distribution flattening.

        T = T_base + ε * (L_vote - τ) for L_vote > τ

        Args:
            epsilon: Defense intensity
            vote_leakage: Vote leakage score
            threshold: Leakage threshold

        Returns:
            Temperature value
        """
        if vote_leakage <= threshold:
            return self.temperature_base

        # Scale temperature with leakage above threshold
        excess_leakage = vote_leakage - threshold
        temperature = self.temperature_base + epsilon * excess_leakage * 4.0

        return float(np.clip(temperature, self.min_temperature, self.max_temperature))

    def _temperature_flatten(
        self,
        distribution: np.ndarray,
        temperature: float,
    ) -> np.ndarray:
        """Apply temperature-scaled softmax flattening.

        v' = softmax(log(v) / T)

        Args:
            distribution: Original distribution
            temperature: Temperature parameter

        Returns:
            Flattened distribution
        """
        if temperature <= 0:
            return distribution

        # Use log-space for numerical stability
        # Add small constant to avoid log(0)
        log_dist = np.log(distribution + 1e-10)

        # Temperature scaling
        scaled = log_dist / temperature

        # Softmax
        exp_scaled = np.exp(scaled - np.max(scaled))  # Subtract max for stability
        flattened = exp_scaled / exp_scaled.sum()

        return flattened

    def _compute_entropy(self, distribution: np.ndarray) -> float:
        """Compute Shannon entropy.

        Args:
            distribution: Probability distribution

        Returns:
            Entropy value
        """
        mask = distribution > 0
        if not np.any(mask):
            return 0.0
        entropy = -np.sum(distribution[mask] * np.log(distribution[mask]))
        return float(entropy)

    def _implicit_reward_resample(
        self,
        distribution: np.ndarray,
        candidates: List[str],
    ) -> tuple:
        """Resample output using implicit rewards.

        Inspired by PRIME (Process Reinforcement through Implicit Rewards),
        this method resamples from the flattened distribution while
        considering implicit quality signals.

        Args:
            distribution: Flattened vote distribution
            candidates: Candidate output strings

        Returns:
            Tuple of (resampled_output, resample_info)
        """
        if len(candidates) != len(distribution):
            # Mismatch - return most likely
            idx = int(np.argmax(distribution))
            return candidates[idx] if candidates else "", {"error": "length_mismatch"}

        # Compute implicit rewards for candidates
        implicit_rewards = self._compute_implicit_rewards(candidates)

        # Combine with flattened distribution
        # Combined = α * flattened + (1-α) * reward_normalized
        alpha = 0.7  # Weight for privacy-protected distribution

        reward_normalized = implicit_rewards / (implicit_rewards.sum() + 1e-10)
        combined = alpha * distribution + (1 - alpha) * reward_normalized
        combined = combined / combined.sum()

        # Sample from combined distribution
        sampled_idx = np.random.choice(len(combined), p=combined)
        resampled_output = candidates[sampled_idx]

        resample_info = {
            "sampled_index": int(sampled_idx),
            "implicit_rewards": implicit_rewards.tolist(),
            "combined_distribution": combined.tolist(),
            "alpha": alpha,
        }

        return resampled_output, resample_info

    def _compute_implicit_rewards(self, candidates: List[str]) -> np.ndarray:
        """Compute implicit quality rewards for candidates.

        Uses heuristics to estimate output quality without explicit labels.

        Args:
            candidates: Candidate outputs

        Returns:
            Array of reward scores
        """
        rewards = []

        for candidate in candidates:
            reward = 1.0  # Base reward

            # Length-based heuristic (moderate length preferred)
            length = len(candidate)
            if 10 <= length <= 500:
                reward += 0.2
            elif length > 500:
                reward -= 0.1

            # Completeness heuristic (ends with punctuation)
            if candidate and candidate[-1] in ".!?)":
                reward += 0.1

            # Consistency heuristic (contains numbers if math-like)
            has_math = any(c.isdigit() for c in candidate)
            if has_math:
                reward += 0.1

            rewards.append(reward)

        return np.array(rewards)
