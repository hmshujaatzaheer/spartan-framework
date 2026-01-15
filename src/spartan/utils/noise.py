"""
Noise Generation Utilities

Noise generators for differential privacy and defense mechanisms.
"""

from typing import Optional, Tuple, Union

import numpy as np


def gaussian_noise(
    shape: Union[int, Tuple[int, ...]],
    scale: float = 1.0,
    mean: float = 0.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate Gaussian noise.

    Args:
        shape: Output shape
        scale: Standard deviation
        mean: Mean value
        seed: Random seed

    Returns:
        Array of Gaussian noise
    """
    if seed is not None:
        np.random.seed(seed)

    if isinstance(shape, int):
        shape = (shape,)

    noise = np.random.normal(mean, scale, shape)

    return noise


def laplace_noise(
    shape: Union[int, Tuple[int, ...]],
    scale: float = 1.0,
    location: float = 0.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate Laplace noise for differential privacy.

    The Laplace mechanism adds noise scaled to sensitivity/epsilon.

    Args:
        shape: Output shape
        scale: Scale parameter (b = sensitivity/epsilon)
        location: Location parameter
        seed: Random seed

    Returns:
        Array of Laplace noise
    """
    if seed is not None:
        np.random.seed(seed)

    if isinstance(shape, int):
        shape = (shape,)

    noise = np.random.laplace(location, scale, shape)

    return noise


def calibrated_noise(
    shape: Union[int, Tuple[int, ...]],
    sensitivity: float,
    epsilon: float,
    mechanism: str = "laplace",
    delta: float = 0.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate calibrated DP noise.

    Automatically calibrates noise scale based on privacy parameters.

    Args:
        shape: Output shape
        sensitivity: Query sensitivity
        epsilon: Privacy parameter
        mechanism: "laplace" or "gaussian"
        delta: Delta parameter (for Gaussian mechanism)
        seed: Random seed

    Returns:
        Calibrated noise array
    """
    if seed is not None:
        np.random.seed(seed)

    if isinstance(shape, int):
        shape = (shape,)

    if mechanism == "laplace":
        # Laplace mechanism: scale = sensitivity / epsilon
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale, shape)

    elif mechanism == "gaussian":
        # Gaussian mechanism: sigma >= sqrt(2 * ln(1.25/delta)) * sensitivity / epsilon
        if delta <= 0:
            delta = 1e-5  # Default delta

        sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
        noise = np.random.normal(0, sigma, shape)

    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")

    return noise


def truncated_noise(
    shape: Union[int, Tuple[int, ...]],
    scale: float = 1.0,
    lower: float = -np.inf,
    upper: float = np.inf,
    distribution: str = "gaussian",
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate truncated noise.

    Args:
        shape: Output shape
        scale: Scale parameter
        lower: Lower truncation bound
        upper: Upper truncation bound
        distribution: "gaussian" or "laplace"
        seed: Random seed

    Returns:
        Truncated noise array
    """
    if seed is not None:
        np.random.seed(seed)

    if isinstance(shape, int):
        shape = (shape,)

    # Generate initial noise
    if distribution == "gaussian":
        noise = np.random.normal(0, scale, shape)
    elif distribution == "laplace":
        noise = np.random.laplace(0, scale, shape)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    # Truncate
    noise = np.clip(noise, lower, upper)

    return noise


def exponential_mechanism_noise(
    scores: np.ndarray,
    sensitivity: float,
    epsilon: float,
    seed: Optional[int] = None,
) -> int:
    """Sample from exponential mechanism.

    For discrete selection with differential privacy.
    Probability proportional to exp(epsilon * score / (2 * sensitivity))

    Args:
        scores: Quality scores for each option
        sensitivity: Sensitivity of scoring function
        epsilon: Privacy parameter
        seed: Random seed

    Returns:
        Selected index
    """
    if seed is not None:
        np.random.seed(seed)

    scores = np.array(scores)

    # Compute probabilities
    log_probs = epsilon * scores / (2 * sensitivity)

    # Numerical stability
    log_probs = log_probs - np.max(log_probs)
    probs = np.exp(log_probs)
    probs = probs / probs.sum()

    # Sample
    selected = np.random.choice(len(scores), p=probs)

    return int(selected)


def adaptive_noise(
    shape: Union[int, Tuple[int, ...]],
    risk_score: float,
    epsilon_min: float = 0.01,
    epsilon_max: float = 0.5,
    mechanism: str = "gaussian",
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate adaptive noise based on risk score.

    Higher risk → more noise (lower privacy budget).

    Args:
        shape: Output shape
        risk_score: Risk score in [0, 1]
        epsilon_min: Minimum noise level
        epsilon_max: Maximum noise level
        mechanism: Noise distribution
        seed: Random seed

    Returns:
        Adaptive noise array
    """
    if seed is not None:
        np.random.seed(seed)

    # Scale epsilon inversely with risk
    # High risk → low epsilon → more noise
    scale = epsilon_min + (epsilon_max - epsilon_min) * risk_score

    if mechanism == "gaussian":
        noise = gaussian_noise(shape, scale=scale, seed=seed)
    elif mechanism == "laplace":
        noise = laplace_noise(shape, scale=scale, seed=seed)
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")

    return noise


# Last updated: 2026-01-15
