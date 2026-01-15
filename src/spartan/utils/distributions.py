"""
Distribution Utilities

Statistical distribution computations for privacy analysis.
"""

from typing import Optional

import numpy as np


def kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    smoothing: float = 1e-10,
) -> float:
    """Compute KL divergence D_KL(P || Q).

    Args:
        p: Distribution P
        q: Distribution Q
        smoothing: Smoothing factor to avoid log(0)

    Returns:
        KL divergence value
    """
    p = np.array(p)
    q = np.array(q)

    # Normalize
    p = p / (p.sum() + smoothing)
    q = q / (q.sum() + smoothing)

    # Add smoothing
    p = p + smoothing
    q = q + smoothing
    p = p / p.sum()
    q = q / q.sum()

    # Compute KL divergence
    kl = np.sum(p * np.log(p / q))

    return float(kl)


def js_divergence(
    p: np.ndarray,
    q: np.ndarray,
    smoothing: float = 1e-10,
) -> float:
    """Compute Jensen-Shannon divergence.

    JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q)

    Args:
        p: Distribution P
        q: Distribution Q
        smoothing: Smoothing factor

    Returns:
        JS divergence value (bounded in [0, log(2)])
    """
    p = np.array(p)
    q = np.array(q)

    # Normalize
    p = p / (p.sum() + smoothing)
    q = q / (q.sum() + smoothing)

    # Compute mixture
    m = 0.5 * (p + q)

    # Compute JS divergence
    js = 0.5 * kl_divergence(p, m, smoothing) + 0.5 * kl_divergence(q, m, smoothing)

    return float(js)


def entropy(
    distribution: np.ndarray,
    base: Optional[float] = None,
) -> float:
    """Compute Shannon entropy.

    H(X) = -Σ p(x) * log(p(x))

    Args:
        distribution: Probability distribution
        base: Logarithm base (default: natural log)

    Returns:
        Entropy value
    """
    distribution = np.array(distribution)

    # Normalize
    total = distribution.sum()
    if total > 0:
        distribution = distribution / total

    # Compute entropy (only for non-zero values)
    mask = distribution > 0
    if not np.any(mask):
        return 0.0

    h = -np.sum(distribution[mask] * np.log(distribution[mask]))

    # Convert base if specified
    if base is not None and base > 0:
        h = h / np.log(base)

    return float(h)


def cross_entropy(
    p: np.ndarray,
    q: np.ndarray,
    smoothing: float = 1e-10,
) -> float:
    """Compute cross-entropy H(P, Q).

    H(P, Q) = -Σ p(x) * log(q(x))

    Args:
        p: True distribution P
        q: Predicted distribution Q
        smoothing: Smoothing factor

    Returns:
        Cross-entropy value
    """
    p = np.array(p)
    q = np.array(q)

    # Normalize
    p = p / (p.sum() + smoothing)
    q = q / (q.sum() + smoothing)

    # Add smoothing
    q = q + smoothing
    q = q / q.sum()

    # Compute cross-entropy
    ce = -np.sum(p * np.log(q))

    return float(ce)


def total_variation_distance(
    p: np.ndarray,
    q: np.ndarray,
) -> float:
    """Compute total variation distance.

    TV(P, Q) = 0.5 * Σ |p(x) - q(x)|

    Args:
        p: Distribution P
        q: Distribution Q

    Returns:
        Total variation distance (bounded in [0, 1])
    """
    p = np.array(p)
    q = np.array(q)

    # Normalize
    p = p / (p.sum() + 1e-10)
    q = q / (q.sum() + 1e-10)

    # Compute TV distance
    tv = 0.5 * np.sum(np.abs(p - q))

    return float(tv)


def hellinger_distance(
    p: np.ndarray,
    q: np.ndarray,
) -> float:
    """Compute Hellinger distance.

    H(P, Q) = sqrt(0.5 * Σ (sqrt(p(x)) - sqrt(q(x)))^2)

    Args:
        p: Distribution P
        q: Distribution Q

    Returns:
        Hellinger distance (bounded in [0, 1])
    """
    p = np.array(p)
    q = np.array(q)

    # Normalize
    p = p / (p.sum() + 1e-10)
    q = q / (q.sum() + 1e-10)

    # Compute Hellinger distance
    h = np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))

    return float(h)


def renyi_divergence(
    p: np.ndarray,
    q: np.ndarray,
    alpha: float = 2.0,
    smoothing: float = 1e-10,
) -> float:
    """Compute Rényi divergence.

    D_α(P || Q) = (1/(α-1)) * log(Σ p(x)^α * q(x)^(1-α))

    Args:
        p: Distribution P
        q: Distribution Q
        alpha: Order parameter (α > 0, α ≠ 1)
        smoothing: Smoothing factor

    Returns:
        Rényi divergence value
    """
    if alpha == 1:
        # Limit as α → 1 is KL divergence
        return kl_divergence(p, q, smoothing)

    p = np.array(p)
    q = np.array(q)

    # Normalize
    p = p / (p.sum() + smoothing)
    q = q / (q.sum() + smoothing)

    # Add smoothing
    p = p + smoothing
    q = q + smoothing
    p = p / p.sum()
    q = q / q.sum()

    # Compute Rényi divergence
    term = np.sum((p**alpha) * (q ** (1 - alpha)))
    renyi = np.log(term) / (alpha - 1)

    return float(renyi)


# Last updated: 2026-01-15
