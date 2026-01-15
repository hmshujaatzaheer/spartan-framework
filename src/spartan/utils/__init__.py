"""
SPARTAN Utilities

Common utilities for metrics, distributions, and noise generation.
"""

from spartan.utils.distributions import (
    entropy,
    js_divergence,
    kl_divergence,
)
from spartan.utils.metrics import (
    compute_accuracy,
    compute_auc_roc,
    compute_f1_score,
    compute_tpr_at_fpr,
)
from spartan.utils.noise import (
    calibrated_noise,
    gaussian_noise,
    laplace_noise,
)

__all__ = [
    "compute_auc_roc",
    "compute_accuracy",
    "compute_tpr_at_fpr",
    "compute_f1_score",
    "kl_divergence",
    "js_divergence",
    "entropy",
    "gaussian_noise",
    "laplace_noise",
    "calibrated_noise",
]

# Last updated: 2026-01-15
