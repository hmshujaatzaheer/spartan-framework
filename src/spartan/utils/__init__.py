"""
SPARTAN Utilities

Common utilities for metrics, distributions, and noise generation.
"""

from spartan.utils.metrics import (
    compute_auc_roc,
    compute_accuracy,
    compute_tpr_at_fpr,
    compute_f1_score,
)
from spartan.utils.distributions import (
    kl_divergence,
    js_divergence,
    entropy,
)
from spartan.utils.noise import (
    gaussian_noise,
    laplace_noise,
    calibrated_noise,
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
