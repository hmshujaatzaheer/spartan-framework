"""
RPPO: Reasoning-Privacy Pareto Optimization

This module implements multi-objective optimization for balancing:
- Reasoning accuracy
- Privacy protection
- Computational efficiency
"""

from spartan.rppo.optimizer import RPPO, RPPOResult
from spartan.rppo.bandit import UCBBandit
from spartan.rppo.pareto import ParetoFront

__all__ = [
    "RPPO",
    "RPPOResult",
    "UCBBandit",
    "ParetoFront",
]
