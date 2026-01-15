"""
RAAS: Reasoning-Aware Adaptive Sanitization

This module implements adaptive defense mechanisms for reasoning LLMs:
- PRM noise injection with NL-blindness exploitation
- Vote distribution flattening with implicit rewards
- MCTS value network perturbation
"""

from spartan.raas.mcts_defense import MCTSDefense
from spartan.raas.prm_defense import PRMDefense
from spartan.raas.sanitizer import RAAS, RAASResult
from spartan.raas.vote_defense import VoteDefense

__all__ = [
    "RAAS",
    "RAASResult",
    "PRMDefense",
    "VoteDefense",
    "MCTSDefense",
]

# Last updated: 2026-01-15
