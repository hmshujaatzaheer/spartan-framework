"""
Attack Implementations

This module provides implementations of novel attack vectors targeting
reasoning LLM test-time compute mechanisms:
- NLBA: Natural Language Blindness Attack (targets PRMs)
- SMVA: Single-Model Voting Attack (targets self-consistency)
- MVNA: MCTS Value Network Attack (targets MCTS search)
"""

from spartan.attacks.base import BaseAttack, AttackResult
from spartan.attacks.nlba import NLBAAttack
from spartan.attacks.smva import SMVAAttack
from spartan.attacks.mvna import MVNAAttack

__all__ = [
    "BaseAttack",
    "AttackResult",
    "NLBAAttack",
    "SMVAAttack",
    "MVNAAttack",
]
