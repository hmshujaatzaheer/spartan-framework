"""
SPARTAN: Secure Privacy-Adaptive Reasoning with Test-time Attack Neutralization

A unified framework for detecting and defending against mechanistic privacy
attacks in Reasoning Large Language Models (LLMs).
"""

from spartan.core import SPARTAN, SPARTANResult
from spartan.config import SPARTANConfig
from spartan.mplq import MPLQ, MPLQResult
from spartan.raas import RAAS, RAASResult
from spartan.rppo import RPPO, RPPOResult

__version__ = "1.0.0"
__author__ = "SPARTAN Team"

__all__ = [
    "SPARTAN",
    "SPARTANResult",
    "SPARTANConfig",
    "MPLQ",
    "MPLQResult",
    "RAAS",
    "RAASResult",
    "RPPO",
    "RPPOResult",
]
