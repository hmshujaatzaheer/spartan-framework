"""
Model Interfaces

Provides base classes and mock implementations for reasoning LLMs.
"""

from spartan.models.base import BaseReasoningLLM, LLMOutput
from spartan.models.mock import MockReasoningLLM

__all__ = [
    "BaseReasoningLLM",
    "LLMOutput",
    "MockReasoningLLM",
]
