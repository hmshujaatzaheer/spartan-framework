"""
Base Reasoning LLM Interface

Abstract interface for reasoning LLMs with TTC mechanisms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LLMOutput:
    """Output from a reasoning LLM.

    Attributes:
        output: Final output text
        reasoning_steps: List of reasoning step strings
        prm_scores: PRM scores for each step (if available)
        vote_distribution: Vote distribution from self-consistency
        mcts_values: Value scores from MCTS nodes
        mcts_tree: Full MCTS tree structure
        metadata: Additional metadata
    """

    output: str
    reasoning_steps: List[str] = field(default_factory=list)
    prm_scores: Optional[List[float]] = None
    vote_distribution: Optional[List[float]] = None
    mcts_values: Optional[List[float]] = None
    mcts_tree: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "output": self.output,
            "reasoning_steps": self.reasoning_steps,
            "prm_scores": self.prm_scores,
            "vote_distribution": self.vote_distribution,
            "mcts_values": self.mcts_values,
            "mcts_tree": self.mcts_tree,
            "metadata": self.metadata,
        }


class BaseReasoningLLM(ABC):
    """Abstract base class for reasoning LLMs.

    Implementations should provide access to TTC mechanisms:
    - PRM scoring
    - Self-consistency voting
    - MCTS search
    """

    def __init__(self, name: str = "BaseReasoningLLM"):
        """Initialize LLM.

        Args:
            name: Model name
        """
        self._name = name

    @property
    def name(self) -> str:
        """Get model name."""
        return self._name

    @abstractmethod
    def generate(
        self,
        query: str,
        num_samples: int = 1,
        use_prm: bool = True,
        use_mcts: bool = False,
        **kwargs: Any,
    ) -> LLMOutput:
        """Generate response with reasoning.

        Args:
            query: Input query
            num_samples: Number of samples for voting
            use_prm: Whether to use PRM verification
            use_mcts: Whether to use MCTS search
            **kwargs: Additional generation arguments

        Returns:
            LLMOutput containing response and TTC information
        """
        pass

    @abstractmethod
    def get_prm_scores(
        self,
        reasoning_steps: List[str],
    ) -> List[float]:
        """Get PRM scores for reasoning steps.

        Args:
            reasoning_steps: List of reasoning step strings

        Returns:
            List of PRM scores
        """
        pass

    def uses_prm(self) -> bool:
        """Check if model uses PRM."""
        return True

    def uses_voting(self) -> bool:
        """Check if model uses self-consistency voting."""
        return True

    def uses_mcts(self) -> bool:
        """Check if model uses MCTS."""
        return False
