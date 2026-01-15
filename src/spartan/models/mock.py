"""
Mock Reasoning LLM Implementation

Provides mock implementations for testing SPARTAN without real LLMs.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from spartan.models.base import BaseReasoningLLM, LLMOutput


class MockReasoningLLM(BaseReasoningLLM):
    """Mock reasoning LLM for testing.

    Generates synthetic reasoning outputs with configurable
    TTC mechanism behaviors.

    Example:
        >>> model = MockReasoningLLM(member_mode=True)
        >>> output = model.generate("What is 2+2?")
        >>> print(output.prm_scores)  # High scores for member
    """

    def __init__(
        self,
        name: str = "MockReasoningLLM",
        member_mode: bool = False,
        num_reasoning_steps: int = 5,
        use_mcts: bool = True,
        seed: Optional[int] = None,
    ):
        """Initialize mock LLM.

        Args:
            name: Model name
            member_mode: If True, simulate member (memorized) behavior
            num_reasoning_steps: Number of reasoning steps to generate
            use_mcts: Whether to include MCTS outputs
            seed: Random seed for reproducibility
        """
        super().__init__(name=name)
        self.member_mode = member_mode
        self.num_reasoning_steps = num_reasoning_steps
        self._use_mcts = use_mcts

        if seed is not None:
            np.random.seed(seed)

    def generate(
        self,
        query: str,
        num_samples: int = 1,
        use_prm: bool = True,
        use_mcts: bool = False,
        **kwargs: Any,
    ) -> LLMOutput:
        """Generate mock response.

        Args:
            query: Input query
            num_samples: Number of samples for voting
            use_prm: Whether to include PRM scores
            use_mcts: Whether to include MCTS values
            **kwargs: Additional arguments

        Returns:
            LLMOutput with synthetic TTC information
        """
        # Generate reasoning steps
        reasoning_steps = self._generate_reasoning_steps(query)

        # Generate PRM scores
        prm_scores = None
        if use_prm:
            prm_scores = self._generate_prm_scores(len(reasoning_steps))

        # Generate vote distribution
        vote_distribution = self._generate_vote_distribution(num_samples)

        # Generate MCTS values if enabled
        mcts_values = None
        mcts_tree = None
        if use_mcts or self._use_mcts:
            mcts_values, mcts_tree = self._generate_mcts_data()

        # Generate output
        output = self._generate_output(query)

        return LLMOutput(
            output=output,
            reasoning_steps=reasoning_steps,
            prm_scores=prm_scores,
            vote_distribution=vote_distribution,
            mcts_values=mcts_values,
            mcts_tree=mcts_tree,
            metadata={
                "query": query,
                "member_mode": self.member_mode,
            },
        )

    def get_prm_scores(
        self,
        reasoning_steps: List[str],
    ) -> List[float]:
        """Get mock PRM scores.

        Args:
            reasoning_steps: Reasoning steps

        Returns:
            Mock PRM scores
        """
        return self._generate_prm_scores(len(reasoning_steps))

    def uses_mcts(self) -> bool:
        """Check if using MCTS."""
        return self._use_mcts

    def _generate_reasoning_steps(self, query: str) -> List[str]:
        """Generate mock reasoning steps."""
        steps = []

        templates = [
            "Let me analyze the problem: {}",
            "First, I'll identify key components",
            "Applying the relevant formula/method",
            "Computing the intermediate result",
            "Therefore, the answer is",
        ]

        num_steps = min(self.num_reasoning_steps, len(templates))

        for i in range(num_steps):
            if i == 0:
                step = templates[i].format(query[:50])
            else:
                step = templates[i % len(templates)]

            # Add some randomness
            step += f" [step {i+1}]"
            steps.append(step)

        return steps

    def _generate_prm_scores(self, num_steps: int) -> List[float]:
        """Generate mock PRM scores based on member mode."""
        if self.member_mode:
            # Members: high scores, low variance
            base = 0.9
            noise = np.random.normal(0, 0.03, num_steps)
        else:
            # Non-members: moderate scores, higher variance
            base = 0.7
            noise = np.random.normal(0, 0.1, num_steps)

        scores = base + noise
        scores = np.clip(scores, 0, 1)

        return scores.tolist()

    def _generate_vote_distribution(self, num_samples: int) -> List[float]:
        """Generate mock vote distribution."""
        num_candidates = max(3, num_samples // 2)

        if self.member_mode:
            # Members: concentrated distribution
            dist = np.zeros(num_candidates)
            dist[0] = 0.8
            remaining = 0.2
            for i in range(1, num_candidates):
                dist[i] = remaining / (num_candidates - 1)
        else:
            # Non-members: more spread distribution
            dist = np.random.dirichlet(np.ones(num_candidates) * 2)

        return dist.tolist()

    def _generate_mcts_data(self) -> tuple:
        """Generate mock MCTS data."""
        num_nodes = np.random.randint(10, 30)

        # Generate values
        if self.member_mode:
            # Members: higher values
            values = np.random.beta(8, 2, num_nodes)
        else:
            # Non-members: centered values
            values = np.random.beta(5, 5, num_nodes)

        # Generate depths
        depths = []
        for i in range(num_nodes):
            depth = min(i // 3, 5)
            depths.append(depth)

        mcts_tree = {
            "depths": depths,
            "values": values.tolist(),
            "trajectories": [
                {"node_indices": list(range(0, num_nodes, 3))},
                {"node_indices": list(range(1, num_nodes, 3))},
            ],
            "outputs": ["Answer A", "Answer B"],
            "node_outputs": [f"Node {i} output" for i in range(num_nodes)],
        }

        return values.tolist(), mcts_tree

    def _generate_output(self, query: str) -> str:
        """Generate mock output."""
        # Simple deterministic output based on query
        if "+" in query:
            return "The result is 4"
        elif "?" in query:
            return "Yes, that is correct"
        else:
            return f"Analysis complete for: {query[:30]}..."
