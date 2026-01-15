"""
Test Configuration and Fixtures

Shared fixtures for SPARTAN test suite.
"""

import pytest
import numpy as np

from spartan.config import SPARTANConfig
from spartan.models.mock import MockReasoningLLM


@pytest.fixture
def config():
    """Default SPARTAN configuration."""
    return SPARTANConfig()


@pytest.fixture
def custom_config():
    """Custom configuration for testing."""
    return SPARTANConfig(
        prm_threshold=0.25,
        vote_threshold=0.35,
        mcts_threshold=0.45,
        epsilon_min=0.02,
        epsilon_max=0.6,
        learning_rate=0.005,
    )


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    return MockReasoningLLM(seed=42)


@pytest.fixture
def member_mock_llm():
    """Mock LLM simulating member behavior."""
    return MockReasoningLLM(member_mode=True, seed=42)


@pytest.fixture
def non_member_mock_llm():
    """Mock LLM simulating non-member behavior."""
    return MockReasoningLLM(member_mode=False, seed=42)


@pytest.fixture
def sample_prm_scores():
    """Sample PRM scores for testing."""
    return [0.85, 0.9, 0.88, 0.92, 0.87]


@pytest.fixture
def sample_reasoning_steps():
    """Sample reasoning steps for testing."""
    return [
        "Step 1: Let x = 5",
        "Step 2: Compute x + 3 = 8",
        "Step 3: Therefore y = 8",
        "Step 4: Verify: 5 + 3 = 8 ✓",
        "Step 5: The answer is 8",
    ]


@pytest.fixture
def sample_vote_distribution():
    """Sample vote distribution for testing."""
    return [0.7, 0.15, 0.1, 0.05]


@pytest.fixture
def concentrated_vote_distribution():
    """Concentrated vote distribution (member-like)."""
    return [0.95, 0.03, 0.01, 0.01]


@pytest.fixture
def uniform_vote_distribution():
    """Uniform vote distribution (non-member-like)."""
    return [0.25, 0.25, 0.25, 0.25]


@pytest.fixture
def sample_mcts_values():
    """Sample MCTS values for testing."""
    return [0.6, 0.65, 0.55, 0.7, 0.62, 0.58, 0.72, 0.61]


@pytest.fixture
def sample_mcts_tree():
    """Sample MCTS tree structure."""
    return {
        "depths": [0, 1, 1, 2, 2, 2, 3, 3],
        "values": [0.6, 0.65, 0.55, 0.7, 0.62, 0.58, 0.72, 0.61],
        "trajectories": [
            {"node_indices": [0, 1, 3, 6]},
            {"node_indices": [0, 2, 4, 7]},
        ],
        "outputs": ["Answer A", "Answer B"],
        "node_outputs": [f"Node {i}" for i in range(8)],
    }


@pytest.fixture
def sample_query():
    """Sample query for testing."""
    return "What is the integral of x^2 from 0 to 1?"


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return 42
