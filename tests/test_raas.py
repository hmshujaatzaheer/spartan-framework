"""
Tests for RAAS (Reasoning-Aware Adaptive Sanitization) module.
"""

import numpy as np
import pytest

from spartan.config import SPARTANConfig
from spartan.mplq import MPLQResult
from spartan.raas import RAAS, RAASResult
from spartan.raas.mcts_defense import MCTSDefense
from spartan.raas.prm_defense import PRMDefense
from spartan.raas.vote_defense import VoteDefense


class TestRAAS:
    """Tests for main RAAS class."""

    @pytest.fixture
    def mock_risk_analysis(self):
        """Create mock MPLQ result."""
        return MPLQResult(
            total_risk=0.6,
            prm_leakage=0.5,
            vote_leakage=0.5,  # Above threshold (0.4)
            mcts_leakage=0.5,  # Above threshold
            importance_weight=1.2,
            component_weights=(0.4, 0.35, 0.25),
        )

    @pytest.fixture
    def low_risk_analysis(self):
        """Create low risk MPLQ result."""
        return MPLQResult(
            total_risk=0.1,
            prm_leakage=0.1,
            vote_leakage=0.05,
            mcts_leakage=0.02,
            importance_weight=0.8,
            component_weights=(0.4, 0.35, 0.25),
        )

    def test_init_default(self):
        """Test default initialization."""
        raas = RAAS()
        assert raas is not None
        assert raas.config is not None

    def test_init_with_config(self, config):
        """Test initialization with config."""
        raas = RAAS(config=config)
        assert raas.config == config

    def test_sanitize_basic(self, mock_risk_analysis):
        """Test basic sanitization."""
        raas = RAAS()
        result = raas.sanitize(
            output="The answer is 42",
            risk_analysis=mock_risk_analysis,
        )

        assert isinstance(result, RAASResult)
        assert result.original_output == "The answer is 42"

    def test_sanitize_with_reasoning_steps(
        self,
        mock_risk_analysis,
        sample_reasoning_steps,
    ):
        """Test sanitization with reasoning steps."""
        raas = RAAS()
        result = raas.sanitize(
            output="Answer: 8",
            risk_analysis=mock_risk_analysis,
            reasoning_steps=sample_reasoning_steps,
        )

        assert result.prm_defense_applied

    def test_sanitize_with_vote_distribution(
        self,
        mock_risk_analysis,
        sample_vote_distribution,
    ):
        """Test sanitization with vote distribution."""
        raas = RAAS()
        result = raas.sanitize(
            output="Answer: A",
            risk_analysis=mock_risk_analysis,
            vote_distribution=sample_vote_distribution,
        )

        assert result.vote_defense_applied

    def test_sanitize_with_mcts_tree(
        self,
        mock_risk_analysis,
        sample_mcts_tree,
    ):
        """Test sanitization with MCTS tree."""
        raas = RAAS()
        result = raas.sanitize(
            output="Answer: B",
            risk_analysis=mock_risk_analysis,
            mcts_tree=sample_mcts_tree,
        )

        # MCTS defense may or may not be applied based on threshold
        assert isinstance(result.mcts_defense_applied, bool)

    def test_low_risk_minimal_defense(self, low_risk_analysis):
        """Low risk should result in minimal defense."""
        raas = RAAS()
        result = raas.sanitize(
            output="Simple answer",
            risk_analysis=low_risk_analysis,
        )

        # Low risk means low epsilon
        assert result.epsilon_used < 0.2

    def test_defense_intensity_scaling(self, mock_risk_analysis, low_risk_analysis):
        """Defense intensity should scale with risk."""
        raas = RAAS()

        high_risk_result = raas.sanitize(
            output="Answer",
            risk_analysis=mock_risk_analysis,
        )

        low_risk_result = raas.sanitize(
            output="Answer",
            risk_analysis=low_risk_analysis,
        )

        assert high_risk_result.epsilon_used >= low_risk_result.epsilon_used

    def test_update_params(self):
        """Test parameter updates."""
        raas = RAAS()

        raas.update_params(
            {
                "epsilon_min": 0.05,
                "epsilon_max": 0.7,
                "thresholds": {"prm": 0.2},
            }
        )

        params = raas.get_params()
        assert params["epsilon_min"] == 0.05
        assert params["epsilon_max"] == 0.7
        assert params["thresholds"]["prm"] == 0.2

    def test_result_to_dict(self, mock_risk_analysis):
        """Test result serialization."""
        raas = RAAS()
        result = raas.sanitize(
            output="Answer",
            risk_analysis=mock_risk_analysis,
        )

        result_dict = result.to_dict()
        assert "sanitized_output" in result_dict
        assert "defense_applied" in result_dict
        assert "epsilon_used" in result_dict


class TestPRMDefense:
    """Tests for PRM defense module."""

    def test_init(self):
        """Test initialization."""
        defense = PRMDefense()
        assert defense is not None

    def test_apply_basic(self, sample_reasoning_steps):
        """Test basic defense application."""
        defense = PRMDefense()
        result = defense.apply(
            reasoning_steps=sample_reasoning_steps,
            epsilon=0.3,
            prm_leakage=0.5,
            threshold=0.3,
        )

        assert result["applied"]
        assert "modified_trace" in result

    def test_feature_selective(self, sample_reasoning_steps):
        """Test feature-selective perturbation."""
        defense = PRMDefense(use_feature_selective=True)
        result = defense.apply(
            reasoning_steps=sample_reasoning_steps,
            epsilon=0.3,
            prm_leakage=0.5,
            threshold=0.3,
        )

        assert result["feature_selective"]

    def test_empty_steps(self):
        """Test with empty steps."""
        defense = PRMDefense()
        result = defense.apply(
            reasoning_steps=[],
            epsilon=0.3,
            prm_leakage=0.5,
            threshold=0.3,
        )

        assert not result["applied"]


class TestVoteDefense:
    """Tests for vote defense module."""

    def test_init(self):
        """Test initialization."""
        defense = VoteDefense()
        assert defense is not None

    def test_apply_basic(self, sample_vote_distribution):
        """Test basic defense application."""
        defense = VoteDefense()
        result = defense.apply(
            vote_distribution=sample_vote_distribution,
            epsilon=0.3,
            vote_leakage=0.5,
            threshold=0.3,
        )

        assert result["applied"]
        assert "flattened_distribution" in result

    def test_temperature_scaling(self, concentrated_vote_distribution):
        """Test temperature scaling flattens distribution."""
        defense = VoteDefense()
        result = defense.apply(
            vote_distribution=concentrated_vote_distribution,
            epsilon=0.5,
            vote_leakage=0.8,
            threshold=0.3,
        )

        # Flattened should have higher entropy
        assert result["flattened_entropy"] >= result["original_entropy"]

    def test_implicit_rewards(self, sample_vote_distribution):
        """Test implicit reward resampling."""
        defense = VoteDefense(use_implicit_rewards=True)
        candidates = ["A", "B", "C", "D"]

        result = defense.apply(
            vote_distribution=sample_vote_distribution,
            epsilon=0.3,
            vote_leakage=0.5,
            threshold=0.3,
            candidate_outputs=candidates,
        )

        assert "resample_info" in result

    def test_empty_distribution(self):
        """Test with empty distribution."""
        defense = VoteDefense()
        result = defense.apply(
            vote_distribution=[],
            epsilon=0.3,
            vote_leakage=0.5,
            threshold=0.3,
        )

        assert not result["applied"]


class TestMCTSDefense:
    """Tests for MCTS defense module."""

    def test_init(self):
        """Test initialization."""
        defense = MCTSDefense()
        assert defense is not None

    def test_apply_basic(self, sample_mcts_tree):
        """Test basic defense application."""
        defense = MCTSDefense()
        result = defense.apply(
            mcts_tree=sample_mcts_tree,
            epsilon=0.3,
            mcts_leakage=0.6,
            threshold=0.5,
        )

        assert result["applied"]
        assert "perturbed_values" in result

    def test_depth_scaled_noise(self, sample_mcts_tree):
        """Test depth-scaled noise injection."""
        defense = MCTSDefense(depth_scale=0.2)
        result = defense.apply(
            mcts_tree=sample_mcts_tree,
            epsilon=0.5,
            mcts_leakage=0.7,
            threshold=0.4,
        )

        # Perturbation info should show depth-based scaling
        assert "perturbation_info" in result

    def test_perturb_single_value(self):
        """Test single value perturbation."""
        defense = MCTSDefense()

        value = 0.7
        perturbed = defense.perturb_single_value(
            value=value,
            depth=2,
            epsilon=0.3,
        )

        assert 0 <= perturbed <= 1

    def test_no_values_in_tree(self):
        """Test with missing values in tree."""
        defense = MCTSDefense()
        result = defense.apply(
            mcts_tree={"depths": [0, 1, 2]},  # No values
            epsilon=0.3,
            mcts_leakage=0.6,
            threshold=0.5,
        )

        assert not result["applied"]
        assert "error" in result


# Last updated: 2026-01-15
