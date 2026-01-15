"""
Final Push to 100% Coverage

Targeting exact uncovered lines in sanitizer.py, vote_defense.py, mcts_defense.py
"""

import numpy as np
import pytest

from spartan.mplq.analyzer import MPLQResult
from spartan.raas.mcts_defense import MCTSDefense
from spartan.raas.sanitizer import RAAS
from spartan.raas.vote_defense import VoteDefense


# ============== sanitizer.py uncovered lines ==============
class TestSanitizerUncovered:
    """Tests for sanitizer.py uncovered lines."""

    def test_update_params_submodules(self):
        """Test update_params for sub-module parameters - lines 267, 269, 271."""
        raas = RAAS()

        # Update sub-module params
        raas.update_params(
            {
                "prm_noise_scale": 2.0,
                "temperature_base": 1.5,
                "mcts_depth_scale": 0.2,
            }
        )

        params = raas.get_params()
        assert params["prm_noise_scale"] == 2.0
        assert params["temperature_base"] == 1.5
        assert params["mcts_depth_scale"] == 0.2

    def test_update_params_thresholds(self):
        """Test update_params with thresholds dict."""
        raas = RAAS()

        raas.update_params(
            {
                "thresholds": {"prm": 0.5, "vote": 0.6},
            }
        )

        params = raas.get_params()
        assert params["thresholds"]["prm"] == 0.5
        assert params["thresholds"]["vote"] == 0.6


# ============== vote_defense.py uncovered lines ==============
class TestVoteDefenseUncovered:
    """Tests for vote_defense.py uncovered lines."""

    def test_apply_zero_vote_sum(self):
        """Test when vote_sum is zero - line 84."""
        defense = VoteDefense()
        result = defense.apply(
            vote_distribution=[0.0, 0.0, 0.0],
            epsilon=0.5,
            vote_leakage=0.7,
            threshold=0.3,
        )
        assert result["applied"] == True
        # Should normalize to uniform distribution
        assert len(result["flattened_distribution"]) == 3

    def test_temperature_at_threshold(self):
        """Test temperature when leakage equals threshold - line 168."""
        defense = VoteDefense(temperature_base=1.0)
        result = defense.apply(
            vote_distribution=[0.5, 0.3, 0.2],
            epsilon=0.5,
            vote_leakage=0.3,  # Equal to threshold
            threshold=0.3,
        )
        assert result["temperature_used"] == 1.0  # Should return temperature_base

    def test_temperature_below_threshold(self):
        """Test temperature when leakage is below threshold."""
        defense = VoteDefense(temperature_base=1.0)
        result = defense.apply(
            vote_distribution=[0.5, 0.3, 0.2],
            epsilon=0.5,
            vote_leakage=0.1,  # Below threshold
            threshold=0.3,
        )
        assert result["temperature_used"] == 1.0

    def test_temperature_zero_or_negative(self):
        """Test _temperature_flatten with temperature <= 0 - line 194."""
        defense = VoteDefense()
        dist = np.array([0.6, 0.3, 0.1])

        # Call _temperature_flatten directly with temperature=0
        result = defense._temperature_flatten(dist, temperature=0)

        # Should return original distribution unchanged
        assert np.allclose(result, dist)

    def test_implicit_resample_empty_candidates(self):
        """Test _implicit_reward_resample with empty candidates."""
        defense = VoteDefense(use_implicit_rewards=True)
        result = defense.apply(
            vote_distribution=[0.8, 0.2],
            epsilon=0.5,
            vote_leakage=0.7,
            threshold=0.3,
            candidate_outputs=[],  # Empty
        )
        # Should handle gracefully
        assert result["applied"] == True

    def test_compute_entropy_all_zeros(self):
        """Test _compute_entropy when all values are zero."""
        defense = VoteDefense()
        dist = np.array([0.0, 0.0, 0.0])
        entropy = defense._compute_entropy(dist)
        assert entropy == 0.0

    def test_implicit_rewards_short_candidate(self):
        """Test implicit rewards with short candidate (< 10 chars)."""
        defense = VoteDefense(use_implicit_rewards=True)
        result = defense.apply(
            vote_distribution=[0.5, 0.5],
            epsilon=0.5,
            vote_leakage=0.7,
            threshold=0.3,
            candidate_outputs=["Hi", "Hello there, this is a longer response."],
        )
        assert "resampled_output" in result

    def test_implicit_rewards_long_candidate(self):
        """Test implicit rewards with long candidate (> 500 chars)."""
        defense = VoteDefense(use_implicit_rewards=True)
        long_text = "A" * 600
        result = defense.apply(
            vote_distribution=[0.5, 0.5],
            epsilon=0.5,
            vote_leakage=0.7,
            threshold=0.3,
            candidate_outputs=[long_text, "Short answer."],
        )
        assert "resampled_output" in result

    def test_implicit_rewards_no_punctuation(self):
        """Test implicit rewards with candidate not ending in punctuation."""
        defense = VoteDefense(use_implicit_rewards=True)
        result = defense.apply(
            vote_distribution=[0.5, 0.5],
            epsilon=0.5,
            vote_leakage=0.7,
            threshold=0.3,
            candidate_outputs=["No punctuation here", "Has punctuation!"],
        )
        assert "resampled_output" in result


# ============== mcts_defense.py uncovered lines ==============
class TestMCTSDefenseUncovered:
    """Tests for mcts_defense.py uncovered lines."""

    def test_simulate_rerun_trajectory_as_list(self):
        """Test _simulate_rerun with trajectory as plain list - lines 175-178."""
        defense = MCTSDefense(rerun_search=True)

        mcts_tree = {
            "values": [0.9, 0.8, 0.7, 0.6],
            "depths": [0, 1, 1, 2],
            "trajectories": [
                [0, 1, 3],  # Plain list of indices (not dict)
                [0, 2],
            ],
            "outputs": ["Answer A", "Answer B"],
        }

        result = defense.apply(
            mcts_tree=mcts_tree,
            epsilon=0.5,
            mcts_leakage=0.7,
            threshold=0.3,
        )

        assert result["applied"] == True
        assert "rerun_result" in result

    def test_simulate_rerun_invalid_trajectory(self):
        """Test _simulate_rerun with invalid trajectory format - line 185."""
        defense = MCTSDefense(rerun_search=True)

        mcts_tree = {
            "values": [0.9, 0.8],
            "trajectories": [
                "invalid_trajectory",  # Neither dict nor list with valid format
                123,  # Also invalid
            ],
        }

        result = defense.apply(
            mcts_tree=mcts_tree,
            epsilon=0.5,
            mcts_leakage=0.7,
            threshold=0.3,
        )

        assert result["applied"] == True

    def test_simulate_rerun_empty_valid_indices(self):
        """Test when valid_indices is empty - line 185."""
        defense = MCTSDefense(rerun_search=True)

        mcts_tree = {
            "values": [0.9],
            "trajectories": [
                {"node_indices": [10, 20, 30]},  # All indices out of range
            ],
        }

        result = defense.apply(
            mcts_tree=mcts_tree,
            epsilon=0.5,
            mcts_leakage=0.7,
            threshold=0.3,
        )

        assert result["applied"] == True

    def test_simulate_rerun_with_node_outputs(self):
        """Test _simulate_rerun with node_outputs - line 204."""
        defense = MCTSDefense(rerun_search=True)

        mcts_tree = {
            "values": [0.9, 0.8, 0.95],
            "node_outputs": ["Output 0", "Output 1", "Output 2"],
            # No trajectories - will fall back to best node selection
        }

        result = defense.apply(
            mcts_tree=mcts_tree,
            epsilon=0.5,
            mcts_leakage=0.7,
            threshold=0.3,
        )

        assert result["applied"] == True
        assert "rerun_result" in result
        # Should have selected from node_outputs
        if "output" in result["rerun_result"]:
            assert result["rerun_result"]["output"] in ["Output 0", "Output 1", "Output 2"]

    def test_simulate_rerun_no_outputs_available(self):
        """Test _simulate_rerun when no outputs are available."""
        defense = MCTSDefense(rerun_search=True)

        mcts_tree = {
            "values": [0.9, 0.8],
            # No trajectories, no outputs, no node_outputs
        }

        result = defense.apply(
            mcts_tree=mcts_tree,
            epsilon=0.5,
            mcts_leakage=0.7,
            threshold=0.3,
        )

        assert result["applied"] == True
        assert "best_node_index" in result["rerun_result"]


# Last updated: 2026-01-15
