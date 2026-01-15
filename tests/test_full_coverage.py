"""
Final Coverage Tests

Tests to achieve 100% code coverage for all remaining uncovered lines.
"""

import numpy as np
import pytest

from spartan.attacks.base import AttackResult
from spartan.attacks.mvna import MVNAAttack
from spartan.attacks.nlba import NLBAAttack
from spartan.attacks.smva import SMVAAttack
from spartan.config import SPARTANConfig
from spartan.models.mock import MockReasoningLLM
from spartan.mplq.mcts_leakage import MCTSLeakageAnalyzer
from spartan.mplq.prm_leakage import PRMLeakageAnalyzer
from spartan.mplq.vote_leakage import VoteLeakageAnalyzer
from spartan.raas.mcts_defense import MCTSDefense
from spartan.raas.prm_defense import PRMDefense
from spartan.raas.vote_defense import VoteDefense
from spartan.rppo.bandit import UCBBandit
from spartan.rppo.pareto import ParetoFront
from spartan.utils.noise import (
    calibrated_noise,
    gaussian_noise,
    laplace_noise,
)


# ============== noise.py coverage ==============
class TestNoiseFull:
    """Full coverage tests for noise module."""

    def test_gaussian_with_seed(self):
        """Test Gaussian noise with seed for reproducibility."""
        n1 = gaussian_noise((10,), scale=1.0, seed=42)
        n2 = gaussian_noise((10,), scale=1.0, seed=42)
        assert np.allclose(n1, n2)

    def test_laplace_with_seed(self):
        """Test Laplace noise with seed."""
        n1 = laplace_noise((10,), scale=1.0, seed=42)
        n2 = laplace_noise((10,), scale=1.0, seed=42)
        assert np.allclose(n1, n2)

    def test_calibrated_laplace_with_delta(self):
        """Test calibrated noise with delta parameter."""
        noise = calibrated_noise(
            (50,), sensitivity=1.0, epsilon=0.5, delta=0.01, mechanism="laplace"
        )
        assert noise.shape == (50,)

    def test_calibrated_gaussian_with_delta(self):
        """Test calibrated Gaussian with delta."""
        noise = calibrated_noise(
            (50,), sensitivity=1.0, epsilon=0.5, delta=0.01, mechanism="gaussian"
        )
        assert noise.shape == (50,)

    def test_calibrated_with_seed(self):
        """Test calibrated noise with seed."""
        n1 = calibrated_noise((10,), sensitivity=1.0, epsilon=0.5, mechanism="laplace", seed=42)
        n2 = calibrated_noise((10,), sensitivity=1.0, epsilon=0.5, mechanism="laplace", seed=42)
        assert np.allclose(n1, n2)


# ============== vote_leakage.py coverage ==============
class TestVoteLeakageFull:
    """Full coverage tests for vote leakage."""

    def test_empty_distribution(self):
        """Test with empty distribution."""
        analyzer = VoteLeakageAnalyzer()
        result = analyzer.analyze(vote_distribution=[])
        assert result["leakage_score"] == 0.0

    def test_all_zeros(self):
        """Test with all zero votes."""
        analyzer = VoteLeakageAnalyzer()
        result = analyzer.analyze(vote_distribution=[0, 0, 0])
        assert "leakage_score" in result

    def test_negative_values(self):
        """Test handling negative values."""
        analyzer = VoteLeakageAnalyzer()
        result = analyzer.analyze(vote_distribution=[-0.1, 0.5, 0.6])
        assert "leakage_score" in result

    def test_very_concentrated(self):
        """Test with very concentrated distribution."""
        analyzer = VoteLeakageAnalyzer()
        result = analyzer.analyze(vote_distribution=[0.99, 0.005, 0.005])
        assert result["leakage_score"] > 0.8

    def test_two_candidates(self):
        """Test with only two candidates."""
        analyzer = VoteLeakageAnalyzer()
        result = analyzer.analyze(vote_distribution=[0.7, 0.3])
        assert "leakage_score" in result

    def test_many_candidates(self):
        """Test with many candidates."""
        analyzer = VoteLeakageAnalyzer()
        dist = [1.0 / 20] * 20
        result = analyzer.analyze(vote_distribution=dist)
        assert result["leakage_score"] < 0.1


# ============== mcts_leakage.py coverage ==============
class TestMCTSLeakageFull:
    """Full coverage tests for MCTS leakage."""

    def test_empty_values(self):
        """Test with empty values."""
        analyzer = MCTSLeakageAnalyzer()
        result = analyzer.analyze(mcts_values=[])
        assert result["leakage_score"] == 0.0

    def test_single_value(self):
        """Test with single value."""
        analyzer = MCTSLeakageAnalyzer()
        result = analyzer.analyze(mcts_values=[0.5])
        assert "leakage_score" in result

    def test_negative_values(self):
        """Test with negative values."""
        analyzer = MCTSLeakageAnalyzer()
        result = analyzer.analyze(mcts_values=[-0.5, 0.5, 1.0])
        assert "leakage_score" in result

    def test_tree_traversal(self):
        """Test deep tree traversal."""
        analyzer = MCTSLeakageAnalyzer()
        tree = {
            "value": 0.5,
            "visit_count": 10,
            "children": [
                {
                    "value": 0.6,
                    "visit_count": 5,
                    "children": [
                        {"value": 0.7, "visit_count": 2, "children": []},
                    ],
                },
            ],
        }
        result = analyzer.analyze(mcts_values=[0.5, 0.6, 0.7], mcts_tree=tree)
        assert "leakage_score" in result


# ============== prm_leakage.py coverage ==============
class TestPRMLeakageFull:
    """Full coverage tests for PRM leakage."""

    def test_empty_scores(self):
        """Test with empty scores."""
        analyzer = PRMLeakageAnalyzer()
        result = analyzer.analyze(prm_scores=[])
        assert result["leakage_score"] == 0.0

    def test_single_score(self):
        """Test with single score."""
        analyzer = PRMLeakageAnalyzer()
        result = analyzer.analyze(prm_scores=[0.9])
        assert "leakage_score" in result

    def test_all_same_scores(self):
        """Test with identical scores."""
        analyzer = PRMLeakageAnalyzer()
        result = analyzer.analyze(prm_scores=[0.5, 0.5, 0.5])
        assert "leakage_score" in result

    def test_with_reasoning_steps(self):
        """Test with reasoning steps provided."""
        analyzer = PRMLeakageAnalyzer()
        result = analyzer.analyze(
            prm_scores=[0.9, 0.8, 0.85],
            reasoning_steps=["Step 1", "Step 2", "Step 3"],
        )
        assert "leakage_score" in result


# ============== mcts_defense.py coverage ==============
class TestMCTSDefenseFull:
    """Full coverage tests for MCTS defense."""

    def test_empty_tree(self):
        """Test with empty tree - no values key."""
        defense = MCTSDefense()
        result = defense.apply(
            mcts_tree={},
            mcts_leakage=0.5,
            threshold=0.3,
            epsilon=0.5,
        )
        assert result["applied"] is False
        assert result["error"] == "no_values_in_tree"

    def test_tree_with_values_list(self):
        """Test with tree containing values list."""
        defense = MCTSDefense()
        tree = {"values": [0.8, 0.7, 0.9, 0.6]}
        result = defense.apply(
            mcts_tree=tree,
            mcts_leakage=0.6,
            threshold=0.3,
            epsilon=0.5,
        )
        assert result["applied"] is True
        assert "perturbed_values" in result

    def test_tree_with_depths(self):
        """Test with tree containing depths."""
        defense = MCTSDefense()
        tree = {
            "values": [0.8, 0.7, 0.9],
            "depths": [0, 1, 2],
        }
        result = defense.apply(
            mcts_tree=tree,
            mcts_leakage=0.6,
            threshold=0.3,
            epsilon=0.5,
        )
        assert result["applied"] is True
        assert result["num_nodes"] == 3

    def test_perturb_single_value(self):
        """Test single value perturbation method."""
        defense = MCTSDefense()
        perturbed = defense.perturb_single_value(
            value=0.8,
            depth=2,
            epsilon=0.5,
        )
        assert 0 <= perturbed <= 1


# ============== prm_defense.py coverage ==============
class TestPRMDefenseFull:
    """Full coverage tests for PRM defense."""

    def test_empty_steps(self):
        """Test with empty reasoning steps."""
        defense = PRMDefense()
        result = defense.apply(
            reasoning_steps=[],
            prm_leakage=0.5,
            threshold=0.3,
            epsilon=0.5,
        )
        assert result["applied"] is False

    def test_with_steps(self):
        """Test with reasoning steps - always applies."""
        defense = PRMDefense()
        result = defense.apply(
            reasoning_steps=["Step 1: Calculate", "Step 2: Verify"],
            prm_leakage=0.9,
            threshold=0.3,
            epsilon=0.8,
        )
        assert result["applied"] is True
        assert "modified_trace" in result

    def test_feature_selective_off(self):
        """Test with feature-selective disabled."""
        defense = PRMDefense(use_feature_selective=False)
        result = defense.apply(
            reasoning_steps=["Calculate x = 5", "Then y = x + 3"],
            prm_leakage=0.8,
            threshold=0.3,
            epsilon=0.9,
        )
        assert result["applied"] is True
        assert result["feature_selective"] is False


# ============== vote_defense.py coverage ==============
class TestVoteDefenseFull:
    """Full coverage tests for vote defense."""

    def test_empty_distribution(self):
        """Test with empty distribution."""
        defense = VoteDefense()
        result = defense.apply(
            vote_distribution=[],
            vote_leakage=0.5,
            threshold=0.3,
            epsilon=0.5,
        )
        assert result["applied"] is False

    def test_with_distribution(self):
        """Test with distribution - always applies."""
        defense = VoteDefense()
        result = defense.apply(
            vote_distribution=[0.9, 0.05, 0.05],
            vote_leakage=0.9,
            threshold=0.3,
            epsilon=0.7,
        )
        assert result["applied"] is True
        assert "flattened_distribution" in result

    def test_with_candidates(self):
        """Test with candidate outputs for resampling."""
        defense = VoteDefense(use_implicit_rewards=True)
        result = defense.apply(
            vote_distribution=[0.8, 0.15, 0.05],
            vote_leakage=0.7,
            threshold=0.3,
            epsilon=0.5,
            candidate_outputs=["Answer A", "Answer B", "Answer C"],
        )
        assert result["applied"] is True
        assert "resampled_output" in result

    def test_low_leakage_temperature(self):
        """Test temperature calculation with low leakage."""
        defense = VoteDefense()
        result = defense.apply(
            vote_distribution=[0.5, 0.5],
            vote_leakage=0.1,  # Below threshold
            threshold=0.5,
            epsilon=0.5,
        )
        # Temperature should be base (1.0) when leakage <= threshold
        assert result["temperature_used"] == 1.0


# ============== pareto.py coverage ==============
class TestParetoFull:
    """Full coverage tests for Pareto front."""

    def test_empty_front(self):
        """Test operations on empty front."""
        pareto = ParetoFront()
        front = pareto.get_front()
        assert len(front) == 0

    def test_clear(self):
        """Test clearing the front."""
        pareto = ParetoFront()
        pareto.add_point(np.array([0.5, 0.5]), {"id": 1})
        pareto.clear()
        assert len(pareto.get_front()) == 0

    def test_get_hypervolume_empty(self):
        """Test hypervolume on empty front."""
        pareto = ParetoFront()
        hv = pareto.get_hypervolume()
        assert hv == 0.0

    def test_get_hypervolume_with_reference(self):
        """Test hypervolume with reference point."""
        pareto = ParetoFront()
        pareto.add_point(np.array([0.8, 0.8]), {"id": 1})
        hv = pareto.get_hypervolume(reference_point=np.array([0, 0]))
        assert hv >= 0

    def test_get_closest_to_ideal_empty(self):
        """Test closest to ideal on empty front."""
        pareto = ParetoFront()
        closest = pareto.get_closest_to_ideal()
        assert closest is None

    def test_get_closest_to_ideal_with_point(self):
        """Test finding closest point to ideal."""
        pareto = ParetoFront()
        pareto.add_point(np.array([0.9, 0.9]), {"id": 1})
        pareto.add_point(np.array([0.5, 0.5]), {"id": 2})
        closest = pareto.get_closest_to_ideal(ideal_point=np.array([1, 1]))
        assert closest is not None
        assert closest[1]["id"] == 1  # 0.9,0.9 is closer to 1,1

    def test_dominates(self):
        """Test dominance checking."""
        pareto = ParetoFront()
        assert pareto._dominates(np.array([0.9, 0.9]), np.array([0.5, 0.5]))
        assert not pareto._dominates(np.array([0.5, 0.5]), np.array([0.9, 0.9]))

    def test_is_pareto_optimal(self):
        """Test pareto optimality check."""
        pareto = ParetoFront()
        pareto.add_point(np.array([0.8, 0.8]), {"id": 1})
        assert pareto.is_pareto_optimal(np.array([0.9, 0.9])) is True
        assert pareto.is_pareto_optimal(np.array([0.5, 0.5])) is False

    def test_size(self):
        """Test size method."""
        pareto = ParetoFront()
        assert pareto.size() == 0
        pareto.add_point(np.array([0.8, 0.8]), {"id": 1})
        assert pareto.size() == 1


# ============== bandit.py coverage ==============
class TestBanditFull:
    """Full coverage tests for UCB bandit."""

    def test_all_arms_explored(self):
        """Test that all arms get explored initially."""
        bandit = UCBBandit(num_arms=5)
        selected = set()
        for _ in range(5):
            arm = bandit.select_arm()
            bandit.update(arm, np.random.random())
            selected.add(arm)
        assert len(selected) == 5

    def test_get_arm_stats(self):
        """Test getting arm statistics."""
        bandit = UCBBandit(num_arms=3)
        for i in range(3):
            bandit.update(i, 0.5 + i * 0.1)
        stats = bandit.get_arm_stats()
        assert len(stats) == 3
        assert "average_reward" in stats[0]

    def test_get_exploration_rate(self):
        """Test exploration rate computation."""
        bandit = UCBBandit(num_arms=3)
        rate = bandit.get_exploration_rate()
        assert rate == 1.0  # All exploration initially

        for i in range(3):
            bandit.update(i, 0.5)
        rate = bandit.get_exploration_rate()
        assert 0 <= rate <= 1

    def test_get_current_arm(self):
        """Test current arm getter."""
        bandit = UCBBandit(num_arms=3)
        arm = bandit.select_arm()
        assert bandit.get_current_arm() == arm


# ============== attacks coverage ==============
class TestAttacksFull:
    """Full coverage tests for attacks."""

    def test_nlba_execute(self):
        """Test NLBA attack execute."""
        attack = NLBAAttack()
        model = MockReasoningLLM()
        result = attack.execute(
            query="Test query",
            target_model=model,
            is_member=True,
        )
        assert isinstance(result, AttackResult)

    def test_smva_execute(self):
        """Test SMVA attack execute."""
        attack = SMVAAttack()
        model = MockReasoningLLM()
        result = attack.execute(
            query="Test query",
            target_model=model,
            is_member=True,
        )
        assert isinstance(result, AttackResult)

    def test_mvna_execute(self):
        """Test MVNA attack execute."""
        attack = MVNAAttack()
        model = MockReasoningLLM()
        result = attack.execute(
            query="Test query",
            target_model=model,
            is_member=True,
        )
        assert isinstance(result, AttackResult)

    def test_attack_result_str(self):
        """Test AttackResult string representation."""
        result = AttackResult(
            success_score=0.8,
            membership_prediction=True,
            confidence=0.9,
        )
        s = str(result)
        assert len(s) > 0


# ============== config coverage ==============
class TestConfigFull:
    """Full coverage tests for config."""

    def test_config_to_dict(self):
        """Test config serialization."""
        config = SPARTANConfig()
        d = config.to_dict()
        assert "mplq" in d
        assert "raas" in d
        assert "rppo" in d

    def test_config_from_dict(self):
        """Test loading config from dict."""
        d = {
            "mplq": {"prm_threshold": 0.5},
            "raas": {"epsilon_min": 0.05},
        }
        config = SPARTANConfig.from_dict(d)
        assert config.prm_threshold == 0.5

    def test_config_defaults(self):
        """Test config default values."""
        config = SPARTANConfig()
        assert config.epsilon_min >= 0
        assert config.epsilon_max <= 1
        assert config.prm_threshold >= 0
