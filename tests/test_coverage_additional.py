"""
Additional Tests for Full Coverage

Tests for edge cases and uncovered lines in existing modules.
"""

import pytest
import numpy as np

from spartan.utils.noise import (
    gaussian_noise,
    laplace_noise,
    calibrated_noise,
    adaptive_noise,
    feature_selective_noise,
)
from spartan.utils.distributions import (
    kl_divergence,
    js_divergence,
    entropy,
    total_variation,
    hellinger_distance,
    wasserstein_distance,
    normalize_distribution,
)
from spartan.utils.metrics import (
    compute_auc_roc,
    compute_accuracy,
    compute_tpr_at_fpr,
    compute_f1_score,
    compute_precision_recall_curve,
)
from spartan.mplq.vote_leakage import VoteLeakageAnalyzer
from spartan.mplq.mcts_leakage import MCTSLeakageAnalyzer
from spartan.mplq.prm_leakage import PRMLeakageAnalyzer
from spartan.raas.mcts_defense import MCTSDefense
from spartan.raas.prm_defense import PRMDefense
from spartan.raas.vote_defense import VoteDefense
from spartan.rppo.bandit import UCBBandit
from spartan.rppo.pareto import ParetoFront
from spartan.models.base import BaseReasoningLLM, LLMOutput
from spartan.attacks.base import BaseAttack, AttackResult
from spartan.core import SPARTAN, SPARTANResult
from spartan.config import SPARTANConfig


class TestNoiseEdgeCases:
    """Additional tests for noise module."""

    def test_gaussian_noise_different_shapes(self):
        """Test Gaussian noise with different shapes."""
        noise_1d = gaussian_noise((10,), scale=1.0)
        assert noise_1d.shape == (10,)

        noise_2d = gaussian_noise((5, 5), scale=1.0)
        assert noise_2d.shape == (5, 5)

        noise_3d = gaussian_noise((2, 3, 4), scale=0.5)
        assert noise_3d.shape == (2, 3, 4)

    def test_laplace_noise_scale(self):
        """Test Laplace noise with different scales."""
        noise_small = laplace_noise((1000,), scale=0.1, seed=42)
        noise_large = laplace_noise((1000,), scale=1.0, seed=42)

        assert np.std(noise_small) < np.std(noise_large)

    def test_calibrated_noise_invalid_mechanism(self):
        """Test calibrated noise with invalid mechanism."""
        with pytest.raises(ValueError):
            calibrated_noise((10,), sensitivity=1.0, epsilon=0.1, mechanism="invalid")

    def test_adaptive_noise_high_risk(self):
        """Test adaptive noise with high risk."""
        noise = adaptive_noise((100,), risk_score=0.9, base_scale=1.0)
        assert noise.shape == (100,)
        # High risk should produce larger noise
        assert np.std(noise) > 0

    def test_adaptive_noise_low_risk(self):
        """Test adaptive noise with low risk."""
        noise = adaptive_noise((100,), risk_score=0.1, base_scale=1.0)
        assert noise.shape == (100,)

    def test_feature_selective_noise(self):
        """Test feature selective noise."""
        noise = feature_selective_noise(
            shape=(100,),
            nl_ratio=10.0,
            math_ratio=1.0,
            seed=42,
        )
        assert noise.shape == (100,)


class TestDistributionsEdgeCases:
    """Additional tests for distributions module."""

    def test_kl_divergence_zero_values(self):
        """Test KL divergence with near-zero values."""
        p = np.array([0.5, 0.5, 0.0001])
        q = np.array([0.33, 0.33, 0.34])
        p = p / p.sum()
        q = q / q.sum()

        kl = kl_divergence(p, q)
        assert np.isfinite(kl)

    def test_js_divergence_symmetric(self):
        """Test JS divergence is symmetric."""
        p = np.array([0.7, 0.2, 0.1])
        q = np.array([0.3, 0.4, 0.3])

        js_pq = js_divergence(p, q)
        js_qp = js_divergence(q, p)

        assert abs(js_pq - js_qp) < 1e-10

    def test_entropy_single_element(self):
        """Test entropy with single element."""
        dist = np.array([1.0])
        e = entropy(dist)
        assert e == 0.0

    def test_wasserstein_distance(self):
        """Test Wasserstein distance."""
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 0.0, 1.0])

        wd = wasserstein_distance(p, q)
        assert wd > 0

    def test_normalize_distribution(self):
        """Test distribution normalization."""
        unnorm = np.array([2.0, 3.0, 5.0])
        norm = normalize_distribution(unnorm)

        assert abs(norm.sum() - 1.0) < 1e-10


class TestMetricsEdgeCases:
    """Additional tests for metrics module."""

    def test_auc_roc_same_labels(self):
        """Test AUC-ROC with all same labels."""
        labels = [1, 1, 1, 1]
        scores = [0.1, 0.5, 0.7, 0.9]

        auc = compute_auc_roc(labels, scores)
        assert auc == 0.5  # Undefined case

    def test_accuracy_empty(self):
        """Test accuracy with empty lists."""
        acc = compute_accuracy([], [])
        assert acc == 0.0

    def test_tpr_at_fpr_edge_cases(self):
        """Test TPR@FPR with edge cases."""
        labels = [0, 0, 0, 0]
        scores = [0.1, 0.2, 0.3, 0.4]

        tpr = compute_tpr_at_fpr(labels, scores, fpr_threshold=0.5)
        assert tpr == 0.0  # No positives

    def test_f1_score_no_positives(self):
        """Test F1 score with no positive predictions."""
        labels = [1, 1, 1]
        predictions = [0, 0, 0]

        f1 = compute_f1_score(labels, predictions)
        assert f1 == 0.0

    def test_precision_recall_curve(self):
        """Test precision-recall curve computation."""
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.4, 0.6, 0.9]

        precisions, recalls, thresholds = compute_precision_recall_curve(
            labels, scores, num_thresholds=10
        )

        assert len(precisions) == 10
        assert len(recalls) == 10
        assert len(thresholds) == 10


class TestVoteLeakageEdgeCases:
    """Additional tests for vote leakage analyzer."""

    def test_uniform_distribution(self):
        """Test with perfectly uniform distribution."""
        analyzer = VoteLeakageAnalyzer()
        result = analyzer.analyze(vote_distribution=[0.25, 0.25, 0.25, 0.25])

        assert result["leakage_score"] < 0.1  # Low leakage for uniform

    def test_single_candidate(self):
        """Test with single candidate."""
        analyzer = VoteLeakageAnalyzer()
        result = analyzer.analyze(vote_distribution=[1.0])

        assert "leakage_score" in result

    def test_analyze_patterns(self):
        """Test pattern analysis."""
        analyzer = VoteLeakageAnalyzer()
        result = analyzer.analyze(
            vote_distribution=[0.6, 0.3, 0.1],
            analyze_patterns=True,
        )

        assert "leakage_score" in result


class TestMCTSLeakageEdgeCases:
    """Additional tests for MCTS leakage analyzer."""

    def test_with_deep_tree(self):
        """Test with deep MCTS tree."""
        analyzer = MCTSLeakageAnalyzer()
        tree = {
            "value": 0.5,
            "children": [
                {
                    "value": 0.6,
                    "children": [
                        {"value": 0.7, "children": []},
                        {"value": 0.8, "children": []},
                    ],
                },
            ],
        }

        result = analyzer.analyze(mcts_values=[0.5, 0.6, 0.7, 0.8], mcts_tree=tree)
        assert "leakage_score" in result

    def test_extreme_values(self):
        """Test with extreme value deviations."""
        analyzer = MCTSLeakageAnalyzer(baseline_mean=0.5, baseline_std=0.1)
        result = analyzer.analyze(mcts_values=[0.99, 0.98, 0.97])

        assert result["leakage_score"] > 0.5  # High leakage


class TestDefenseEdgeCases:
    """Additional tests for defense modules."""

    def test_prm_defense_no_nl_content(self):
        """Test PRM defense with no NL content."""
        defense = PRMDefense()
        result = defense.apply(
            reasoning_steps=["x = 5", "y = x + 3"],
            epsilon=0.5,
        )

        assert "sanitized_steps" in result

    def test_vote_defense_high_temperature(self):
        """Test vote defense with high temperature."""
        defense = VoteDefense()
        result = defense.apply(
            vote_distribution=[0.9, 0.05, 0.05],
            epsilon=0.9,
            use_temperature=True,
        )

        sanitized = result["sanitized_distribution"]
        # Should be more uniform after high temperature
        assert max(sanitized) < 0.9

    def test_mcts_defense_empty_tree(self):
        """Test MCTS defense with empty tree."""
        defense = MCTSDefense()
        result = defense.apply(
            mcts_tree={},
            epsilon=0.5,
        )

        assert result["defense_applied"] is False


class TestBanditEdgeCases:
    """Additional tests for UCB bandit."""

    def test_many_arms(self):
        """Test with many arms."""
        bandit = UCBBandit(num_arms=100, exploration_constant=2.0)

        # Pull each arm once
        for _ in range(100):
            arm = bandit.select_arm()
            bandit.update(arm, np.random.random())

        best = bandit.get_best_arm()
        assert 0 <= best < 100

    def test_zero_exploration(self):
        """Test with zero exploration constant."""
        bandit = UCBBandit(num_arms=5, exploration_constant=0.0)

        # With no exploration, should exploit immediately after initial pulls
        for i in range(5):
            bandit.update(i, i * 0.1)

        # Should select arm with highest mean
        selected = bandit.select_arm()
        assert selected == 4


class TestParetoEdgeCases:
    """Additional tests for Pareto front."""

    def test_many_objectives(self):
        """Test with many objectives."""
        pareto = ParetoFront(num_objectives=5)

        for i in range(10):
            point = np.random.random(5)
            pareto.add_point(point, {"id": i})

        front = pareto.get_front()
        assert len(front) > 0

    def test_identical_points(self):
        """Test with identical points."""
        pareto = ParetoFront(num_objectives=2)

        pareto.add_point(np.array([0.5, 0.5]), {"id": 1})
        pareto.add_point(np.array([0.5, 0.5]), {"id": 2})

        front = pareto.get_front()
        assert len(front) >= 1


class TestCoreEdgeCases:
    """Additional tests for core SPARTAN class."""

    def test_result_to_dict(self):
        """Test SPARTANResult to_dict method."""
        from spartan.mplq import MPLQResult
        from spartan.raas import RAASResult

        mplq_result = MPLQResult(
            total_risk=0.5,
            prm_leakage=0.3,
            vote_leakage=0.4,
            mcts_leakage=0.2,
            importance_weight=1.0,
            component_weights=(0.4, 0.35, 0.25),
        )

        raas_result = RAASResult(
            sanitized_output="test",
            original_output="test",
            defense_applied=True,
            epsilon_used=0.1,
        )

        result = SPARTANResult(
            output="test",
            original_output="test",
            risk_score=0.5,
            risk_analysis=mplq_result,
            defense_result=raas_result,
        )

        d = result.to_dict()
        assert "output" in d
        assert "risk_score" in d

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "mplq": {"prm_threshold": 0.4},
            "raas": {"epsilon_max": 0.7},
            "rppo": {"learning_rate": 0.005},
        }

        config = SPARTANConfig.from_dict(config_dict)
        assert config.prm_threshold == 0.4
        assert config.epsilon_max == 0.7


class TestBaseModelAbstract:
    """Tests for abstract base model."""

    def test_base_llm_abstract(self):
        """Test that BaseReasoningLLM is abstract."""
        with pytest.raises(TypeError):
            BaseReasoningLLM()

    def test_base_attack_abstract(self):
        """Test that BaseAttack is abstract."""
        with pytest.raises(TypeError):
            BaseAttack()
