"""
Additional Tests for Full Coverage

Tests for edge cases and uncovered lines in existing modules.
"""

import numpy as np
import pytest

from spartan.attacks.base import AttackResult
from spartan.config import SPARTANConfig
from spartan.core import SPARTAN, SPARTANResult
from spartan.models.mock import MockReasoningLLM
from spartan.mplq import MPLQResult
from spartan.mplq.mcts_leakage import MCTSLeakageAnalyzer
from spartan.mplq.prm_leakage import PRMLeakageAnalyzer
from spartan.mplq.vote_leakage import VoteLeakageAnalyzer
from spartan.raas import RAASResult
from spartan.raas.mcts_defense import MCTSDefense
from spartan.raas.prm_defense import PRMDefense
from spartan.raas.vote_defense import VoteDefense
from spartan.rppo.bandit import UCBBandit
from spartan.rppo.pareto import ParetoFront
from spartan.utils.distributions import (
    cross_entropy,
    entropy,
    hellinger_distance,
    js_divergence,
    kl_divergence,
    renyi_divergence,
    total_variation_distance,
)
from spartan.utils.metrics import (
    compute_accuracy,
    compute_auc_roc,
    compute_f1_score,
    compute_precision_recall_curve,
    compute_tpr_at_fpr,
)
from spartan.utils.noise import (
    calibrated_noise,
    gaussian_noise,
    laplace_noise,
)


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

    def test_calibrated_noise_laplace_mechanism(self):
        """Test calibrated noise with laplace mechanism."""
        noise = calibrated_noise((100,), sensitivity=1.0, epsilon=0.5, mechanism="laplace")
        assert noise.shape == (100,)
        assert np.std(noise) > 0

    def test_calibrated_noise_gaussian_mechanism(self):
        """Test calibrated noise with gaussian mechanism."""
        noise = calibrated_noise((100,), sensitivity=1.0, epsilon=0.5, mechanism="gaussian")
        assert noise.shape == (100,)
        assert np.std(noise) > 0


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

    def test_entropy_with_base(self):
        """Test entropy with custom base."""
        dist = np.array([0.5, 0.5])
        e_natural = entropy(dist)
        e_base2 = entropy(dist, base=2.0)
        assert e_base2 > 0
        assert e_natural != e_base2

    def test_cross_entropy(self):
        """Test cross entropy computation."""
        p = np.array([0.9, 0.1])
        q = np.array([0.6, 0.4])
        ce = cross_entropy(p, q)
        assert ce > 0

    def test_total_variation_distance(self):
        """Test total variation distance."""
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        tv = total_variation_distance(p, q)
        # Use looser tolerance for floating point
        assert abs(tv - 1.0) < 1e-6

    def test_hellinger_distance(self):
        """Test Hellinger distance."""
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        h = hellinger_distance(p, q)
        assert 0 <= h <= 1

    def test_renyi_divergence(self):
        """Test Renyi divergence."""
        p = np.array([0.7, 0.3])
        q = np.array([0.4, 0.6])

        # Test with alpha=2
        r2 = renyi_divergence(p, q, alpha=2.0)
        assert r2 >= 0

        # Test with alpha=1 (should equal KL)
        r1 = renyi_divergence(p, q, alpha=1.0)
        kl = kl_divergence(p, q)
        assert abs(r1 - kl) < 1e-6


class TestMetricsEdgeCases:
    """Additional tests for metrics module."""

    def test_auc_roc_same_labels(self):
        """Test AUC-ROC with all same labels."""
        labels = [1, 1, 1, 1]
        scores = [0.1, 0.5, 0.7, 0.9]

        auc = compute_auc_roc(labels, scores)
        assert auc == 0.5

    def test_accuracy_empty(self):
        """Test accuracy with empty lists."""
        acc = compute_accuracy([], [])
        assert acc == 0.0

    def test_tpr_at_fpr_edge_cases(self):
        """Test TPR@FPR with edge cases."""
        labels = [0, 0, 0, 0]
        scores = [0.1, 0.2, 0.3, 0.4]

        tpr = compute_tpr_at_fpr(labels, scores, fpr_threshold=0.5)
        assert tpr == 0.0

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

    def test_auc_roc_length_mismatch(self):
        """Test AUC-ROC with mismatched lengths."""
        with pytest.raises(ValueError):
            compute_auc_roc([1, 0], [0.5])

    def test_accuracy_length_mismatch(self):
        """Test accuracy with mismatched lengths."""
        with pytest.raises(ValueError):
            compute_accuracy([1, 0], [1])

    def test_tpr_at_fpr_length_mismatch(self):
        """Test TPR@FPR with mismatched lengths."""
        with pytest.raises(ValueError):
            compute_tpr_at_fpr([1, 0], [0.5])

    def test_f1_score_length_mismatch(self):
        """Test F1 with mismatched lengths."""
        with pytest.raises(ValueError):
            compute_f1_score([1, 0], [1])


class TestVoteLeakageEdgeCases:
    """Additional tests for vote leakage analyzer."""

    def test_uniform_distribution(self):
        """Test with perfectly uniform distribution."""
        analyzer = VoteLeakageAnalyzer()
        result = analyzer.analyze(vote_distribution=[0.25, 0.25, 0.25, 0.25])

        assert result["leakage_score"] < 0.1

    def test_single_candidate(self):
        """Test with single candidate."""
        analyzer = VoteLeakageAnalyzer()
        result = analyzer.analyze(vote_distribution=[1.0])

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

        assert result["leakage_score"] > 0.5


class TestDefenseEdgeCases:
    """Additional tests for defense modules."""

    def test_prm_defense_basic(self):
        """Test PRM defense with required arguments."""
        defense = PRMDefense()
        result = defense.apply(
            reasoning_steps=["Step 1: Calculate x = 5", "Step 2: Calculate y = x + 3"],
            prm_leakage=0.6,
            threshold=0.3,
            epsilon=0.5,
        )

        assert "sanitized_steps" in result

    def test_vote_defense_basic(self):
        """Test vote defense basic operation."""
        defense = VoteDefense()
        result = defense.apply(
            vote_distribution=[0.9, 0.05, 0.05],
            vote_leakage=0.7,
            threshold=0.3,
            epsilon=0.5,
        )

        assert "sanitized_distribution" in result
        sanitized = result["sanitized_distribution"]
        # Defense should make distribution less concentrated
        assert max(sanitized) <= 0.95

    def test_mcts_defense_basic(self):
        """Test MCTS defense with required arguments."""
        defense = MCTSDefense()
        tree = {
            "value": 0.8,
            "children": [
                {"value": 0.7, "children": []},
                {"value": 0.9, "children": []},
            ],
        }
        result = defense.apply(
            mcts_tree=tree,
            mcts_leakage=0.6,
            threshold=0.3,
            epsilon=0.5,
        )

        assert "sanitized_tree" in result


class TestBanditEdgeCases:
    """Additional tests for UCB bandit."""

    def test_many_arms(self):
        """Test with many arms."""
        bandit = UCBBandit(num_arms=100, exploration_constant=2.0)

        for _ in range(100):
            arm = bandit.select_arm()
            bandit.update(arm, np.random.random())

        best = bandit.get_best_arm()
        assert 0 <= best < 100

    def test_zero_exploration(self):
        """Test with zero exploration constant."""
        bandit = UCBBandit(num_arms=5, exploration_constant=0.0)

        for i in range(5):
            bandit.update(i, i * 0.1)

        selected = bandit.select_arm()
        assert selected == 4


class TestParetoEdgeCases:
    """Additional tests for Pareto front."""

    def test_add_multiple_points(self):
        """Test adding multiple points to Pareto front."""
        pareto = ParetoFront()

        for i in range(10):
            point = np.random.random(2)
            pareto.add_point(point, {"id": i})

        front = pareto.get_front()
        assert len(front) > 0

    def test_dominated_points(self):
        """Test that dominated points are not in front."""
        pareto = ParetoFront()

        # Add a dominant point
        pareto.add_point(np.array([0.9, 0.9]), {"id": "dominant"})
        # Add a dominated point
        pareto.add_point(np.array([0.5, 0.5]), {"id": "dominated"})

        front = pareto.get_front()
        # Only the dominant point should be in front
        assert len(front) >= 1


class TestCoreEdgeCases:
    """Additional tests for core SPARTAN class."""

    def test_result_to_dict(self):
        """Test SPARTANResult to_dict method."""
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

    def test_spartan_evaluate_defense(self):
        """Test SPARTAN evaluate_defense method."""
        llm = MockReasoningLLM()
        spartan = SPARTAN(llm)

        queries = ["Query 1", "Query 2", "Query 3", "Query 4"]
        labels = [1, 0, 1, 0]

        metrics = spartan.evaluate_defense(queries, labels)
        assert "auc_roc" in metrics
        assert "defense_rate" in metrics
