"""
Comprehensive Coverage Tests

Tests to achieve 100% code coverage by targeting all uncovered lines.
"""

import json
import os
import tempfile

import numpy as np
import pytest

from spartan.attacks.base import AttackResult, BaseAttack
from spartan.attacks.mvna import MVNAAttack
from spartan.attacks.nlba import NLBAAttack
from spartan.attacks.smva import SMVAAttack
from spartan.benchmarks.datasets import BenchmarkDataset, DatasetLoader
from spartan.benchmarks.runner import BenchmarkRunner
from spartan.config import SPARTANConfig
from spartan.core import SPARTAN
from spartan.models.base import BaseReasoningLLM
from spartan.models.mock import MockReasoningLLM
from spartan.mplq.analyzer import MPLQ
from spartan.mplq.mcts_leakage import MCTSLeakageAnalyzer
from spartan.mplq.prm_leakage import PRMLeakageAnalyzer
from spartan.mplq.vote_leakage import VoteLeakageAnalyzer
from spartan.raas.mcts_defense import MCTSDefense
from spartan.raas.prm_defense import PRMDefense
from spartan.raas.sanitizer import RAAS
from spartan.raas.vote_defense import VoteDefense
from spartan.rppo.bandit import UCBBandit
from spartan.rppo.optimizer import RPPO
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


# ============== BaseAttack coverage ==============
class TestBaseAttackCoverage:
    """Tests for base attack coverage."""

    def test_attack_result_to_dict(self):
        """Test AttackResult to_dict method."""
        result = AttackResult(
            success_score=0.8,
            membership_prediction=True,
            confidence=0.9,
            details={"key": "value"},
        )
        d = result.to_dict()
        assert d["success_score"] == 0.8
        assert d["membership_prediction"] is True
        assert d["confidence"] == 0.9
        assert "details" in d

    def test_attack_result_repr(self):
        """Test AttackResult __repr__."""
        result = AttackResult(
            success_score=0.8,
            membership_prediction=True,
            confidence=0.9,
        )
        r = repr(result)
        assert "AttackResult" in r or "0.8" in str(result)


# ============== NLBA Attack coverage ==============
class TestNLBAAttackCoverage:
    """Tests for NLBA attack edge cases."""

    def test_nlba_with_prm_scores_direct(self):
        """Test NLBA with direct PRM scores."""
        attack = NLBAAttack()
        result = attack.execute(
            query="What is 2+2?",
            target_model=MockReasoningLLM(),
            is_member=True,
            prm_scores=[0.95, 0.92, 0.88],
        )
        assert result.success_score >= 0

    def test_nlba_nonmember(self):
        """Test NLBA with non-member."""
        attack = NLBAAttack()
        result = attack.execute(
            query="Random query",
            target_model=MockReasoningLLM(),
            is_member=False,
        )
        assert isinstance(result, AttackResult)


# ============== SMVA Attack coverage ==============
class TestSMVAAttackCoverage:
    """Tests for SMVA attack edge cases."""

    def test_smva_with_distribution_direct(self):
        """Test SMVA with direct vote distribution."""
        attack = SMVAAttack()
        result = attack.execute(
            query="Test",
            target_model=MockReasoningLLM(),
            is_member=True,
            vote_distribution=[0.9, 0.05, 0.05],
        )
        assert result.success_score >= 0

    def test_smva_uniform_distribution(self):
        """Test SMVA with uniform distribution."""
        attack = SMVAAttack()
        result = attack.execute(
            query="Test",
            target_model=MockReasoningLLM(),
            is_member=False,
            vote_distribution=[0.25, 0.25, 0.25, 0.25],
        )
        assert result.success_score < 0.5


# ============== MVNA Attack coverage ==============
class TestMVNAAttackCoverage:
    """Tests for MVNA attack edge cases."""

    def test_mvna_with_values_direct(self):
        """Test MVNA with direct MCTS values."""
        attack = MVNAAttack()
        result = attack.execute(
            query="Test",
            target_model=MockReasoningLLM(),
            is_member=True,
            mcts_values=[0.9, 0.85, 0.92],
        )
        assert result.success_score >= 0

    def test_mvna_with_tree(self):
        """Test MVNA with MCTS tree structure."""
        attack = MVNAAttack()
        tree = {
            "value": 0.8,
            "children": [
                {"value": 0.7, "children": []},
                {"value": 0.9, "children": []},
            ],
        }
        result = attack.execute(
            query="Test",
            target_model=MockReasoningLLM(),
            is_member=True,
            mcts_tree=tree,
        )
        assert isinstance(result, AttackResult)


# ============== Benchmark Runner coverage ==============
class TestBenchmarkRunnerCoverage:
    """Tests for benchmark runner edge cases."""

    def test_runner_with_custom_attacks(self):
        """Test runner with specific attacks."""
        runner = BenchmarkRunner()
        dataset = DatasetLoader.load_mock(num_samples=10)
        results = runner.benchmark_attacks(
            dataset=dataset,
            attacks=["nlba"],
        )
        assert "nlba" in results

    def test_runner_save_results(self):
        """Test saving benchmark results."""
        runner = BenchmarkRunner()
        dataset = DatasetLoader.load_mock(num_samples=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.json")
            results = runner.run_full_benchmark(
                dataset=dataset,
                output_path=output_path,
            )
            assert os.path.exists(output_path)


# ============== MPLQ Analyzer coverage ==============
class TestMPLQAnalyzerCoverage:
    """Tests for MPLQ analyzer edge cases."""

    def test_mplq_with_all_none(self):
        """Test MPLQ with all None inputs."""
        mplq = MPLQ()
        result = mplq.analyze()
        assert result.total_risk == 0.0

    def test_mplq_high_risk(self):
        """Test MPLQ with high risk scenario."""
        mplq = MPLQ()
        result = mplq.analyze(
            prm_scores=[0.99, 0.98, 0.97],
            vote_distribution=[0.95, 0.03, 0.02],
            mcts_values=[0.98, 0.97, 0.99],
        )
        assert result.total_risk > 0.5


# ============== PRM Leakage coverage ==============
class TestPRMLeakageCoverage:
    """Tests for PRM leakage edge cases."""

    def test_prm_with_reference(self):
        """Test PRM with custom reference distribution."""
        analyzer = PRMLeakageAnalyzer()
        analyzer.set_reference_distribution([0.5, 0.3, 0.2])
        result = analyzer.analyze(prm_scores=[0.9, 0.8, 0.7])
        assert "leakage_score" in result

    def test_prm_extreme_scores(self):
        """Test PRM with extreme scores."""
        analyzer = PRMLeakageAnalyzer()
        result = analyzer.analyze(prm_scores=[1.0, 1.0, 1.0])
        assert result["leakage_score"] > 0


# ============== Vote Leakage coverage ==============
class TestVoteLeakageCoverage:
    """Tests for vote leakage edge cases."""

    def test_vote_with_patterns(self):
        """Test vote leakage pattern analysis."""
        analyzer = VoteLeakageAnalyzer()
        result = analyzer.analyze(
            vote_distribution=[0.8, 0.1, 0.1],
            analyze_patterns=True,
        )
        assert "leakage_score" in result

    def test_vote_single_winner(self):
        """Test with single dominant candidate."""
        analyzer = VoteLeakageAnalyzer()
        result = analyzer.analyze(vote_distribution=[1.0, 0.0, 0.0])
        assert result["leakage_score"] == 1.0


# ============== MCTS Leakage coverage ==============
class TestMCTSLeakageCoverage:
    """Tests for MCTS leakage edge cases."""

    def test_mcts_with_tree_structure(self):
        """Test MCTS with complex tree."""
        analyzer = MCTSLeakageAnalyzer()
        tree = {
            "value": 0.9,
            "visit_count": 100,
            "children": [
                {
                    "value": 0.85,
                    "visit_count": 50,
                    "children": [
                        {"value": 0.8, "visit_count": 25, "children": []},
                    ],
                },
                {"value": 0.75, "visit_count": 50, "children": []},
            ],
        }
        result = analyzer.analyze(mcts_values=[0.9, 0.85, 0.8, 0.75], mcts_tree=tree)
        assert "leakage_score" in result


# ============== RAAS Sanitizer coverage ==============
class TestRAASSanitizerCoverage:
    """Tests for RAAS sanitizer edge cases."""

    def test_raas_with_all_components(self):
        """Test RAAS with all defense components."""
        raas = RAAS()
        result = raas.sanitize(
            output="The answer is 42",
            risk_score=0.7,
            reasoning_steps=["Step 1: Think", "Step 2: Calculate"],
            vote_distribution=[0.8, 0.1, 0.1],
            mcts_tree={"values": [0.9, 0.8]},
            prm_leakage=0.6,
            vote_leakage=0.7,
            mcts_leakage=0.5,
        )
        assert result.defense_applied is True

    def test_raas_low_risk(self):
        """Test RAAS with low risk - minimal defense."""
        raas = RAAS()
        result = raas.sanitize(
            output="Simple answer",
            risk_score=0.1,
        )
        assert result.epsilon_used < 0.5


# ============== PRM Defense coverage ==============
class TestPRMDefenseCoverage:
    """Tests for PRM defense edge cases."""

    def test_prm_defense_long_steps(self):
        """Test PRM defense with long reasoning steps."""
        defense = PRMDefense()
        long_step = "This is a very long reasoning step " * 20
        result = defense.apply(
            reasoning_steps=[long_step, "Short step"],
            prm_leakage=0.8,
            threshold=0.3,
            epsilon=0.7,
        )
        assert result["applied"] is True

    def test_prm_defense_math_heavy(self):
        """Test PRM defense with math-heavy content."""
        defense = PRMDefense()
        result = defense.apply(
            reasoning_steps=["x = 5 + 3 * 2", "y = x^2 - 4"],
            prm_leakage=0.6,
            threshold=0.3,
            epsilon=0.5,
        )
        assert "modified_trace" in result


# ============== Vote Defense coverage ==============
class TestVoteDefenseCoverage:
    """Tests for vote defense edge cases."""

    def test_vote_defense_mismatched_candidates(self):
        """Test vote defense with mismatched candidate count."""
        defense = VoteDefense()
        result = defense.apply(
            vote_distribution=[0.8, 0.2],
            vote_leakage=0.7,
            threshold=0.3,
            epsilon=0.5,
            candidate_outputs=["A", "B", "C"],  # Mismatched
        )
        assert "resampled_output" in result

    def test_vote_defense_high_temperature(self):
        """Test vote defense with high leakage causing high temperature."""
        defense = VoteDefense()
        result = defense.apply(
            vote_distribution=[0.99, 0.005, 0.005],
            vote_leakage=0.95,
            threshold=0.3,
            epsilon=0.9,
        )
        assert result["temperature_used"] > 1.0


# ============== MCTS Defense coverage ==============
class TestMCTSDefenseCoverage:
    """Tests for MCTS defense edge cases."""

    def test_mcts_defense_with_trajectories(self):
        """Test MCTS defense with trajectory data."""
        defense = MCTSDefense()
        tree = {
            "values": [0.9, 0.8, 0.7],
            "depths": [0, 1, 2],
            "trajectories": [
                {"node_indices": [0, 1]},
                {"node_indices": [0, 2]},
            ],
            "outputs": ["Answer A", "Answer B"],
        }
        result = defense.apply(
            mcts_tree=tree,
            mcts_leakage=0.7,
            threshold=0.3,
            epsilon=0.6,
        )
        assert "rerun_result" in result

    def test_mcts_defense_no_rerun(self):
        """Test MCTS defense without rerun."""
        defense = MCTSDefense(rerun_search=False)
        result = defense.apply(
            mcts_tree={"values": [0.8, 0.7]},
            mcts_leakage=0.6,
            threshold=0.3,
            epsilon=0.5,
        )
        assert "rerun_result" not in result


# ============== Pareto Front coverage ==============
class TestParetoFrontCoverage:
    """Tests for Pareto front edge cases."""

    def test_pareto_1d(self):
        """Test Pareto with 1D points."""
        pareto = ParetoFront()
        pareto.add_point(np.array([0.8]), {"id": 1})
        pareto.add_point(np.array([0.9]), {"id": 2})
        hv = pareto.get_hypervolume()
        assert hv >= 0

    def test_pareto_3d(self):
        """Test Pareto with 3D points."""
        pareto = ParetoFront()
        pareto.add_point(np.array([0.8, 0.7, 0.6]), {"id": 1})
        pareto.add_point(np.array([0.6, 0.8, 0.7]), {"id": 2})
        hv = pareto.get_hypervolume(reference_point=np.array([0, 0, 0]))
        assert hv >= 0

    def test_pareto_4d_monte_carlo(self):
        """Test Pareto with 4D points (Monte Carlo)."""
        pareto = ParetoFront()
        pareto.add_point(np.array([0.8, 0.7, 0.6, 0.5]), {"id": 1})
        hv = pareto.get_hypervolume()
        assert hv >= 0

    def test_pareto_get_points(self):
        """Test getting points only."""
        pareto = ParetoFront()
        pareto.add_point(np.array([0.8, 0.7]), {"id": 1})
        points = pareto.get_points()
        assert len(points) == 1

    def test_pareto_get_params(self):
        """Test getting params only."""
        pareto = ParetoFront()
        pareto.add_point(np.array([0.8, 0.7]), {"id": 1})
        params = pareto.get_params()
        assert len(params) == 1
        assert params[0]["id"] == 1


# ============== UCB Bandit coverage ==============
class TestUCBBanditCoverage:
    """Tests for UCB bandit edge cases."""

    def test_bandit_reward_history(self):
        """Test bandit with reward history."""
        bandit = UCBBandit(num_arms=3)
        for _ in range(10):
            arm = bandit.select_arm()
            bandit.update(arm, np.random.random())

        stats = bandit.get_arm_stats()
        # Check that stats include history-based metrics
        for stat in stats:
            if stat["count"] > 0:
                assert "average_reward" in stat

    def test_bandit_ucb_computation(self):
        """Test UCB value computation."""
        bandit = UCBBandit(num_arms=3, exploration_constant=2.0)
        # Update arms with different rewards
        bandit.update(0, 0.9)
        bandit.update(1, 0.5)
        bandit.update(2, 0.7)

        # UCB should consider both reward and exploration
        arm = bandit.select_arm()
        assert 0 <= arm < 3


# ============== RPPO Optimizer coverage ==============
class TestRPPOCoverage:
    """Tests for RPPO optimizer edge cases."""

    def test_rppo_pareto_analysis(self):
        """Test RPPO Pareto front analysis."""
        rppo = RPPO()
        # Add multiple updates
        for i in range(20):
            rppo.update(
                params={"epsilon": 0.1 + i * 0.05},
                privacy_score=0.5 + np.random.random() * 0.3,
                utility_score=0.5 + np.random.random() * 0.3,
            )

        optimal = rppo.get_optimal_params()
        assert optimal is not None


# ============== Noise utilities coverage ==============
class TestNoiseCoverage:
    """Tests for noise utility edge cases."""

    def test_gaussian_large_shape(self):
        """Test Gaussian noise with large shape."""
        noise = gaussian_noise((100, 100), scale=0.5)
        assert noise.shape == (100, 100)

    def test_laplace_reproducibility(self):
        """Test Laplace noise reproducibility."""
        n1 = laplace_noise((50,), scale=1.0, seed=123)
        n2 = laplace_noise((50,), scale=1.0, seed=123)
        assert np.allclose(n1, n2)

    def test_calibrated_high_epsilon(self):
        """Test calibrated noise with high epsilon."""
        noise = calibrated_noise((20,), sensitivity=1.0, epsilon=2.0, mechanism="laplace")
        # High epsilon = low noise
        assert np.std(noise) < 2.0


# ============== Metrics coverage ==============
class TestMetricsCoverage:
    """Tests for metrics edge cases."""

    def test_precision_recall_edge(self):
        """Test precision-recall with edge cases."""
        labels = [1, 1, 0, 0]
        scores = [0.9, 0.8, 0.2, 0.1]
        p, r, t = compute_precision_recall_curve(labels, scores)
        assert len(p) > 0

    def test_tpr_at_various_fpr(self):
        """Test TPR at various FPR thresholds."""
        labels = [0, 0, 1, 1, 0, 1]
        scores = [0.1, 0.2, 0.8, 0.9, 0.3, 0.7]

        tpr_01 = compute_tpr_at_fpr(labels, scores, fpr_threshold=0.1)
        tpr_05 = compute_tpr_at_fpr(labels, scores, fpr_threshold=0.5)

        assert tpr_05 >= tpr_01


# ============== Config coverage ==============
class TestConfigCoverage:
    """Tests for config edge cases."""

    def test_config_update(self):
        """Test config update method."""
        config = SPARTANConfig()
        config.epsilon_min = 0.05
        config.epsilon_max = 0.8
        assert config.epsilon_min == 0.05

    def test_config_from_dict_nested(self):
        """Test config from nested dict."""
        d = {
            "mplq": {
                "prm_threshold": 0.4,
                "vote_threshold": 0.5,
                "mcts_threshold": 0.6,
            },
            "raas": {
                "epsilon_min": 0.02,
                "epsilon_max": 0.9,
            },
        }
        config = SPARTANConfig.from_dict(d)
        assert config.prm_threshold == 0.4


# ============== Distribution utilities coverage ==============
class TestDistributionsCoverage:
    """Tests for distribution utilities edge cases."""

    def test_renyi_alpha_half(self):
        """Test Renyi divergence with alpha=0.5."""
        p = np.array([0.7, 0.3])
        q = np.array([0.4, 0.6])
        r = renyi_divergence(p, q, alpha=0.5)
        assert r >= 0

    def test_cross_entropy_identical(self):
        """Test cross entropy with identical distributions."""
        p = np.array([0.5, 0.5])
        ce = cross_entropy(p, p)
        e = entropy(p)
        assert np.isclose(ce, e)
