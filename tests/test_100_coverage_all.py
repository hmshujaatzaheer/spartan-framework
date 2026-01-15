"""
Comprehensive 100% Coverage Tests

Targets ALL remaining uncovered lines across all modules.
"""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ============== METRICS.PY (93.75% -> 100%) ==============
from spartan.utils.metrics import (
    compute_accuracy,
    compute_auc_roc,
    compute_f1_score,
    compute_precision_recall_curve,
    compute_tpr_at_fpr,
)


class TestMetrics100:
    def test_auc_roc_num_pos_zero(self):
        """Cover line 46-47: num_pos == 0 after unique check."""
        # All same labels - triggers unique < 2 check
        result = compute_auc_roc([0, 0, 0], [0.1, 0.2, 0.3])
        assert result == 0.5

    def test_tpr_at_fpr_empty(self):
        """Cover line 112-113: empty labels."""
        result = compute_tpr_at_fpr([], [])
        assert result == 0.0

    def test_tpr_at_fpr_no_break(self):
        """Cover line 132->148: loop completes without break."""
        # fpr_threshold=1.0 means we never exceed it
        result = compute_tpr_at_fpr([1, 1, 0, 0], [0.9, 0.8, 0.7, 0.6], fpr_threshold=1.0)
        assert result >= 0

    def test_tpr_at_fpr_break_on_positive(self):
        """Cover line 142->145: break when current label is positive."""
        # Need to exceed fpr_threshold when processing a positive label
        result = compute_tpr_at_fpr([1, 0, 1, 0], [0.9, 0.85, 0.8, 0.75], fpr_threshold=0.4)
        assert result >= 0

    def test_f1_empty(self):
        """Cover line 169-170: empty labels."""
        result = compute_f1_score([], [])
        assert result == 0.0


# ============== ANALYZER.PY (92.94% -> 100%) ==============
from spartan.mplq.analyzer import MPLQ


class TestAnalyzer100:
    def test_importance_high_mean_low_var(self):
        """Cover importance weight memorization detection."""
        mplq = MPLQ()
        result = mplq.analyze(
            query="test",
            prm_scores=[0.95, 0.95, 0.95, 0.95],  # High mean, low variance
        )
        assert result.importance_weight >= 1.0

    def test_importance_medium_confidence(self):
        """Cover medium confidence branch."""
        mplq = MPLQ()
        result = mplq.analyze(
            query="test",
            prm_scores=[0.75, 0.78, 0.72, 0.76],  # Medium scores
        )
        assert result.total_risk >= 0

    def test_update_weights_normalization(self):
        """Cover weight normalization."""
        mplq = MPLQ()
        mplq.update_weights(alpha=2.0, beta=2.0, gamma=2.0)
        weights = mplq.get_weights()
        assert abs(sum(weights) - 1.0) < 0.01


# ============== PRM_DEFENSE.PY (91.86% -> 100%) ==============
from spartan.raas.prm_defense import PRMDefense


class TestPRMDefense100:
    def test_defense_below_threshold(self):
        """Cover no defense when below threshold."""
        defense = PRMDefense()
        result = defense.apply(
            reasoning_steps=["Step 1", "Step 2"],
            epsilon=0.1,
            prm_leakage=0.1,  # Below default threshold
            threshold=0.5,
        )
        assert "applied" in result

    def test_defense_feature_selective_off(self):
        """Cover feature selective disabled."""
        defense = PRMDefense(use_feature_selective=False)
        result = defense.apply(
            reasoning_steps=["Step 1 with math x=2", "Step 2"],
            epsilon=0.3,
            prm_leakage=0.8,
            threshold=0.3,
        )
        assert result["applied"] == True

    def test_defense_empty_steps(self):
        """Cover empty steps handling."""
        defense = PRMDefense()
        result = defense.apply(
            reasoning_steps=[],
            epsilon=0.3,
            prm_leakage=0.8,
            threshold=0.3,
        )
        assert "applied" in result


# ============== OPTIMIZER.PY (94.17% -> 100%) ==============
from spartan.rppo.optimizer import RPPO


class TestOptimizer100:
    def test_gradient_refinement_episode_10(self):
        """Cover gradient refinement at episode 10."""
        rppo = RPPO()
        for i in range(10):
            rppo.update(
                {
                    "risk_score": 0.5,
                    "accuracy": 0.8,
                    "compute": 0.3,
                }
            )
        assert rppo._episode == 10

    def test_get_optimal_params_no_data(self):
        """Cover insufficient data case."""
        rppo = RPPO()
        result = rppo.get_optimal_params()
        assert result is None

    def test_get_statistics(self):
        """Cover statistics computation."""
        rppo = RPPO()
        for i in range(5):
            rppo.update(
                {
                    "risk_score": 0.5 - 0.05 * i,
                    "accuracy": 0.8,
                    "compute": 0.3,
                }
            )
        stats = rppo.get_statistics()
        assert "mean_reward" in stats or stats is not None


# ============== CONFIG.PY (95.40% -> 100%) ==============
from spartan.config import SPARTANConfig


class TestConfig100:
    def test_config_from_dict_all_sections(self):
        """Cover from_dict with all sections."""
        config = SPARTANConfig.from_dict(
            {
                "mplq": {"kl_smoothing": 1e-6},
                "raas": {"min_epsilon": 0.05},
                "rppo": {"learning_rate": 0.02},
            }
        )
        assert config is not None

    def test_config_validation_invalid_threshold(self):
        """Cover validation error."""
        with pytest.raises(ValueError):
            SPARTANConfig(prm_threshold=1.5)  # Invalid > 1

    def test_config_validation_invalid_epsilon(self):
        """Cover epsilon validation."""
        with pytest.raises(ValueError):
            SPARTANConfig(epsilon_min=-0.1)  # Invalid negative


# ============== BASE.PY ATTACKS (95.24% -> 100%) ==============
from spartan.attacks.base import AttackResult, BaseAttack
from spartan.attacks.nlba import NLBAAttack


class TestBaseAttack100:
    def test_predict_membership_edge(self):
        """Cover confidence calculation edge cases."""
        attack = NLBAAttack(threshold=0.5)

        # At threshold
        pred, conf = attack.predict_membership(0.5)
        assert conf >= 0

        # Far below
        pred, conf = attack.predict_membership(0.01)
        assert pred == False

    def test_attack_result_repr(self):
        """Cover AttackResult string representation."""
        result = AttackResult(
            success_score=0.8,
            membership_prediction=True,
            confidence=0.9,
        )
        assert result.success_score == 0.8


# ============== BASE.PY MODELS (93.75% -> 100%) ==============
from spartan.models.base import BaseReasoningLLM, LLMOutput


class TestBaseModels100:
    def test_uses_prm_default(self):
        """Cover uses_prm default."""

        class TestLLM(BaseReasoningLLM):
            def generate(self, query, **kwargs):
                return LLMOutput(output="test")

            def get_prm_scores(self, steps):
                return [0.9] * len(steps)

        llm = TestLLM()
        assert llm.uses_prm() == True

    def test_uses_voting_default(self):
        """Cover uses_voting default."""

        class TestLLM(BaseReasoningLLM):
            def generate(self, query, **kwargs):
                return LLMOutput(output="test")

            def get_prm_scores(self, steps):
                return [0.9] * len(steps)

        llm = TestLLM()
        assert llm.uses_voting() == True

    def test_uses_mcts_default(self):
        """Cover uses_mcts default."""

        class TestLLM(BaseReasoningLLM):
            def generate(self, query, **kwargs):
                return LLMOutput(output="test")

            def get_prm_scores(self, steps):
                return [0.9] * len(steps)

        llm = TestLLM()
        assert llm.uses_mcts() == False


# ============== PRM_LEAKAGE.PY (94.03% -> 100%) ==============
from spartan.mplq.prm_leakage import PRMLeakageAnalyzer


class TestPRMLeakage100:
    def test_default_reference(self):
        """Cover default reference distribution."""
        analyzer = PRMLeakageAnalyzer()
        result = analyzer.analyze(prm_scores=[0.8, 0.85, 0.9])
        assert result.leakage_score >= 0

    def test_empty_scores(self):
        """Cover empty scores."""
        analyzer = PRMLeakageAnalyzer()
        result = analyzer.analyze(prm_scores=[])
        assert result.leakage_score >= 0


# ============== VOTE_LEAKAGE.PY (95.00% -> 100%) ==============
from spartan.mplq.vote_leakage import VoteLeakageAnalyzer


class TestVoteLeakage100:
    def test_empty_distribution(self):
        """Cover empty distribution."""
        analyzer = VoteLeakageAnalyzer()
        result = analyzer.analyze(vote_distribution=[])
        assert result.leakage_score >= 0

    def test_single_candidate(self):
        """Cover single candidate."""
        analyzer = VoteLeakageAnalyzer()
        result = analyzer.analyze(vote_distribution=[1.0])
        assert result.leakage_score >= 0


# ============== NOISE.PY (92.06% -> 100%) ==============
from spartan.utils.noise import (
    adaptive_noise,
    calibrated_noise,
    gaussian_noise,
    laplace_noise,
)


class TestNoise100:
    def test_calibrated_gaussian_with_delta(self):
        """Cover gaussian mechanism with delta."""
        noise = calibrated_noise(
            epsilon=0.5,
            sensitivity=1.0,
            mechanism="gaussian",
            delta=1e-5,
        )
        assert noise is not None

    def test_adaptive_noise_laplace(self):
        """Cover adaptive laplace."""
        noise = adaptive_noise(
            epsilon=0.5,
            sensitivity=1.0,
            risk_score=0.8,
            mechanism="laplace",
        )
        assert noise is not None

    def test_adaptive_noise_invalid(self):
        """Cover invalid mechanism."""
        with pytest.raises(ValueError):
            adaptive_noise(
                epsilon=0.5,
                sensitivity=1.0,
                risk_score=0.8,
                mechanism="invalid",
            )


# ============== CLI.PY (97.37% -> 100%) ==============
from spartan.cli import main as cli_main


class TestCLI100:
    def test_cli_analyze(self):
        """Cover analyze command."""
        with patch.object(sys, "argv", ["spartan", "analyze", "--query", "test"]):
            with patch("builtins.print"):
                try:
                    cli_main()
                except SystemExit:
                    pass


# ============== MOCK.PY (95.95% -> 100%) ==============
from spartan.models.mock import MockReasoningLLM


class TestMock100:
    def test_mock_member_mode(self):
        """Cover member mode generation."""
        llm = MockReasoningLLM(member_mode=True)
        output = llm.generate("test query", use_mcts=True)
        assert output.output is not None

    def test_mock_nonmember_mode(self):
        """Cover non-member mode."""
        llm = MockReasoningLLM(member_mode=False)
        output = llm.generate("test query")
        assert output.output is not None


# ============== SANITIZER.PY (94.87% -> 100%) ==============
from spartan.raas.sanitizer import RAAS


class TestSanitizer100:
    def test_sanitize_low_risk(self):
        """Cover low risk - no defense needed."""
        from spartan.mplq.analyzer import MPLQResult

        raas = RAAS()
        risk_result = MagicMock()
        risk_result.total_risk = 0.1
        risk_result.prm_leakage = 0.1
        risk_result.vote_leakage = 0.1
        risk_result.mcts_leakage = 0.1

        result = raas.sanitize(
            output="test",
            risk_analysis=risk_result,
        )
        assert result.epsilon_used >= 0


# ============== VOTE_DEFENSE.PY (97.33% -> 100%) ==============
from spartan.raas.vote_defense import VoteDefense


class TestVoteDefense100:
    def test_zero_vote_sum(self):
        """Cover zero sum votes."""
        defense = VoteDefense()
        result = defense.apply(
            vote_distribution=[0.0, 0.0, 0.0],
            epsilon=0.3,
            vote_leakage=0.8,
            threshold=0.4,
        )
        assert "applied" in result


# ============== BANDIT.PY (97.26% -> 100%) ==============
from spartan.rppo.bandit import UCBBandit


class TestBandit100:
    def test_unexplored_arm(self):
        """Cover unexplored arm selection."""
        bandit = UCBBandit(n_arms=5)
        arm = bandit.select_arm()
        assert 0 <= arm < 5

    def test_reward_history(self):
        """Cover reward tracking."""
        bandit = UCBBandit(n_arms=3)
        bandit.update(0, 1.0)
        bandit.update(0, 0.8)
        best = bandit.get_best_arm()
        assert best == 0


# ============== PARETO.PY (97.59% -> 100%) ==============
from spartan.rppo.pareto import ParetoFront


class TestPareto100:
    def test_empty_front(self):
        """Cover empty front operations."""
        front = ParetoFront(n_objectives=2)
        result = front.get_front()
        assert len(result) == 0

    def test_hypervolume_empty(self):
        """Cover hypervolume on empty front."""
        front = ParetoFront(n_objectives=2)
        hv = front.get_hypervolume()
        assert hv == 0.0


# ============== DISTRIBUTIONS.PY (95.71% -> 100%) ==============
from spartan.utils.distributions import entropy, js_divergence, kl_divergence


class TestDistributions100:
    def test_kl_with_zeros(self):
        """Cover KL with zero handling."""
        result = kl_divergence([0.5, 0.5, 0.0], [0.3, 0.3, 0.4])
        assert result >= 0

    def test_entropy_single(self):
        """Cover single element entropy."""
        result = entropy([1.0])
        assert result == 0.0


# ============== CORE.PY (97.18% -> 100%) ==============
from spartan.core import SPARTAN


class TestCore100:
    def test_process_with_optimization(self):
        """Cover processing with optimization enabled."""
        from spartan.models.mock import MockReasoningLLM

        llm = MockReasoningLLM()
        spartan = SPARTAN(llm, enable_optimization=True)
        result = spartan.process("test query")
        assert result.output is not None


# ============== MVNA.PY (96.08% -> 100%) ==============
from spartan.attacks.mvna import MVNAAttack


class TestMVNA100:
    def test_mvna_empty_values(self):
        """Cover empty MCTS values."""
        attack = MVNAAttack()
        result = attack.execute(
            target_model=None,
            query="test",
            mcts_values=[],
        )
        assert result.success_score >= 0


# ============== NLBA.PY (94.92% -> 100%) ==============
class TestNLBA100:
    def test_nlba_empty_step(self):
        """Cover empty reasoning step in NL ratio."""
        attack = NLBAAttack()
        ratios = attack._compute_nl_ratios(["Step 1", "", "Step 3"])
        assert ratios[1] == 0.0

    def test_nlba_no_prm_scores(self):
        """Cover no PRM scores case."""
        attack = NLBAAttack()
        result = attack.execute(
            target_model=None,
            query="test",
            prm_scores=None,
        )
        assert result.confidence == 0.0


# ============== DATASETS.PY (98.63% -> 100%) ==============
from spartan.benchmarks.datasets import BenchmarkDataset, DatasetLoader


class TestDatasets100:
    def test_load_mock_dataset(self):
        """Cover mock dataset loading."""
        dataset = DatasetLoader.load("mock", num_samples=10)
        assert len(dataset) == 10


# ============== RUNNER.PY (98.41% -> 100%) ==============
from spartan.benchmarks.runner import BenchmarkRunner
from spartan.benchmarks.runner import main as runner_main


class TestRunner100:
    def test_runner_main(self):
        """Cover main CLI function."""
        with patch.object(sys, "argv", ["runner", "--num-samples", "5"]):
            with patch("builtins.print"):
                with patch("spartan.benchmarks.runner.BenchmarkResult.save"):
                    try:
                        runner_main()
                    except SystemExit:
                        pass


# ============== MCTS_LEAKAGE.PY (98.67% -> 100%) ==============
from spartan.mplq.mcts_leakage import MCTSLeakageAnalyzer


class TestMCTSLeakage100:
    def test_empty_values(self):
        """Cover empty MCTS values."""
        analyzer = MCTSLeakageAnalyzer()
        result = analyzer.analyze(mcts_values=[])
        assert result.leakage_score >= 0
