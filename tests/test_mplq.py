"""
Tests for MPLQ (Mechanistic Privacy Leakage Quantification) module.
"""

import numpy as np
import pytest

from spartan.config import SPARTANConfig
from spartan.mplq import MPLQ, MPLQResult
from spartan.mplq.mcts_leakage import MCTSLeakageAnalyzer
from spartan.mplq.prm_leakage import PRMLeakageAnalyzer
from spartan.mplq.vote_leakage import VoteLeakageAnalyzer


class TestMPLQ:
    """Tests for main MPLQ class."""

    def test_init_default(self):
        """Test default initialization."""
        mplq = MPLQ()
        assert mplq is not None
        assert mplq.config is not None

    def test_init_with_config(self, config):
        """Test initialization with custom config."""
        mplq = MPLQ(config=config)
        assert mplq.config == config

    def test_analyze_basic(self, sample_query, sample_prm_scores):
        """Test basic analysis."""
        mplq = MPLQ()
        result = mplq.analyze(
            query=sample_query,
            prm_scores=sample_prm_scores,
        )

        assert isinstance(result, MPLQResult)
        assert 0 <= result.total_risk <= 1
        assert 0 <= result.prm_leakage <= 1

    def test_analyze_with_all_inputs(
        self,
        sample_query,
        sample_prm_scores,
        sample_reasoning_steps,
        sample_vote_distribution,
        sample_mcts_values,
    ):
        """Test analysis with all inputs."""
        mplq = MPLQ()
        result = mplq.analyze(
            query=sample_query,
            reasoning_steps=sample_reasoning_steps,
            prm_scores=sample_prm_scores,
            vote_distribution=sample_vote_distribution,
            mcts_values=sample_mcts_values,
        )

        assert result.total_risk >= 0
        assert result.prm_leakage >= 0
        assert result.vote_leakage >= 0
        assert result.mcts_leakage >= 0

    def test_analyze_empty_inputs(self, sample_query):
        """Test analysis with empty inputs."""
        mplq = MPLQ()
        result = mplq.analyze(query=sample_query)

        assert result.prm_leakage == 0.0
        assert result.vote_leakage == 0.0
        assert result.mcts_leakage == 0.0

    def test_importance_weight(self, sample_query, sample_prm_scores):
        """Test importance weight computation."""
        mplq = MPLQ()
        result = mplq.analyze(
            query=sample_query,
            prm_scores=sample_prm_scores,
        )

        assert 0.5 <= result.importance_weight <= 2.0

    def test_component_weights(self):
        """Test component weight access and update."""
        mplq = MPLQ()

        alpha, beta, gamma = mplq.get_weights()
        assert alpha + beta + gamma == pytest.approx(1.0)

        mplq.update_weights(0.5, 0.3, 0.2)
        new_alpha, new_beta, new_gamma = mplq.get_weights()
        assert new_alpha + new_beta + new_gamma == pytest.approx(1.0)

    def test_result_to_dict(self, sample_query, sample_prm_scores):
        """Test result serialization."""
        mplq = MPLQ()
        result = mplq.analyze(
            query=sample_query,
            prm_scores=sample_prm_scores,
        )

        result_dict = result.to_dict()
        assert "total_risk" in result_dict
        assert "prm_leakage" in result_dict
        assert "component_weights" in result_dict

    def test_exceeds_threshold(self, sample_query, sample_prm_scores):
        """Test threshold checking."""
        mplq = MPLQ()
        result = mplq.analyze(
            query=sample_query,
            prm_scores=sample_prm_scores,
        )

        thresholds = result.exceeds_threshold()
        assert "prm" in thresholds
        assert "vote" in thresholds
        assert "mcts" in thresholds


class TestPRMLeakageAnalyzer:
    """Tests for PRM leakage analyzer."""

    def test_init(self):
        """Test initialization."""
        analyzer = PRMLeakageAnalyzer()
        assert analyzer is not None

    def test_analyze_basic(self, sample_prm_scores):
        """Test basic analysis."""
        analyzer = PRMLeakageAnalyzer()
        result = analyzer.analyze(prm_scores=sample_prm_scores)

        assert "leakage_score" in result
        assert 0 <= result["leakage_score"] <= 1

    def test_analyze_with_steps(self, sample_prm_scores, sample_reasoning_steps):
        """Test analysis with reasoning steps."""
        analyzer = PRMLeakageAnalyzer()
        result = analyzer.analyze(
            prm_scores=sample_prm_scores,
            reasoning_steps=sample_reasoning_steps,
        )

        assert "nl_blindness" in result

    def test_high_scores_high_leakage(self):
        """High scores should indicate higher leakage."""
        analyzer = PRMLeakageAnalyzer()

        high_scores = [0.95, 0.98, 0.97, 0.96, 0.99]
        low_scores = [0.5, 0.55, 0.48, 0.52, 0.51]

        high_result = analyzer.analyze(prm_scores=high_scores)
        low_result = analyzer.analyze(prm_scores=low_scores)

        # High scores should generally indicate more leakage
        # (deviation from expected distribution)
        assert high_result["mean_score"] > low_result["mean_score"]

    def test_empty_scores(self):
        """Test with empty scores."""
        analyzer = PRMLeakageAnalyzer()
        result = analyzer.analyze(prm_scores=[])

        assert result["leakage_score"] == 0.0
        assert result["num_steps"] == 0

    def test_set_reference_distribution(self):
        """Test setting reference distribution."""
        analyzer = PRMLeakageAnalyzer()

        ref_dist = np.random.uniform(0, 1, 50)
        ref_dist = ref_dist / ref_dist.sum()

        analyzer.set_reference_distribution(ref_dist)

        stored = analyzer.get_reference_distribution()
        assert len(stored) == 50


class TestVoteLeakageAnalyzer:
    """Tests for vote leakage analyzer."""

    def test_init(self):
        """Test initialization."""
        analyzer = VoteLeakageAnalyzer()
        assert analyzer is not None

    def test_analyze_basic(self, sample_vote_distribution):
        """Test basic analysis."""
        analyzer = VoteLeakageAnalyzer()
        result = analyzer.analyze(vote_distribution=sample_vote_distribution)

        assert "leakage_score" in result
        assert "entropy" in result
        assert 0 <= result["leakage_score"] <= 1

    def test_concentrated_distribution_high_leakage(
        self,
        concentrated_vote_distribution,
        uniform_vote_distribution,
    ):
        """Concentrated distribution should have higher leakage."""
        analyzer = VoteLeakageAnalyzer()

        conc_result = analyzer.analyze(vote_distribution=concentrated_vote_distribution)
        unif_result = analyzer.analyze(vote_distribution=uniform_vote_distribution)

        assert conc_result["leakage_score"] > unif_result["leakage_score"]

    def test_entropy_computation(self, uniform_vote_distribution):
        """Test entropy computation."""
        analyzer = VoteLeakageAnalyzer()
        result = analyzer.analyze(vote_distribution=uniform_vote_distribution)

        # Uniform distribution should have high entropy
        assert result["normalized_entropy"] > 0.8

    def test_empty_distribution(self):
        """Test with empty distribution."""
        analyzer = VoteLeakageAnalyzer()
        result = analyzer.analyze(vote_distribution=[])

        assert result["leakage_score"] == 0.0
        assert result["num_candidates"] == 0

    def test_patterns_analysis(self, concentrated_vote_distribution):
        """Test pattern detection."""
        analyzer = VoteLeakageAnalyzer()
        result = analyzer.analyze(vote_distribution=concentrated_vote_distribution)

        assert "patterns" in result
        assert result["patterns"]["is_dominant"]


class TestMCTSLeakageAnalyzer:
    """Tests for MCTS leakage analyzer."""

    def test_init(self):
        """Test initialization."""
        analyzer = MCTSLeakageAnalyzer()
        assert analyzer is not None

    def test_analyze_basic(self, sample_mcts_values):
        """Test basic analysis."""
        analyzer = MCTSLeakageAnalyzer()
        result = analyzer.analyze(mcts_values=sample_mcts_values)

        assert "leakage_score" in result
        assert 0 <= result["leakage_score"] <= 1

    def test_analyze_with_tree(self, sample_mcts_values, sample_mcts_tree):
        """Test analysis with tree structure."""
        analyzer = MCTSLeakageAnalyzer()
        result = analyzer.analyze(
            mcts_values=sample_mcts_values,
            mcts_tree=sample_mcts_tree,
        )

        assert "trajectory_analysis" in result

    def test_high_values_high_leakage(self):
        """High values should indicate memorization."""
        analyzer = MCTSLeakageAnalyzer()

        high_values = [0.9, 0.92, 0.88, 0.95, 0.91]
        normal_values = [0.5, 0.55, 0.48, 0.52, 0.51]

        high_result = analyzer.analyze(mcts_values=high_values)
        normal_result = analyzer.analyze(mcts_values=normal_values)

        assert high_result["leakage_score"] >= normal_result["leakage_score"]

    def test_empty_values(self):
        """Test with empty values."""
        analyzer = MCTSLeakageAnalyzer()
        result = analyzer.analyze(mcts_values=[])

        assert result["leakage_score"] == 0.0
        assert result["num_nodes"] == 0

    def test_baseline_params(self):
        """Test baseline parameter access."""
        analyzer = MCTSLeakageAnalyzer()

        params = analyzer.get_baseline_params()
        assert "mean" in params
        assert "std" in params

        analyzer.set_baseline_params(0.6, 0.2)
        new_params = analyzer.get_baseline_params()
        assert new_params["mean"] == 0.6
        assert new_params["std"] == 0.2


# Last updated: 2026-01-15
