"""
Tests for attack implementations.
"""

import numpy as np
import pytest

from spartan.attacks import (
    AttackResult,
    BaseAttack,
    MVNAAttack,
    NLBAAttack,
    SMVAAttack,
)
from spartan.models.mock import MockReasoningLLM


class TestAttackResult:
    """Tests for AttackResult."""

    def test_creation(self):
        """Test result creation."""
        result = AttackResult(
            success_score=0.75,
            membership_prediction=True,
            confidence=0.8,
        )

        assert result.success_score == 0.75
        assert result.membership_prediction is True
        assert result.confidence == 0.8

    def test_to_dict(self):
        """Test serialization."""
        result = AttackResult(
            success_score=0.75,
            membership_prediction=True,
            confidence=0.8,
            leakage_signals={"signal1": 0.5},
        )

        result_dict = result.to_dict()
        assert "success_score" in result_dict
        assert "leakage_signals" in result_dict


class TestNLBAAttack:
    """Tests for Natural Language Blindness Attack."""

    def test_init(self):
        """Test initialization."""
        attack = NLBAAttack()
        assert attack.name == "NLBAAttack"

    def test_execute_with_scores(self, sample_prm_scores, sample_reasoning_steps):
        """Test execution with provided scores."""
        attack = NLBAAttack()
        result = attack.execute(
            target_model=None,
            query="test",
            prm_scores=sample_prm_scores,
            reasoning_steps=sample_reasoning_steps,
        )

        assert isinstance(result, AttackResult)
        assert 0 <= result.success_score <= 1

    def test_execute_with_mock_model(self, mock_llm, sample_query):
        """Test execution with mock model."""
        attack = NLBAAttack()
        result = attack.execute(
            target_model=mock_llm,
            query=sample_query,
        )

        assert isinstance(result, AttackResult)

    def test_member_vs_nonmember(self, member_mock_llm, non_member_mock_llm, sample_query):
        """Members should have higher attack scores."""
        attack = NLBAAttack()

        member_result = attack.execute(
            target_model=member_mock_llm,
            query=sample_query,
        )

        nonmember_result = attack.execute(
            target_model=non_member_mock_llm,
            query=sample_query,
        )

        # Member should generally have higher score
        assert member_result.success_score >= 0
        assert nonmember_result.success_score >= 0

    def test_compute_attack_signal(self, sample_prm_scores):
        """Test direct signal computation."""
        attack = NLBAAttack()
        signal = attack.compute_attack_signal(prm_scores=sample_prm_scores)

        assert 0 <= signal <= 1

    def test_no_prm_scores(self):
        """Test handling of missing PRM scores."""
        attack = NLBAAttack()
        result = attack.execute(
            target_model=None,
            query="test",
            prm_scores=None,
        )

        assert result.confidence == 0.0


class TestSMVAAttack:
    """Tests for Single-Model Voting Attack."""

    def test_init(self):
        """Test initialization."""
        attack = SMVAAttack()
        assert attack.name == "SMVAAttack"

    def test_execute_with_distribution(self, sample_vote_distribution):
        """Test execution with provided distribution."""
        attack = SMVAAttack()
        result = attack.execute(
            target_model=None,
            query="test",
            vote_distribution=sample_vote_distribution,
        )

        assert isinstance(result, AttackResult)
        assert 0 <= result.success_score <= 1

    def test_concentrated_distribution_high_score(
        self,
        concentrated_vote_distribution,
        uniform_vote_distribution,
    ):
        """Concentrated distribution should have higher attack score."""
        attack = SMVAAttack()

        conc_result = attack.execute(
            target_model=None,
            query="test",
            vote_distribution=concentrated_vote_distribution,
        )

        unif_result = attack.execute(
            target_model=None,
            query="test",
            vote_distribution=uniform_vote_distribution,
        )

        assert conc_result.success_score > unif_result.success_score

    def test_compute_attack_signal(self, sample_vote_distribution):
        """Test direct signal computation."""
        attack = SMVAAttack()
        signal = attack.compute_attack_signal(vote_distribution=sample_vote_distribution)

        assert 0 <= signal <= 1

    def test_no_distribution(self):
        """Test handling of missing distribution."""
        attack = SMVAAttack()
        result = attack.execute(
            target_model=None,
            query="test",
            vote_distribution=None,
        )

        assert result.confidence == 0.0


class TestMVNAAttack:
    """Tests for MCTS Value Network Attack."""

    def test_init(self):
        """Test initialization."""
        attack = MVNAAttack()
        assert attack.name == "MVNAAttack"

    def test_execute_with_values(self, sample_mcts_values):
        """Test execution with provided values."""
        attack = MVNAAttack()
        result = attack.execute(
            target_model=None,
            query="test",
            mcts_values=sample_mcts_values,
        )

        assert isinstance(result, AttackResult)
        assert 0 <= result.success_score <= 1

    def test_execute_with_tree(self, sample_mcts_values, sample_mcts_tree):
        """Test execution with tree structure."""
        attack = MVNAAttack()
        result = attack.execute(
            target_model=None,
            query="test",
            mcts_values=sample_mcts_values,
            mcts_tree=sample_mcts_tree,
        )

        assert "depth_correlation" in result.leakage_signals

    def test_high_values_high_score(self):
        """High values should indicate memorization."""
        attack = MVNAAttack()

        high_values = [0.95, 0.92, 0.98, 0.94, 0.97]
        normal_values = [0.5, 0.55, 0.48, 0.52, 0.51]

        high_result = attack.execute(
            target_model=None,
            query="test",
            mcts_values=high_values,
        )

        normal_result = attack.execute(
            target_model=None,
            query="test",
            mcts_values=normal_values,
        )

        assert high_result.success_score >= normal_result.success_score

    def test_compute_attack_signal(self, sample_mcts_values):
        """Test direct signal computation."""
        attack = MVNAAttack()
        signal = attack.compute_attack_signal(mcts_values=sample_mcts_values)

        assert 0 <= signal <= 1

    def test_no_values(self):
        """Test handling of missing values."""
        attack = MVNAAttack()
        result = attack.execute(
            target_model=None,
            query="test",
            mcts_values=None,
        )

        assert result.confidence == 0.0


class TestAttackEvaluation:
    """Tests for attack evaluation functionality."""

    def test_evaluate(self):
        """Test evaluation metrics computation."""
        attack = NLBAAttack()

        # Create mock results
        results = [
            AttackResult(success_score=0.8, membership_prediction=True, confidence=0.9),
            AttackResult(success_score=0.3, membership_prediction=False, confidence=0.8),
            AttackResult(success_score=0.7, membership_prediction=True, confidence=0.7),
            AttackResult(success_score=0.2, membership_prediction=False, confidence=0.9),
        ]

        ground_truth = [True, False, True, False]

        metrics = attack.evaluate(results, ground_truth)

        assert "auc_roc" in metrics
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_predict_membership(self):
        """Test membership prediction."""
        attack = NLBAAttack(threshold=0.5)

        pred1, conf1 = attack.predict_membership(0.8)
        assert pred1 is True

        pred2, conf2 = attack.predict_membership(0.3)
        assert pred2 is False


# Last updated: 2026-01-15
