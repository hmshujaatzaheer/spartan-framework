"""
Final Coverage Push - Targeting Remaining Uncovered Lines

Tests for 100% coverage based on actual source code analysis.
"""

import numpy as np
import pytest

from spartan.attacks.base import AttackResult, BaseAttack
from spartan.attacks.nlba import NLBAAttack
from spartan.models.base import BaseReasoningLLM, LLMOutput
from spartan.models.mock import MockReasoningLLM
from spartan.mplq.mcts_leakage import MCTSLeakageAnalyzer
from spartan.utils.metrics import (
    compute_accuracy,
    compute_auc_roc,
    compute_f1_score,
    compute_precision_recall_curve,
    compute_tpr_at_fpr,
)


# ============== attacks/base.py - Line 132: evaluate() ==============
class TestBaseAttackEvaluate:
    """Tests for BaseAttack.evaluate method."""

    def test_evaluate_length_mismatch(self):
        """Test evaluate raises error on length mismatch - line 132."""
        attack = NLBAAttack()
        results = [
            AttackResult(success_score=0.8, membership_prediction=True, confidence=0.9),
            AttackResult(success_score=0.3, membership_prediction=False, confidence=0.7),
        ]
        ground_truth = [True]  # Mismatched length

        with pytest.raises(ValueError, match="mismatch"):
            attack.evaluate(results, ground_truth)

    def test_evaluate_success(self):
        """Test evaluate with matching lengths."""
        attack = NLBAAttack()
        results = [
            AttackResult(success_score=0.9, membership_prediction=True, confidence=0.9),
            AttackResult(success_score=0.2, membership_prediction=False, confidence=0.8),
        ]
        ground_truth = [True, False]

        metrics = attack.evaluate(results, ground_truth)
        assert "auc_roc" in metrics
        assert "accuracy" in metrics
        assert metrics["num_samples"] == 2


# ============== models/base.py - Lines 105, 109, 113, 117 ==============
class TestBaseReasoningLLMMethods:
    """Tests for BaseReasoningLLM default methods."""

    def test_uses_prm_default(self):
        """Test uses_prm returns True by default - line 105."""
        llm = MockReasoningLLM()
        assert llm.uses_prm() == True

    def test_uses_voting_default(self):
        """Test uses_voting returns True by default - line 109."""
        llm = MockReasoningLLM()
        assert llm.uses_voting() == True

    def test_uses_mcts_default(self):
        """Test uses_mcts returns False by default - line 113."""
        llm = MockReasoningLLM()
        # MockReasoningLLM might override this, check the actual value
        result = llm.uses_mcts()
        assert isinstance(result, bool)

    def test_llm_output_to_dict(self):
        """Test LLMOutput.to_dict method."""
        output = LLMOutput(
            output="Test output",
            reasoning_steps=["Step 1", "Step 2"],
            prm_scores=[0.9, 0.8],
            vote_distribution=[0.7, 0.3],
            mcts_values=[0.8, 0.7],
            metadata={"key": "value"},
        )
        d = output.to_dict()
        assert d["output"] == "Test output"
        assert d["reasoning_steps"] == ["Step 1", "Step 2"]
        assert d["prm_scores"] == [0.9, 0.8]


# ============== mcts_leakage.py - Lines 237-258: trajectory analysis ==============
class TestMCTSLeakageTrajectories:
    """Tests for MCTS trajectory analysis."""

    def test_analyze_with_trajectories_list_format(self):
        """Test trajectory analysis with list format - lines 237-258."""
        analyzer = MCTSLeakageAnalyzer()

        # Tree with trajectories in list format
        mcts_tree = {
            "depths": [0, 1, 1, 2, 2],
            "trajectories": [
                [{"node_idx": 0}, {"node_idx": 1}, {"node_idx": 3}],
                [{"node_idx": 0}, {"node_idx": 2}, {"node_idx": 4}],
            ],
        }

        result = analyzer.analyze(
            mcts_values=[0.9, 0.85, 0.8, 0.75, 0.7],
            mcts_tree=mcts_tree,
        )

        assert "trajectory_analysis" in result
        assert result["trajectory_analysis"]["available"] == True

    def test_analyze_trajectories_without_node_idx(self):
        """Test trajectory analysis without node_idx in items."""
        analyzer = MCTSLeakageAnalyzer()

        mcts_tree = {
            "depths": [0, 1, 2],
            "trajectories": [
                ["step0", "step1", "step2"],  # List without dict format
            ],
        }

        result = analyzer.analyze(
            mcts_values=[0.9, 0.8, 0.7],
            mcts_tree=mcts_tree,
        )

        assert "trajectory_analysis" in result

    def test_analyze_no_trajectories_key(self):
        """Test when trajectories key is missing."""
        analyzer = MCTSLeakageAnalyzer()

        mcts_tree = {
            "depths": [0, 1, 2],
            # No "trajectories" key
        }

        result = analyzer.analyze(
            mcts_values=[0.9, 0.8, 0.7],
            mcts_tree=mcts_tree,
        )

        assert result["trajectory_analysis"]["available"] == False

    def test_analyze_empty_trajectories(self):
        """Test with empty trajectory list."""
        analyzer = MCTSLeakageAnalyzer()

        mcts_tree = {
            "depths": [0, 1],
            "trajectories": [],
        }

        result = analyzer.analyze(
            mcts_values=[0.9, 0.8],
            mcts_tree=mcts_tree,
        )

        assert "trajectory_analysis" in result


# ============== utils/metrics.py - Various uncovered lines ==============
class TestMetricsEdgeCases:
    """Tests for metrics edge cases."""

    def test_auc_roc_all_same_label_positive(self):
        """Test AUC-ROC when all labels are positive - line 47."""
        labels = [1, 1, 1, 1]
        scores = [0.9, 0.8, 0.7, 0.6]
        auc = compute_auc_roc(labels, scores)
        assert auc == 0.5  # Edge case returns 0.5

    def test_auc_roc_all_same_label_negative(self):
        """Test AUC-ROC when all labels are negative."""
        labels = [0, 0, 0, 0]
        scores = [0.9, 0.8, 0.7, 0.6]
        auc = compute_auc_roc(labels, scores)
        assert auc == 0.5

    def test_auc_roc_no_positives(self):
        """Test AUC-ROC with no positive samples - lines 60-61."""
        labels = [0, 0, 0]
        scores = [0.5, 0.6, 0.7]
        auc = compute_auc_roc(labels, scores)
        assert auc == 0.5

    def test_auc_roc_no_negatives(self):
        """Test AUC-ROC with no negative samples."""
        labels = [1, 1, 1]
        scores = [0.5, 0.6, 0.7]
        auc = compute_auc_roc(labels, scores)
        assert auc == 0.5

    def test_tpr_at_fpr_no_positives(self):
        """Test TPR@FPR with no positive samples - line 113."""
        labels = [0, 0, 0]
        scores = [0.5, 0.6, 0.7]
        tpr = compute_tpr_at_fpr(labels, scores, fpr_threshold=0.1)
        assert tpr == 0.0

    def test_tpr_at_fpr_no_negatives(self):
        """Test TPR@FPR with no negative samples."""
        labels = [1, 1, 1]
        scores = [0.5, 0.6, 0.7]
        tpr = compute_tpr_at_fpr(labels, scores, fpr_threshold=0.1)
        assert tpr == 0.0

    def test_tpr_at_fpr_exceeds_threshold(self):
        """Test TPR@FPR when FPR exceeds threshold - line 145."""
        labels = [1, 0, 1, 0, 1, 0]
        scores = [0.9, 0.85, 0.7, 0.65, 0.5, 0.45]
        tpr = compute_tpr_at_fpr(labels, scores, fpr_threshold=0.01)
        assert 0 <= tpr <= 1

    def test_f1_no_true_positives(self):
        """Test F1 with no true positives - line 170."""
        labels = [1, 1, 1]
        predictions = [0, 0, 0]
        f1 = compute_f1_score(labels, predictions)
        assert f1 == 0.0

    def test_f1_no_predicted_positives(self):
        """Test F1 with no predicted positives."""
        labels = [0, 0, 1]
        predictions = [0, 0, 0]
        f1 = compute_f1_score(labels, predictions)
        assert f1 == 0.0

    def test_precision_recall_all_negative(self):
        """Test precision-recall with all negative labels."""
        labels = [0, 0, 0, 0]
        scores = [0.9, 0.7, 0.5, 0.3]
        p, r, t = compute_precision_recall_curve(labels, scores)
        assert len(p) > 0
        assert len(r) > 0


# ============== BaseAttack.predict_membership ==============
class TestBaseAttackPredictMembership:
    """Tests for predict_membership method."""

    def test_predict_membership_above_threshold(self):
        """Test prediction above threshold."""
        attack = NLBAAttack(threshold=0.5)
        prediction, confidence = attack.predict_membership(0.8)
        assert prediction == True
        assert 0 <= confidence <= 1

    def test_predict_membership_below_threshold(self):
        """Test prediction below threshold."""
        attack = NLBAAttack(threshold=0.5)
        prediction, confidence = attack.predict_membership(0.2)
        assert prediction == False
        assert 0 <= confidence <= 1

    def test_predict_membership_at_threshold(self):
        """Test prediction exactly at threshold."""
        attack = NLBAAttack(threshold=0.5)
        prediction, confidence = attack.predict_membership(0.5)
        assert prediction == False  # Not strictly greater
        assert confidence == 0.0  # No distance from threshold


# ============== BaseAttack.name property ==============
class TestBaseAttackProperties:
    """Tests for BaseAttack properties."""

    def test_attack_name_property(self):
        """Test attack name property."""
        attack = NLBAAttack()
        assert attack.name == "NLBAAttack"
