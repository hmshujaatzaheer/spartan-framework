"""
Tests for Evaluate Module

Comprehensive tests for evaluation functions.
"""

import json
import os
import tempfile

import pytest

from spartan.config import SPARTANConfig
from spartan.evaluate import evaluate_attack, evaluate_defense, main


class TestEvaluateAttack:
    """Tests for attack evaluation."""

    def test_evaluate_all_attacks(self):
        """Test evaluating all attack types."""
        metrics = evaluate_attack(attack_type="all", num_samples=10)

        assert "nlba_auc_roc" in metrics
        assert "smva_auc_roc" in metrics
        assert "mvna_auc_roc" in metrics

        for key, value in metrics.items():
            if "auc" in key:
                assert 0.0 <= value <= 1.0

    def test_evaluate_nlba(self):
        """Test evaluating NLBA attack."""
        metrics = evaluate_attack(attack_type="nlba", num_samples=10)
        assert "nlba_auc_roc" in metrics
        assert 0.0 <= metrics["nlba_auc_roc"] <= 1.0

    def test_evaluate_smva(self):
        """Test evaluating SMVA attack."""
        metrics = evaluate_attack(attack_type="smva", num_samples=10)
        assert "smva_auc_roc" in metrics
        assert 0.0 <= metrics["smva_auc_roc"] <= 1.0

    def test_evaluate_mvna(self):
        """Test evaluating MVNA attack."""
        metrics = evaluate_attack(attack_type="mvna", num_samples=10)
        assert "mvna_auc_roc" in metrics
        assert 0.0 <= metrics["mvna_auc_roc"] <= 1.0

    def test_evaluate_with_output(self):
        """Test evaluation with output file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            metrics = evaluate_attack(
                attack_type="nlba",
                num_samples=10,
                output_path=temp_path,
            )

            assert os.path.exists(temp_path)
            with open(temp_path) as f:
                loaded = json.load(f)
            assert "nlba_auc_roc" in loaded
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_evaluate_invalid_attack(self):
        """Test evaluating invalid attack type."""
        with pytest.raises(ValueError):
            evaluate_attack(attack_type="invalid", num_samples=10)


class TestEvaluateDefense:
    """Tests for defense evaluation."""

    def test_evaluate_defense_basic(self):
        """Test basic defense evaluation."""
        metrics = evaluate_defense(num_samples=10)

        assert "avg_member_risk" in metrics
        assert "avg_nonmember_risk" in metrics
        assert "defense_rate" in metrics

    def test_evaluate_defense_with_config(self):
        """Test defense evaluation with custom config."""
        config = SPARTANConfig(epsilon_max=0.8)
        metrics = evaluate_defense(num_samples=10, config=config)

        assert "avg_member_risk" in metrics
        assert "avg_epsilon" in metrics

    def test_evaluate_defense_with_output(self):
        """Test defense evaluation with output file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            metrics = evaluate_defense(
                num_samples=10,
                output_path=temp_path,
            )

            assert os.path.exists(temp_path)
            with open(temp_path) as f:
                loaded = json.load(f)
            assert "defense_rate" in loaded
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestEvaluateMain:
    """Tests for evaluate main function."""

    def test_main_attack_mode(self):
        """Test main with attack mode."""
        import sys
        from io import StringIO
        from unittest.mock import patch

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            with patch(
                "sys.argv",
                ["evaluate", "--mode", "attack", "--num-samples", "10", "--output", temp_path],
            ):
                with patch("sys.stdout", new_callable=StringIO):
                    try:
                        main()
                    except SystemExit:
                        pass
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_main_defense_mode(self):
        """Test main with defense mode."""
        import sys
        from io import StringIO
        from unittest.mock import patch

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            with patch(
                "sys.argv",
                ["evaluate", "--mode", "defense", "--num-samples", "10", "--output", temp_path],
            ):
                with patch("sys.stdout", new_callable=StringIO):
                    try:
                        main()
                    except SystemExit:
                        pass
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
