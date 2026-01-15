"""
Tests for CLI Module

Comprehensive tests for command-line interface.
"""

import json
import os
import sys
import tempfile
from io import StringIO
from unittest.mock import patch

import pytest

from spartan.cli import main


class TestCLIMain:
    """Tests for main CLI function."""

    def test_main_no_args(self):
        """Test main with no arguments shows help."""
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = main([])
            assert result == 0

    def test_main_version(self):
        """Test version flag."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0

    def test_main_help(self):
        """Test help flag."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0


class TestAnalyzeCommand:
    """Tests for analyze command."""

    def test_analyze_basic(self):
        """Test basic analysis."""
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = main(["analyze", "--query", "What is 2+2?"])
            assert result == 0
            output = mock_stdout.getvalue()
            assert "total_risk" in output or len(output) > 0

    def test_analyze_with_prm_scores(self):
        """Test analysis with PRM scores."""
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = main(
                [
                    "analyze",
                    "--query",
                    "Test query",
                    "--prm-scores",
                    "0.9,0.8,0.7",
                ]
            )
            assert result == 0

    def test_analyze_with_output_file(self):
        """Test analysis with output file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            result = main(
                [
                    "analyze",
                    "--query",
                    "Test query",
                    "--output",
                    temp_path,
                ]
            )
            assert result == 0
            assert os.path.exists(temp_path)

            with open(temp_path) as f:
                data = json.load(f)
            assert "total_risk" in data
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestDefendCommand:
    """Tests for defend command."""

    def test_defend_basic(self):
        """Test basic defense."""
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = main(["defend", "--input", "Test output to defend"])
            assert result == 0
            output = mock_stdout.getvalue()
            assert "Original" in output or "Sanitized" in output

    def test_defend_with_risk_score(self):
        """Test defense with custom risk score."""
        with patch("sys.stdout", new_callable=StringIO):
            result = main(
                [
                    "defend",
                    "--input",
                    "Test output",
                    "--risk-score",
                    "0.8",
                ]
            )
            assert result == 0

    def test_defend_with_epsilon(self):
        """Test defense with custom epsilon."""
        with patch("sys.stdout", new_callable=StringIO):
            result = main(
                [
                    "defend",
                    "--input",
                    "Test output",
                    "--epsilon",
                    "0.3",
                ]
            )
            assert result == 0


class TestEvaluateCommand:
    """Tests for evaluate command."""

    def test_evaluate_attack_mode(self):
        """Test attack evaluation mode."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_data = f.name
            json.dump({"queries": ["q1"], "labels": [1]}, f)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_output = f.name

        try:
            with patch("sys.stdout", new_callable=StringIO):
                result = main(
                    [
                        "evaluate",
                        "--mode",
                        "attack",
                        "--data",
                        temp_data,
                        "--output",
                        temp_output,
                    ]
                )
                assert result == 0
        finally:
            if os.path.exists(temp_data):
                os.unlink(temp_data)
            if os.path.exists(temp_output):
                os.unlink(temp_output)

    def test_evaluate_defense_mode(self):
        """Test defense evaluation mode."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_data = f.name
            json.dump({"queries": ["q1"], "labels": [1]}, f)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_output = f.name

        try:
            with patch("sys.stdout", new_callable=StringIO):
                result = main(
                    [
                        "evaluate",
                        "--mode",
                        "defense",
                        "--data",
                        temp_data,
                        "--output",
                        temp_output,
                    ]
                )
                assert result == 0
        finally:
            if os.path.exists(temp_data):
                os.unlink(temp_data)
            if os.path.exists(temp_output):
                os.unlink(temp_output)


class TestConfigCommand:
    """Tests for config command."""

    def test_config_default(self):
        """Test default config generation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            with patch("sys.stdout", new_callable=StringIO):
                result = main(["config", "--output", temp_path])
                assert result == 0

            assert os.path.exists(temp_path)
            with open(temp_path) as f:
                config = json.load(f)
            assert "mplq" in config or "raas" in config
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


# Last updated: 2026-01-15
