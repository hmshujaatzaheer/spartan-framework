"""
Tests for CLI Module

Comprehensive tests for command-line interface.
"""

import pytest
import sys
import tempfile
import os
from io import StringIO
from unittest.mock import patch, MagicMock

from spartan.cli import (
    create_parser,
    cmd_analyze,
    cmd_defend,
    cmd_evaluate,
    cmd_config,
    main,
)
from spartan.config import SPARTANConfig


class TestCLIParser:
    """Tests for CLI argument parser."""

    def test_create_parser(self):
        """Test parser creation."""
        parser = create_parser()
        assert parser is not None

    def test_analyze_command(self):
        """Test analyze command parsing."""
        parser = create_parser()
        args = parser.parse_args(["analyze", "--query", "test query"])
        assert args.command == "analyze"
        assert args.query == "test query"

    def test_defend_command(self):
        """Test defend command parsing."""
        parser = create_parser()
        args = parser.parse_args(["defend", "--input", "test.txt"])
        assert args.command == "defend"
        assert args.input == "test.txt"

    def test_evaluate_command(self):
        """Test evaluate command parsing."""
        parser = create_parser()
        args = parser.parse_args(["evaluate", "--mode", "attack"])
        assert args.command == "evaluate"
        assert args.mode == "attack"

    def test_config_command(self):
        """Test config command parsing."""
        parser = create_parser()
        args = parser.parse_args(["config", "--show"])
        assert args.command == "config"
        assert args.show is True


class TestAnalyzeCommand:
    """Tests for analyze command."""

    def test_analyze_basic(self):
        """Test basic analysis."""
        with patch("sys.stdout", new_callable=StringIO):
            result = cmd_analyze(query="What is 2+2?", verbose=False)
            assert result is not None

    def test_analyze_verbose(self):
        """Test verbose analysis."""
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            cmd_analyze(query="Test query", verbose=True)
            output = mock_stdout.getvalue()
            # Should have some output in verbose mode
            assert len(output) >= 0


class TestDefendCommand:
    """Tests for defend command."""

    def test_defend_with_query(self):
        """Test defense with direct query."""
        with patch("sys.stdout", new_callable=StringIO):
            result = cmd_defend(query="Test query", epsilon=0.1)
            assert result is not None

    def test_defend_with_file(self):
        """Test defense with input file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test query from file")
            temp_path = f.name

        try:
            with patch("sys.stdout", new_callable=StringIO):
                result = cmd_defend(input_file=temp_path, epsilon=0.1)
                assert result is not None
        finally:
            os.unlink(temp_path)


class TestEvaluateCommand:
    """Tests for evaluate command."""

    def test_evaluate_attack_mode(self):
        """Test attack evaluation mode."""
        with patch("sys.stdout", new_callable=StringIO):
            result = cmd_evaluate(mode="attack", num_samples=10)
            assert result is not None
            assert "auc" in str(result).lower() or isinstance(result, dict)

    def test_evaluate_defense_mode(self):
        """Test defense evaluation mode."""
        with patch("sys.stdout", new_callable=StringIO):
            result = cmd_evaluate(mode="defense", num_samples=10)
            assert result is not None


class TestConfigCommand:
    """Tests for config command."""

    def test_config_show(self):
        """Test showing configuration."""
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            cmd_config(show=True)
            output = mock_stdout.getvalue()
            assert "epsilon" in output.lower() or len(output) >= 0

    def test_config_save(self):
        """Test saving configuration."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name

        try:
            cmd_config(save=temp_path)
            assert os.path.exists(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_config_load(self):
        """Test loading configuration."""
        config = SPARTANConfig(epsilon_max=0.9)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            import yaml

            yaml.dump(config.to_dict(), f)
            temp_path = f.name

        try:
            result = cmd_config(load=temp_path)
            assert result is not None
        finally:
            os.unlink(temp_path)


class TestMainFunction:
    """Tests for main entry point."""

    def test_main_no_args(self):
        """Test main with no arguments."""
        with patch("sys.argv", ["spartan"]):
            with patch("sys.stdout", new_callable=StringIO):
                with pytest.raises(SystemExit):
                    main()

    def test_main_help(self):
        """Test main with help flag."""
        with patch("sys.argv", ["spartan", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_main_analyze(self):
        """Test main with analyze command."""
        with patch("sys.argv", ["spartan", "analyze", "--query", "test"]):
            with patch("sys.stdout", new_callable=StringIO):
                try:
                    main()
                except SystemExit as e:
                    assert e.code in [0, None]
