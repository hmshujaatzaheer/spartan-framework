"""
Tests for Benchmarks Module

Comprehensive tests for BenchmarkRunner and DatasetLoader.
"""

import pytest
import numpy as np
import json
import tempfile
import os

from spartan.benchmarks import BenchmarkRunner, DatasetLoader
from spartan.benchmarks.datasets import BenchmarkDataset
from spartan.benchmarks.runner import BenchmarkResult
from spartan.config import SPARTANConfig


class TestBenchmarkDataset:
    """Tests for BenchmarkDataset class."""

    def test_creation(self):
        """Test dataset creation."""
        dataset = BenchmarkDataset(
            name="test",
            queries=["q1", "q2", "q3"],
            labels=[1, 0, 1],
        )
        assert dataset.name == "test"
        assert len(dataset.queries) == 3
        assert len(dataset.labels) == 3

    def test_len(self):
        """Test __len__ method."""
        dataset = BenchmarkDataset(
            name="test",
            queries=["q1", "q2"],
            labels=[1, 0],
        )
        assert len(dataset) == 2

    def test_split(self):
        """Test dataset splitting."""
        dataset = BenchmarkDataset(
            name="test",
            queries=[f"q{i}" for i in range(10)],
            labels=[i % 2 for i in range(10)],
        )
        train, test = dataset.split(train_ratio=0.8, seed=42)
        assert len(train) == 8
        assert len(test) == 2
        assert train.name == "test_train"
        assert test.name == "test_test"


class TestDatasetLoader:
    """Tests for DatasetLoader class."""

    def test_load_mock(self):
        """Test loading mock dataset."""
        dataset = DatasetLoader.load_mock(num_samples=100, seed=42)
        assert len(dataset) == 100
        assert dataset.name == "mock"
        assert sum(dataset.labels) > 0  # Has positive labels
        assert sum(1 - l for l in dataset.labels) > 0  # Has negative labels

    def test_load_mock_member_ratio(self):
        """Test mock dataset with custom member ratio."""
        dataset = DatasetLoader.load_mock(
            num_samples=100,
            member_ratio=0.7,
            seed=42,
        )
        assert len(dataset) == 100
        # Member ratio should be approximately 0.7
        member_count = sum(dataset.labels)
        assert 60 <= member_count <= 80

    def test_load_from_json_file(self):
        """Test loading dataset from JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "queries": ["q1", "q2", "q3"],
                    "labels": [1, 0, 1],
                    "metadata": {"source": "test"},
                },
                f,
            )
            temp_path = f.name

        try:
            dataset = DatasetLoader.load_from_file(temp_path)
            assert len(dataset) == 3
            assert dataset.labels == [1, 0, 1]
        finally:
            os.unlink(temp_path)

    def test_load_from_csv_file(self):
        """Test loading dataset from CSV file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("query,label\n")
            f.write("question 1,1\n")
            f.write("question 2,0\n")
            temp_path = f.name

        try:
            dataset = DatasetLoader.load_from_file(temp_path)
            assert len(dataset) == 2
            assert dataset.labels == [1, 0]
        finally:
            os.unlink(temp_path)

    def test_load_unsupported_format(self):
        """Test loading unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                DatasetLoader.load_from_file(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_prm800k_sample(self):
        """Test loading PRM800K sample dataset."""
        dataset = DatasetLoader.load_prm800k_sample(num_samples=50, seed=42)
        assert len(dataset) == 50
        assert dataset.name == "prm800k_sample"
        assert dataset.metadata["synthetic"] is True


class TestBenchmarkResult:
    """Tests for BenchmarkResult class."""

    def test_creation(self):
        """Test result creation."""
        result = BenchmarkResult(
            attack_metrics={"auc": 0.85},
            defense_metrics={"reduction": 0.6},
            timing_metrics={"avg_ms": 100},
            config={"epsilon": 0.1},
        )
        assert result.attack_metrics["auc"] == 0.85
        assert result.defense_metrics["reduction"] == 0.6

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = BenchmarkResult(
            attack_metrics={"auc": 0.85},
            defense_metrics={"reduction": 0.6},
        )
        d = result.to_dict()
        assert "attack_metrics" in d
        assert "defense_metrics" in d
        assert d["attack_metrics"]["auc"] == 0.85

    def test_save(self):
        """Test saving results to file."""
        result = BenchmarkResult(
            attack_metrics={"auc": 0.85},
            defense_metrics={"reduction": 0.6},
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            result.save(temp_path)
            with open(temp_path) as f:
                loaded = json.load(f)
            assert loaded["attack_metrics"]["auc"] == 0.85
        finally:
            os.unlink(temp_path)


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner class."""

    def test_init_default(self):
        """Test default initialization."""
        runner = BenchmarkRunner()
        assert runner.config is not None
        assert runner.seed == 42

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = SPARTANConfig(epsilon_max=0.8)
        runner = BenchmarkRunner(config=config, seed=123)
        assert runner.config.epsilon_max == 0.8
        assert runner.seed == 123

    def test_benchmark_attacks(self):
        """Test attack benchmarking."""
        runner = BenchmarkRunner(seed=42)
        metrics = runner.benchmark_attacks(num_member=10, num_nonmember=10)

        assert "nlba_auc_roc" in metrics
        assert "smva_auc_roc" in metrics
        assert "mvna_auc_roc" in metrics
        assert "combined_auc_roc" in metrics

        # AUC should be between 0 and 1
        for key, value in metrics.items():
            if "auc" in key:
                assert 0.0 <= value <= 1.0

    def test_benchmark_defense(self):
        """Test defense benchmarking."""
        runner = BenchmarkRunner(seed=42)
        metrics = runner.benchmark_defense(num_member=10, num_nonmember=10)

        assert "avg_member_risk" in metrics
        assert "avg_nonmember_risk" in metrics
        assert "defense_rate" in metrics
        assert "avg_epsilon" in metrics

    def test_benchmark_timing(self):
        """Test timing benchmarking."""
        runner = BenchmarkRunner(seed=42)
        metrics = runner.benchmark_timing(num_samples=5)

        assert "avg_time_ms" in metrics
        assert "std_time_ms" in metrics
        assert "min_time_ms" in metrics
        assert "max_time_ms" in metrics
        assert "throughput_qps" in metrics

        assert metrics["avg_time_ms"] > 0
        assert metrics["throughput_qps"] > 0

    def test_run_full_benchmark(self):
        """Test full benchmark suite."""
        runner = BenchmarkRunner(seed=42)
        result = runner.run_full_benchmark(num_samples=20)

        assert isinstance(result, BenchmarkResult)
        assert len(result.attack_metrics) > 0
        assert len(result.defense_metrics) > 0
        assert len(result.timing_metrics) > 0
