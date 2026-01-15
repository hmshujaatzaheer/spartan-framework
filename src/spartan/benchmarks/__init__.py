"""
SPARTAN Benchmarks Module

Provides benchmark utilities for evaluating attack and defense performance.
"""

from spartan.benchmarks.datasets import DatasetLoader
from spartan.benchmarks.runner import BenchmarkRunner

__all__ = [
    "BenchmarkRunner",
    "DatasetLoader",
]
