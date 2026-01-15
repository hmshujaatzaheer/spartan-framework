"""
SPARTAN Benchmarks Module

Provides benchmark utilities for evaluating attack and defense performance.
"""

from spartan.benchmarks.runner import BenchmarkRunner
from spartan.benchmarks.datasets import DatasetLoader

__all__ = [
    "BenchmarkRunner",
    "DatasetLoader",
]
