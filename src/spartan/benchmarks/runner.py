"""
Benchmark Runner

Runs comprehensive benchmarks for SPARTAN attack and defense evaluation.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from spartan import SPARTAN, SPARTANConfig
from spartan.attacks import MVNAAttack, NLBAAttack, SMVAAttack
from spartan.models import MockReasoningLLM
from spartan.utils.metrics import compute_accuracy, compute_auc_roc, compute_tpr_at_fpr


@dataclass
class BenchmarkResult:
    """Result from benchmark run.

    Attributes:
        attack_metrics: Attack effectiveness metrics
        defense_metrics: Defense performance metrics
        timing_metrics: Computational timing metrics
        config: Configuration used
    """

    attack_metrics: Dict[str, float] = field(default_factory=dict)
    defense_metrics: Dict[str, float] = field(default_factory=dict)
    timing_metrics: Dict[str, float] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "attack_metrics": self.attack_metrics,
            "defense_metrics": self.defense_metrics,
            "timing_metrics": self.timing_metrics,
            "config": self.config,
        }

    def save(self, path: str) -> None:
        """Save results to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class BenchmarkRunner:
    """Runs SPARTAN benchmarks.

    Example:
        >>> runner = BenchmarkRunner()
        >>> result = runner.run_full_benchmark(num_samples=100)
        >>> result.save("benchmark_results.json")
    """

    def __init__(
        self,
        config: Optional[SPARTANConfig] = None,
        seed: int = 42,
    ):
        """Initialize benchmark runner.

        Args:
            config: SPARTAN configuration
            seed: Random seed for reproducibility
        """
        self.config = config or SPARTANConfig()
        self.seed = seed
        np.random.seed(seed)

    def run_full_benchmark(
        self,
        num_samples: int = 100,
        num_member: Optional[int] = None,
        num_nonmember: Optional[int] = None,
    ) -> BenchmarkResult:
        """Run full benchmark suite.

        Args:
            num_samples: Total number of samples
            num_member: Number of member samples (default: half)
            num_nonmember: Number of non-member samples (default: half)

        Returns:
            BenchmarkResult with all metrics
        """
        if num_member is None:
            num_member = num_samples // 2
        if num_nonmember is None:
            num_nonmember = num_samples - num_member

        # Generate test data
        queries = [f"Test query {i}" for i in range(num_samples)]

        # Run attack benchmarks
        attack_metrics = self.benchmark_attacks(
            num_member=num_member,
            num_nonmember=num_nonmember,
        )

        # Run defense benchmarks
        defense_metrics = self.benchmark_defense(
            num_member=num_member,
            num_nonmember=num_nonmember,
        )

        # Run timing benchmarks
        timing_metrics = self.benchmark_timing(num_samples=min(20, num_samples))

        return BenchmarkResult(
            attack_metrics=attack_metrics,
            defense_metrics=defense_metrics,
            timing_metrics=timing_metrics,
            config=self.config.to_dict(),
        )

    def benchmark_attacks(
        self,
        num_member: int = 50,
        num_nonmember: int = 50,
    ) -> Dict[str, float]:
        """Benchmark attack effectiveness.

        Args:
            num_member: Number of member samples
            num_nonmember: Number of non-member samples

        Returns:
            Dictionary of attack metrics
        """
        # Initialize models
        member_llm = MockReasoningLLM(member_mode=True, seed=self.seed)
        nonmember_llm = MockReasoningLLM(member_mode=False, seed=self.seed + 1)

        # Initialize attacks
        nlba = NLBAAttack()
        smva = SMVAAttack()
        mvna = MVNAAttack()

        metrics = {}

        # Collect scores
        for attack_name, attack in [("nlba", nlba), ("smva", smva), ("mvna", mvna)]:
            scores = []
            labels = []

            # Member samples
            for i in range(num_member):
                output = member_llm.generate(f"Member query {i}", use_mcts=True)

                if attack_name == "nlba":
                    result = attack.execute(None, "", prm_scores=output.prm_scores)
                elif attack_name == "smva":
                    result = attack.execute(None, "", vote_distribution=output.vote_distribution)
                else:
                    result = attack.execute(None, "", mcts_values=output.mcts_values)

                scores.append(result.success_score)
                labels.append(1)

            # Non-member samples
            for i in range(num_nonmember):
                output = nonmember_llm.generate(f"Non-member query {i}", use_mcts=True)

                if attack_name == "nlba":
                    result = attack.execute(None, "", prm_scores=output.prm_scores)
                elif attack_name == "smva":
                    result = attack.execute(None, "", vote_distribution=output.vote_distribution)
                else:
                    result = attack.execute(None, "", mcts_values=output.mcts_values)

                scores.append(result.success_score)
                labels.append(0)

            # Compute metrics
            auc = compute_auc_roc(labels, scores)
            tpr = compute_tpr_at_fpr(labels, scores, fpr_threshold=0.01)

            metrics[f"{attack_name}_auc_roc"] = auc
            metrics[f"{attack_name}_tpr_at_fpr_0.01"] = tpr

        # Combined attack (max of all)
        # Simulated - would need actual combination logic
        metrics["combined_auc_roc"] = max(
            metrics["nlba_auc_roc"],
            metrics["smva_auc_roc"],
            metrics["mvna_auc_roc"],
        )

        return metrics

    def benchmark_defense(
        self,
        num_member: int = 50,
        num_nonmember: int = 50,
    ) -> Dict[str, float]:
        """Benchmark defense performance.

        Args:
            num_member: Number of member samples
            num_nonmember: Number of non-member samples

        Returns:
            Dictionary of defense metrics
        """
        # Initialize models with defense
        member_llm = MockReasoningLLM(member_mode=True, seed=self.seed)
        nonmember_llm = MockReasoningLLM(member_mode=False, seed=self.seed + 1)

        spartan_member = SPARTAN(member_llm, config=self.config)
        spartan_nonmember = SPARTAN(nonmember_llm, config=self.config)

        metrics = {}

        # Process samples
        member_results = []
        nonmember_results = []

        for i in range(num_member):
            result = spartan_member.process(f"Member query {i}")
            member_results.append(result)

        for i in range(num_nonmember):
            result = spartan_nonmember.process(f"Non-member query {i}")
            nonmember_results.append(result)

        # Compute metrics
        member_risks = [r.risk_score for r in member_results]
        nonmember_risks = [r.risk_score for r in nonmember_results]

        metrics["avg_member_risk"] = float(np.mean(member_risks))
        metrics["avg_nonmember_risk"] = float(np.mean(nonmember_risks))
        metrics["risk_separation"] = metrics["avg_member_risk"] - metrics["avg_nonmember_risk"]

        # Defense rate
        all_results = member_results + nonmember_results
        metrics["defense_rate"] = sum(r.defense_applied for r in all_results) / len(all_results)

        # Average epsilon used
        epsilons = [r.defense_result.epsilon_used for r in all_results]
        metrics["avg_epsilon"] = float(np.mean(epsilons))

        return metrics

    def benchmark_timing(
        self,
        num_samples: int = 20,
    ) -> Dict[str, float]:
        """Benchmark computational timing.

        Args:
            num_samples: Number of samples for timing

        Returns:
            Dictionary of timing metrics
        """
        llm = MockReasoningLLM(seed=self.seed)
        spartan = SPARTAN(llm, config=self.config)

        # Warm up
        spartan.process("Warmup query")

        # Time processing
        times = []
        for i in range(num_samples):
            start = time.time()
            spartan.process(f"Timing query {i}")
            elapsed = time.time() - start
            times.append(elapsed)

        return {
            "avg_time_ms": float(np.mean(times) * 1000),
            "std_time_ms": float(np.std(times) * 1000),
            "min_time_ms": float(np.min(times) * 1000),
            "max_time_ms": float(np.max(times) * 1000),
            "throughput_qps": 1.0 / np.mean(times) if np.mean(times) > 0 else 0,
        }


def main():
    """Run benchmarks from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Run SPARTAN benchmarks")
    parser.add_argument("--model", type=str, default="mock", help="Model to use")
    parser.add_argument("--dataset", type=str, default="mock", help="Dataset to use")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output file")

    args = parser.parse_args()

    runner = BenchmarkRunner()
    result = runner.run_full_benchmark(num_samples=args.num_samples)
    result.save(args.output)

    print(f"Benchmark results saved to {args.output}")
    print(f"\nAttack Metrics:")
    for k, v in result.attack_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nDefense Metrics:")
    for k, v in result.defense_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nTiming Metrics:")
    for k, v in result.timing_metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
