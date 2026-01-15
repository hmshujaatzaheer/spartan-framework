"""
SPARTAN Evaluation Module

Provides evaluation utilities for attack and defense performance.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from spartan import SPARTAN, SPARTANConfig
from spartan.attacks import NLBAAttack, SMVAAttack, MVNAAttack
from spartan.benchmarks import BenchmarkRunner, DatasetLoader
from spartan.models import MockReasoningLLM
from spartan.utils.metrics import compute_auc_roc, compute_accuracy, compute_tpr_at_fpr


def evaluate_attack(
    attack_type: str = "all",
    num_samples: int = 100,
    output_path: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate attack effectiveness.
    
    Args:
        attack_type: Type of attack ("nlba", "smva", "mvna", or "all")
        num_samples: Number of samples per class
        output_path: Path to save results
        
    Returns:
        Dictionary of evaluation metrics
    """
    runner = BenchmarkRunner()
    
    if attack_type == "all":
        metrics = runner.benchmark_attacks(
            num_member=num_samples,
            num_nonmember=num_samples,
        )
    else:
        # Single attack evaluation
        member_llm = MockReasoningLLM(member_mode=True, seed=42)
        nonmember_llm = MockReasoningLLM(member_mode=False, seed=43)
        
        attacks = {
            "nlba": NLBAAttack(),
            "smva": SMVAAttack(),
            "mvna": MVNAAttack(),
        }
        
        attack = attacks.get(attack_type)
        if attack is None:
            raise ValueError(f"Unknown attack type: {attack_type}")
        
        scores = []
        labels = []
        
        for i in range(num_samples):
            # Member
            output = member_llm.generate(f"Member {i}", use_mcts=True)
            if attack_type == "nlba":
                result = attack.execute(None, "", prm_scores=output.prm_scores)
            elif attack_type == "smva":
                result = attack.execute(None, "", vote_distribution=output.vote_distribution)
            else:
                result = attack.execute(None, "", mcts_values=output.mcts_values)
            scores.append(result.success_score)
            labels.append(1)
            
            # Non-member
            output = nonmember_llm.generate(f"Non-member {i}", use_mcts=True)
            if attack_type == "nlba":
                result = attack.execute(None, "", prm_scores=output.prm_scores)
            elif attack_type == "smva":
                result = attack.execute(None, "", vote_distribution=output.vote_distribution)
            else:
                result = attack.execute(None, "", mcts_values=output.mcts_values)
            scores.append(result.success_score)
            labels.append(0)
        
        metrics = {
            f"{attack_type}_auc_roc": compute_auc_roc(labels, scores),
            f"{attack_type}_tpr_at_fpr_0.01": compute_tpr_at_fpr(labels, scores, 0.01),
            f"{attack_type}_accuracy": compute_accuracy(
                labels, [1 if s > 0.5 else 0 for s in scores]
            ),
        }
    
    if output_path:
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
    
    return metrics


def evaluate_defense(
    num_samples: int = 100,
    config: Optional[SPARTANConfig] = None,
    output_path: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate defense performance.
    
    Args:
        num_samples: Number of samples per class
        config: SPARTAN configuration
        output_path: Path to save results
        
    Returns:
        Dictionary of evaluation metrics
    """
    config = config or SPARTANConfig()
    runner = BenchmarkRunner(config=config)
    
    metrics = runner.benchmark_defense(
        num_member=num_samples,
        num_nonmember=num_samples,
    )
    
    # Add timing metrics
    timing = runner.benchmark_timing(num_samples=min(20, num_samples))
    metrics.update(timing)
    
    if output_path:
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
    
    return metrics


def main():
    """Run evaluation from command line."""
    parser = argparse.ArgumentParser(description="SPARTAN Evaluation")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["attack", "defense"],
        required=True,
        help="Evaluation mode",
    )
    parser.add_argument(
        "--attack-type",
        type=str,
        default="all",
        choices=["nlba", "smva", "mvna", "all"],
        help="Attack type to evaluate",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples per class",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output file path",
    )
    
    args = parser.parse_args()
    
    if args.mode == "attack":
        metrics = evaluate_attack(
            attack_type=args.attack_type,
            num_samples=args.num_samples,
            output_path=args.output,
        )
        print("\nAttack Evaluation Results:")
    else:
        metrics = evaluate_defense(
            num_samples=args.num_samples,
            output_path=args.output,
        )
        print("\nDefense Evaluation Results:")
    
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
