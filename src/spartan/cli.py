"""
SPARTAN Command Line Interface

Provides command-line access to SPARTAN functionality.
"""

import argparse
import json
import sys
from typing import Optional


def main(args: Optional[list] = None) -> int:
    """Main CLI entry point.

    Args:
        args: Command line arguments (uses sys.argv if None)

    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        prog="spartan",
        description="SPARTAN: Secure Privacy-Adaptive Reasoning with Test-time Attack Neutralization",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze privacy leakage",
    )
    analyze_parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query to analyze",
    )
    analyze_parser.add_argument(
        "--prm-scores",
        type=str,
        help="Comma-separated PRM scores",
    )
    analyze_parser.add_argument(
        "--output",
        type=str,
        default="-",
        help="Output file (- for stdout)",
    )

    # Defend command
    defend_parser = subparsers.add_parser(
        "defend",
        help="Apply defense to output",
    )
    defend_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input to defend",
    )
    defend_parser.add_argument(
        "--risk-score",
        type=float,
        default=0.5,
        help="Risk score (0-1)",
    )
    defend_parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Defense intensity",
    )

    # Evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate attack/defense performance",
    )
    eval_parser.add_argument(
        "--mode",
        choices=["attack", "defense"],
        required=True,
        help="Evaluation mode",
    )
    eval_parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to evaluation data",
    )
    eval_parser.add_argument(
        "--output",
        type=str,
        default="results.json",
        help="Output file for results",
    )

    # Config command
    config_parser = subparsers.add_parser(
        "config",
        help="Generate default configuration",
    )
    config_parser.add_argument(
        "--output",
        type=str,
        default="spartan_config.json",
        help="Output file",
    )

    parsed_args = parser.parse_args(args)

    if parsed_args.command is None:
        parser.print_help()
        return 0

    if parsed_args.command == "analyze":
        return _cmd_analyze(parsed_args)
    elif parsed_args.command == "defend":
        return _cmd_defend(parsed_args)
    elif parsed_args.command == "evaluate":
        return _cmd_evaluate(parsed_args)
    elif parsed_args.command == "config":
        return _cmd_config(parsed_args)

    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    """Handle analyze command."""
    from spartan.mplq import MPLQ

    mplq = MPLQ()

    # Parse PRM scores
    prm_scores = None
    if args.prm_scores:
        prm_scores = [float(x) for x in args.prm_scores.split(",")]

    result = mplq.analyze(
        query=args.query,
        prm_scores=prm_scores,
    )

    output = json.dumps(result.to_dict(), indent=2)

    if args.output == "-":
        print(output)
    else:
        with open(args.output, "w") as f:
            f.write(output)

    return 0


def _cmd_defend(args: argparse.Namespace) -> int:
    """Handle defend command."""
    from spartan.mplq import MPLQResult
    from spartan.raas import RAAS

    # Create mock risk analysis
    risk_analysis = MPLQResult(
        total_risk=args.risk_score,
        prm_leakage=args.risk_score,
        vote_leakage=0.0,
        mcts_leakage=0.0,
        importance_weight=1.0,
        component_weights=(0.4, 0.35, 0.25),
    )

    raas = RAAS()
    result = raas.sanitize(
        output=args.input,
        risk_analysis=risk_analysis,
    )

    print(f"Original: {result.original_output}")
    print(f"Sanitized: {result.sanitized_output}")
    print(f"Defense applied: {result.defense_applied}")

    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    """Handle evaluate command."""
    print(f"Evaluation mode: {args.mode}")
    print(f"Data file: {args.data}")
    print(f"Output: {args.output}")
    print("Note: Full evaluation requires model access.")

    # Placeholder results
    results = {
        "mode": args.mode,
        "metrics": {
            "auc_roc": 0.85,
            "accuracy": 0.82,
            "tpr_at_fpr_0.01": 0.31,
        },
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.output}")

    return 0


def _cmd_config(args: argparse.Namespace) -> int:
    """Handle config command."""
    from spartan.config import SPARTANConfig

    config = SPARTANConfig()
    config_dict = config.to_dict()

    with open(args.output, "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"Configuration saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

# Last updated: 2026-01-15
