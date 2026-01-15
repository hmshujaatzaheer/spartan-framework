"""
SPARTAN Basic Usage Example

Demonstrates basic usage of the SPARTAN framework for
privacy protection in reasoning LLMs.
"""

from spartan import SPARTAN, SPARTANConfig
from spartan.models import MockReasoningLLM


def main():
    """Run basic SPARTAN example."""
    print("=" * 60)
    print("SPARTAN Framework - Basic Usage Example")
    print("=" * 60)

    # Initialize a mock reasoning LLM
    # In practice, replace with actual model
    llm = MockReasoningLLM(seed=42)

    # Create SPARTAN instance with default config
    spartan = SPARTAN(llm)

    # Process a query
    query = "What is the integral of x^2 from 0 to 1?"
    print(f"\nProcessing query: {query}")
    print("-" * 40)

    result = spartan.process(query, num_samples=5)

    # Display results
    print(f"\nOriginal Output: {result.original_output}")
    print(f"Sanitized Output: {result.output}")
    print(f"\nPrivacy Risk Score: {result.risk_score:.4f}")
    print(f"Defense Applied: {result.defense_applied}")
    print(f"Defense Intensity (ε): {result.defense_result.epsilon_used:.4f}")

    # Show leakage components
    print("\nLeakage Analysis:")
    print(f"  - PRM Leakage: {result.risk_analysis.prm_leakage:.4f}")
    print(f"  - Vote Leakage: {result.risk_analysis.vote_leakage:.4f}")
    print(f"  - MCTS Leakage: {result.risk_analysis.mcts_leakage:.4f}")
    print(f"  - Importance Weight: {result.risk_analysis.importance_weight:.4f}")

    # Custom configuration example
    print("\n" + "=" * 60)
    print("Custom Configuration Example")
    print("=" * 60)

    custom_config = SPARTANConfig(
        prm_threshold=0.25,
        vote_threshold=0.35,
        epsilon_min=0.02,
        epsilon_max=0.6,
        accuracy_weight=0.5,
        privacy_weight=0.3,
        compute_weight=0.2,
    )

    spartan_custom = SPARTAN(llm, config=custom_config)
    result_custom = spartan_custom.process(query)

    print(f"\nWith custom config:")
    print(f"  Risk Score: {result_custom.risk_score:.4f}")
    print(f"  Defense Intensity: {result_custom.defense_result.epsilon_used:.4f}")

    # Batch processing example
    print("\n" + "=" * 60)
    print("Batch Processing Example")
    print("=" * 60)

    queries = [
        "What is 2 + 2?",
        "Solve: x^2 - 4 = 0",
        "Find the derivative of sin(x)",
    ]

    results = spartan.batch_process(queries)

    print("\nBatch Results:")
    for q, r in zip(queries, results):
        print(f"  Query: {q[:30]}...")
        print(f"    Risk: {r.risk_score:.4f}, Defense: {r.defense_applied}")

    print("\n" + "=" * 60)
    print("Example Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
