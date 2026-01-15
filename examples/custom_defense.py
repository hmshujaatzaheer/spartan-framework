"""
SPARTAN Custom Defense Example

Demonstrates how to customize defense parameters and
use individual defense modules.
"""

from spartan import SPARTAN, SPARTANConfig
from spartan.mplq import MPLQ, MPLQResult
from spartan.raas import RAAS
from spartan.rppo import RPPO
from spartan.models import MockReasoningLLM


def main():
    """Run custom defense example."""
    print("=" * 60)
    print("SPARTAN Framework - Custom Defense Example")
    print("=" * 60)
    
    # Initialize mock LLM with member mode (high risk)
    llm = MockReasoningLLM(member_mode=True, seed=42)
    query = "Solve: x^2 + 2x - 3 = 0"
    
    # Generate output
    output = llm.generate(query, num_samples=5, use_mcts=True)
    
    print(f"\nQuery: {query}")
    print(f"Output: {output.output}")
    
    # Manual MPLQ Analysis
    print("\n" + "-" * 60)
    print("Step 1: Manual MPLQ Analysis")
    print("-" * 60)
    
    mplq = MPLQ()
    risk_analysis = mplq.analyze(
        query=query,
        reasoning_steps=output.reasoning_steps,
        prm_scores=output.prm_scores,
        vote_distribution=output.vote_distribution,
        mcts_values=output.mcts_values,
    )
    
    print(f"Total Risk: {risk_analysis.total_risk:.4f}")
    print(f"PRM Leakage: {risk_analysis.prm_leakage:.4f}")
    print(f"Vote Leakage: {risk_analysis.vote_leakage:.4f}")
    print(f"MCTS Leakage: {risk_analysis.mcts_leakage:.4f}")
    print(f"Importance Weight: {risk_analysis.importance_weight:.4f}")
    
    # Update MPLQ weights
    print("\nUpdating component weights...")
    mplq.update_weights(alpha=0.5, beta=0.3, gamma=0.2)
    
    risk_analysis_updated = mplq.analyze(
        query=query,
        prm_scores=output.prm_scores,
        vote_distribution=output.vote_distribution,
        mcts_values=output.mcts_values,
    )
    
    print(f"Updated Total Risk: {risk_analysis_updated.total_risk:.4f}")
    
    # Custom RAAS Configuration
    print("\n" + "-" * 60)
    print("Step 2: Custom RAAS Defense")
    print("-" * 60)
    
    config = SPARTANConfig(
        epsilon_min=0.05,
        epsilon_max=0.8,
        prm_threshold=0.2,
        vote_threshold=0.3,
        mcts_threshold=0.4,
        use_feature_selective=True,
        use_implicit_rewards=True,
    )
    
    raas = RAAS(config=config)
    
    defense_result = raas.sanitize(
        output=output.output,
        risk_analysis=risk_analysis,
        reasoning_steps=output.reasoning_steps,
        vote_distribution=output.vote_distribution,
        mcts_tree=output.mcts_tree,
    )
    
    print(f"Defense Applied: {defense_result.defense_applied}")
    print(f"PRM Defense: {defense_result.prm_defense_applied}")
    print(f"Vote Defense: {defense_result.vote_defense_applied}")
    print(f"MCTS Defense: {defense_result.mcts_defense_applied}")
    print(f"Epsilon Used: {defense_result.epsilon_used:.4f}")
    
    # RPPO Optimization
    print("\n" + "-" * 60)
    print("Step 3: RPPO Parameter Optimization")
    print("-" * 60)
    
    rppo = RPPO(config=config)
    
    # Simulate optimization over multiple iterations
    print("Running optimization iterations...")
    for i in range(20):
        rppo.update({
            "risk_score": 0.3 + 0.02 * (i % 10),
            "accuracy": 0.85 - 0.01 * (i % 5),
            "compute": 0.4,
            "defense_intensity": 0.1 + 0.02 * i,
        })
    
    opt_result = rppo.get_optimal_params()
    
    if opt_result:
        print(f"\nOptimized Parameters:")
        print(f"  Epsilon Min: {opt_result.params.get('epsilon_min', 'N/A'):.4f}")
        print(f"  Epsilon Max: {opt_result.params.get('epsilon_max', 'N/A'):.4f}")
        print(f"  Total Reward: {opt_result.reward:.4f}")
        print(f"  Is Pareto Optimal: {opt_result.is_pareto_optimal}")
        
        # Apply optimized params to RAAS
        raas.update_params(opt_result.params)
        print("\nApplied optimized params to RAAS")
    
    # Get statistics
    stats = rppo.get_statistics()
    print(f"\nOptimization Statistics:")
    print(f"  Total Episodes: {stats['total_episodes']}")
    print(f"  Best Arm: {stats['best_arm']}")
    print(f"  Pareto Front Size: {stats['pareto_front_size']}")
    
    # Full pipeline with custom config
    print("\n" + "-" * 60)
    print("Step 4: Full Pipeline with Custom Config")
    print("-" * 60)
    
    spartan = SPARTAN(llm, config=config)
    result = spartan.process(query)
    
    print(f"Final Output: {result.output}")
    print(f"Final Risk Score: {result.risk_score:.4f}")
    print(f"Defense Applied: {result.defense_applied}")
    
    print("\n" + "=" * 60)
    print("Example Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
