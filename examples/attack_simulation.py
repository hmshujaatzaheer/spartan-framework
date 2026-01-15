"""
SPARTAN Attack Simulation Example

Demonstrates how to use SPARTAN's attack implementations
to evaluate privacy vulnerabilities.
"""

from spartan.attacks import MVNAAttack, NLBAAttack, SMVAAttack
from spartan.models import MockReasoningLLM


def main():
    """Run attack simulation example."""
    print("=" * 60)
    print("SPARTAN Framework - Attack Simulation Example")
    print("=" * 60)

    # Create member and non-member models
    member_llm = MockReasoningLLM(member_mode=True, seed=42)
    nonmember_llm = MockReasoningLLM(member_mode=False, seed=42)

    query = "What is the derivative of x^3?"

    # Generate outputs
    member_output = member_llm.generate(query, use_mcts=True)
    nonmember_output = nonmember_llm.generate(query, use_mcts=True)

    # Initialize attacks
    nlba = NLBAAttack(threshold=0.5)
    smva = SMVAAttack(threshold=0.5)
    mvna = MVNAAttack(threshold=0.5)

    print("\n" + "-" * 60)
    print("NLBA (Natural Language Blindness Attack)")
    print("-" * 60)

    nlba_member = nlba.execute(
        target_model=None,
        query=query,
        prm_scores=member_output.prm_scores,
        reasoning_steps=member_output.reasoning_steps,
    )

    nlba_nonmember = nlba.execute(
        target_model=None,
        query=query,
        prm_scores=nonmember_output.prm_scores,
        reasoning_steps=nonmember_output.reasoning_steps,
    )

    print(f"Member Score: {nlba_member.success_score:.4f}")
    print(f"Member Prediction: {nlba_member.membership_prediction}")
    print(f"Non-Member Score: {nlba_nonmember.success_score:.4f}")
    print(f"Non-Member Prediction: {nlba_nonmember.membership_prediction}")

    print("\n" + "-" * 60)
    print("SMVA (Single-Model Voting Attack)")
    print("-" * 60)

    smva_member = smva.execute(
        target_model=None,
        query=query,
        vote_distribution=member_output.vote_distribution,
    )

    smva_nonmember = smva.execute(
        target_model=None,
        query=query,
        vote_distribution=nonmember_output.vote_distribution,
    )

    print(f"Member Score: {smva_member.success_score:.4f}")
    print(f"Member Prediction: {smva_member.membership_prediction}")
    print(f"Non-Member Score: {smva_nonmember.success_score:.4f}")
    print(f"Non-Member Prediction: {smva_nonmember.membership_prediction}")

    print("\n" + "-" * 60)
    print("MVNA (MCTS Value Network Attack)")
    print("-" * 60)

    mvna_member = mvna.execute(
        target_model=None,
        query=query,
        mcts_values=member_output.mcts_values,
        mcts_tree=member_output.mcts_tree,
    )

    mvna_nonmember = mvna.execute(
        target_model=None,
        query=query,
        mcts_values=nonmember_output.mcts_values,
        mcts_tree=nonmember_output.mcts_tree,
    )

    print(f"Member Score: {mvna_member.success_score:.4f}")
    print(f"Member Prediction: {mvna_member.membership_prediction}")
    print(f"Non-Member Score: {mvna_nonmember.success_score:.4f}")
    print(f"Non-Member Prediction: {mvna_nonmember.membership_prediction}")

    # Evaluation
    print("\n" + "=" * 60)
    print("Attack Evaluation")
    print("=" * 60)

    # Simulate multiple samples
    results = []
    ground_truth = []

    for _ in range(20):
        # Member samples
        m_out = member_llm.generate(query)
        m_result = nlba.execute(
            target_model=None,
            query=query,
            prm_scores=m_out.prm_scores,
        )
        results.append(m_result)
        ground_truth.append(True)

        # Non-member samples
        nm_out = nonmember_llm.generate(query)
        nm_result = nlba.execute(
            target_model=None,
            query=query,
            prm_scores=nm_out.prm_scores,
        )
        results.append(nm_result)
        ground_truth.append(False)

    metrics = nlba.evaluate(results, ground_truth)

    print(f"\nNLBA Attack Metrics:")
    print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  TPR@FPR=0.01: {metrics['tpr_at_fpr_0.01']:.4f}")

    print("\n" + "=" * 60)
    print("Example Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

# Last updated: 2026-01-15
