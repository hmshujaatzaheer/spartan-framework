# Examples

## Attack Detection Examples

### NLBA (Natural Language Blindness Attack)
```python
from spartan.attacks import NLBAAttack
from spartan.models import MockReasoningLLM

# Initialize attack
attack = NLBAAttack(threshold=0.6)

# Execute attack with PRM scores
result = attack.execute(
    target_model=None,
    query="Solve: x + 2 = 5",
    prm_scores=[0.95, 0.92, 0.98],
    reasoning_steps=["Let x + 2 = 5", "x = 5 - 2", "x = 3"],
)

print(f"Attack Score: {result.success_score:.4f}")
print(f"Membership Prediction: {result.membership_prediction}")
print(f"Confidence: {result.confidence:.4f}")
```

### SMVA (Single-Model Voting Attack)
```python
from spartan.attacks import SMVAAttack

# Initialize attack
attack = SMVAAttack(threshold=0.55)

# Execute with vote distribution
result = attack.execute(
    target_model=None,
    query="What is 2+2?",
    vote_distribution=[0.8, 0.1, 0.05, 0.05],
)

print(f"Attack Score: {result.success_score:.4f}")
print(f"Leakage Signals: {result.leakage_signals}")
```

### MVNA (MCTS Value Network Attack)
```python
from spartan.attacks import MVNAAttack

# Initialize attack
attack = MVNAAttack(threshold=0.5)

# Execute with MCTS values
result = attack.execute(
    target_model=None,
    query="Complex reasoning task",
    mcts_values=[0.92, 0.88, 0.95, 0.91],
)

print(f"Attack Score: {result.success_score:.4f}")
```

## Defense Examples

### PRM Defense
```python
from spartan.raas.prm_defense import PRMDefense

# Initialize defense
defense = PRMDefense(
    noise_scale=1.0,
    nl_perturbation_ratio=10.0,
    use_feature_selective=True,
)

# Apply defense
result = defense.apply(
    reasoning_steps=["Step 1: Calculate", "Step 2: Verify"],
    epsilon=0.3,
    prm_leakage=0.7,
    threshold=0.3,
)

print(f"Defense Applied: {result['applied']}")
```

### Vote Defense
```python
from spartan.raas.vote_defense import VoteDefense

# Initialize defense
defense = VoteDefense(
    temperature_base=1.0,
    use_implicit_rewards=True,
)

# Apply defense
result = defense.apply(
    vote_distribution=[0.8, 0.1, 0.1],
    epsilon=0.3,
    vote_leakage=0.6,
    threshold=0.4,
    candidate_outputs=["Answer A", "Answer B", "Answer C"],
)

print(f"Temperature Used: {result['temperature_used']:.4f}")
print(f"Entropy Increase: {result['entropy_increase']:.4f}")
```

### MCTS Defense
```python
from spartan.raas.mcts_defense import MCTSDefense

# Initialize defense
defense = MCTSDefense(
    depth_scale=0.1,
    max_perturbation=0.3,
)

# Apply defense
result = defense.apply(
    mcts_tree={
        "values": [0.9, 0.85, 0.88, 0.92],
        "depths": [0, 1, 1, 2],
    },
    epsilon=0.3,
    mcts_leakage=0.6,
    threshold=0.5,
)

print(f"Mean Perturbation: {result['perturbation_info']['mean_perturbation']:.4f}")
```

## Full Pipeline Example
```python
from spartan import SPARTAN, SPARTANConfig
from spartan.models import MockReasoningLLM

# Configure SPARTAN
config = SPARTANConfig(
    prm_threshold=0.3,
    vote_threshold=0.4,
    mcts_threshold=0.5,
    epsilon_min=0.01,
    epsilon_max=0.5,
)

# Initialize
llm = MockReasoningLLM(member_mode=True)
spartan = SPARTAN(llm, config=config)

# Process multiple queries
queries = [
    "What is the derivative of x^2?",
    "Solve the equation: 2x + 5 = 15",
    "Calculate the integral of sin(x)",
]

for query in queries:
    result = spartan.process(query)
    print(f"\nQuery: {query}")
    print(f"  Risk Score: {result.risk_score:.4f}")
    print(f"  Defense Applied: {result.defense_applied}")
    print(f"  Output: {result.output[:50]}...")
```

## Benchmarking Example
```python
from spartan.benchmarks import BenchmarkRunner

# Initialize runner
runner = BenchmarkRunner(seed=42)

# Run full benchmark
result = runner.run_full_benchmark(num_samples=100)

# Print results
print("Attack Metrics:")
for metric, value in result.attack_metrics.items():
    print(f"  {metric}: {value:.4f}")

print("\nDefense Metrics:")
for metric, value in result.defense_metrics.items():
    print(f"  {metric}: {value:.4f}")

# Save results
result.save("benchmark_results.json")
```

## Optimization with RPPO
```python
from spartan.rppo import RPPO

# Initialize optimizer
rppo = RPPO()

# Simulate optimization loop
for i in range(20):
    # Simulate observation from processing
    observation = {
        "risk_score": 0.5 - 0.02 * i,
        "accuracy": 0.85 + 0.005 * i,
        "compute": 0.3,
    }
    rppo.update(observation)

# Get optimal parameters
optimal = rppo.get_optimal_params()
if optimal:
    print(f"Optimal Parameters: {optimal.params}")
    print(f"Total Reward: {optimal.reward:.4f}")
```

## Integration with Custom Models
```python
from spartan import SPARTAN
from spartan.models.base import BaseReasoningLLM, LLMOutput

class MyCustomLLM(BaseReasoningLLM):
    """Custom LLM implementation."""
    
    def __init__(self):
        super().__init__(name="MyCustomLLM")
    
    def generate(self, query, **kwargs):
        # Your custom generation logic
        return LLMOutput(
            output="Generated response",
            reasoning_steps=["Step 1", "Step 2"],
            prm_scores=[0.9, 0.85],
            vote_distribution=[0.7, 0.2, 0.1],
        )
    
    def get_prm_scores(self, reasoning_steps):
        # Your PRM scoring logic
        return [0.9] * len(reasoning_steps)

# Use with SPARTAN
llm = MyCustomLLM()
spartan = SPARTAN(llm)
result = spartan.process("Test query")
```
