# Getting Started

## Installation

Install SPARTAN using pip:
```bash
pip install spartan-framework
```

Or install from source:
```bash
git clone https://github.com/hmshujaatzaheer/spartan-framework.git
cd spartan-framework
pip install -e .
```

## Requirements

- Python 3.9+
- NumPy >= 1.21.0

## Quick Start

### Basic Usage
```python
from spartan import SPARTAN, SPARTANConfig
from spartan.models import MockReasoningLLM

# Initialize a reasoning LLM (use MockReasoningLLM for testing)
llm = MockReasoningLLM()

# Create SPARTAN instance
spartan = SPARTAN(llm)

# Process a query with privacy protection
result = spartan.process("What is 2 + 2?")

print(f"Output: {result.output}")
print(f"Risk Score: {result.risk_score:.4f}")
print(f"Defense Applied: {result.defense_applied}")
```

### Custom Configuration
```python
from spartan import SPARTAN, SPARTANConfig

# Create custom configuration
config = SPARTANConfig(
    prm_threshold=0.3,      # PRM leakage threshold
    vote_threshold=0.4,     # Vote leakage threshold
    mcts_threshold=0.5,     # MCTS leakage threshold
    epsilon_min=0.01,       # Minimum defense intensity
    epsilon_max=0.5,        # Maximum defense intensity
)

# Initialize with custom config
spartan = SPARTAN(llm, config=config)
```

### Analyzing Privacy Risk
```python
from spartan.mplq import MPLQ

# Initialize MPLQ analyzer
mplq = MPLQ()

# Analyze privacy leakage
result = mplq.analyze(
    query="Solve: x + 2 = 5",
    prm_scores=[0.95, 0.92, 0.98],
    vote_distribution=[0.8, 0.1, 0.1],
    mcts_values=[0.9, 0.85, 0.88],
)

print(f"Total Risk: {result.total_risk:.4f}")
print(f"PRM Leakage: {result.prm_leakage:.4f}")
print(f"Vote Leakage: {result.vote_leakage:.4f}")
print(f"MCTS Leakage: {result.mcts_leakage:.4f}")
```

### Applying Defense
```python
from spartan.raas import RAAS
from spartan.mplq import MPLQ

# Analyze risk first
mplq = MPLQ()
risk_analysis = mplq.analyze(
    query="test query",
    prm_scores=[0.9, 0.85, 0.95],
)

# Apply defense
raas = RAAS()
defense_result = raas.sanitize(
    output="The answer is 4",
    risk_analysis=risk_analysis,
    reasoning_steps=["Step 1: ...", "Step 2: ..."],
)

print(f"Defense Applied: {defense_result.defense_applied}")
print(f"Epsilon Used: {defense_result.epsilon_used:.4f}")
```

## Command Line Interface

SPARTAN provides a CLI for common operations:
```bash
# Analyze privacy risk
spartan analyze --query "What is 2+2?" --prm-scores 0.9,0.85,0.95

# Apply defense
spartan defend --output "The answer is 4" --risk-score 0.7

# Run evaluation
spartan evaluate --mode attack --num-samples 100
```

## Next Steps

- Explore the [API Reference](api_reference.md) for detailed documentation
- Learn about the [Algorithms](algorithms.md) behind SPARTAN
- See [Examples](examples.md) for more use cases

<!-- Last updated: 2026-01-15 -->
