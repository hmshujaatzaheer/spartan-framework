# SPARTAN Documentation

Welcome to the SPARTAN framework documentation.

## Contents

* [Getting Started](getting_started.md)
* [API Reference](api_reference.md)
* [Algorithms](algorithms.md)
* [Examples](examples.md)

## Overview

SPARTAN (Secure Privacy-Adaptive Reasoning with Test-time Attack Neutralization) is a unified framework for detecting and defending against mechanistic privacy attacks in Reasoning Large Language Models.

## Key Features

- **MPLQ**: Mechanistic Privacy Leakage Quantification for detecting privacy risks
- **RAAS**: Reasoning-Aware Adaptive Sanitization for applying targeted defenses
- **RPPO**: Reasoning-Privacy Pareto Optimization for balancing utility and privacy
- **Attack Suite**: NLBA, SMVA, and MVNA attacks for comprehensive evaluation

## Quick Links

* [GitHub Repository](https://github.com/hmshujaatzaheer/spartan-framework)
* [PyPI Package](https://pypi.org/project/spartan-framework/)
* [Issue Tracker](https://github.com/hmshujaatzaheer/spartan-framework/issues)

## Installation
```bash
pip install spartan-framework
```

## Quick Example
```python
from spartan import SPARTAN
from spartan.models import MockReasoningLLM

llm = MockReasoningLLM()
spartan = SPARTAN(llm)
result = spartan.process("What is 2 + 2?")

print(f"Risk Score: {result.risk_score:.4f}")
print(f"Defense Applied: {result.defense_applied}")
```
