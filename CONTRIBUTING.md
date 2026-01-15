# Contributing to SPARTAN

Thank you for your interest in contributing to SPARTAN! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## Getting Started

### Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/spartan-framework.git
cd spartan-framework
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=spartan --cov-report=html

# Run specific test file
pytest tests/test_mplq.py -v
```

### Code Style

We use the following tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

Run all checks:
```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
```

## How to Contribute

### Reporting Bugs

1. Check existing issues to avoid duplicates
2. Use the bug report template
3. Include:
   - Python version
   - Package versions
   - Minimal reproducible example
   - Expected vs actual behavior

### Suggesting Features

1. Check existing issues and discussions
2. Use the feature request template
3. Describe the use case and proposed solution

### Pull Requests

1. Create a feature branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes following our coding standards

3. Add tests for new functionality

4. Update documentation as needed

5. Run the test suite:
```bash
pytest --cov=spartan
```

6. Commit with clear messages:
```bash
git commit -m "feat: add new privacy metric for PRM analysis"
```

7. Push and create a pull request

### Commit Message Convention

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

## Architecture Guidelines

### Module Structure

Each major component (MPLQ, RAAS, RPPO) follows this pattern:

```
module/
├── __init__.py      # Public API exports
├── core.py          # Core implementation
├── utils.py         # Module-specific utilities
└── types.py         # Type definitions
```

### Adding New Attacks

1. Create a new file in `src/spartan/attacks/`
2. Inherit from `BaseAttack`
3. Implement required methods
4. Add tests in `tests/test_attacks.py`

### Adding New Defenses

1. Add defense logic in appropriate RAAS submodule
2. Update `RAAS.sanitize()` to include new defense
3. Add configuration options
4. Add tests

## Testing Guidelines

- Maintain 100% test coverage
- Use descriptive test names
- Include edge cases
- Use fixtures for common setups

Example test structure:
```python
class TestMPLQ:
    """Tests for MPLQ module."""
    
    def test_prm_leakage_score_normal_distribution(self, mock_llm):
        """PRM leakage score should be low for normal distributions."""
        ...
    
    def test_prm_leakage_score_memorized_pattern(self, mock_llm):
        """PRM leakage score should be high for memorized patterns."""
        ...
```

## Documentation

- Use Google-style docstrings
- Update README for user-facing changes
- Add examples for new features

## Questions?

Feel free to open an issue for any questions or concerns.

Thank you for contributing to SPARTAN!

<!-- Last updated: 2026-01-15 -->
