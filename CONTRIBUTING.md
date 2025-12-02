# Contributing to AutoML Classification System

Thank you for your interest in contributing to the AutoML Classification System! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

Please be respectful and constructive in your interactions with other contributors.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/AutoML.git
   cd AutoML
   ```

3. Create a branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Install Dependencies

1. Install production dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality:

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually on all files
pre-commit run --all-files
```

## Code Style

We follow these coding standards:

### Python Style Guide

- **PEP 8**: Follow PEP 8 style guidelines
- **Line Length**: Maximum 120 characters
- **Imports**: Use absolute imports, organized with `isort`
- **Type Hints**: Add type hints to function signatures
- **Docstrings**: Use Google-style docstrings

### Code Formatting

We use the following tools:

- **Black**: Code formatter
- **isort**: Import sorter
- **Flake8**: Linter
- **MyPy**: Type checker

Run formatting before committing:

```bash
# Format code with Black
black .

# Sort imports
isort .

# Check linting
flake8 .

# Type checking
mypy modules utils
```

## Testing

### Running Tests

Run the test suite with pytest:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=modules --cov=utils --cov-report=html

# Run specific test file
pytest tests/test_data_loader.py

# Run with verbose output
pytest -v
```

### Writing Tests

- Write tests for all new features
- Maintain or improve code coverage
- Use descriptive test names
- Follow the Arrange-Act-Assert pattern

Example test:

```python
def test_function_name():
    """Test description."""
    # Arrange
    input_data = create_test_data()
    
    # Act
    result = function_to_test(input_data)
    
    # Assert
    assert result == expected_output
```

## Security

### Security Checks

Run security checks before committing:

```bash
# Run Bandit security scanner
bandit -r . -c .bandit

# Check for known vulnerabilities in dependencies
safety check
```

### Reporting Security Issues

If you discover a security vulnerability, please email the maintainers directly rather than creating a public issue.

## Submitting Changes

### Commit Messages

Write clear, descriptive commit messages:

- Use the imperative mood ("Add feature" not "Added feature")
- First line: brief summary (50 chars or less)
- Add detailed explanation if needed
- Reference related issues

Example:
```
Add type hints to data_loader module

- Added type hints to all functions
- Improved error handling
- Added logging

Fixes #123
```

### Pull Request Process

1. Update the documentation if needed
2. Add tests for new features
3. Ensure all tests pass
4. Run pre-commit hooks
5. Create a Pull Request with:
   - Clear title and description
   - Reference to related issues
   - Screenshots (if UI changes)
   - Test results

### Review Process

- Maintainers will review your PR
- Address feedback and make requested changes
- Once approved, your PR will be merged

## Additional Resources

### Project Structure

```
AutoML/
├── app.py                  # Main Streamlit application
├── config.py              # Configuration management
├── modules/               # Core functionality modules
│   ├── data_loader.py
│   ├── eda.py
│   ├── issue_detector.py
│   ├── preprocessor.py
│   ├── model_trainer.py
│   ├── model_evaluator.py
│   └── report_generator.py
├── utils/                 # Utility functions
│   └── helpers.py
├── tests/                 # Test suite
└── sample_data/          # Sample datasets
```

### Useful Commands

```bash
# Run application
streamlit run app.py

# Run linters
flake8 .
pylint modules utils

# Check types
mypy modules utils

# Format code
black . && isort .

# Full quality check
black . && isort . && flake8 . && pytest
```

## Questions?

If you have questions, feel free to:
- Open an issue on GitHub
- Contact the maintainers

Thank you for contributing!
