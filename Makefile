.PHONY: help install install-dev run test coverage lint format security clean

# Default target
help:
	@echo "AutoML Classification System - Available Commands:"
	@echo ""
	@echo "  make install       - Install production dependencies"
	@echo "  make install-dev   - Install development dependencies"
	@echo "  make run          - Run the Streamlit application"
	@echo "  make test         - Run test suite"
	@echo "  make coverage     - Run tests with coverage report"
	@echo "  make lint         - Run linting checks"
	@echo "  make format       - Format code with black and isort"
	@echo "  make security     - Run security checks"
	@echo "  make clean        - Clean up cache and temporary files"
	@echo "  make all-checks   - Run all quality checks (format, lint, security, test)"
	@echo ""

# Install production dependencies
install:
	pip install -r requirements.txt

# Install development dependencies
install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install

# Run the application
run:
	streamlit run app.py

# Run tests
test:
	pytest -v

# Run tests with coverage
coverage:
	pytest --cov=modules --cov=utils --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

# Run linting
lint:
	@echo "Running Flake8..."
	flake8 .
	@echo "Running Pylint..."
	pylint modules utils --exit-zero
	@echo "Running MyPy..."
	mypy modules utils --ignore-missing-imports

# Format code
format:
	@echo "Formatting with Black..."
	black .
	@echo "Sorting imports with isort..."
	isort .

# Security checks
security:
	@echo "Running Bandit security scan..."
	bandit -r . -c .bandit
	@echo "Checking dependencies for vulnerabilities..."
	safety check || true

# Run all checks
all-checks: format lint security test
	@echo "All checks completed!"

# Clean cache and temporary files
clean:
	@echo "Cleaning cache and temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete
	@echo "Cleanup complete!"

# Setup development environment from scratch
setup: install-dev
	@echo "Development environment setup complete!"
	@echo "Run 'make run' to start the application"
