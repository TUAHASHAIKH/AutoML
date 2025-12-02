# Changelog

All notable changes to the AutoML Classification System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Configuration management system (`config.py`)
  - Centralized configuration for models, data processing, visualization, and UI
  - Easy configuration updates through `get_config()` and `update_config()` functions
  
- Development tooling and code quality
  - Pre-commit hooks configuration (`.pre-commit-config.yaml`)
  - Flake8 linting configuration (`.flake8`)
  - Bandit security scanning configuration (`.bandit`)
  - Black and isort formatting configuration (`pyproject.toml`)
  - Development dependencies (`requirements-dev.txt`)
  
- Testing infrastructure
  - Unit test framework with pytest
  - Test coverage configuration
  - Sample tests for `data_loader` and `config` modules
  
- Type hints and improved error handling
  - Added type hints to `data_loader.py` functions
  - Enhanced error handling with specific exception catching
  - Logging infrastructure for better debugging
  
- Documentation
  - Contributing guidelines (`CONTRIBUTING.md`)
  - This changelog (`CHANGELOG.md`)

### Changed
- Enhanced `data_loader.py` module:
  - Improved error handling with specific exception types
  - Added logging for better debugging
  - Added validation for columns with all NaN values
  - Better type hints for function signatures

### Security
- Added Bandit security scanner configuration
- Added Safety check for dependency vulnerabilities

## [1.0.1] - Previous Version

### Features
- Streamlit-based AutoML web application
- 7 classification algorithms support
- Automated EDA capabilities
- Data quality issue detection
- Interactive preprocessing pipeline
- Model comparison dashboard
- Comprehensive report generation

### Modules
- `data_loader.py`: Dataset upload and basic information
- `eda.py`: Exploratory data analysis
- `issue_detector.py`: Data quality issue detection
- `preprocessor.py`: Data preprocessing pipeline
- `model_trainer.py`: Model training and optimization
- `model_evaluator.py`: Model evaluation and comparison
- `report_generator.py`: Report generation

### Supported Models
1. Logistic Regression
2. K-Nearest Neighbors
3. Decision Tree
4. Naive Bayes
5. Random Forest
6. Support Vector Machine
7. Rule-based Classifier

---

## Version History Notes

### Legend
- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security improvements
