# AutoML System for Classification 🤖

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Educational-green.svg)](LICENSE)

## Project Overview

An automated Machine Learning system built with Streamlit that performs end-to-end classification tasks including exploratory data analysis (EDA), preprocessing, model training, hyperparameter optimization, and comprehensive evaluation. Perfect for rapid prototyping and educational purposes.

## ✨ Key Features

- **📁 Dataset Upload**: CSV file upload with automatic metadata extraction and validation
- **🔍 Automated EDA**: Missing values analysis, outlier detection, correlation matrix, distribution plots
- **⚠️ Issue Detection**: Automatic flagging of data quality issues with user approval workflow
- **⚙️ Smart Preprocessing**: Imputation, scaling, encoding with user control
- **🤖 Model Training**: 7 classification algorithms with hyperparameter optimization
- **📊 Model Comparison**: Interactive dashboard with metrics, visualizations, and rankings
- **📄 Auto Report Generation**: Downloadable comprehensive evaluation report (Markdown/HTML)
- **🔧 Configuration Management**: Centralized configuration for easy customization
- **🧪 Testing Infrastructure**: Unit tests with pytest for reliability
- **🔒 Security**: Built-in security scanning and input validation

## 🎯 Supported Models

1. **Logistic Regression** - Linear model for binary/multiclass classification
2. **K-Nearest Neighbors** - Instance-based learning algorithm
3. **Decision Tree** - Tree-based model with interpretable rules
4. **Naive Bayes** - Probabilistic classifier based on Bayes' theorem
5. **Random Forest** - Ensemble of decision trees
6. **Support Vector Machine** - Maximum margin classifier
7. **Rule-Based Classifier** - Simple threshold-based rules

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/TUAHASHAIKH/AutoML.git
cd AutoML
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install development dependencies (optional):**
```bash
pip install -r requirements-dev.txt
```

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Project Structure
```
PROJECTS/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── modules/
│   ├── __init__.py
│   ├── data_loader.py         # Dataset upload and basic info
│   ├── eda.py                 # Exploratory Data Analysis
│   ├── issue_detector.py      # Data quality issue detection
│   ├── preprocessor.py        # Data preprocessing pipeline
│   ├── model_trainer.py       # Model training and optimization
│   ├── model_evaluator.py     # Model evaluation and comparison
│   └── report_generator.py    # Report generation
├── sample_data/
│   └── sample_dataset.csv     # Sample dataset for testing
├── screenshots/                # Application screenshots
└── utils/
    ├── __init__.py
    └── helpers.py             # Utility functions
```

## Usage Guide

1. **Upload Dataset**: Click "Browse files" to upload your CSV file
2. **Review Basic Info**: Check dataset summary statistics and class distribution
3. **Explore Data**: Review automated EDA findings
4. **Handle Issues**: Approve or reject suggested fixes for detected issues
5. **Configure Preprocessing**: Select imputation methods, scaling, and encoding options
6. **Train Models**: Click "Train Models" to start training all classifiers
7. **Compare Results**: Review model comparison dashboard
8. **Download Report**: Generate and download comprehensive evaluation report

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=modules --cov=utils --cov-report=html

# Run specific test file
pytest tests/test_data_loader.py -v
```

## 🔧 Development

### Code Quality Tools

We use several tools to maintain code quality:

```bash
# Format code
black .
isort .

# Lint code
flake8 .

# Security scan
bandit -r . -c .bandit

# Type checking
mypy modules utils
```

### Pre-commit Hooks

Install pre-commit hooks for automatic code quality checks:

```bash
pre-commit install
pre-commit run --all-files
```

## 📊 Configuration

The system uses a centralized configuration system in `config.py`:

```python
from config import get_config, update_config

# Get configuration
model_config = get_config('model')
print(model_config.DEFAULT_TEST_SIZE)  # 0.2

# Update configuration
update_config('model', DEFAULT_TEST_SIZE=0.3)
```

Available configurations:
- `model`: Model training parameters
- `data`: Data validation and processing rules
- `viz`: Visualization settings
- `ui`: User interface settings
- `log`: Logging configuration

## 🔒 Security

- Input validation for all user inputs
- Security scanning with Bandit
- Dependency vulnerability checking with Safety
- Sanitization of file uploads
- No execution of arbitrary code

## 📈 Performance

- Memory-efficient DataFrame operations
- Caching of expensive computations
- Batch processing for large datasets
- Progress monitoring and logging

## 🌐 Deployment

This application is deployed on Streamlit Cloud:
- **Live URL**: https://automlnust.streamlit.app/

### Deploy Your Own

1. Fork this repository
2. Sign up at [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Deploy with one click!

## 📚 Documentation

- [Contributing Guidelines](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)
- [Project Guide](PROJECT_GUIDE.md)
- [Quick Start Guide](QUICKSTART.md)

## 🛠️ Technologies Used

- **Framework**: Streamlit 1.28+
- **ML Libraries**: scikit-learn, pandas, numpy, scipy
- **Visualization**: matplotlib, seaborn, plotly
- **Report Generation**: markdown, FPDF
- **Testing**: pytest, pytest-cov
- **Code Quality**: black, flake8, isort, mypy, bandit

## 👥 Team Members

- Muhammad Tuaha - CMS 463124
- Muhammad Taaha Bin Zaheer - CMS 465788

## 🎓 Academic Context

This project is developed as part of CS-245 Machine Learning course at NUST.

## 📝 License

This project is for educational purposes. See individual library licenses for dependencies.

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📧 Contact

For questions or feedback:
- Open an issue on GitHub
- Contact the maintainers

## 🙏 Acknowledgments

- NUST CS Department for academic support
- Streamlit community for excellent documentation
- scikit-learn team for comprehensive ML library
