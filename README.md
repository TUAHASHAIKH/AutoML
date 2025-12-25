# AutoML System for Classification

## Project Overview
This is an automated Machine Learning system built with Streamlit that performs end-to-end classification tasks including EDA, preprocessing, model training, hyperparameter optimization, and evaluation.

## Features
- **Dataset Upload**: CSV file upload with automatic metadata extraction
- **Automated EDA**: Missing values analysis, outlier detection, correlation matrix, distribution plots
- **Issue Detection**: Automatic flagging of data quality issues with user approval workflow
- **Smart Preprocessing**: Imputation, scaling, encoding with user control
- **Model Training**: 7 classification algorithms with hyperparameter optimization
- **Model Comparison**: Interactive dashboard with metrics, visualizations, and rankings
- **Auto Report Generation**: Downloadable comprehensive evaluation report

## Supported Models
1. Logistic Regression
2. K-Nearest Neighbors
3. Decision Tree
4. Naive Bayes
5. Random Forest
6. Support Vector Machines
7. Rule-based Classifier

## Installation & Local Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps
1. Clone the repository:
```bash
git clone <your-repo-url>
cd PROJECTS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

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

## Deployment

This application is deployed on Streamlit Cloud:
- **Live URL**: https://automlnust.streamlit.app/

## Technologies Used
- **Framework**: Streamlit
- **ML Libraries**: scikit-learn, pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Report Generation**: reportlab/pdfkit or markdown

## Team Members
- [Muhammad Tuaha CMS 463124]
- [Muhammad Taaha Bin Zaheer CMS 465788]

## License
This project is for educational purposes as part of CS-245 Machine Learning course.

