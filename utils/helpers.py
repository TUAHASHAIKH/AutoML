"""
Utility Helper Functions
Common utility functions used across the application.
"""

import streamlit as st
import pandas as pd
import numpy as np


def set_page_config():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="AutoML Classification System",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def apply_custom_css():
    """Apply custom CSS styling."""
    st.markdown("""
        <style>
        /* Main header */
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            padding: 1rem 0;
        }
        
        /* Sub header */
        .sub-header {
            font-size: 1.5rem;
            color: #666666;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        /* Metric cards */
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            border: 1px solid #e0e0e0;
        }
        
        /* Progress bar */
        .stProgress > div > div > div > div {
            background-color: #1f77b4;
        }
        </style>
    """, unsafe_allow_html=True)


def display_header():
    """Display application header."""
    st.markdown('<h1 class="main-header">AutoML Classification System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Automated Machine Learning for Classification Tasks</p>', unsafe_allow_html=True)
    st.markdown("---")


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    if 'eda_complete' not in st.session_state:
        st.session_state.eda_complete = False
    
    if 'issues_detected' not in st.session_state:
        st.session_state.issues_detected = False
    
    if 'preprocessing_complete' not in st.session_state:
        st.session_state.preprocessing_complete = False
    
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    
    if 'df' not in st.session_state:
        st.session_state.df = None
    
    if 'target_col' not in st.session_state:
        st.session_state.target_col = None
    
    if 'numerical_cols' not in st.session_state:
        st.session_state.numerical_cols = []
    
    if 'categorical_cols' not in st.session_state:
        st.session_state.categorical_cols = []


def create_sidebar_navigation():
    """Create sidebar navigation menu."""
    st.sidebar.title("Navigation")
    
    st.sidebar.markdown("---")
    
    steps = [
        "Upload Dataset",
        "Exploratory Data Analysis",
        "Issue Detection",
        "Preprocessing",
        "Model Training",
        "Model Comparison",
        "Generate Report"
    ]
    
    # Display progress
    progress_status = []
    if st.session_state.data_loaded:
        progress_status.append("Data Loaded")
    if st.session_state.eda_complete:
        progress_status.append("EDA Complete")
    if st.session_state.get('issues_detected', False):
        progress_status.append("Issues Detected")
    if st.session_state.preprocessing_complete:
        progress_status.append("Preprocessing Complete")
    if st.session_state.models_trained:
        progress_status.append("Models Trained")
    
    st.sidebar.subheader("Progress")
    if progress_status:
        for status in progress_status:
            st.sidebar.write(status)
    else:
        st.sidebar.info("Start by uploading a dataset")
    
    st.sidebar.markdown("---")
    
    # Information
    st.sidebar.subheader("Information")
    st.sidebar.info(
        """
        This AutoML system automates the entire machine learning pipeline:
        
        1. Upload your CSV dataset
        2. Review automated EDA
        3. Handle data quality issues
        4. Configure preprocessing
        5. Train multiple models
        6. Compare results
        7. Download report
        """
    )
    
    st.sidebar.markdown("---")
    st.sidebar.caption("CS-245 Machine Learning Project")


def validate_csv_upload(df):
    """
    Validate uploaded CSV file.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if df is None:
        return False, "Failed to load file"
    
    if df.empty:
        return False, "Dataset is empty"
    
    if len(df) < 10:
        return True, "Warning: Dataset has fewer than 10 rows"
    
    if len(df.columns) < 2:
        return False, "Dataset must have at least 2 columns (features + target)"
    
    return True, "Valid dataset"


def format_number(num, decimals=2):
    """
    Format number for display.
    
    Args:
        num: Number to format
        decimals: Number of decimal places
        
    Returns:
        str: Formatted number
    """
    if isinstance(num, (int, float)):
        return f"{num:.{decimals}f}"
    return str(num)


def create_download_link(data, filename, text):
    """
    Create a download link for data.
    
    Args:
        data: Data to download
        filename: Name of the file
        text: Link text
        
    Returns:
        str: HTML download link
    """
    import base64
    
    b64 = base64.b64encode(data.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'
    return href


def show_success_message(message):
    """Display a success message with custom styling."""
    st.success(f"{message}")


def show_warning_message(message):
    """Display a warning message with custom styling."""
    st.warning(f"{message}")


def show_error_message(message):
    """Display an error message with custom styling."""
    st.error(f"{message}")


def show_info_message(message):
    """Display an info message with custom styling."""
    st.info(f"{message}")


def create_expandable_section(title, content_func):
    """
    Create an expandable section.
    
    Args:
        title: Section title
        content_func: Function to render content
    """
    with st.expander(title):
        content_func()


def safe_divide(numerator, denominator, default=0):
    """
    Safely divide two numbers.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division fails
        
    Returns:
        float: Result of division or default
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except:
        return default


def get_dataframe_summary(df):
    """
    Get a summary of a DataFrame.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        dict: Summary information
    """
    return {
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2),
        'column_names': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict()
    }
