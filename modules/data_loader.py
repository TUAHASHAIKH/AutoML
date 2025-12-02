"""
Data Loader Module
Handles dataset upload and displays basic information about the dataset.
"""

import pandas as pd
import streamlit as st
import numpy as np
from typing import Optional, Tuple, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Load CSV file and return as pandas DataFrame.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        pd.DataFrame: Loaded dataset, or None if loading fails
        
    Raises:
        None: Exceptions are caught and logged
    """
    try:
        df = pd.read_csv(uploaded_file)
        logger.info(f"Successfully loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        return df
    except pd.errors.EmptyDataError:
        error_msg = "The uploaded file is empty"
        st.error(f"Error loading file: {error_msg}")
        logger.error(error_msg)
        return None
    except pd.errors.ParserError as e:
        error_msg = f"Error parsing CSV file: {str(e)}"
        st.error(f"Error loading file: {error_msg}")
        logger.error(error_msg)
        return None
    except Exception as e:
        error_msg = f"Unexpected error loading file: {str(e)}"
        st.error(f"Error loading file: {error_msg}")
        logger.error(error_msg)
        return None


def display_basic_info(df, target_column=None):
    """
    Display basic metadata and statistics about the dataset.
    
    Args:
        df: pandas DataFrame
        target_column: Name of the target column for classification
    """
    st.subheader("📊 Dataset Basic Information")
    
    # Create columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Number of Rows", df.shape[0])
    with col2:
        st.metric("Number of Columns", df.shape[1])
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Column types
    st.subheader("🔤 Column Types")
    col_types = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.values,
        'Non-Null Count': df.count().values,
        'Null Count': df.isnull().sum().values
    })
    st.dataframe(col_types, use_container_width=True)
    
    # Summary statistics
    st.subheader("📈 Summary Statistics")
    st.dataframe(df.describe(include='all').T, use_container_width=True)
    
    # Class distribution (if target column is specified)
    if target_column and target_column in df.columns:
        st.subheader("🎯 Class Distribution")
        class_dist = df[target_column].value_counts()
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(pd.DataFrame({
                'Class': class_dist.index,
                'Count': class_dist.values,
                'Percentage': (class_dist.values / len(df) * 100).round(2)
            }), use_container_width=True)
        
        with col2:
            import plotly.express as px
            fig = px.pie(values=class_dist.values, names=class_dist.index, 
                        title='Class Distribution')
            st.plotly_chart(fig, use_container_width=True)
    
    # Data preview
    st.subheader("👀 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    return col_types


def get_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Categorize columns into numerical and categorical.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        tuple: (numerical_columns, categorical_columns)
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    logger.info(f"Identified {len(numerical_cols)} numerical and {len(categorical_cols)} categorical columns")
    return numerical_cols, categorical_cols


def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Perform basic validation on the dataset.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        dict: Dictionary containing validation results with keys:
            - is_valid (bool): Whether dataset passes validation
            - errors (list): List of error messages
            - warnings (list): List of warning messages
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check if dataset is empty
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append("Dataset is empty")
        logger.error("Dataset validation failed: empty dataset")
    
    # Check for minimum rows
    if len(df) < 10:
        validation_results['warnings'].append("Dataset has fewer than 10 rows. Results may not be reliable.")
        logger.warning(f"Dataset has only {len(df)} rows")
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        validation_results['warnings'].append(f"Found {duplicates} duplicate rows")
        logger.warning(f"Found {duplicates} duplicate rows in dataset")
    
    # Check for all NaN columns
    all_nan_cols = df.columns[df.isnull().all()].tolist()
    if all_nan_cols:
        validation_results['warnings'].append(f"Columns with all missing values: {', '.join(all_nan_cols)}")
        logger.warning(f"Found columns with all NaN values: {all_nan_cols}")
    
    return validation_results
