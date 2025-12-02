"""
Input Validation Utilities
Provides validation functions for data integrity and security.
"""

import pandas as pd
import numpy as np
from typing import Union, List, Optional, Tuple
import re
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_dataframe(df: pd.DataFrame, 
                       min_rows: int = 1, 
                       min_cols: int = 1,
                       max_missing_pct: float = 100.0) -> Tuple[bool, str]:
    """
    Validate a pandas DataFrame for basic requirements.
    
    Args:
        df: DataFrame to validate
        min_rows: Minimum number of rows required
        min_cols: Minimum number of columns required
        max_missing_pct: Maximum percentage of missing values allowed
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df is None:
        return False, "DataFrame is None"
    
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has {len(df)} rows, minimum {min_rows} required"
    
    if len(df.columns) < min_cols:
        return False, f"DataFrame has {len(df.columns)} columns, minimum {min_cols} required"
    
    # Check missing values
    missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    if missing_pct > max_missing_pct:
        return False, f"DataFrame has {missing_pct:.1f}% missing values, max {max_missing_pct}% allowed"
    
    return True, "Valid DataFrame"


def validate_column_name(column_name: str, valid_columns: List[str]) -> bool:
    """
    Validate that a column name exists in the list of valid columns.
    
    Args:
        column_name: Column name to validate
        valid_columns: List of valid column names
        
    Returns:
        True if valid, raises ValidationError otherwise
    """
    if not column_name:
        raise ValidationError("Column name cannot be empty")
    
    if column_name not in valid_columns:
        raise ValidationError(f"Column '{column_name}' not found. Available: {valid_columns}")
    
    return True


def validate_numeric_range(value: Union[int, float],
                           min_val: Optional[float] = None,
                           max_val: Optional[float] = None,
                           param_name: str = "value") -> bool:
    """
    Validate that a numeric value is within specified range.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        param_name: Name of the parameter for error messages
        
    Returns:
        True if valid, raises ValidationError otherwise
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{param_name} must be numeric, got {type(value)}")
    
    if np.isnan(value) or np.isinf(value):
        raise ValidationError(f"{param_name} cannot be NaN or Inf")
    
    if min_val is not None and value < min_val:
        raise ValidationError(f"{param_name} must be >= {min_val}, got {value}")
    
    if max_val is not None and value > max_val:
        raise ValidationError(f"{param_name} must be <= {max_val}, got {value}")
    
    return True


def validate_percentage(value: Union[int, float], param_name: str = "percentage") -> bool:
    """
    Validate that a value is a valid percentage (0-100).
    
    Args:
        value: Value to validate
        param_name: Name of the parameter for error messages
        
    Returns:
        True if valid, raises ValidationError otherwise
    """
    return validate_numeric_range(value, 0.0, 100.0, param_name)


def validate_probability(value: Union[int, float], param_name: str = "probability") -> bool:
    """
    Validate that a value is a valid probability (0-1).
    
    Args:
        value: Value to validate
        param_name: Name of the parameter for error messages
        
    Returns:
        True if valid, raises ValidationError otherwise
    """
    return validate_numeric_range(value, 0.0, 1.0, param_name)


def validate_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
    """
    Validate file extension.
    
    Args:
        filename: Name of the file
        allowed_extensions: List of allowed extensions (e.g., ['.csv', '.txt'])
        
    Returns:
        True if valid, raises ValidationError otherwise
    """
    if not filename:
        raise ValidationError("Filename cannot be empty")
    
    file_ext = filename.lower().split('.')[-1] if '.' in filename else ''
    
    # Normalize extensions
    allowed_exts = [ext.lower().replace('.', '') for ext in allowed_extensions]
    
    if file_ext not in allowed_exts:
        raise ValidationError(
            f"Invalid file extension '.{file_ext}'. Allowed: {', '.join(allowed_extensions)}"
        )
    
    return True


def validate_model_name(model_name: str, available_models: List[str]) -> bool:
    """
    Validate model name.
    
    Args:
        model_name: Name of the model
        available_models: List of available model names
        
    Returns:
        True if valid, raises ValidationError otherwise
    """
    if not model_name:
        raise ValidationError("Model name cannot be empty")
    
    if model_name not in available_models:
        raise ValidationError(
            f"Unknown model '{model_name}'. Available: {', '.join(available_models)}"
        )
    
    return True


def sanitize_string(input_str: str, max_length: int = 1000) -> str:
    """
    Sanitize user input string to prevent injection attacks.
    
    Args:
        input_str: String to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
        
    Note:
        Allows alphanumeric characters, spaces, hyphens, underscores, periods, 
        commas, and parentheses. Brackets and braces are removed for security.
    """
    if not isinstance(input_str, str):
        return str(input_str)
    
    # Trim to max length
    sanitized = input_str[:max_length]
    
    # Remove potentially dangerous characters
    # Allow only alphanumeric, spaces, and safe punctuation (no brackets/braces)
    sanitized = re.sub(r'[^\w\s\-_.,()]', '', sanitized)
    
    return sanitized.strip()


def validate_categorical_feature(df: pd.DataFrame, 
                                 column: str,
                                 max_unique: int = 100) -> Tuple[bool, str]:
    """
    Validate that a categorical feature has reasonable cardinality.
    
    Args:
        df: DataFrame containing the feature
        column: Column name
        max_unique: Maximum number of unique values allowed
        
    Returns:
        Tuple of (is_valid, message)
    """
    if column not in df.columns:
        return False, f"Column '{column}' not found"
    
    n_unique = df[column].nunique()
    
    if n_unique > max_unique:
        return False, f"Column '{column}' has {n_unique} unique values (max {max_unique})"
    
    if n_unique == 1:
        return False, f"Column '{column}' has only one unique value"
    
    return True, f"Valid categorical feature with {n_unique} unique values"


def validate_target_column(df: pd.DataFrame, 
                          target_col: str,
                          min_classes: int = 2,
                          max_classes: int = 100) -> Tuple[bool, str]:
    """
    Validate target column for classification.
    
    Args:
        df: DataFrame containing the target
        target_col: Target column name
        min_classes: Minimum number of classes required
        max_classes: Maximum number of classes allowed
        
    Returns:
        Tuple of (is_valid, message)
    """
    if target_col not in df.columns:
        return False, f"Target column '{target_col}' not found"
    
    # Check for missing values
    if df[target_col].isnull().any():
        missing_count = df[target_col].isnull().sum()
        return False, f"Target column has {missing_count} missing values"
    
    # Check number of classes
    n_classes = df[target_col].nunique()
    
    if n_classes < min_classes:
        return False, f"Target has {n_classes} classes, minimum {min_classes} required for classification"
    
    if n_classes > max_classes:
        return False, f"Target has {n_classes} classes, maximum {max_classes} allowed"
    
    # Check class distribution
    class_counts = df[target_col].value_counts()
    min_count = class_counts.min()
    
    if min_count < 2:
        return False, f"Some classes have fewer than 2 samples"
    
    return True, f"Valid target column with {n_classes} classes"


def validate_train_test_split(test_size: float, 
                              n_samples: int,
                              min_test_samples: int = 2,
                              min_train_samples: int = 5) -> Tuple[bool, str]:
    """
    Validate train-test split parameters.
    
    Args:
        test_size: Proportion of test set (0-1)
        n_samples: Total number of samples
        min_test_samples: Minimum samples in test set
        min_train_samples: Minimum samples in train set
        
    Returns:
        Tuple of (is_valid, message)
    """
    try:
        validate_probability(test_size, "test_size")
    except ValidationError as e:
        return False, str(e)
    
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test
    
    if n_test < min_test_samples:
        return False, f"Test set would have {n_test} samples, minimum {min_test_samples} required"
    
    if n_train < min_train_samples:
        return False, f"Train set would have {n_train} samples, minimum {min_train_samples} required"
    
    return True, f"Valid split: {n_train} train, {n_test} test samples"
