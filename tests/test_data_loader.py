"""
Unit tests for data_loader module.
"""

import pytest
import pandas as pd
import numpy as np
from io import StringIO
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.data_loader import get_column_types, validate_dataset


class TestGetColumnTypes:
    """Test cases for get_column_types function."""
    
    def test_numerical_columns_only(self):
        """Test with only numerical columns."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4.5, 5.5, 6.5],
            'col3': [7, 8, 9]
        })
        numerical_cols, categorical_cols = get_column_types(df)
        assert len(numerical_cols) == 3
        assert len(categorical_cols) == 0
        assert set(numerical_cols) == {'col1', 'col2', 'col3'}
    
    def test_categorical_columns_only(self):
        """Test with only categorical columns."""
        df = pd.DataFrame({
            'col1': ['a', 'b', 'c'],
            'col2': ['x', 'y', 'z']
        })
        numerical_cols, categorical_cols = get_column_types(df)
        assert len(numerical_cols) == 0
        assert len(categorical_cols) == 2
        assert set(categorical_cols) == {'col1', 'col2'}
    
    def test_mixed_columns(self):
        """Test with mixed column types."""
        df = pd.DataFrame({
            'numeric': [1, 2, 3],
            'text': ['a', 'b', 'c'],
            'float': [1.1, 2.2, 3.3]
        })
        numerical_cols, categorical_cols = get_column_types(df)
        assert len(numerical_cols) == 2
        assert len(categorical_cols) == 1
        assert set(numerical_cols) == {'numeric', 'float'}
        assert set(categorical_cols) == {'text'}


class TestValidateDataset:
    """Test cases for validate_dataset function."""
    
    def test_valid_dataset(self):
        """Test with a valid dataset."""
        df = pd.DataFrame({
            'col1': range(20),
            'col2': range(20, 40)
        })
        result = validate_dataset(df)
        assert result['is_valid'] is True
        assert len(result['errors']) == 0
    
    def test_empty_dataset(self):
        """Test with an empty dataset."""
        df = pd.DataFrame()
        result = validate_dataset(df)
        assert result['is_valid'] is False
        assert 'empty' in result['errors'][0].lower()
    
    def test_small_dataset_warning(self):
        """Test with a dataset having fewer than 10 rows."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        result = validate_dataset(df)
        assert result['is_valid'] is True
        assert len(result['warnings']) > 0
        assert 'fewer than 10 rows' in result['warnings'][0]
    
    def test_duplicate_rows_warning(self):
        """Test with duplicate rows."""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 1, 2],
            'col2': [4, 5, 6, 4, 5]
        })
        result = validate_dataset(df)
        assert result['is_valid'] is True
        assert any('duplicate' in w.lower() for w in result['warnings'])
    
    def test_all_nan_column_warning(self):
        """Test with columns containing all NaN values."""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'col2': [np.nan] * 10
        })
        result = validate_dataset(df)
        assert result['is_valid'] is True
        assert any('all missing' in w.lower() for w in result['warnings'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
