"""
Unit tests for config module.
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import (
    ModelConfig, DataConfig, VisualizationConfig, UIConfig, LoggingConfig,
    get_config, update_config
)


class TestGetConfig:
    """Test cases for get_config function."""
    
    def test_get_model_config(self):
        """Test retrieving model configuration."""
        config = get_config('model')
        assert isinstance(config, ModelConfig)
        assert hasattr(config, 'DEFAULT_TEST_SIZE')
        assert hasattr(config, 'DEFAULT_RANDOM_STATE')
    
    def test_get_data_config(self):
        """Test retrieving data configuration."""
        config = get_config('data')
        assert isinstance(config, DataConfig)
        assert hasattr(config, 'MIN_ROWS')
        assert hasattr(config, 'MIN_COLUMNS')
    
    def test_get_viz_config(self):
        """Test retrieving visualization configuration."""
        config = get_config('viz')
        assert isinstance(config, VisualizationConfig)
        assert hasattr(config, 'DEFAULT_PLOT_HEIGHT')
    
    def test_get_ui_config(self):
        """Test retrieving UI configuration."""
        config = get_config('ui')
        assert isinstance(config, UIConfig)
        assert hasattr(config, 'PAGE_TITLE')
    
    def test_get_log_config(self):
        """Test retrieving logging configuration."""
        config = get_config('log')
        assert isinstance(config, LoggingConfig)
        assert hasattr(config, 'LOG_LEVEL')
    
    def test_invalid_config_type(self):
        """Test retrieving invalid configuration type."""
        with pytest.raises(ValueError):
            get_config('invalid')


class TestUpdateConfig:
    """Test cases for update_config function."""
    
    def test_update_model_config(self):
        """Test updating model configuration."""
        original_value = get_config('model').DEFAULT_TEST_SIZE
        update_config('model', DEFAULT_TEST_SIZE=0.3)
        assert get_config('model').DEFAULT_TEST_SIZE == 0.3
        # Reset to original
        update_config('model', DEFAULT_TEST_SIZE=original_value)
    
    def test_update_invalid_attribute(self):
        """Test updating non-existent attribute."""
        with pytest.raises(AttributeError):
            update_config('model', NONEXISTENT_ATTR=123)


class TestModelConfig:
    """Test cases for ModelConfig class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ModelConfig()
        assert config.DEFAULT_TEST_SIZE == 0.2
        assert config.DEFAULT_RANDOM_STATE == 42
        assert config.DEFAULT_CV_FOLDS == 5
        assert config.PRIMARY_METRIC == 'f1_weighted'
    
    def test_classification_metrics_initialization(self):
        """Test that classification metrics are initialized."""
        config = ModelConfig()
        assert config.CLASSIFICATION_METRICS is not None
        assert len(config.CLASSIFICATION_METRICS) > 0
        assert 'accuracy' in config.CLASSIFICATION_METRICS


class TestDataConfig:
    """Test cases for DataConfig class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = DataConfig()
        assert config.MIN_ROWS == 10
        assert config.MIN_COLUMNS == 2
        assert config.OUTLIER_IQR_MULTIPLIER == 1.5
        assert config.HIGH_CORRELATION_THRESHOLD == 0.7


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
