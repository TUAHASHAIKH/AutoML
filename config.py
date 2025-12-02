"""
Configuration Module
Centralized configuration management for AutoML system.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import os


@dataclass
class ModelConfig:
    """Configuration for model training."""
    
    # Training configuration
    DEFAULT_TEST_SIZE: float = 0.2
    DEFAULT_RANDOM_STATE: int = 42
    DEFAULT_CV_FOLDS: int = 5
    MIN_CV_FOLDS: int = 3
    CV_FOLD_THRESHOLD: int = 2000  # Use fewer folds for datasets larger than this
    
    # Hyperparameter optimization
    DEFAULT_N_ITER_RANDOM_SEARCH: int = 10
    MAX_OPTIMIZATION_TIME: int = 3600  # seconds
    
    # Model-specific settings
    LOGISTIC_REGRESSION_MAX_ITER: int = 1000
    SVM_PROBABILITY: bool = True
    
    # Evaluation metrics
    PRIMARY_METRIC: str = 'f1_weighted'
    CLASSIFICATION_METRICS: List[str] = None
    
    def __post_init__(self):
        """Initialize default metrics if not provided."""
        if self.CLASSIFICATION_METRICS is None:
            self.CLASSIFICATION_METRICS = [
                'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'
            ]


@dataclass
class DataConfig:
    """Configuration for data processing."""
    
    # Data validation
    MIN_ROWS: int = 10
    MIN_COLUMNS: int = 2
    MAX_MISSING_PERCENTAGE: float = 90.0
    
    # Outlier detection
    OUTLIER_IQR_MULTIPLIER: float = 1.5
    OUTLIER_ZSCORE_THRESHOLD: float = 3.0
    
    # Feature engineering
    HIGH_CARDINALITY_THRESHOLD: int = 50
    HIGH_CARDINALITY_PERCENTAGE: float = 50.0
    NEAR_CONSTANT_THRESHOLD: float = 95.0
    
    # Class imbalance
    IMBALANCE_RATIO_THRESHOLD: float = 3.0
    HIGH_IMBALANCE_RATIO: float = 10.0
    
    # Correlation analysis
    HIGH_CORRELATION_THRESHOLD: float = 0.7


@dataclass
class VisualizationConfig:
    """Configuration for visualizations."""
    
    # Plot settings
    DEFAULT_PLOT_HEIGHT: int = 500
    BOXPLOT_COLS: int = 3
    HISTOGRAM_BINS: int = 30
    
    # Color schemes
    PRIMARY_COLOR: str = '#1f77b4'
    HEATMAP_COLORSCALE: str = 'Blues'
    RADAR_CHART_MAX_MODELS: int = 5


@dataclass
class UIConfig:
    """Configuration for UI settings."""
    
    # Streamlit page config
    PAGE_TITLE: str = "AutoML Classification System"
    PAGE_ICON: str = "🤖"
    LAYOUT: str = "wide"
    INITIAL_SIDEBAR_STATE: str = "expanded"
    
    # Progress indicators
    SHOW_PROGRESS_BAR: bool = True
    SHOW_TRAINING_VERBOSE: bool = True


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: str = "automl.log"
    ENABLE_FILE_LOGGING: bool = False


# Global configuration instances
MODEL_CONFIG = ModelConfig()
DATA_CONFIG = DataConfig()
VIZ_CONFIG = VisualizationConfig()
UI_CONFIG = UIConfig()
LOG_CONFIG = LoggingConfig()


def get_config(config_type: str) -> Any:
    """
    Get configuration by type.
    
    Args:
        config_type: Type of configuration ('model', 'data', 'viz', 'ui', 'log')
        
    Returns:
        Configuration object
        
    Raises:
        ValueError: If config_type is not recognized
    """
    config_map = {
        'model': MODEL_CONFIG,
        'data': DATA_CONFIG,
        'viz': VIZ_CONFIG,
        'ui': UI_CONFIG,
        'log': LOG_CONFIG
    }
    
    if config_type not in config_map:
        raise ValueError(f"Unknown config type: {config_type}. Available: {list(config_map.keys())}")
    
    return config_map[config_type]


def update_config(config_type: str, **kwargs) -> None:
    """
    Update configuration values.
    
    Args:
        config_type: Type of configuration to update
        **kwargs: Configuration key-value pairs to update
        
    Example:
        update_config('model', DEFAULT_TEST_SIZE=0.3, DEFAULT_RANDOM_STATE=123)
    """
    config = get_config(config_type)
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise AttributeError(f"Configuration {config_type} has no attribute {key}")
