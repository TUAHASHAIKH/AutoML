"""
Performance Optimization Utilities
Provides caching and performance monitoring utilities.
"""

import functools
import time
import logging
from typing import Any, Callable, Dict
import streamlit as st

logger = logging.getLogger(__name__)


def timeit(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function with timing
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"{func.__name__} took {elapsed_time:.2f} seconds to execute")
        return result
    return wrapper


def memoize(func: Callable) -> Callable:
    """
    Simple memoization decorator for expensive computations.
    
    Args:
        func: Function to memoize
        
    Returns:
        Memoized function
    """
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args):
        if args in cache:
            logger.debug(f"Cache hit for {func.__name__}")
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result
    
    return wrapper


def streamlit_cache(func: Callable) -> Callable:
    """
    Wrapper for Streamlit's caching mechanism.
    
    Args:
        func: Function to cache
        
    Returns:
        Cached function
    """
    return st.cache_data(func)


class PerformanceMonitor:
    """Monitor and log performance metrics."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics: Dict[str, float] = {}
        self.start_times: Dict[str, float] = {}
    
    def start(self, operation: str) -> None:
        """
        Start timing an operation.
        
        Args:
            operation: Name of the operation
        """
        self.start_times[operation] = time.time()
        logger.debug(f"Started timing: {operation}")
    
    def end(self, operation: str) -> float:
        """
        End timing an operation and store the result.
        
        Args:
            operation: Name of the operation
            
        Returns:
            Elapsed time in seconds
        """
        if operation not in self.start_times:
            logger.warning(f"Operation {operation} was not started")
            return 0.0
        
        elapsed = time.time() - self.start_times[operation]
        self.metrics[operation] = elapsed
        logger.info(f"{operation} completed in {elapsed:.2f} seconds")
        return elapsed
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get all recorded metrics.
        
        Returns:
            Dictionary of operation names to elapsed times
        """
        return self.metrics.copy()
    
    def display_metrics(self) -> None:
        """Display metrics in Streamlit."""
        if self.metrics:
            st.subheader("⏱️ Performance Metrics")
            for operation, time_taken in self.metrics.items():
                st.metric(operation, f"{time_taken:.2f}s")


def optimize_dataframe_memory(df: 'pd.DataFrame') -> 'pd.DataFrame':
    """
    Optimize DataFrame memory usage by downcasting numeric types.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        Optimized DataFrame
    """
    import pandas as pd
    import numpy as np
    
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"Initial memory usage: {start_mem:.2f} MB")
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                # Use float32 for most cases as float16 has limited precision
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"Final memory usage: {end_mem:.2f} MB")
    logger.info(f"Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%")
    
    return df


def batch_process(items: list, batch_size: int = 100) -> list:
    """
    Process items in batches for better performance.
    
    Args:
        items: List of items to process
        batch_size: Size of each batch
        
    Returns:
        List of processed batches
    """
    batches = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batches.append(batch)
    
    logger.info(f"Split {len(items)} items into {len(batches)} batches of size {batch_size}")
    return batches
