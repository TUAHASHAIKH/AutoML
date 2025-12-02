"""
Exploratory Data Analysis Module
Performs automated EDA including missing values, outliers, correlations, and visualizations.
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats


def perform_eda(df, numerical_cols, categorical_cols):
    """
    Perform comprehensive exploratory data analysis.
    
    Args:
        df: pandas DataFrame
        numerical_cols: List of numerical column names
        categorical_cols: List of categorical column names
    """
    st.header("🔍 Automated Exploratory Data Analysis")
    
    # Missing value analysis
    missing_analysis(df)
    
    # Outlier detection
    if numerical_cols:
        outlier_analysis(df, numerical_cols)
    
    # Correlation matrix
    if len(numerical_cols) > 1:
        correlation_analysis(df, numerical_cols)
    
    # Distribution plots
    if numerical_cols:
        distribution_plots(df, numerical_cols)
    
    # Categorical features analysis
    if categorical_cols:
        categorical_analysis(df, categorical_cols)


def missing_analysis(df):
    """
    Analyze and visualize missing values in the dataset.
    
    Args:
        df: pandas DataFrame
    """
    st.subheader("❓ Missing Value Analysis")
    
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing Count': df.isnull().sum().values,
        'Missing Percentage': (df.isnull().sum().values / len(df) * 100).round(2)
    })
    missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values(
        'Missing Percentage', ascending=False
    )
    
    if len(missing_data) == 0:
        st.success("✅ No missing values found in the dataset!")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(missing_data, use_container_width=True)
            total_missing = df.isnull().sum().sum()
            total_cells = df.shape[0] * df.shape[1]
            st.metric("Global Missing Percentage", 
                     f"{(total_missing / total_cells * 100):.2f}%")
        
        with col2:
            fig = px.bar(missing_data, x='Column', y='Missing Percentage',
                        title='Missing Values by Column',
                        labels={'Missing Percentage': 'Missing %'})
            st.plotly_chart(fig, use_container_width=True)


def outlier_analysis(df, numerical_cols):
    """
    Detect outliers using IQR and Z-score methods.
    
    Args:
        df: pandas DataFrame
        numerical_cols: List of numerical column names
        
    Returns:
        dict: Dictionary containing outlier information
    """
    st.subheader("📉 Outlier Detection")
    
    outlier_info = {}
    
    # Method selection
    method = st.selectbox("Select Outlier Detection Method", 
                         ["IQR Method", "Z-Score Method", "Both"])
    
    for col in numerical_cols:
        outliers_iqr = []
        outliers_zscore = []
        
        # IQR Method
        if method in ["IQR Method", "Both"]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_iqr = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
        
        # Z-Score Method
        if method in ["Z-Score Method", "Both"]:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outliers_zscore = df[col].dropna().index[z_scores > 3].tolist()
        
        outlier_info[col] = {
            'iqr_outliers': outliers_iqr,
            'zscore_outliers': outliers_zscore,
            'iqr_count': len(outliers_iqr),
            'zscore_count': len(outliers_zscore)
        }
    
    # Display outlier summary
    outlier_summary = pd.DataFrame({
        'Column': numerical_cols,
        'IQR Outliers': [outlier_info[col]['iqr_count'] for col in numerical_cols],
        'Z-Score Outliers': [outlier_info[col]['zscore_count'] for col in numerical_cols]
    })
    
    st.dataframe(outlier_summary, use_container_width=True)
    
    # Box plots for visualization
    if len(numerical_cols) > 0:
        st.write("**Box Plots for Outlier Visualization**")
        n_cols = min(3, len(numerical_cols))
        
        for i in range(0, len(numerical_cols), n_cols):
            cols = st.columns(n_cols)
            for j, col in enumerate(numerical_cols[i:i+n_cols]):
                with cols[j]:
                    fig = px.box(df, y=col, title=f'{col}')
                    st.plotly_chart(fig, use_container_width=True)
    
    return outlier_info


def correlation_analysis(df, numerical_cols):
    """
    Compute and visualize correlation matrix.
    
    Args:
        df: pandas DataFrame
        numerical_cols: List of numerical column names
    """
    st.subheader("🔗 Correlation Matrix")
    
    # Compute correlation
    corr_matrix = df[numerical_cols].corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                fmt='.2f', square=True, ax=ax)
    plt.title('Feature Correlation Matrix')
    st.pyplot(fig)
    plt.close()
    
    # Find highly correlated features
    st.write("**Highly Correlated Features (|correlation| > 0.7)**")
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                high_corr.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
    
    if high_corr:
        st.dataframe(pd.DataFrame(high_corr), use_container_width=True)
    else:
        st.info("No highly correlated features found.")


def distribution_plots(df, numerical_cols):
    """
    Create distribution plots for numerical features.
    
    Args:
        df: pandas DataFrame
        numerical_cols: List of numerical column names
    """
    st.subheader("📊 Distribution Plots for Numerical Features")
    
    # Select columns to plot
    selected_cols = st.multiselect(
        "Select features to visualize",
        numerical_cols,
        default=numerical_cols[:min(4, len(numerical_cols))]
    )
    
    if selected_cols:
        n_cols = 2
        n_rows = (len(selected_cols) + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for idx, col in enumerate(selected_cols):
            if idx < len(axes):
                axes[idx].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
                axes[idx].set_title(f'Distribution of {col}')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequency')
        
        # Hide extra subplots
        for idx in range(len(selected_cols), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


def categorical_analysis(df, categorical_cols):
    """
    Create bar plots for categorical features.
    
    Args:
        df: pandas DataFrame
        categorical_cols: List of categorical column names
    """
    st.subheader("📊 Categorical Features Analysis")
    
    # Select columns to plot
    selected_cols = st.multiselect(
        "Select categorical features to visualize",
        categorical_cols,
        default=categorical_cols[:min(3, len(categorical_cols))],
        key='cat_analysis'
    )
    
    if selected_cols:
        for col in selected_cols:
            value_counts = df[col].value_counts().head(10)  # Limit to top 10
            
            fig = px.bar(x=value_counts.index, y=value_counts.values,
                        title=f'Distribution of {col}',
                        labels={'x': col, 'y': 'Count'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Show cardinality warning
            unique_count = df[col].nunique()
            if unique_count > 20:
                st.warning(f"⚠️ High cardinality detected: {col} has {unique_count} unique values")


def get_train_test_split_summary(X_train, X_test, y_train, y_test):
    """
    Display train/test split summary.
    
    Args:
        X_train, X_test: Feature sets
        y_train, y_test: Target sets
    """
    st.subheader("✂️ Train/Test Split Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Training Samples", len(X_train))
    with col2:
        st.metric("Testing Samples", len(X_test))
    with col3:
        split_ratio = len(X_train) / (len(X_train) + len(X_test)) * 100
        st.metric("Train/Test Ratio", f"{split_ratio:.1f}% / {100-split_ratio:.1f}%")
    
    # Class distribution in train and test
    train_dist = pd.Series(y_train).value_counts()
    test_dist = pd.Series(y_test).value_counts()
    
    dist_df = pd.DataFrame({
        'Class': train_dist.index,
        'Train Count': train_dist.values,
        'Test Count': test_dist.index.map(lambda x: test_dist.get(x, 0))
    })
    
    st.dataframe(dist_df, use_container_width=True)
