"""
Issue Detection Module
Automatically detects data quality issues and manages user approval for fixes.
"""

import pandas as pd
import numpy as np
import streamlit as st
from scipy import stats


class IssueDetector:
    """Class to detect and manage data quality issues."""
    
    def __init__(self, df, numerical_cols, categorical_cols, target_col=None):
        """
        Initialize the issue detector.
        
        Args:
            df: pandas DataFrame
            numerical_cols: List of numerical column names
            categorical_cols: List of categorical column names
            target_col: Target column name
        """
        self.df = df
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.target_col = target_col
        self.issues = []
        self.user_decisions = {}
    
    def detect_all_issues(self):
        """Detect all data quality issues."""
        self.detect_missing_values()
        self.detect_outliers()
        self.detect_class_imbalance()
        self.detect_high_cardinality()
        self.detect_constant_features()
        
        return self.issues
    
    def detect_missing_values(self):
        """Detect missing values in the dataset."""
        missing_cols = self.df.columns[self.df.isnull().any()].tolist()
        
        if missing_cols:
            for col in missing_cols:
                missing_count = self.df[col].isnull().sum()
                missing_pct = (missing_count / len(self.df)) * 100
                
                self.issues.append({
                    'type': 'missing_values',
                    'column': col,
                    'severity': 'high' if missing_pct > 50 else 'medium' if missing_pct > 10 else 'low',
                    'description': f"Column '{col}' has {missing_count} missing values ({missing_pct:.2f}%)",
                    'suggestions': self._get_imputation_suggestions(col)
                })
    
    def detect_outliers(self):
        """Detect outliers in numerical columns using IQR method."""
        for col in self.numerical_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outlier_count = len(outliers)
            
            if outlier_count > 0:
                outlier_pct = (outlier_count / len(self.df)) * 100
                self.issues.append({
                    'type': 'outliers',
                    'column': col,
                    'severity': 'high' if outlier_pct > 10 else 'medium' if outlier_pct > 5 else 'low',
                    'description': f"Column '{col}' has {outlier_count} outliers ({outlier_pct:.2f}%)",
                    'suggestions': ['Remove outliers', 'Cap outliers at bounds', 'Keep outliers'],
                    'bounds': (lower_bound, upper_bound)
                })
    
    def detect_class_imbalance(self):
        """Detect class imbalance in target column."""
        if self.target_col and self.target_col in self.df.columns:
            class_counts = self.df[self.target_col].value_counts()
            
            if len(class_counts) > 1:
                imbalance_ratio = class_counts.max() / class_counts.min()
                
                if imbalance_ratio > 3:
                    self.issues.append({
                        'type': 'class_imbalance',
                        'column': self.target_col,
                        'severity': 'high' if imbalance_ratio > 10 else 'medium',
                        'description': f"Class imbalance detected. Ratio: {imbalance_ratio:.2f}:1",
                        'suggestions': ['Apply SMOTE', 'Use class weights', 'No action'],
                        'class_distribution': class_counts.to_dict()
                    })
    
    def detect_high_cardinality(self):
        """Detect high cardinality in categorical features."""
        for col in self.categorical_cols:
            if col != self.target_col:
                unique_count = self.df[col].nunique()
                cardinality_pct = (unique_count / len(self.df)) * 100
                
                if unique_count > 50 or cardinality_pct > 50:
                    self.issues.append({
                        'type': 'high_cardinality',
                        'column': col,
                        'severity': 'medium' if unique_count > 100 else 'low',
                        'description': f"Column '{col}' has {unique_count} unique values ({cardinality_pct:.2f}% of rows)",
                        'suggestions': ['Drop column', 'Group rare categories', 'Keep as is']
                    })
    
    def detect_constant_features(self):
        """Detect constant or near-constant features."""
        for col in self.df.columns:
            if col != self.target_col:
                value_counts = self.df[col].value_counts()
                if len(value_counts) == 1:
                    self.issues.append({
                        'type': 'constant_feature',
                        'column': col,
                        'severity': 'high',
                        'description': f"Column '{col}' has only one unique value",
                        'suggestions': ['Drop column']
                    })
                elif len(value_counts) > 1:
                    # Near-constant: one value appears in >95% of rows
                    top_pct = (value_counts.iloc[0] / len(self.df)) * 100
                    if top_pct > 95:
                        self.issues.append({
                            'type': 'near_constant_feature',
                            'column': col,
                            'severity': 'medium',
                            'description': f"Column '{col}' is near-constant ({top_pct:.2f}% same value)",
                            'suggestions': ['Drop column', 'Keep as is']
                        })
    
    def _get_imputation_suggestions(self, col):
        """Get appropriate imputation suggestions based on column type."""
        if col in self.numerical_cols:
            return ['Mean imputation', 'Median imputation', 'Forward fill', 'Drop rows']
        else:
            return ['Mode imputation', 'Constant value', 'Drop rows']
    
    def display_issues_and_get_approval(self):
        """
        Display detected issues and get user approval for fixes.
        
        Returns:
            dict: User decisions for each issue
        """
        if not self.issues:
            st.success("✅ No data quality issues detected!")
            return {}
        
        st.header("⚠️ Data Quality Issues Detected")
        st.write(f"Found **{len(self.issues)}** potential issues:")
        
        # Group issues by severity
        high_severity = [i for i in self.issues if i['severity'] == 'high']
        medium_severity = [i for i in self.issues if i['severity'] == 'medium']
        low_severity = [i for i in self.issues if i['severity'] == 'low']
        
        # Display issues by severity
        for severity, issues_list, color in [
            ('High', high_severity, '🔴'),
            ('Medium', medium_severity, '🟡'),
            ('Low', low_severity, '🟢')
        ]:
            if issues_list:
                st.subheader(f"{color} {severity} Severity Issues ({len(issues_list)})")
                
                for idx, issue in enumerate(issues_list):
                    with st.expander(f"{issue['type'].replace('_', ' ').title()}: {issue['column']}"):
                        st.write(f"**Description:** {issue['description']}")
                        
                        # Get user decision
                        key = f"{issue['type']}_{issue['column']}_{idx}"
                        
                        if 'suggestions' in issue:
                            action = st.radio(
                                "Select action:",
                                issue['suggestions'],
                                key=key
                            )
                            
                            self.user_decisions[key] = {
                                'issue': issue,
                                'action': action
                            }
                            
                            # Additional parameters for specific actions
                            if issue['type'] == 'missing_values':
                                if 'Constant value' in action:
                                    const_val = st.text_input(
                                        "Enter constant value:",
                                        key=f"{key}_const"
                                    )
                                    self.user_decisions[key]['constant_value'] = const_val
        
        return self.user_decisions
    
    def get_issues_summary(self):
        """
        Get a summary of all detected issues.
        
        Returns:
            pd.DataFrame: Summary of issues
        """
        if not self.issues:
            return pd.DataFrame()
        
        summary = pd.DataFrame([{
            'Type': issue['type'].replace('_', ' ').title(),
            'Column': issue['column'],
            'Severity': issue['severity'].upper(),
            'Description': issue['description']
        } for issue in self.issues])
        
        return summary


def create_issue_report(issues):
    """
    Create a formatted report of all issues.
    
    Args:
        issues: List of detected issues
        
    Returns:
        str: Formatted report
    """
    if not issues:
        return "No issues detected."
    
    report = "## Data Quality Issues Report\n\n"
    
    for issue in issues:
        report += f"### {issue['type'].replace('_', ' ').title()}\n"
        report += f"- **Column:** {issue['column']}\n"
        report += f"- **Severity:** {issue['severity'].upper()}\n"
        report += f"- **Description:** {issue['description']}\n\n"
    
    return report
