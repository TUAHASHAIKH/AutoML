"""
Preprocessing Module
Handles data preprocessing including imputation, scaling, encoding, and train-test split.
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """Class to handle all preprocessing operations."""
    
    def __init__(self, df, numerical_cols, categorical_cols, target_col):
        """
        Initialize the preprocessor.
        
        Args:
            df: pandas DataFrame
            numerical_cols: List of numerical column names
            categorical_cols: List of categorical column names
            target_col: Target column name
        """
        self.df = df.copy()
        self.numerical_cols = [col for col in numerical_cols if col != target_col]
        self.categorical_cols = [col for col in categorical_cols if col != target_col]
        self.target_col = target_col
        self.preprocessing_steps = []
    
    def apply_user_decisions(self, user_decisions):
        """
        Apply preprocessing based on user decisions from issue detection.
        
        Args:
            user_decisions: Dictionary of user decisions from IssueDetector
        """
        st.subheader("Applying Preprocessing Steps")
        
        for key, decision in user_decisions.items():
            issue = decision['issue']
            action = decision['action']
            
            if issue['type'] == 'missing_values':
                self._handle_missing_values(issue['column'], action, decision)
            
            elif issue['type'] == 'outliers':
                self._handle_outliers(issue['column'], action, issue.get('bounds'))
            
            elif issue['type'] in ['constant_feature', 'near_constant_feature']:
                if 'Drop column' in action:
                    self._drop_column(issue['column'])
            
            elif issue['type'] == 'high_cardinality':
                if 'Drop column' in action:
                    self._drop_column(issue['column'])
    
    def _handle_missing_values(self, column, action, decision):
        """Handle missing values based on user action."""
        if 'Mean' in action:
            imputer = SimpleImputer(strategy='mean')
            self.df[column] = imputer.fit_transform(self.df[[column]])
            self.preprocessing_steps.append(f"Imputed '{column}' with mean")
            st.success(f"✓ Imputed '{column}' with mean")
        
        elif 'Median' in action:
            imputer = SimpleImputer(strategy='median')
            self.df[column] = imputer.fit_transform(self.df[[column]])
            self.preprocessing_steps.append(f"Imputed '{column}' with median")
            st.success(f"✓ Imputed '{column}' with median")
        
        elif 'Mode' in action:
            imputer = SimpleImputer(strategy='most_frequent')
            self.df[column] = imputer.fit_transform(self.df[[column]]).ravel()
            self.preprocessing_steps.append(f"Imputed '{column}' with mode")
            st.success(f"✓ Imputed '{column}' with mode")
        
        elif 'Constant' in action:
            const_val = decision.get('constant_value', 0)
            self.df[column].fillna(const_val, inplace=True)
            self.preprocessing_steps.append(f"Imputed '{column}' with constant: {const_val}")
            st.success(f"✓ Imputed '{column}' with constant value")
        
        elif 'Drop rows' in action:
            before_count = len(self.df)
            self.df.dropna(subset=[column], inplace=True)
            after_count = len(self.df)
            self.preprocessing_steps.append(f"Dropped {before_count - after_count} rows with missing '{column}'")
            st.success(f"✓ Dropped {before_count - after_count} rows with missing '{column}'")
    
    def _handle_outliers(self, column, action, bounds):
        """Handle outliers based on user action."""
        if bounds is None:
            return
        
        lower_bound, upper_bound = bounds
        
        if 'Remove' in action:
            before_count = len(self.df)
            self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
            after_count = len(self.df)
            self.preprocessing_steps.append(f"Removed {before_count - after_count} outliers from '{column}'")
            st.success(f"✓ Removed {before_count - after_count} outliers from '{column}'")
        
        elif 'Cap' in action:
            outlier_count = ((self.df[column] < lower_bound) | (self.df[column] > upper_bound)).sum()
            self.df[column] = self.df[column].clip(lower=lower_bound, upper=upper_bound)
            self.preprocessing_steps.append(f"Capped {outlier_count} outliers in '{column}'")
            st.success(f"✓ Capped {outlier_count} outliers in '{column}'")
    
    def _drop_column(self, column):
        """Drop a column from the dataset."""
        if column in self.df.columns:
            self.df.drop(columns=[column], inplace=True)
            if column in self.numerical_cols:
                self.numerical_cols.remove(column)
            if column in self.categorical_cols:
                self.categorical_cols.remove(column)
            self.preprocessing_steps.append(f"Dropped column '{column}'")
            st.success(f"✓ Dropped column '{column}'")
    
    def configure_preprocessing(self):
        """
        Allow user to configure preprocessing options.
        
        Returns:
            dict: Preprocessing configuration
        """
        st.header("Preprocessing Configuration")
        
        config = {}
        
        # Scaling
        if self.numerical_cols:
            st.subheader("Feature Scaling")
            config['scaling'] = st.selectbox(
                "Select scaling method for numerical features:",
                ["None", "StandardScaler (Z-score normalization)", "MinMaxScaler (0-1 normalization)"]
            )
        
        # Encoding
        if self.categorical_cols:
            st.subheader(" Categorical Encoding")
            config['encoding'] = st.selectbox(
                "Select encoding method for categorical features:",
                ["One-Hot Encoding", "Label Encoding"]
            )
        
        # Train-test split
        st.subheader("Train-Test Split")
        config['test_size'] = st.slider(
            "Test set size (%)",
            min_value=10,
            max_value=40,
            value=20,
            step=5
        ) / 100
        
        config['random_state'] = st.number_input(
            "Random state (for reproducibility)",
            min_value=0,
            max_value=1000,
            value=42
        )
        
        return config
    
    def apply_preprocessing(self, config):
        """
        Apply preprocessing transformations.
        
        Args:
            config: Preprocessing configuration dictionary
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, preprocessors)
        """
        st.subheader("🔄 Applying Preprocessing Transformations")
        
        # Separate features and target
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        
        # Encode target if categorical
        le_target = None
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
            self.preprocessing_steps.append("Encoded target variable")
            st.info(f"Target classes: {list(le_target.classes_)}")
        
        # Handle categorical features
        if self.categorical_cols and 'encoding' in config:
            if config['encoding'] == "One-Hot Encoding":
                X = pd.get_dummies(X, columns=self.categorical_cols, drop_first=True)
                self.preprocessing_steps.append(f"Applied One-Hot Encoding to {len(self.categorical_cols)} categorical features")
                st.success(f"✓ Applied One-Hot Encoding")
            else:  # Label Encoding
                le_dict = {}
                for col in self.categorical_cols:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    le_dict[col] = le
                self.preprocessing_steps.append(f"Applied Label Encoding to {len(self.categorical_cols)} categorical features")
                st.success(f"✓ Applied Label Encoding")
        
        # Train-test split
        # Check if stratification is possible
        try:
            # Try stratified split first
            unique_classes, class_counts = np.unique(y, return_counts=True)
            min_class_count = np.min(class_counts)
            
            # Only use stratify if each class has at least 2 samples
            if len(unique_classes) > 1 and min_class_count >= 2:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=config['test_size'],
                    random_state=int(config['random_state']),
                    stratify=y
                )
                self.preprocessing_steps.append(f"Split data with stratification: {len(X_train)} train, {len(X_test)} test")
            else:
                # Use regular split without stratification
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=config['test_size'],
                    random_state=int(config['random_state'])
                )
                self.preprocessing_steps.append(f"Split data without stratification: {len(X_train)} train, {len(X_test)} test")
                if min_class_count < 2:
                    st.warning(f"Some classes have very few samples. Stratification disabled.")
        except Exception as e:
            # Fallback to simple split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=config['test_size'],
                random_state=int(config['random_state'])
            )
            self.preprocessing_steps.append(f"Split data: {len(X_train)} train, {len(X_test)} test")
        
        st.success(f"✓ Split data into train ({len(X_train)}) and test ({len(X_test)}) sets")
        
        # Scaling
        scaler = None
        if 'scaling' in config and config['scaling'] != "None":
            if "StandardScaler" in config['scaling']:
                scaler = StandardScaler()
            else:  # MinMaxScaler
                scaler = MinMaxScaler()
            
            # Get numerical columns after encoding
            num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
            
            if num_cols:
                X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
                X_test[num_cols] = scaler.transform(X_test[num_cols])
                self.preprocessing_steps.append(f"Applied {config['scaling'].split()[0]} to numerical features")
                st.success(f"✓ Applied {config['scaling'].split()[0]}")
        
        # Store preprocessors
        preprocessors = {
            'scaler': scaler,
            'label_encoder_target': le_target,
            'feature_names': X_train.columns.tolist()
        }
        
        st.success("All preprocessing completed successfully!")
        
        return X_train, X_test, y_train, y_test, preprocessors
    
    def get_preprocessing_summary(self):
        """
        Get a summary of all preprocessing steps.
        
        Returns:
            list: List of preprocessing steps
        """
        return self.preprocessing_steps
    
    def display_preprocessing_summary(self):
        """Display a summary of preprocessing steps."""
        if self.preprocessing_steps:
            st.subheader("Preprocessing Summary")
            for i, step in enumerate(self.preprocessing_steps, 1):
                st.write(f"{i}. {step}")


def handle_remaining_missing_values(X_train, X_test, numerical_cols, categorical_cols):
    """
    Handle any remaining missing values after user decisions.
    
    Args:
        X_train, X_test: Training and testing features
        numerical_cols: List of numerical column names
        categorical_cols: List of categorical column names
        
    Returns:
        tuple: (X_train, X_test) with missing values handled
    """
    # Numerical imputation
    if numerical_cols:
        num_imputer = SimpleImputer(strategy='median')
        num_cols_in_data = [col for col in numerical_cols if col in X_train.columns]
        if num_cols_in_data:
            X_train[num_cols_in_data] = num_imputer.fit_transform(X_train[num_cols_in_data])
            X_test[num_cols_in_data] = num_imputer.transform(X_test[num_cols_in_data])
    
    # Categorical imputation
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        cat_cols_in_data = [col for col in categorical_cols if col in X_train.columns]
        if cat_cols_in_data:
            X_train[cat_cols_in_data] = cat_imputer.fit_transform(X_train[cat_cols_in_data])
            X_test[cat_cols_in_data] = cat_imputer.transform(X_test[cat_cols_in_data])
    
    return X_train, X_test
