"""
Model Training Module
Handles training of multiple classification models with hyperparameter optimization.
"""

import numpy as np
import pandas as pd
import streamlit as st
import time
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, roc_auc_score, 
                            classification_report, roc_curve)
import warnings
warnings.filterwarnings('ignore')


class RuleBasedClassifier:
    """Simple rule-based classifier implementation."""
    
    def __init__(self):
        self.rules = {}
        self.default_class = None
    
    def fit(self, X, y):
        """
        Fit the rule-based classifier.
        Creates simple threshold-based rules for each feature.
        """
        self.default_class = pd.Series(y).mode()[0]
        
        # For simplicity, use the first feature to create a rule
        if len(X.shape) > 1 and X.shape[1] > 0:
            feature_0 = X.iloc[:, 0] if isinstance(X, pd.DataFrame) else X[:, 0]
            threshold = np.median(feature_0)
            
            # Create rule: if feature_0 > threshold, predict class A, else class B
            classes = np.unique(y)
            if len(classes) >= 2:
                self.rules['feature_0'] = {
                    'threshold': threshold,
                    'class_above': classes[1],
                    'class_below': classes[0]
                }
        
        return self
    
    def predict(self, X):
        """Predict using rule-based logic."""
        predictions = []
        
        if 'feature_0' in self.rules:
            feature_0 = X.iloc[:, 0] if isinstance(X, pd.DataFrame) else X[:, 0]
            rule = self.rules['feature_0']
            
            for val in feature_0:
                if val > rule['threshold']:
                    predictions.append(rule['class_above'])
                else:
                    predictions.append(rule['class_below'])
        else:
            predictions = [self.default_class] * len(X)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """Return probability estimates (simplified)."""
        predictions = self.predict(X)
        classes = np.unique(predictions)
        n_classes = len(classes)
        n_samples = len(predictions)
        
        # Create a simple probability matrix
        proba = np.zeros((n_samples, max(2, n_classes)))
        for i, pred in enumerate(predictions):
            class_idx = np.where(classes == pred)[0][0] if len(np.where(classes == pred)[0]) > 0 else 0
            proba[i, class_idx] = 0.8
            # Distribute remaining probability
            if n_classes > 1:
                proba[i, 1-class_idx] = 0.2
        
        return proba


class ModelTrainer:
    """Class to handle training and optimization of multiple models."""
    
    def __init__(self, X_train, X_test, y_train, y_test):
        """
        Initialize the model trainer.
        
        Args:
            X_train, X_test: Training and testing features
            y_train, y_test: Training and testing targets
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.results = {}
        self.trained_models = {}
    
    def get_model_configurations(self):
        """
        Get model configurations with hyperparameter grids.
        
        Returns:
            dict: Dictionary of models and their hyperparameter grids
        """
        return {
            'Logistic Regression': {
                'model': LogisticRegression(max_iter=1000, random_state=42),
                'params': {
                    'C': [0.01, 0.1, 1, 10],
                    'solver': ['lbfgs', 'liblinear']
                }
            },
            'K-Nearest Neighbors': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }
            },
            'Decision Tree': {
                'model': DecisionTreeClassifier(random_state=42),
                'params': {
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'criterion': ['gini', 'entropy']
                }
            },
            'Naive Bayes': {
                'model': GaussianNB(),
                'params': {
                    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5]
                }
            },
            'Support Vector Machine': {
                'model': SVC(probability=True, random_state=42),
                'params': {
                    'C': [0.1, 1],  # Reduced from 3 to 2 values
                    'kernel': ['rbf'],  # Only RBF kernel (best for most cases)
                    'gamma': ['scale']  # Only scale (recommended default)
                }
            },
            'Rule-Based Classifier': {
                'model': RuleBasedClassifier(),
                'params': {}  # No hyperparameters for simple rule-based
            }
        }
    
    def train_all_models(self, optimization_method='grid', n_iter=10):
        """
        Train all models with hyperparameter optimization.
        
        Args:
            optimization_method: 'grid' or 'random'
            n_iter: Number of iterations for RandomizedSearchCV
            
        Returns:
            dict: Training results for all models
        """
        st.header("Model Training & Hyperparameter Optimization")
        
        model_configs = self.get_model_configurations()
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (model_name, config) in enumerate(model_configs.items()):
            status_text.text(f"Training {model_name}...")
            
            start_time = time.time()
            
            try:
                # Train model
                if config['params'] and model_name != 'Rule-Based Classifier':
                    # Use fewer CV folds for large datasets to speed up training
                    n_samples = len(self.X_train)
                    cv_folds = 3 if n_samples > 2000 else min(5, n_samples)
                    
                    if optimization_method == 'grid':
                        search = GridSearchCV(
                            config['model'],
                            config['params'],
                            cv=cv_folds,
                            scoring='f1_weighted',
                            n_jobs=-1,
                            verbose=1  # Show progress
                        )
                    else:  # random
                        search = RandomizedSearchCV(
                            config['model'],
                            config['params'],
                            n_iter=n_iter,
                            cv=cv_folds,
                            scoring='f1_weighted',
                            random_state=42,
                            n_jobs=-1,
                            verbose=1  # Show progress
                        )
                    
                    search.fit(self.X_train, self.y_train)
                    best_model = search.best_estimator_
                    best_params = search.best_params_
                else:
                    # For models without hyperparameters
                    best_model = config['model']
                    best_model.fit(self.X_train, self.y_train)
                    best_params = {}
                
                training_time = time.time() - start_time
                
                # Make predictions
                y_pred = best_model.predict(self.X_test)
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_pred, best_model)
                
                # Store results
                self.results[model_name] = {
                    'model': best_model,
                    'best_params': best_params,
                    'training_time': training_time,
                    'metrics': metrics,
                    'predictions': y_pred
                }
                
                self.trained_models[model_name] = best_model
                
                st.success(f"✓ {model_name} trained successfully in {training_time:.2f}s")
                
            except Exception as e:
                st.error(f"✗ Error training {model_name}: {str(e)}")
                self.results[model_name] = {
                    'error': str(e),
                    'training_time': 0,
                    'metrics': {}
                }
            
            # Update progress
            progress_bar.progress((idx + 1) / len(model_configs))
        
        status_text.text("Training complete!")
        st.success("All models trained successfully!")
        
        return self.results
    
    def _calculate_metrics(self, y_pred, model):
        """
        Calculate evaluation metrics for a model.
        
        Args:
            y_pred: Predictions
            model: Trained model
            
        Returns:
            dict: Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(self.y_test, y_pred)
        
        # Handle different averaging strategies based on problem type
        n_classes = len(np.unique(self.y_test))
        average = 'binary' if n_classes == 2 else 'weighted'
        
        metrics['precision'] = precision_score(self.y_test, y_pred, average=average, zero_division=0)
        metrics['recall'] = recall_score(self.y_test, y_pred, average=average, zero_division=0)
        metrics['f1_score'] = f1_score(self.y_test, y_pred, average=average, zero_division=0)
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(self.y_test, y_pred)
        
        # ROC-AUC for binary classification
        if n_classes == 2:
            try:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(self.X_test)[:, 1]
                    metrics['roc_auc'] = roc_auc_score(self.y_test, y_proba)
                    metrics['roc_curve'] = roc_curve(self.y_test, y_proba)
                else:
                    metrics['roc_auc'] = None
            except:
                metrics['roc_auc'] = None
        else:
            try:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(self.X_test)
                    metrics['roc_auc'] = roc_auc_score(self.y_test, y_proba, 
                                                       multi_class='ovr', average='weighted')
                else:
                    metrics['roc_auc'] = None
            except:
                metrics['roc_auc'] = None
        
        # Classification report
        metrics['classification_report'] = classification_report(
            self.y_test, y_pred, output_dict=True, zero_division=0
        )
        
        return metrics
    
    def display_individual_results(self, model_name):
        """
        Display detailed results for a specific model.
        
        Args:
            model_name: Name of the model
        """
        if model_name not in self.results:
            st.error(f"No results found for {model_name}")
            return
        
        result = self.results[model_name]
        
        if 'error' in result:
            st.error(f"Error: {result['error']}")
            return
        
        st.subheader(f"{model_name} - Detailed Results")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        metrics = result['metrics']
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.4f}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.4f}")
        with col4:
            st.metric("F1-Score", f"{metrics['f1_score']:.4f}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if metrics.get('roc_auc') is not None:
                st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
            st.metric("Training Time", f"{result['training_time']:.2f}s")
        
        with col2:
            # Best parameters
            if result['best_params']:
                st.write("**Best Hyperparameters:**")
                for param, value in result['best_params'].items():
                    st.write(f"- {param}: {value}")
        
        # Confusion Matrix
        st.write("**Confusion Matrix:**")
        import plotly.figure_factory as ff
        
        cm = metrics['confusion_matrix']
        fig = ff.create_annotated_heatmap(
            z=cm,
            colorscale='Blues',
            showscale=True
        )
        fig.update_layout(title='Confusion Matrix', 
                         xaxis_title='Predicted', 
                         yaxis_title='Actual')
        st.plotly_chart(fig, use_container_width=True)
        
        # ROC Curve for binary classification
        if metrics.get('roc_curve') is not None:
            st.write("**ROC Curve:**")
            fpr, tpr, _ = metrics['roc_curve']
            
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                                    name='Random Classifier', line=dict(dash='dash')))
            fig.update_layout(
                title=f'ROC Curve (AUC = {metrics["roc_auc"]:.4f})',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate'
            )
            st.plotly_chart(fig, use_container_width=True)
