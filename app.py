"""
Main Streamlit Application
AutoML System for Classification Tasks
Version: 1.0.1
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.helpers import (
    set_page_config, apply_custom_css, display_header,
    initialize_session_state, create_sidebar_navigation,
    validate_csv_upload, show_success_message, show_error_message,
    show_info_message
)
from modules.data_loader import (
    load_dataset, display_basic_info, get_column_types, validate_dataset
)
from modules.eda import perform_eda, get_train_test_split_summary
from modules.issue_detector import IssueDetector, create_issue_report
from modules.preprocessor import DataPreprocessor
from modules.model_trainer import ModelTrainer
from modules.model_evaluator import ModelEvaluator
from modules.report_generator import ReportGenerator

# Page configuration
set_page_config()

# Apply custom CSS
apply_custom_css()

# Initialize session state
initialize_session_state()

# Display header
display_header()

# Create sidebar navigation
create_sidebar_navigation()

# Main application flow
def main():
    """Main application function."""
    
    # Step 1: Dataset Upload
    st.header("📁 Step 1: Upload Dataset")
    
    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=['csv'],
        help="Upload a CSV file containing your classification dataset"
    )
    
    if uploaded_file is not None:
        # Load dataset
        df = load_dataset(uploaded_file)
        
        if df is not None:
            # Validate dataset
            is_valid, message = validate_csv_upload(df)
            
            if not is_valid:
                show_error_message(message)
                return
            elif "Warning" in message:
                show_info_message(message)
            
            # Store in session state
            st.session_state.df = df
            st.session_state.data_loaded = True
            
            show_success_message(f"Dataset loaded successfully! ({len(df)} rows, {len(df.columns)} columns)")
            
            # Select target column
            st.subheader("🎯 Select Target Column")
            target_col = st.selectbox(
                "Select the target column for classification:",
                options=df.columns.tolist(),
                index=len(df.columns) - 1  # Default to last column
            )
            
            st.session_state.target_col = target_col
            
            # Get column types
            numerical_cols, categorical_cols = get_column_types(df)
            st.session_state.numerical_cols = numerical_cols
            st.session_state.categorical_cols = categorical_cols
            
            # Display basic info
            display_basic_info(df, target_col)
            
            # Validation
            validation_results = validate_dataset(df)
            if validation_results['errors']:
                for error in validation_results['errors']:
                    show_error_message(error)
            if validation_results['warnings']:
                for warning in validation_results['warnings']:
                    show_info_message(warning)
            
            st.markdown("---")
            
            # Step 2: Exploratory Data Analysis
            st.header("🔍 Step 2: Exploratory Data Analysis")
            
            if st.button("🔍 Run Automated EDA", type="primary"):
                with st.spinner("Performing EDA..."):
                    perform_eda(df, numerical_cols, categorical_cols)
                    st.session_state.eda_complete = True
                    show_success_message("EDA completed successfully!")
            
            st.markdown("---")
            
            # Step 3: Issue Detection
            st.header("⚠️ Step 3: Data Quality Issue Detection")
            
            if st.button("🔍 Detect Data Issues", type="primary"):
                with st.spinner("Detecting data quality issues..."):
                    # Create issue detector
                    issue_detector = IssueDetector(
                        df, numerical_cols, categorical_cols, target_col
                    )
                    
                    # Detect all issues
                    issues = issue_detector.detect_all_issues()
                    st.session_state.issues = issues
                    st.session_state.issue_detector = issue_detector
                    
                    # Display issues and get user approval
                    user_decisions = issue_detector.display_issues_and_get_approval()
                    st.session_state.user_decisions = user_decisions
                    
                    # Display issue summary
                    if issues:
                        summary = issue_detector.get_issues_summary()
                        st.subheader("📋 Issue Summary")
                        st.dataframe(summary, use_container_width=True)
            
            st.markdown("---")
            
            # Step 4: Preprocessing
            st.header("⚙️ Step 4: Data Preprocessing")
            
            if st.button("⚙️ Configure Preprocessing", type="primary"):
                # Create preprocessor
                preprocessor = DataPreprocessor(
                    df, numerical_cols, categorical_cols, target_col
                )
                
                # Apply user decisions from issue detection
                if hasattr(st.session_state, 'user_decisions'):
                    preprocessor.apply_user_decisions(st.session_state.user_decisions)
                
                # Get preprocessing configuration
                config = preprocessor.configure_preprocessing()
                st.session_state.preprocessing_config = config
                st.session_state.preprocessor = preprocessor
                
                show_info_message("Preprocessing configuration saved. Click 'Apply Preprocessing' to proceed.")
            
            # Apply preprocessing
            if hasattr(st.session_state, 'preprocessing_config'):
                if st.button("✅ Apply Preprocessing", type="primary"):
                    with st.spinner("Applying preprocessing transformations..."):
                        preprocessor = st.session_state.preprocessor
                        config = st.session_state.preprocessing_config
                        
                        # Apply preprocessing
                        X_train, X_test, y_train, y_test, preprocessors = preprocessor.apply_preprocessing(config)
                        
                        # Store in session state
                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test
                        st.session_state.preprocessors = preprocessors
                        st.session_state.preprocessing_complete = True
                        
                        # Display preprocessing summary
                        preprocessor.display_preprocessing_summary()
                        
                        # Display train/test split summary
                        get_train_test_split_summary(X_train, X_test, y_train, y_test)
                        
                        show_success_message("Preprocessing completed successfully!")
            
            st.markdown("---")
            
            # Step 5: Model Training
            st.header("🤖 Step 5: Model Training & Hyperparameter Optimization")
            
            if st.session_state.preprocessing_complete:
                col1, col2 = st.columns(2)
                
                with col1:
                    optimization_method = st.selectbox(
                        "Hyperparameter Optimization Method:",
                        ["Grid Search", "Randomized Search"]
                    )
                
                with col2:
                    if optimization_method == "Randomized Search":
                        n_iter = st.number_input(
                            "Number of iterations:",
                            min_value=5,
                            max_value=50,
                            value=10
                        )
                    else:
                        n_iter = 10
                
                if st.button("🚀 Train All Models", type="primary"):
                    with st.spinner("Training models... This may take a few minutes..."):
                        # Create model trainer
                        trainer = ModelTrainer(
                            st.session_state.X_train,
                            st.session_state.X_test,
                            st.session_state.y_train,
                            st.session_state.y_test
                        )
                        
                        # Train all models
                        method = 'grid' if optimization_method == "Grid Search" else 'random'
                        results = trainer.train_all_models(method, n_iter)
                        
                        # Store results
                        st.session_state.model_results = results
                        st.session_state.trainer = trainer
                        st.session_state.models_trained = True
                        
                        show_success_message(f"All models trained successfully using {optimization_method}!")
                
                # Display individual model results
                if st.session_state.models_trained:
                    st.subheader("🔍 View Individual Model Results")
                    
                    model_names = list(st.session_state.model_results.keys())
                    selected_model = st.selectbox("Select a model to view details:", model_names)
                    
                    if selected_model:
                        st.session_state.trainer.display_individual_results(selected_model)
            else:
                show_info_message("Please complete preprocessing before training models.")
            
            st.markdown("---")
            
            # Step 6: Model Comparison
            st.header("📊 Step 6: Model Comparison Dashboard")
            
            if st.session_state.models_trained:
                # Create evaluator
                evaluator = ModelEvaluator(st.session_state.model_results)
                
                # Display comparison dashboard
                comparison_df, best_model_name = evaluator.display_comparison_dashboard()
                
                # Store in session state
                st.session_state.comparison_df = comparison_df
                st.session_state.best_model_name = best_model_name
                st.session_state.evaluator = evaluator
                
                # Display confusion matrices
                if st.checkbox("Show Confusion Matrices for All Models"):
                    evaluator.display_confusion_matrices()
            else:
                show_info_message("Please train models first.")
            
            st.markdown("---")
            
            # Step 7: Generate Report
            st.header("📄 Step 7: Generate Evaluation Report")
            
            if st.session_state.models_trained:
                if st.button("📄 Generate Comprehensive Report", type="primary"):
                    with st.spinner("Generating report..."):
                        # Prepare data for report
                        dataset_info = {
                            'n_rows': len(df),
                            'n_cols': len(df.columns),
                            'target_col': target_col,
                            'n_classes': df[target_col].nunique(),
                            'class_distribution': df[target_col].value_counts().to_dict()
                        }
                        
                        # EDA info
                        eda_info = {
                            'missing_values': {col: (df[col].isnull().sum() / len(df) * 100) 
                                             for col in df.columns if df[col].isnull().sum() > 0},
                            'outliers': {},
                            'high_correlation': []
                        }
                        
                        # Get best model info
                        best_model_info = st.session_state.evaluator.get_best_model_info(
                            st.session_state.comparison_df
                        )
                        
                        # Get preprocessing steps
                        preprocessing_steps = st.session_state.preprocessor.get_preprocessing_summary()
                        
                        # Create report generator
                        report_gen = ReportGenerator(
                            dataset_info=dataset_info,
                            eda_info=eda_info,
                            issues=st.session_state.get('issues', []),
                            preprocessing_steps=preprocessing_steps,
                            model_results=st.session_state.model_results,
                            best_model_info=best_model_info
                        )
                        
                        # Display and download report
                        report_gen.display_and_download_report()
                        
                        show_success_message("Report generated successfully!")
            else:
                show_info_message("Please train models before generating report.")
    
    else:
        # Display instructions
        st.info("""
        👆 **Get Started:**
        
        1. Upload a CSV file containing your classification dataset
        2. The system will automatically guide you through:
           - Exploratory Data Analysis
           - Data Quality Issue Detection
           - Preprocessing Configuration
           - Model Training & Optimization
           - Performance Comparison
           - Report Generation
        
        **Supported Features:**
        - 7 Classification Algorithms
        - Automated Hyperparameter Tuning
        - Comprehensive Performance Metrics
        - Interactive Visualizations
        - Downloadable Reports
        """)
        
        # Display sample data format
        with st.expander("📝 Sample Dataset Format"):
            st.write("Your CSV should have the following structure:")
            sample_data = pd.DataFrame({
                'feature_1': [1, 2, 3, 4, 5],
                'feature_2': [10, 20, 30, 40, 50],
                'feature_3': ['A', 'B', 'A', 'B', 'A'],
                'target': [0, 1, 0, 1, 0]
            })
            st.dataframe(sample_data, use_container_width=True)
            st.caption("The last column typically contains the target/label for classification")


# Run the application
if __name__ == "__main__":
    main()
