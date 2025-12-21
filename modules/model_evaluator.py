"""
Model Evaluator Module
Handles model comparison, visualization, and results export.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


class ModelEvaluator:
    """Class to evaluate and compare multiple models."""
    
    def __init__(self, results):
        """
        Initialize the model evaluator.
        
        Args:
            results: Dictionary of model results from ModelTrainer
        """
        self.results = results
    
    def create_comparison_table(self):
        """
        Create a comparison table of all models.
        
        Returns:
            pd.DataFrame: Comparison table
        """
        comparison_data = []
        
        for model_name, result in self.results.items():
            if 'error' in result:
                continue
            
            metrics = result['metrics']
            row = {
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'Training Time (s)': result['training_time']
            }
            
            if metrics.get('roc_auc') is not None:
                row['ROC-AUC'] = metrics['roc_auc']
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by F1-Score (default primary metric)
        if 'F1-Score' in df.columns:
            df = df.sort_values('F1-Score', ascending=False)
        
        return df
    
    def display_comparison_dashboard(self):
        """Display comprehensive model comparison dashboard."""
        st.header("Model Comparison Dashboard")
        
        # Get comparison table
        comparison_df = self.create_comparison_table()
        
        if comparison_df.empty:
            st.warning("No model results to compare.")
            return
        
        # Display comparison table
        st.subheader("Performance Comparison Table")
        
        # Highlight best values
        styled_df = comparison_df.style.highlight_max(
            subset=[col for col in comparison_df.columns if col != 'Model' and col != 'Training Time (s)'],
            color='lightgreen'
        ).highlight_min(
            subset=['Training Time (s)'],
            color='lightblue'
        )
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Ranking
        st.subheader("🏆 Model Rankings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top 3 by F1-Score:**")
            top_3 = comparison_df.nlargest(3, 'F1-Score')[['Model', 'F1-Score']]
            for idx, row in top_3.iterrows():
                medal = ["🥇", "🥈", "🥉"][top_3.index.get_loc(idx)]
                st.write(f"{medal} {row['Model']}: {row['F1-Score']:.4f}")
        
        with col2:
            st.write("**Top 3 by Accuracy:**")
            top_3_acc = comparison_df.nlargest(3, 'Accuracy')[['Model', 'Accuracy']]
            for idx, row in top_3_acc.iterrows():
                medal = ["🥇", "🥈", "🥉"][top_3_acc.index.get_loc(idx)]
                st.write(f"{medal} {row['Model']}: {row['Accuracy']:.4f}")
        
        # Best model summary
        best_model_name = comparison_df.iloc[0]['Model']
        st.success(f"🌟 **Best Overall Model (by F1-Score):** {best_model_name}")
        
        # Visualization
        self.create_comparison_visualizations(comparison_df)
        
        # Download option
        st.subheader("💾 Download Results")
        csv = comparison_df.to_csv(index=False)
        st.download_button(
            label="Download Comparison Table (CSV)",
            data=csv,
            file_name="model_comparison.csv",
            mime="text/csv"
        )
        
        return comparison_df, best_model_name
    
    def create_comparison_visualizations(self, comparison_df):
        """
        Create visualizations for model comparison.
        
        Args:
            comparison_df: Comparison DataFrame
        """
        st.subheader("Performance Visualizations")
        
        # Metrics bar chart
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]
        
        if available_metrics:
            # Reshape data for plotting
            plot_data = comparison_df.melt(
                id_vars=['Model'],
                value_vars=available_metrics,
                var_name='Metric',
                value_name='Score'
            )
            
            fig = px.bar(
                plot_data,
                x='Model',
                y='Score',
                color='Metric',
                barmode='group',
                title='Model Performance Comparison',
                labels={'Score': 'Score', 'Model': 'Model'},
                height=500
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Individual metric comparisons
        col1, col2 = st.columns(2)
        
        with col1:
            # F1-Score comparison
            fig = px.bar(
                comparison_df,
                x='Model',
                y='F1-Score',
                title='F1-Score Comparison',
                color='F1-Score',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(xaxis_tickangle=-45, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Training time comparison
            fig = px.bar(
                comparison_df,
                x='Model',
                y='Training Time (s)',
                title='Training Time Comparison',
                color='Training Time (s)',
                color_continuous_scale='Reds'
            )
            fig.update_layout(xaxis_tickangle=-45, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Radar chart for overall comparison
        if len(available_metrics) >= 3:
            st.write("**Multi-Metric Radar Chart**")
            self.create_radar_chart(comparison_df, available_metrics)
        
        # ROC Curves comparison (if available)
        if 'ROC-AUC' in comparison_df.columns:
            self.plot_roc_curves_comparison()
    
    def create_radar_chart(self, comparison_df, metrics):
        """
        Create a radar chart for model comparison.
        
        Args:
            comparison_df: Comparison DataFrame
            metrics: List of metrics to include
        """
        # Select top 5 models for clarity
        top_models = comparison_df.head(5)
        
        fig = go.Figure()
        
        for idx, row in top_models.iterrows():
            values = [row[metric] for metric in metrics]
            values.append(values[0])  # Close the radar chart
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                fill='toself',
                name=row['Model']
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Multi-Metric Model Comparison (Top 5 Models)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_roc_curves_comparison(self):
        """Plot ROC curves for all models (binary classification only)."""
        st.write("**ROC Curves Comparison**")
        
        fig = go.Figure()
        
        # Add random classifier baseline
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray')
        ))
        
        # Add ROC curve for each model
        for model_name, result in self.results.items():
            if 'error' in result:
                continue
            
            metrics = result['metrics']
            if metrics.get('roc_curve') is not None:
                fpr, tpr, _ = metrics['roc_curve']
                auc = metrics.get('roc_auc', 0)
                
                fig.add_trace(go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode='lines',
                    name=f"{model_name} (AUC={auc:.3f})"
                ))
        
        fig.update_layout(
            title='ROC Curves - All Models',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def get_best_model_info(self, comparison_df):
        """
        Get detailed information about the best model.
        
        Args:
            comparison_df: Comparison DataFrame
            
        Returns:
            dict: Best model information
        """
        if comparison_df.empty:
            return None
        
        best_model_row = comparison_df.iloc[0]
        best_model_name = best_model_row['Model']
        best_model_result = self.results[best_model_name]
        
        info = {
            'name': best_model_name,
            'metrics': best_model_row.to_dict(),
            'best_params': best_model_result.get('best_params', {}),
            'model_object': best_model_result.get('model')
        }
        
        return info
    
    def display_confusion_matrices(self):
        """Display confusion matrices for all models."""
        st.subheader("🔲 Confusion Matrices")
        
        # Calculate number of columns for layout
        n_models = len([r for r in self.results.values() if 'error' not in r])
        n_cols = min(3, n_models)
        
        models_list = [(name, result) for name, result in self.results.items() if 'error' not in result]
        
        for i in range(0, len(models_list), n_cols):
            cols = st.columns(n_cols)
            
            for j, (model_name, result) in enumerate(models_list[i:i+n_cols]):
                with cols[j]:
                    import plotly.figure_factory as ff
                    
                    cm = result['metrics']['confusion_matrix']
                    
                    fig = ff.create_annotated_heatmap(
                        z=cm,
                        colorscale='Blues',
                        showscale=False
                    )
                    fig.update_layout(
                        title=model_name,
                        xaxis_title='Predicted',
                        yaxis_title='Actual',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    def export_detailed_results(self):
        """
        Export detailed results for all models.
        
        Returns:
            dict: Detailed results dictionary
        """
        detailed_results = {}
        
        for model_name, result in self.results.items():
            if 'error' in result:
                detailed_results[model_name] = {'error': result['error']}
                continue
            
            detailed_results[model_name] = {
                'metrics': result['metrics'],
                'best_params': result.get('best_params', {}),
                'training_time': result['training_time']
            }
        
        return detailed_results
