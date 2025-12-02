"""
Report Generator Module
Generates comprehensive evaluation reports in PDF/HTML/Markdown format.
"""

import pandas as pd
import streamlit as st
from datetime import datetime
import base64


class ReportGenerator:
    """Class to generate comprehensive AutoML reports."""
    
    def __init__(self, dataset_info, eda_info, issues, preprocessing_steps, 
                 model_results, best_model_info):
        """
        Initialize the report generator.
        
        Args:
            dataset_info: Dictionary with dataset information
            eda_info: Dictionary with EDA findings
            issues: List of detected issues
            preprocessing_steps: List of preprocessing steps
            model_results: Dictionary of model results
            best_model_info: Information about the best model
        """
        self.dataset_info = dataset_info
        self.eda_info = eda_info
        self.issues = issues
        self.preprocessing_steps = preprocessing_steps
        self.model_results = model_results
        self.best_model_info = best_model_info
        self.report_content = ""
    
    def generate_markdown_report(self):
        """
        Generate a comprehensive report in Markdown format.
        
        Returns:
            str: Markdown formatted report
        """
        report = []
        
        # Header
        report.append("# AutoML Classification Report")
        report.append(f"\n**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append("---\n")
        
        # 1. Dataset Overview
        report.append("## 1. Dataset Overview\n")
        report.append(f"- **Number of Rows:** {self.dataset_info.get('n_rows', 'N/A')}")
        report.append(f"- **Number of Columns:** {self.dataset_info.get('n_cols', 'N/A')}")
        report.append(f"- **Target Column:** {self.dataset_info.get('target_col', 'N/A')}")
        report.append(f"- **Number of Classes:** {self.dataset_info.get('n_classes', 'N/A')}\n")
        
        if 'class_distribution' in self.dataset_info:
            report.append("### Class Distribution\n")
            for class_name, count in self.dataset_info['class_distribution'].items():
                report.append(f"- **{class_name}:** {count}")
            report.append("")
        
        # 2. EDA Findings
        report.append("## 2. Exploratory Data Analysis Findings\n")
        
        if 'missing_values' in self.eda_info:
            report.append("### Missing Values\n")
            if self.eda_info['missing_values']:
                for col, pct in self.eda_info['missing_values'].items():
                    report.append(f"- **{col}:** {pct:.2f}%")
            else:
                report.append("- No missing values detected")
            report.append("")
        
        if 'outliers' in self.eda_info:
            report.append("### Outliers\n")
            if self.eda_info['outliers']:
                for col, count in self.eda_info['outliers'].items():
                    report.append(f"- **{col}:** {count} outliers detected")
            else:
                report.append("- No significant outliers detected")
            report.append("")
        
        if 'high_correlation' in self.eda_info and self.eda_info['high_correlation']:
            report.append("### Highly Correlated Features\n")
            for pair in self.eda_info['high_correlation']:
                report.append(f"- **{pair[0]}** and **{pair[1]}:** {pair[2]:.3f}")
            report.append("")
        
        # 3. Detected Issues
        report.append("## 3. Data Quality Issues\n")
        
        if self.issues:
            issue_types = {}
            for issue in self.issues:
                issue_type = issue['type'].replace('_', ' ').title()
                if issue_type not in issue_types:
                    issue_types[issue_type] = []
                issue_types[issue_type].append(issue)
            
            for issue_type, issues_list in issue_types.items():
                report.append(f"### {issue_type}\n")
                for issue in issues_list:
                    report.append(f"- **{issue['column']}:** {issue['description']}")
                report.append("")
        else:
            report.append("No data quality issues detected.\n")
        
        # 4. Preprocessing Steps
        report.append("## 4. Preprocessing Pipeline\n")
        
        if self.preprocessing_steps:
            for i, step in enumerate(self.preprocessing_steps, 1):
                report.append(f"{i}. {step}")
            report.append("")
        else:
            report.append("No preprocessing steps applied.\n")
        
        # 5. Model Training & Results
        report.append("## 5. Model Training & Evaluation\n")
        
        report.append("### Models Trained\n")
        for model_name in self.model_results.keys():
            report.append(f"- {model_name}")
        report.append("")
        
        # 6. Model Comparison
        report.append("## 6. Model Performance Comparison\n")
        
        # Create comparison table
        comparison_data = []
        for model_name, result in self.model_results.items():
            if 'error' in result:
                continue
            metrics = result['metrics']
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'Training Time': f"{result['training_time']:.2f}s"
            })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            report.append(df.to_markdown(index=False))
            report.append("")
        
        # 7. Best Model
        report.append("## 7. Best Model Summary\n")
        
        if self.best_model_info:
            report.append(f"### {self.best_model_info['name']}\n")
            
            report.append("#### Performance Metrics\n")
            for metric, value in self.best_model_info['metrics'].items():
                if metric != 'Model' and isinstance(value, (int, float)):
                    report.append(f"- **{metric}:** {value:.4f}")
            report.append("")
            
            if self.best_model_info.get('best_params'):
                report.append("#### Best Hyperparameters\n")
                for param, value in self.best_model_info['best_params'].items():
                    report.append(f"- **{param}:** {value}")
                report.append("")
            
            report.append("#### Justification\n")
            report.append(f"This model was selected as the best performer based on the F1-Score metric, "
                        f"achieving {self.best_model_info['metrics'].get('F1-Score', 'N/A')} on the test set. ")
            
            # Add context based on metrics
            f1_score = self.best_model_info['metrics'].get('F1-Score', 0)
            if isinstance(f1_score, (int, float)):
                if f1_score >= 0.9:
                    report.append("The model demonstrates excellent performance with high precision and recall.")
                elif f1_score >= 0.75:
                    report.append("The model shows good performance with balanced precision and recall.")
                elif f1_score >= 0.6:
                    report.append("The model shows moderate performance. Consider feature engineering or gathering more data.")
                else:
                    report.append("The model shows room for improvement. Consider reviewing features, data quality, or trying different algorithms.")
            report.append("")
        
        # 8. Recommendations
        report.append("## 8. Recommendations\n")
        
        recommendations = self._generate_recommendations()
        for rec in recommendations:
            report.append(f"- {rec}")
        report.append("")
        
        # Footer
        report.append("---")
        report.append("\n*This report was automatically generated by the AutoML Classification System.*")
        
        self.report_content = "\n".join(report)
        return self.report_content
    
    def _generate_recommendations(self):
        """Generate recommendations based on results."""
        recommendations = []
        
        # Based on best model performance
        if self.best_model_info:
            f1_score = self.best_model_info['metrics'].get('F1-Score', 0)
            
            if isinstance(f1_score, (int, float)) and f1_score < 0.7:
                recommendations.append("Consider collecting more training data to improve model performance")
                recommendations.append("Explore feature engineering to create more informative features")
        
        # Based on detected issues
        if self.issues:
            has_missing = any(i['type'] == 'missing_values' for i in self.issues)
            if has_missing:
                recommendations.append("Investigate the root cause of missing values")
            
            has_imbalance = any(i['type'] == 'class_imbalance' for i in self.issues)
            if has_imbalance:
                recommendations.append("Consider using techniques like SMOTE or class weights to handle class imbalance")
        
        # Based on model types
        if self.model_results:
            # Check if ensemble models performed well
            ensemble_models = ['Random Forest']
            best_name = self.best_model_info.get('name', '')
            
            if best_name not in ensemble_models:
                recommendations.append("Consider trying ensemble methods like Gradient Boosting or XGBoost")
        
        # General recommendations
        recommendations.append("Perform cross-validation to ensure model stability")
        recommendations.append("Monitor model performance on new data and retrain periodically")
        
        if not recommendations:
            recommendations.append("Continue monitoring model performance in production")
        
        return recommendations
    
    def generate_html_report(self):
        """
        Generate HTML version of the report.
        
        Returns:
            str: HTML formatted report
        """
        # First generate markdown
        if not self.report_content:
            self.generate_markdown_report()
        
        # Convert markdown to HTML (basic conversion)
        import markdown
        html_content = markdown.markdown(self.report_content, extensions=['tables'])
        
        # Wrap in HTML template
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AutoML Classification Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    background-color: white;
                    padding: 30px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 30px; border-bottom: 2px solid #ecf0f1; padding-bottom: 8px; }}
                h3 {{ color: #7f8c8d; }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                th {{
                    background-color: #3498db;
                    color: white;
                }}
                tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                ul {{ line-height: 1.8; }}
                code {{ background-color: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="container">
                {html_content}
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def display_and_download_report(self):
        """Display report and provide download options."""
        st.header("📄 Auto-Generated Evaluation Report")
        
        # Generate reports
        markdown_report = self.generate_markdown_report()
        html_report = self.generate_html_report()
        
        # Display options
        report_format = st.radio(
            "Select report format to preview:",
            ["Markdown", "HTML"]
        )
        
        if report_format == "Markdown":
            st.markdown(markdown_report)
        else:
            st.components.v1.html(html_report, height=800, scrolling=True)
        
        # Download buttons
        st.subheader("💾 Download Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="Download as Markdown (.md)",
                data=markdown_report,
                file_name=f"automl_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
        
        with col2:
            st.download_button(
                label="Download as HTML (.html)",
                data=html_report,
                file_name=f"automl_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html"
            )


def create_summary_report(dataset_name, n_rows, n_cols, target_col, n_models, best_model, best_f1):
    """
    Create a quick summary report.
    
    Args:
        dataset_name: Name of the dataset
        n_rows: Number of rows
        n_cols: Number of columns
        target_col: Target column name
        n_models: Number of models trained
        best_model: Name of best model
        best_f1: F1-score of best model
        
    Returns:
        str: Summary report
    """
    summary = f"""
    ## AutoML Summary Report
    
    **Dataset:** {dataset_name}
    **Samples:** {n_rows} | **Features:** {n_cols}
    **Target:** {target_col}
    
    **Models Trained:** {n_models}
    **Best Model:** {best_model}
    **Best F1-Score:** {best_f1:.4f}
    
    Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
    
    return summary
