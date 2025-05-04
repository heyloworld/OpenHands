import os
import numpy as np
import pandas as pd
import time
from typing import Dict, Any, Tuple
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_enron_spam_dataset
from src.model import SpamClassifier

def tune_hyperparameters(X_train, y_train, param_grid: Dict[str, Any] = None, 
                        cv: int = 5, n_jobs: int = -1) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Perform hyperparameter tuning using GridSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training labels
        param_grid: Grid of parameters to search
        cv: Number of cross-validation folds
        n_jobs: Number of jobs to run in parallel
        
    Returns:
        Tuple of (best parameters, CV results)
    """
    if param_grid is None:
        param_grid = {
            'kernel': ['linear', 'rbf'],
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1]
        }
    
    # Define scoring metrics
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score)
    }
    
    # Create base model
    base_model = SpamClassifier().model
    
    print(f"Starting hyperparameter tuning with {cv}-fold cross-validation...")
    print(f"Parameter grid: {param_grid}")
    
    # Create GridSearchCV
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit='f1',  # Refit on the best parameters based on f1 score
        verbose=1,
        return_train_score=True
    )
    
    # Fit GridSearchCV
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    end_time = time.time()
    
    print(f"Hyperparameter tuning completed in {end_time - start_time:.2f} seconds")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score (F1): {grid_search.best_score_:.4f}")
    
    # Get CV results
    cv_results = pd.DataFrame(grid_search.cv_results_)
    
    return grid_search.best_params_, cv_results

def plot_cv_results(cv_results: pd.DataFrame, output_path: str = None) -> None:
    """
    Plot cross-validation results.
    
    Args:
        cv_results: DataFrame of CV results
        output_path: Path to save the plot
    """
    # Extract relevant columns
    param_cols = [col for col in cv_results.columns if col.startswith('param_')]
    score_cols = [col for col in cv_results.columns if col.startswith('mean_test_')]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot mean test scores for each parameter combination
    for i, score_col in enumerate(score_cols):
        metric_name = score_col.replace('mean_test_', '')
        plt.subplot(2, 2, i+1)
        
        # Sort by score
        sorted_idx = cv_results[score_col].argsort()[::-1]
        top_n = min(10, len(sorted_idx))  # Show top 10 combinations
        
        x_labels = []
        for idx in sorted_idx[:top_n]:
            label = ', '.join([f"{param.replace('param_', '')}: {cv_results.iloc[idx][param]}" 
                              for param in param_cols])
            x_labels.append(label)
        
        plt.bar(range(top_n), cv_results.iloc[sorted_idx[:top_n]][score_col])
        plt.xticks(range(top_n), range(1, top_n+1), rotation=0)
        plt.title(f'Top {top_n} Parameter Combinations - {metric_name.capitalize()}')
        plt.ylabel(f'Mean {metric_name.capitalize()} Score')
        plt.xlabel('Parameter Combination Rank')
    
    plt.tight_layout()
    
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"CV results plot saved to {output_path}")
    
    plt.close()

def generate_classification_report(metrics: Dict[str, Any], model_params: Dict[str, Any], 
                                 cv_results: pd.DataFrame, output_path: str) -> None:
    """
    Generate a PDF report of classification results.
    
    Args:
        metrics: Dictionary of evaluation metrics
        model_params: Model parameters
        cv_results: Cross-validation results
        output_path: Path to save the report
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create PDF document
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=12
    )
    heading_style = ParagraphStyle(
        'Heading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=6
    )
    normal_style = styles['Normal']
    
    # Create content
    content = []
    
    # Title
    content.append(Paragraph("Spam Email Classification Report", title_style))
    content.append(Spacer(1, 0.25*inch))
    
    # Model Information
    content.append(Paragraph("Model Information", heading_style))
    model_info = [
        f"Model: Support Vector Machine (SVM)",
        f"Kernel: {model_params['kernel']}",
        f"C: {model_params['C']}",
        f"Gamma: {model_params['gamma']}"
    ]
    for info in model_info:
        content.append(Paragraph(info, normal_style))
    content.append(Spacer(1, 0.2*inch))
    
    # Performance Metrics
    content.append(Paragraph("Performance Metrics", heading_style))
    metrics_data = [
        ["Metric", "Value"],
        ["Accuracy", f"{metrics['accuracy']:.4f}"],
        ["Precision", f"{metrics['precision']:.4f}"],
        ["Recall", f"{metrics['recall']:.4f}"],
        ["F1 Score", f"{metrics['f1_score']:.4f}"]
    ]
    metrics_table = Table(metrics_data, colWidths=[2*inch, 1.5*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (1, 0), 12),
        ('BACKGROUND', (0, 1), (1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    content.append(metrics_table)
    content.append(Spacer(1, 0.2*inch))
    
    # Classification Report
    content.append(Paragraph("Detailed Classification Report", heading_style))
    report_text = metrics['classification_report'].replace('\n', '<br/>')
    content.append(Paragraph(f"<pre>{report_text}</pre>", normal_style))
    content.append(Spacer(1, 0.2*inch))
    
    # Confusion Matrix
    content.append(Paragraph("Confusion Matrix", heading_style))
    # Add confusion matrix image
    if os.path.exists('results/figures/confusion_matrix.png'):
        content.append(Image('results/figures/confusion_matrix.png', width=5*inch, height=4*inch))
    content.append(Spacer(1, 0.2*inch))
    
    # Hyperparameter Tuning Results
    content.append(Paragraph("Hyperparameter Tuning Results", heading_style))
    
    # Top 5 parameter combinations
    top_params = cv_results.sort_values('mean_test_f1', ascending=False).head(5)
    param_data = [["Rank", "Parameters", "F1 Score", "Accuracy"]]
    
    for i, (_, row) in enumerate(top_params.iterrows(), 1):
        params_str = ', '.join([f"{param.replace('param_', '')}: {row[param]}" 
                               for param in cv_results.columns if param.startswith('param_')])
        param_data.append([
            str(i),
            params_str,
            f"{row['mean_test_f1']:.4f}",
            f"{row['mean_test_accuracy']:.4f}"
        ])
    
    param_table = Table(param_data, colWidths=[0.5*inch, 3.5*inch, 1*inch, 1*inch])
    param_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    content.append(param_table)
    
    # Build PDF
    doc.build(content)
    print(f"Classification report saved to {output_path}")

def train_and_evaluate(data_dir: str = 'data/enron-spam', 
                      model_dir: str = 'models/saved_models',
                      results_dir: str = 'results',
                      tune_params: bool = True,
                      cv: int = 5) -> None:
    """
    Train and evaluate the spam classifier.
    
    Args:
        data_dir: Directory containing the dataset
        model_dir: Directory to save the trained model
        results_dir: Directory to save results
        tune_params: Whether to perform hyperparameter tuning
        cv: Number of cross-validation folds
    """
    # Create directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'figures'), exist_ok=True)
    
    # Load data
    data = load_enron_spam_dataset(data_dir=data_dir)
    
    # Default parameters
    best_params = {
        'kernel': 'linear',
        'C': 1.0,
        'gamma': 'scale'
    }
    
    cv_results = None
    
    # Perform hyperparameter tuning if requested
    if tune_params:
        best_params, cv_results = tune_hyperparameters(
            data['X_train'], 
            data['y_train'],
            cv=cv
        )
        
        # Plot CV results
        plot_cv_results(
            cv_results,
            os.path.join(results_dir, 'figures', 'cv_results.png')
        )
    
    # Create and train model with best parameters
    classifier = SpamClassifier(
        kernel=best_params['kernel'],
        C=best_params['C'],
        gamma=best_params['gamma']
    )
    
    classifier.train(data['X_train'], data['y_train'])
    
    # Save model
    model_path = os.path.join(model_dir, 'spam_classifier.pkl')
    classifier.save_model(model_path)
    
    # Evaluate model
    metrics = classifier.evaluate(data['X_test'], data['y_test'], data['class_names'])
    
    # Print metrics
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    
    print("\nClassification Report:")
    print(metrics['classification_report'])
    
    # Plot confusion matrix
    classifier.plot_confusion_matrix(
        metrics['confusion_matrix'],
        data['class_names'],
        os.path.join(results_dir, 'figures', 'confusion_matrix.png')
    )
    
    # Generate classification report
    generate_classification_report(
        metrics,
        best_params,
        cv_results if cv_results is not None else pd.DataFrame(),
        os.path.join(results_dir, 'classification_report.pdf')
    )
    
    print("\nTraining and evaluation completed successfully!")

if __name__ == "__main__":
    train_and_evaluate(tune_params=True, cv=3)  # Use fewer CV folds for testing