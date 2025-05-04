"""
Training script for the Telco Customer Churn prediction model.
This script handles loading the data, training the model with cross-validation,
and evaluating the model performance.
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import matplotlib.pyplot as plt

from src.data_loader import TelcoChurnDataLoader
from src.model import ChurnPredictionModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Train a Telco Customer Churn prediction model')
    
    # Data arguments
    parser.add_argument('--data-url', type=str, default=None,
                        help='URL to download the dataset from')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to the dataset file')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of the dataset to include in the test split')
    
    # Feature engineering arguments
    parser.add_argument('--num-features', type=int, default=15,
                        help='Number of features to select')
    parser.add_argument('--handle-imbalance', type=str, default='smote',
                        choices=['smote', 'undersample', 'none'],
                        help='Method to handle imbalanced data')
    parser.add_argument('--sampling-strategy', type=float, default=1.0,
                        help='Sampling strategy for handling imbalanced data')
    
    # Model arguments
    parser.add_argument('--model-type', type=str, default='logistic',
                        choices=['logistic'],
                        help='Type of model to train')
    parser.add_argument('--cv', type=int, default=5,
                        help='Number of cross-validation folds')
    parser.add_argument('--class-weight', type=str, default=None,
                        choices=['balanced', 'none'],
                        help='Class weights for the model')
    
    # Output arguments
    parser.add_argument('--model-path', type=str, default='models/logistic_regression.pkl',
                        help='Path to save the trained model')
    parser.add_argument('--metrics-path', type=str, default='results/metrics/classification_report.txt',
                        help='Path to save the classification report')
    parser.add_argument('--roc-curve-path', type=str, default='results/figures/roc_curve.png',
                        help='Path to save the ROC curve')
    parser.add_argument('--pr-curve-path', type=str, default='results/figures/pr_curve.png',
                        help='Path to save the Precision-Recall curve')
    
    # Misc arguments
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state for reproducibility')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity level')
    
    args = parser.parse_args()
    
    # Convert 'none' to None for class_weight
    if args.class_weight == 'none':
        args.class_weight = None
        
    return args

def train_with_cross_validation(X_train, y_train, model_type='logistic', cv=5, class_weight=None, random_state=42, verbose=1):
    """
    Train a model with cross-validation and hyperparameter tuning.
    
    Args:
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        model_type (str, optional): Type of model to train.
        cv (int, optional): Number of cross-validation folds.
        class_weight (str or dict, optional): Class weights for the model.
        random_state (int, optional): Random state for reproducibility.
        verbose (int, optional): Verbosity level.
        
    Returns:
        tuple: Best model and best parameters.
    """
    logger.info(f"Training {model_type} model with {cv}-fold cross-validation")
    
    if model_type == 'logistic':
        # Create the model
        model = ChurnPredictionModel(random_state=random_state)
        
        # Define the parameter grid for grid search
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear'],  # liblinear supports both l1 and l2
            'class_weight': [class_weight, None] if class_weight else [None]
        }
        
        # Create a base model for grid search
        base_model = model.build_model().model
        
        # Create the cross-validation strategy
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        
        # Create the grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv_strategy,
            scoring='f1',  # Use F1 score as the metric
            n_jobs=-1,  # Use all available cores
            verbose=verbose
        )
        
        # Fit the grid search
        grid_search.fit(X_train, y_train)
        
        # Get the best parameters and model
        best_params = grid_search.best_params_
        logger.info(f"Best parameters: {best_params}")
        
        # Create a new model with the best parameters
        best_model = model.build_model(
            C=best_params['C'],
            penalty=best_params['penalty'],
            solver=best_params['solver'],
            class_weight=best_params['class_weight']
        )
        
        # Train the model on the full training set
        best_model.train(X_train, y_train)
        
        return best_model, best_params
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def main():
    """
    Main function to train and evaluate the model.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Create data loader
    data_loader = TelcoChurnDataLoader(data_url=args.data_url, random_state=args.random_state)
    
    # Load and preprocess data
    data_path = '/workspace/OpenHands/data/churn_synthetic.csv'
    data_loader.load_data(filepath=data_path)
    data_loader.preprocess_data(test_size=args.test_size)
    
    # Select features
    X_train_selected, X_test_selected = data_loader.select_features(k=args.num_features)
    
    # Handle imbalanced data if specified
    if args.handle_imbalance != 'none':
        X_train_resampled, y_train_resampled = data_loader.handle_imbalanced_data(
            method=args.handle_imbalance,
            sampling_strategy=args.sampling_strategy
        )
    else:
        X_train_resampled, y_train_resampled = X_train_selected, data_loader.y_train
    
    # Train the model with cross-validation
    best_model, best_params = train_with_cross_validation(
        X_train_resampled, y_train_resampled,
        model_type=args.model_type,
        cv=args.cv,
        class_weight=args.class_weight,
        random_state=args.random_state,
        verbose=args.verbose
    )
    
    # Evaluate the model on the test set
    metrics = best_model.evaluate(X_test_selected, data_loader.y_test)
    
    # Save the model
    if args.model_path:
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        best_model.save_model(args.model_path)
    
    # Save the classification report
    if args.metrics_path:
        best_model.save_classification_report(X_test_selected, data_loader.y_test, args.metrics_path)
    
    # Plot and save the ROC curve
    if args.roc_curve_path:
        best_model.plot_roc_curve(X_test_selected, data_loader.y_test, args.roc_curve_path)
    
    # Plot and save the Precision-Recall curve
    if args.pr_curve_path:
        best_model.plot_precision_recall_curve(X_test_selected, data_loader.y_test, args.pr_curve_path)
    
    # Print a summary of the results
    print("\n" + "="*80)
    print("Model Training Summary")
    print("="*80)
    print(f"Model Type: {args.model_type}")
    print(f"Best Parameters: {best_params}")
    print(f"Number of Features: {args.num_features}")
    print(f"Imbalance Handling: {args.handle_imbalance}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Precision: {metrics['precision']:.4f}")
    print(f"Test Recall: {metrics['recall']:.4f}")
    print(f"Test F1 Score: {metrics['f1']:.4f}")
    print(f"Test ROC AUC: {metrics['roc_auc']:.4f}")
    print("="*80)
    
    # Return the best model and metrics
    return best_model, metrics

if __name__ == "__main__":
    main()