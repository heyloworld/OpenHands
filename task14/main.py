"""
Main script to run the Telco Customer Churn prediction pipeline.
"""

import os
import argparse
import logging
from src.train import main as train_main

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Run the Telco Customer Churn prediction pipeline')
    
    # Data arguments
    parser.add_argument('--data-url', type=str, 
                        default=None,
                        help='URL to download the dataset from')
    parser.add_argument('--data-path', type=str, default="data/churn.csv",
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
    parser.add_argument('--class-weight', type=str, default='balanced',
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

def main():
    """
    Main function to run the pipeline.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Create necessary directories
    os.makedirs(os.path.dirname(args.data_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.roc_curve_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.pr_curve_path), exist_ok=True)
    
    # Run the training pipeline
    logger.info("Starting the Telco Customer Churn prediction pipeline")
    
    # Convert args to a dictionary and pass to train_main
    train_args = argparse.Namespace(**vars(args))
    best_model, metrics = train_main()
    
    logger.info("Pipeline completed successfully")
    
    return best_model, metrics

if __name__ == "__main__":
    main()