"""
Model implementation for the Telco Customer Churn prediction.
This module implements a Logistic Regression model with hyperparameter tuning.
"""

import numpy as np
import pandas as pd
import pickle
import os
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, auc
)
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChurnPredictionModel:
    """
    Logistic Regression model for predicting customer churn.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the model.
        
        Args:
            random_state (int, optional): Random state for reproducibility.
        """
        self.random_state = random_state
        self.model = None
        
    def build_model(self, C=1.0, penalty='l2', solver='liblinear', class_weight=None, max_iter=1000):
        """
        Build a Logistic Regression model with the specified hyperparameters.
        
        Args:
            C (float, optional): Inverse of regularization strength.
            penalty (str, optional): Penalty type ('l1', 'l2', 'elasticnet', or 'none').
            solver (str, optional): Algorithm to use for optimization.
            class_weight (dict or 'balanced', optional): Weights associated with classes.
            max_iter (int, optional): Maximum number of iterations.
            
        Returns:
            self: The model instance.
        """
        logger.info(f"Building Logistic Regression model with C={C}, penalty={penalty}, solver={solver}")
        
        self.model = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            class_weight=class_weight,
            random_state=self.random_state,
            max_iter=max_iter,
            n_jobs=-1  # Use all available cores
        )
        
        return self
        
    def train(self, X_train, y_train):
        """
        Train the model on the given data.
        
        Args:
            X_train (array-like): Training features.
            y_train (array-like): Training labels.
            
        Returns:
            self: The trained model instance.
        """
        if self.model is None:
            logger.info("Model not built yet. Building default model.")
            self.build_model()
            
        logger.info(f"Training model on {X_train.shape[0]} samples with {X_train.shape[1]} features")
        
        self.model.fit(X_train, y_train)
        
        logger.info("Model training completed")
        
        return self
        
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X (array-like): Features to predict on.
            
        Returns:
            array: Predicted class labels.
        """
        if self.model is None:
            logger.error("Model not trained. Call train() first.")
            raise ValueError("Model not trained. Call train() first.")
            
        return self.model.predict(X)
        
    def predict_proba(self, X):
        """
        Predict class probabilities using the trained model.
        
        Args:
            X (array-like): Features to predict on.
            
        Returns:
            array: Predicted class probabilities.
        """
        if self.model is None:
            logger.error("Model not trained. Call train() first.")
            raise ValueError("Model not trained. Call train() first.")
            
        return self.model.predict_proba(X)
        
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the test data.
        
        Args:
            X_test (array-like): Test features.
            y_test (array-like): Test labels.
            
        Returns:
            dict: Dictionary of evaluation metrics.
        """
        if self.model is None:
            logger.error("Model not trained. Call train() first.")
            raise ValueError("Model not trained. Call train() first.")
            
        logger.info(f"Evaluating model on {X_test.shape[0]} samples")
        
        # Make predictions
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]  # Probability of the positive class
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Log metrics
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"ROC AUC: {roc_auc:.4f}")
        logger.info(f"Confusion Matrix:\n{conf_matrix}")
        
        # Return metrics as a dictionary
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': conf_matrix,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
    def save_model(self, filepath):
        """
        Save the trained model to a file.
        
        Args:
            filepath (str): Path to save the model to.
        """
        if self.model is None:
            logger.error("Model not trained. Call train() first.")
            raise ValueError("Model not trained. Call train() first.")
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save the model
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
                
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
            
    def load_model(self, filepath):
        """
        Load a trained model from a file.
        
        Args:
            filepath (str): Path to load the model from.
            
        Returns:
            self: The model instance with the loaded model.
        """
        try:
            # Load the model
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
                
            logger.info(f"Model loaded from {filepath}")
            
            return self
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
    def plot_roc_curve(self, X_test, y_test, filepath=None):
        """
        Plot the ROC curve for the model.
        
        Args:
            X_test (array-like): Test features.
            y_test (array-like): Test labels.
            filepath (str, optional): Path to save the plot to.
                If None, the plot will be displayed but not saved.
                
        Returns:
            tuple: False positive rate, true positive rate, and AUC.
        """
        if self.model is None:
            logger.error("Model not trained. Call train() first.")
            raise ValueError("Model not trained. Call train() first.")
            
        # Predict probabilities
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        # Save or display the plot
        if filepath:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {filepath}")
            plt.close()
        else:
            plt.show()
            
        return fpr, tpr, roc_auc
        
    def plot_precision_recall_curve(self, X_test, y_test, filepath=None):
        """
        Plot the Precision-Recall curve for the model.
        
        Args:
            X_test (array-like): Test features.
            y_test (array-like): Test labels.
            filepath (str, optional): Path to save the plot to.
                If None, the plot will be displayed but not saved.
                
        Returns:
            tuple: Precision, recall, and average precision.
        """
        if self.model is None:
            logger.error("Model not trained. Call train() first.")
            raise ValueError("Model not trained. Call train() first.")
            
        # Predict probabilities
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        # Calculate Precision-Recall curve
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = np.mean(precision)
        
        # Plot Precision-Recall curve
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
        # Save or display the plot
        if filepath:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-Recall curve saved to {filepath}")
            plt.close()
        else:
            plt.show()
            
        return precision, recall, avg_precision
        
    def save_classification_report(self, X_test, y_test, filepath):
        """
        Generate and save a classification report.
        
        Args:
            X_test (array-like): Test features.
            y_test (array-like): Test labels.
            filepath (str): Path to save the report to.
            
        Returns:
            str: The classification report as a string.
        """
        if self.model is None:
            logger.error("Model not trained. Call train() first.")
            raise ValueError("Model not trained. Call train() first.")
            
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Generate classification report
        report = classification_report(y_test, y_pred)
        
        # Add additional metrics
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Create a more detailed report
        detailed_report = f"Classification Report\n{'-' * 80}\n\n"
        detailed_report += report
        detailed_report += f"\n\nROC AUC: {roc_auc:.4f}\n\n"
        detailed_report += f"Confusion Matrix:\n{conf_matrix}\n\n"
        detailed_report += f"Model Parameters:\n{self.model.get_params()}\n"
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save the report
            with open(filepath, 'w') as f:
                f.write(detailed_report)
                
            logger.info(f"Classification report saved to {filepath}")
            
            return detailed_report
            
        except Exception as e:
            logger.error(f"Error saving classification report: {e}")
            raise


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    
    # Generate a synthetic dataset
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10, n_redundant=5,
        n_classes=2, weights=[0.7, 0.3], random_state=42
    )
    
    # Split the data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train the model
    model = ChurnPredictionModel(random_state=42)
    model.build_model(C=1.0, class_weight='balanced')
    model.train(X_train, y_train)
    
    # Evaluate the model
    metrics = model.evaluate(X_test, y_test)
    
    # Plot ROC curve
    model.plot_roc_curve(X_test, y_test)
    
    # Save classification report
    model.save_classification_report(X_test, y_test, 'classification_report.txt')