import numpy as np
import pickle
import os
from typing import Dict, Any, Tuple
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

class SpamClassifier:
    """
    SVM-based spam email classifier.
    """
    
    def __init__(self, kernel: str = 'linear', C: float = 1.0, gamma: str = 'scale'):
        """
        Initialize the classifier.
        
        Args:
            kernel: Kernel type to be used in the SVM algorithm
            C: Regularization parameter
            gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid' kernels
        """
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=True,
            random_state=42
        )
        self.is_trained = False
        self.params = {
            'kernel': kernel,
            'C': C,
            'gamma': gamma
        }
    
    def train(self, X_train, y_train) -> None:
        """
        Train the classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print(f"Training SVM classifier with parameters: {self.params}")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("Training completed")
    
    def predict(self, X) -> np.ndarray:
        """
        Make predictions using the trained classifier.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X) -> np.ndarray:
        """
        Get probability estimates for each class.
        
        Args:
            X: Features to predict on
            
        Returns:
            Probability estimates
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test, class_names: list = None) -> Dict[str, Any]:
        """
        Evaluate the classifier on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            class_names: Names of the classes
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        # Generate classification report
        if class_names is None:
            class_names = ['ham', 'spam']
        
        report = classification_report(y_test, y_pred, target_names=class_names)
        
        # Return metrics
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'y_pred': y_pred
        }
    
    def save_model(self, model_path: str) -> None:
        """
        Save the trained model to a file.
        
        Args:
            model_path: Path to save the model
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'params': self.params,
                'is_trained': self.is_trained
            }, f)
        
        print(f"Model saved to {model_path}")
    
    @classmethod
    def load_model(cls, model_path: str) -> 'SpamClassifier':
        """
        Load a trained model from a file.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded SpamClassifier instance
        """
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        # Create instance
        classifier = cls(
            kernel=data['params']['kernel'],
            C=data['params']['C'],
            gamma=data['params']['gamma']
        )
        
        # Set model and trained flag
        classifier.model = data['model']
        classifier.is_trained = data['is_trained']
        
        print(f"Model loaded from {model_path}")
        return classifier
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: list, 
                             output_path: str = None) -> None:
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            class_names: Names of the classes
            output_path: Path to save the plot
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        if output_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix plot saved to {output_path}")
        
        plt.close()

if __name__ == "__main__":
    # Test the model
    from src.data_loader import load_enron_spam_dataset
    
    # Load data
    data = load_enron_spam_dataset()
    
    # Create and train model
    classifier = SpamClassifier(kernel='linear', C=1.0)
    classifier.train(data['X_train'], data['y_train'])
    
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
        'results/figures/confusion_matrix.png'
    )