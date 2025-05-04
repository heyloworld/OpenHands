import os
import logging
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SentimentClassifier:
    def __init__(self, kernel='linear', C=1.0, random_state=42):
        """
        Initialize the SentimentClassifier.
        
        Args:
            kernel (str, optional): Kernel type to be used in the SVM algorithm.
            C (float, optional): Regularization parameter.
            random_state (int, optional): Random state for reproducibility.
        """
        self.kernel = kernel
        self.C = C
        self.random_state = random_state
        self.model = None
        
    def train(self, X_train, y_train):
        """
        Train the SVM classifier.
        
        Args:
            X_train (numpy.ndarray): Training features.
            y_train (numpy.ndarray): Training labels.
            
        Returns:
            sklearn.svm.SVC: The trained SVM classifier.
        """
        logger.info(f"Training SVM classifier with kernel={self.kernel}, C={self.C}...")
        
        # Initialize the SVM classifier
        self.model = SVC(
            kernel=self.kernel,
            C=self.C,
            random_state=self.random_state,
            probability=True
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Get training accuracy
        y_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_pred)
        
        logger.info(f"SVM classifier trained. Training accuracy: {train_accuracy:.4f}")
        return self.model
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the SVM classifier.
        
        Args:
            X_test (numpy.ndarray): Test features.
            y_test (numpy.ndarray): Test labels.
            
        Returns:
            dict: A dictionary containing evaluation metrics.
        """
        if self.model is None:
            logger.error("Model not trained. Call train() first.")
            raise ValueError("Model not trained. Call train() first.")
        
        logger.info("Evaluating SVM classifier...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Get classification report
        report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])
        
        # Create a dictionary of metrics
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report
        }
        
        logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        return metrics
    
    def predict(self, X):
        """
        Make predictions using the trained SVM classifier.
        
        Args:
            X (numpy.ndarray): Features to predict.
            
        Returns:
            numpy.ndarray: Predicted labels.
        """
        if self.model is None:
            logger.error("Model not trained. Call train() first.")
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get probability estimates for each class.
        
        Args:
            X (numpy.ndarray): Features to predict.
            
        Returns:
            numpy.ndarray: Probability estimates.
        """
        if self.model is None:
            logger.error("Model not trained. Call train() first.")
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict_proba(X)
    
    def save_model(self, path):
        """
        Save the trained SVM classifier.
        
        Args:
            path (str): The path to save the model.
        """
        if self.model is not None:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save the model
            with open(path, 'wb') as f:
                pickle.dump(self.model, f)
            
            logger.info(f"SVM classifier saved to {path}")
        else:
            logger.warning("No model to save.")
    
    def load_model(self, path):
        """
        Load a trained SVM classifier.
        
        Args:
            path (str): The path to load the model from.
            
        Returns:
            sklearn.svm.SVC: The loaded SVM classifier.
        """
        try:
            with open(path, 'rb') as f:
                self.model = pickle.load(f)
            
            logger.info(f"SVM classifier loaded from {path}")
            return self.model
        except Exception as e:
            logger.error(f"Error loading SVM classifier: {e}")
            raise
    
    def plot_confusion_matrix(self, cm, save_path=None):
        """
        Plot the confusion matrix.
        
        Args:
            cm (numpy.ndarray): Confusion matrix.
            save_path (str, optional): Path to save the plot.
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.xticks([0.5, 1.5], ['Negative', 'Positive'])
        plt.yticks([0.5, 1.5], ['Negative', 'Positive'])
        
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save the plot
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix plot saved to {save_path}")
        
        plt.close()
    
    def save_metrics(self, metrics, path):
        """
        Save the evaluation metrics to a text file.
        
        Args:
            metrics (dict): A dictionary containing evaluation metrics.
            path (str): The path to save the metrics.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the metrics
        with open(path, 'w') as f:
            f.write("Sentiment Analysis with SVM Classifier\n")
            f.write("=====================================\n\n")
            f.write(f"Model: SVM with {self.kernel} kernel, C={self.C}\n\n")
            f.write("Performance Metrics:\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
            f.write(f"F1 Score: {metrics['f1_score']:.4f}\n\n")
            f.write("Confusion Matrix:\n")
            f.write(f"{metrics['confusion_matrix']}\n\n")
            f.write("Classification Report:\n")
            f.write(f"{metrics['classification_report']}\n")
        
        logger.info(f"Metrics saved to {path}")

if __name__ == "__main__":
    # Test the SentimentClassifier
    from data_loader import DataLoader
    
    # Load and prepare data
    data_loader = DataLoader(sample_size=10000)  # Use a small sample for testing
    X_train, X_test, y_train, y_test, df = data_loader.prepare_data()
    
    # Train and evaluate the classifier
    classifier = SentimentClassifier()
    classifier.train(X_train, y_train)
    metrics = classifier.evaluate(X_test, y_test)
    
    # Save the model
    classifier.save_model("models/svm_classifier.pkl")
    
    # Save the metrics
    classifier.save_metrics(metrics, "results/metrics/accuracy_score.txt")
    
    # Plot and save the confusion matrix
    classifier.plot_confusion_matrix(metrics['confusion_matrix'], "results/figures/confusion_matrix.png")