import os
import logging
import argparse
from src.data_loader import DataLoader
from src.model import SentimentClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main(args):
    """
    Main function to run the sentiment analysis.
    
    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    # Create directories
    os.makedirs("models/saved_models", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    
    # Initialize the data loader
    data_loader = DataLoader(
        sample_size=args.sample_size,
        test_size=args.test_size,
        random_state=args.random_state,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count
    )
    
    # Load and prepare data
    logger.info("Loading and preparing data...")
    X_train, X_test, y_train, y_test, df = data_loader.prepare_data()
    
    # Save the Word2Vec model
    data_loader.save_word2vec_model("models/saved_models/word2vec_model.bin")
    
    # Initialize the classifier
    classifier = SentimentClassifier(
        kernel=args.kernel,
        C=args.C,
        random_state=args.random_state
    )
    
    # Train the classifier
    classifier.train(X_train, y_train)
    
    # Evaluate the classifier
    metrics = classifier.evaluate(X_test, y_test)
    
    # Save the model
    classifier.save_model("models/saved_models/svm_classifier.pkl")
    
    # Save the metrics
    classifier.save_metrics(metrics, "results/metrics/accuracy_score.txt")
    
    # Plot and save the confusion matrix
    classifier.plot_confusion_matrix(metrics['confusion_matrix'], "results/figures/confusion_matrix.png")
    
    logger.info("Sentiment analysis completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment Analysis on Tweets")
    
    # Data loader arguments
    parser.add_argument("--sample_size", type=int, default=100000, help="Number of samples to use. If None, use all data.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of the dataset to include in the test split.")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility.")
    parser.add_argument("--vector_size", type=int, default=100, help="Dimensionality of the word vectors.")
    parser.add_argument("--window", type=int, default=5, help="Maximum distance between the current and predicted word.")
    parser.add_argument("--min_count", type=int, default=1, help="Ignores all words with total frequency lower than this.")
    
    # Classifier arguments
    parser.add_argument("--kernel", type=str, default="linear", choices=["linear", "poly", "rbf", "sigmoid"], help="Kernel type to be used in the SVM algorithm.")
    parser.add_argument("--C", type=float, default=1.0, help="Regularization parameter.")
    
    args = parser.parse_args()
    
    main(args)