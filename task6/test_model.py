import os
import logging
import argparse
import numpy as np
from src.data_loader import DataLoader
from src.model import SentimentClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_custom_tweets(tweets, word2vec_path, model_path):
    """
    Test the trained model on custom tweets.
    
    Args:
        tweets (list): List of tweets to test.
        word2vec_path (str): Path to the trained Word2Vec model.
        model_path (str): Path to the trained SVM classifier.
    """
    # Initialize the data loader
    data_loader = DataLoader()
    
    # Load the Word2Vec model
    data_loader.load_word2vec_model(word2vec_path)
    
    # Initialize the classifier
    classifier = SentimentClassifier()
    
    # Load the trained model
    classifier.load_model(model_path)
    
    # Process each tweet
    for tweet in tweets:
        # Clean and tokenize the tweet
        tokens = data_loader.clean_text(tweet)
        
        # Vectorize the tweet
        vector = data_loader.get_sentence_vector(tokens)
        
        # Reshape the vector for prediction
        vector = vector.reshape(1, -1)
        
        # Make prediction
        prediction = classifier.predict(vector)[0]
        probabilities = classifier.predict_proba(vector)[0]
        
        # Print the results
        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = probabilities[prediction]
        
        print(f"Tweet: {tweet}")
        print(f"Sentiment: {sentiment}")
        print(f"Confidence: {confidence:.4f}")
        print("-" * 50)

def main(args):
    """
    Main function to test the trained model.
    
    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    # Check if the models exist
    if not os.path.exists(args.word2vec_path):
        logger.error(f"Word2Vec model not found at {args.word2vec_path}")
        return
    
    if not os.path.exists(args.model_path):
        logger.error(f"SVM classifier not found at {args.model_path}")
        return
    
    # Test custom tweets
    test_custom_tweets(args.tweets, args.word2vec_path, args.model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Sentiment Analysis Model")
    
    # Model paths
    parser.add_argument("--word2vec_path", type=str, default="models/saved_models/word2vec_model.bin", help="Path to the trained Word2Vec model.")
    parser.add_argument("--model_path", type=str, default="models/saved_models/svm_classifier.pkl", help="Path to the trained SVM classifier.")
    
    # Custom tweets
    parser.add_argument("--tweets", nargs="+", default=[
        "I love this product! It's amazing!",
        "This is the worst experience I've ever had.",
        "The weather is nice today.",
        "I'm feeling neutral about this situation."
    ], help="List of tweets to test.")
    
    args = parser.parse_args()
    
    main(args)