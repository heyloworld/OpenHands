import argparse
from src.model import train_and_evaluate

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: The parsed arguments
    """
    parser = argparse.ArgumentParser(description='Train and evaluate a Naive Bayes classifier on the 20 Newsgroups dataset.')
    
    # Dataset parameters
    parser.add_argument('--categories', type=str, default=None, 
                        help='Comma-separated list of categories to use. If not specified, all categories are used.')
    
    # Model parameters
    parser.add_argument('--alpha', type=float, default=1.0, 
                        help='Smoothing parameter for Naive Bayes.')
    parser.add_argument('--use-tfidf', action='store_true', default=True,
                        help='Use TF-IDF features instead of raw counts.')
    
    # Training parameters
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of the dataset to include in the test split.')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state for reproducibility.')
    
    return parser.parse_args()

def main():
    """
    Main function to train and evaluate the classifier.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Parse categories
    categories = None
    if args.categories:
        categories = args.categories.split(',')
    
    # Train and evaluate the classifier
    classifier, vectorizer, raw_data = train_and_evaluate(
        categories=categories,
        alpha=args.alpha,
        use_tfidf=args.use_tfidf,
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    print("\nTraining and evaluation completed successfully!")
    print("Results saved to:")
    print("- results/figures/wordcloud_before.png")
    print("- results/figures/wordcloud_after.png")
    print("- results/figures/confusion_matrix.png")
    print("- results/metrics/performance.txt")
    
    # Print top features for each class
    print("\nTop 5 features for each class:")
    top_features = classifier.get_top_features(vectorizer, n_top=5)
    for i, features in top_features.items():
        class_name = raw_data.target_names[i]
        print(f"Class {i} ({class_name}): {', '.join(features)}")

if __name__ == "__main__":
    main()