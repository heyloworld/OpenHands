import argparse
import os
from src.train import train_and_evaluate
from src.model import SpamClassifier
from src.data_loader import load_enron_spam_dataset

def main():
    """
    Main function to run the spam email detection pipeline.
    """
    parser = argparse.ArgumentParser(description='Spam Email Detection using SVM')
    
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate a trained model')
    parser.add_argument('--predict', type=str, help='Predict on a single email file')
    
    parser.add_argument('--data_dir', type=str, default='data/enron-spam',
                        help='Directory containing the dataset')
    parser.add_argument('--model_dir', type=str, default='models/saved_models',
                        help='Directory to save/load the model')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save results')
    
    parser.add_argument('--tune_params', action='store_true', 
                        help='Perform hyperparameter tuning')
    parser.add_argument('--cv', type=int, default=5,
                        help='Number of cross-validation folds')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    if args.train:
        print("Starting training and evaluation...")
        train_and_evaluate(
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            results_dir=args.results_dir,
            tune_params=args.tune_params,
            cv=args.cv
        )
    
    elif args.evaluate:
        print("Evaluating trained model...")
        # Load model
        model_path = os.path.join(args.model_dir, 'spam_classifier.pkl')
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            print("Please train the model first using --train")
            return
        
        classifier = SpamClassifier.load_model(model_path)
        
        # Load data
        data = load_enron_spam_dataset(data_dir=args.data_dir)
        
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
    
    elif args.predict:
        # Check if file exists
        if not os.path.exists(args.predict):
            print(f"Error: File not found at {args.predict}")
            return
        
        # Load model
        model_path = os.path.join(args.model_dir, 'spam_classifier.pkl')
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            print("Please train the model first using --train")
            return
        
        classifier = SpamClassifier.load_model(model_path)
        
        # Load data to get the vectorizer
        data = load_enron_spam_dataset(data_dir=args.data_dir)
        vectorizer = data['vectorizer']
        
        # Read and preprocess the email
        from src.data_loader import process_email_file
        email_text = process_email_file(args.predict)
        
        # Transform to TF-IDF features
        email_features = vectorizer.transform([email_text])
        
        # Make prediction
        prediction = classifier.predict(email_features)[0]
        probabilities = classifier.predict_proba(email_features)[0]
        
        # Print result
        print(f"\nPrediction for {args.predict}:")
        print(f"Class: {data['class_names'][prediction]} (Confidence: {probabilities[prediction]:.2f})")
        print(f"Probabilities: Ham: {probabilities[0]:.4f}, Spam: {probabilities[1]:.4f}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()