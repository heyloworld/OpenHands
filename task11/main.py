import os
import argparse
from src.train import main as train_main

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='House Price Prediction')
    parser.add_argument('--train', action='store_true', help='Train the model')
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Create necessary directories
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)
    
    if args.train:
        print("Starting model training and evaluation...")
        train_main()
    else:
        print("Please specify an action. Use --train to train the model.")

if __name__ == "__main__":
    main()