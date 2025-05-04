import os
import argparse
from src.train import main as train_main

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Fashion-MNIST Classification with ResNet-18')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--no_pretrained', action='store_true', help='Do not use pretrained weights')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory to store the dataset')
    parser.add_argument('--save_dir', type=str, default='./models/saved_models', help='Directory to save the model')
    
    return parser.parse_args()

def main():
    """
    Main function to run the Fashion-MNIST classification system.
    """
    # Create necessary directories
    os.makedirs('./models/saved_models', exist_ok=True)
    os.makedirs('./plots', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    
    # Run the training script
    train_main()

if __name__ == '__main__':
    main()