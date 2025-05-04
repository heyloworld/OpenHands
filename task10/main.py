import os
import logging
import argparse
import torch
import numpy as np
from src.data_loader import create_dataloaders
from src.model import FaceNetModel, FaceRecognitionSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ensure_dir(path):
    """Ensure directory exists."""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def save_metrics(metrics, path):
    """Save evaluation metrics to a file."""
    ensure_dir(path)
    
    with open(path, 'w') as f:
        f.write("Face Recognition Evaluation Metrics\n")
        f.write("=================================\n\n")
        
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    
    logger.info(f"Metrics saved to {path}")

def main(args):
    """Main function."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    train_loader, test_loader, dataset = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        test_split=args.test_split,
        min_images_per_person=args.min_images_per_person,
        image_size=(args.image_size, args.image_size),
        num_workers=args.num_workers,
        random_seed=args.random_seed
    )
    
    if train_loader is None or test_loader is None:
        logger.error("Failed to create data loaders")
        return
    
    # Create model
    model = FaceNetModel(
        embedding_dim=args.embedding_dim,
        pretrained_path=args.model_path if not args.train else None
    )
    
    # Create face recognition system
    face_recognition = FaceRecognitionSystem(
        model=model,
        device=device,
        distance_threshold=args.distance_threshold
    )
    
    # Train or load model
    if args.train:
        logger.info("Training FaceNet model...")
        history = face_recognition.train(
            train_loader=train_loader,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            margin=args.margin,
            save_path=args.model_path
        )
    else:
        logger.info(f"Using pretrained model from {args.model_path}")
    
    # Evaluate model
    logger.info("Evaluating face recognition system...")
    metrics = face_recognition.evaluate(test_loader=test_loader, dataset=dataset)
    
    # Save metrics
    save_metrics(metrics, args.metrics_path)
    
    # Visualize embeddings
    logger.info("Visualizing embeddings...")
    face_recognition.visualize_embeddings(
        data_loader=test_loader,
        dataset=dataset,
        output_path=args.visualization_path,
        max_samples=args.max_visualization_samples
    )
    
    logger.info("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Recognition with FaceNet")
    
    # Data parameters
    parser.add_argument("--data-dir", type=str, default="data/synthetic_faces",
                        help="Directory containing face images")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--test-split", type=float, default=0.2,
                        help="Proportion of data to use for testing")
    parser.add_argument("--min-images-per-person", type=int, default=5,
                        help="Minimum number of images required per person")
    parser.add_argument("--image-size", type=int, default=96,
                        help="Size to resize images to")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of worker threads for data loading")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Model parameters
    parser.add_argument("--embedding-dim", type=int, default=128,
                        help="Dimension of the embedding vectors")
    parser.add_argument("--model-path", type=str, default="models/saved_models/facenet.pt",
                        help="Path to save/load model")
    parser.add_argument("--distance-threshold", type=float, default=0.6,
                        help="Threshold for face recognition")
    
    # Training parameters
    parser.add_argument("--train", action="store_true",
                        help="Train the model")
    parser.add_argument("--num-epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Learning rate for optimization")
    parser.add_argument("--margin", type=float, default=0.2,
                        help="Margin for triplet loss")
    parser.add_argument("--no-cuda", action="store_true",
                        help="Disable CUDA")
    
    # Output parameters
    parser.add_argument("--metrics-path", type=str, default="results/metrics/recognition_accuracy.txt",
                        help="Path to save evaluation metrics")
    parser.add_argument("--visualization-path", type=str, default="results/figures/embedding_visualization.png",
                        help="Path to save embeddings visualization")
    parser.add_argument("--max-visualization-samples", type=int, default=500,
                        help="Maximum number of samples to visualize")
    
    args = parser.parse_args()
    
    main(args)