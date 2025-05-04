import os
import logging
import argparse
import numpy as np
import torch
from src.data_loader import MovieLensDataLoader
from src.model import RecommenderSystem

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

def save_recommendations(recommendations, user_info, path):
    """Save recommendations to a file."""
    ensure_dir(path)
    
    with open(path, 'w') as f:
        f.write(f"Recommendations for User {user_info['user_idx']}\n")
        f.write(f"User Info: {user_info}\n\n")
        
        for i, rec in enumerate(recommendations):
            f.write(f"{i+1}. {rec['title']}\n")
            f.write(f"   Predicted Rating: {rec['predicted_rating']:.2f}\n")
            f.write(f"   Average Rating: {rec['avg_rating']:.2f} ({rec['num_ratings']} ratings)\n\n")
    
    logger.info(f"Recommendations saved to {path}")

def save_metrics(metrics, path):
    """Save evaluation metrics to a file."""
    ensure_dir(path)
    
    with open(path, 'w') as f:
        f.write("Evaluation Metrics\n")
        f.write("=================\n\n")
        
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    
    logger.info(f"Metrics saved to {path}")

def main(args):
    """Main function."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create data loader
    data_loader = MovieLensDataLoader(
        data_path=args.data_path,
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    # Get data loaders
    train_loader, test_loader = data_loader.get_data_loaders(
        batch_size=args.batch_size,
        shuffle=True
    )
    
    # Create recommender systems
    ncf_recommender = RecommenderSystem(
        data_loader=data_loader,
        model_type="ncf",
        embedding_dim=args.embedding_dim,
        layers=args.layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device
    )
    
    mf_recommender = RecommenderSystem(
        data_loader=data_loader,
        model_type="mf",
        embedding_dim=args.embedding_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device
    )
    
    # Train models
    logger.info("Training NCF model...")
    ncf_history = ncf_recommender.train(
        train_loader=train_loader,
        num_epochs=args.num_epochs,
        verbose=True
    )
    
    logger.info("Training MF model...")
    mf_history = mf_recommender.train(
        train_loader=train_loader,
        num_epochs=args.num_epochs,
        verbose=True
    )
    
    # Evaluate models
    logger.info("Evaluating NCF model...")
    ncf_metrics = ncf_recommender.evaluate(test_loader)
    
    logger.info("Evaluating MF model...")
    mf_metrics = mf_recommender.evaluate(test_loader)
    
    # Save metrics
    metrics = {
        "NCF": ncf_metrics,
        "MF": mf_metrics
    }
    save_metrics(metrics, args.metrics_path)
    
    # Find a good test user with sufficient ratings
    active_users = data_loader.get_active_users(n=20)
    test_user_idx = active_users[0]['user_idx']
    test_user_info = data_loader.get_user_info(test_user_idx)
    test_user_stats = data_loader.get_user_stats(test_user_idx)
    test_user_info.update(test_user_stats)
    
    logger.info(f"Generating recommendations for test user {test_user_idx}")
    
    # Generate recommendations
    ncf_recommendations = ncf_recommender.recommend_for_user(
        user_idx=test_user_idx,
        n=args.num_recommendations,
        exclude_rated=True
    )
    
    mf_recommendations = mf_recommender.recommend_for_user(
        user_idx=test_user_idx,
        n=args.num_recommendations,
        exclude_rated=True
    )
    
    # Save recommendations
    recommendations = {
        "NCF": ncf_recommendations,
        "MF": mf_recommendations
    }
    
    # Format and save recommendations
    with open(args.recommendations_path, 'w') as f:
        f.write(f"Recommendations for User {test_user_idx}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("User Information:\n")
        for key, value in test_user_info.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # User's top rated movies
        user_ratings = data_loader.get_user_ratings(test_user_idx)
        sorted_ratings = sorted(user_ratings.items(), key=lambda x: x[1], reverse=True)
        
        f.write("User's Top Rated Movies:\n")
        for i, (movie_idx, rating) in enumerate(sorted_ratings[:5]):
            movie_title = data_loader.get_movie_title(movie_idx)
            f.write(f"  {i+1}. {movie_title} - Rating: {rating}\n")
        f.write("\n")
        
        # NCF Recommendations
        f.write("NCF Recommendations:\n")
        for i, rec in enumerate(ncf_recommendations):
            f.write(f"  {i+1}. {rec['title']}\n")
            f.write(f"     Predicted Rating: {rec['predicted_rating']:.2f}\n")
            f.write(f"     Average Rating: {rec['avg_rating']:.2f} ({rec['num_ratings']} ratings)\n")
        f.write("\n")
        
        # MF Recommendations
        f.write("Matrix Factorization Recommendations:\n")
        for i, rec in enumerate(mf_recommendations):
            f.write(f"  {i+1}. {rec['title']}\n")
            f.write(f"     Predicted Rating: {rec['predicted_rating']:.2f}\n")
            f.write(f"     Average Rating: {rec['avg_rating']:.2f} ({rec['num_ratings']} ratings)\n")
    
    logger.info(f"Recommendations saved to {args.recommendations_path}")
    
    # Save models
    if args.save_models:
        ncf_recommender.save_model(args.ncf_model_path)
        mf_recommender.save_model(args.mf_model_path)
    
    logger.info("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Movie Recommendation System")
    
    # Data parameters
    parser.add_argument("--data-path", type=str, default="data/ml-100k",
                        help="Path to the MovieLens dataset")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Proportion of data to use for testing")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Model parameters
    parser.add_argument("--embedding-dim", type=int, default=32,
                        help="Dimension of the embedding vectors")
    parser.add_argument("--layers", type=int, nargs="+", default=[64, 32, 16, 8],
                        help="List of layer dimensions for the MLP")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout probability")
    
    # Training parameters
    parser.add_argument("--num-epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Learning rate for optimization")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="Weight decay for regularization")
    parser.add_argument("--no-cuda", action="store_true",
                        help="Disable CUDA")
    
    # Recommendation parameters
    parser.add_argument("--num-recommendations", type=int, default=10,
                        help="Number of recommendations to generate")
    
    # Output parameters
    parser.add_argument("--metrics-path", type=str, default="results/metrics/evaluation_metrics.txt",
                        help="Path to save evaluation metrics")
    parser.add_argument("--recommendations-path", type=str, default="results/metrics/top_10_recommendations.txt",
                        help="Path to save recommendations")
    parser.add_argument("--save-models", action="store_true",
                        help="Save trained models")
    parser.add_argument("--ncf-model-path", type=str, default="models/ncf_model.pt",
                        help="Path to save NCF model")
    parser.add_argument("--mf-model-path", type=str, default="models/mf_model.pt",
                        help="Path to save MF model")
    
    args = parser.parse_args()
    
    main(args)