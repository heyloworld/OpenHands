import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MatrixFactorization(nn.Module):
    """
    Matrix Factorization model for collaborative filtering.
    
    This model learns latent factors for users and items to predict ratings.
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 50
    ):
        """
        Initialize the Matrix Factorization model.
        
        Args:
            num_users (int): Number of users in the dataset.
            num_items (int): Number of items in the dataset.
            embedding_dim (int): Dimension of the embedding vectors.
        """
        super(MatrixFactorization, self).__init__()
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, mean=0, std=0.1)
        nn.init.normal_(self.item_embedding.weight, mean=0, std=0.1)
    
    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            user_indices (torch.Tensor): Indices of users.
            item_indices (torch.Tensor): Indices of items.
            
        Returns:
            torch.Tensor: Predicted ratings.
        """
        # Get embeddings
        user_embeds = self.user_embedding(user_indices)
        item_embeds = self.item_embedding(item_indices)
        
        # Calculate dot product
        dot_product = torch.sum(user_embeds * item_embeds, dim=1)
        
        return dot_product
    
    def predict(self, user_idx: int, item_idx: int) -> float:
        """
        Predict the rating for a user-item pair.
        
        Args:
            user_idx (int): User index.
            item_idx (int): Item index.
            
        Returns:
            float: Predicted rating.
        """
        # Convert to tensors
        user_tensor = torch.LongTensor([user_idx])
        item_tensor = torch.LongTensor([item_idx])
        
        # Set model to evaluation mode
        self.eval()
        
        # Make prediction
        with torch.no_grad():
            prediction = self.forward(user_tensor, item_tensor).item()
        
        return prediction
    
    def recommend_items(
        self,
        user_idx: int,
        n: int = 10,
        exclude_rated: bool = True,
        rated_items: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Recommend items for a user.
        
        Args:
            user_idx (int): User index.
            n (int): Number of recommendations.
            exclude_rated (bool): Whether to exclude already rated items.
            rated_items (List[int], optional): List of items already rated by the user.
            
        Returns:
            List[Tuple[int, float]]: List of (item_idx, predicted_rating) tuples.
        """
        # Set model to evaluation mode
        self.eval()
        
        # Get user embedding
        user_tensor = torch.LongTensor([user_idx])
        user_embed = self.user_embedding(user_tensor)
        
        # Get all item embeddings
        all_items = torch.arange(self.item_embedding.num_embeddings)
        item_embeds = self.item_embedding(all_items)
        
        # Calculate dot products
        with torch.no_grad():
            dot_products = torch.matmul(user_embed, item_embeds.t()).squeeze()
        
        # Convert to numpy for easier manipulation
        predictions = dot_products.numpy()
        
        # Exclude rated items if requested
        if exclude_rated and rated_items is not None:
            predictions[rated_items] = -np.inf
        
        # Get top N items
        top_indices = np.argsort(predictions)[::-1][:n]
        
        # Create recommendations list
        recommendations = [(int(idx), float(predictions[idx])) for idx in top_indices]
        
        return recommendations


class NeuralCollaborativeFiltering(nn.Module):
    """
    Neural Collaborative Filtering (NCF) model.
    
    This model combines matrix factorization with neural networks for collaborative filtering.
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 32,
        layers: List[int] = [64, 32, 16, 8],
        dropout: float = 0.2
    ):
        """
        Initialize the NCF model.
        
        Args:
            num_users (int): Number of users in the dataset.
            num_items (int): Number of items in the dataset.
            embedding_dim (int): Dimension of the embedding vectors.
            layers (List[int]): List of layer dimensions for the MLP.
            dropout (float): Dropout probability.
        """
        super(NeuralCollaborativeFiltering, self).__init__()
        
        # GMF embeddings
        self.user_embedding_gmf = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(num_items, embedding_dim)
        
        # MLP embeddings
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers
        self.mlp_layers = nn.ModuleList()
        input_dim = 2 * embedding_dim
        
        for i, layer_dim in enumerate(layers):
            if i == 0:
                self.mlp_layers.append(nn.Linear(input_dim, layer_dim))
            else:
                self.mlp_layers.append(nn.Linear(layers[i-1], layer_dim))
            
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(dropout))
        
        # Output layer
        self.output_layer = nn.Linear(layers[-1] + embedding_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)
    
    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            user_indices (torch.Tensor): Indices of users.
            item_indices (torch.Tensor): Indices of items.
            
        Returns:
            torch.Tensor: Predicted ratings.
        """
        # GMF part
        user_embeds_gmf = self.user_embedding_gmf(user_indices)
        item_embeds_gmf = self.item_embedding_gmf(item_indices)
        gmf_output = user_embeds_gmf * item_embeds_gmf
        
        # MLP part
        user_embeds_mlp = self.user_embedding_mlp(user_indices)
        item_embeds_mlp = self.item_embedding_mlp(item_indices)
        mlp_input = torch.cat([user_embeds_mlp, item_embeds_mlp], dim=1)
        
        for layer in self.mlp_layers:
            mlp_input = layer(mlp_input)
        
        # Concatenate GMF and MLP outputs
        concat_output = torch.cat([gmf_output, mlp_input], dim=1)
        
        # Final prediction
        prediction = self.output_layer(concat_output)
        
        return prediction.squeeze()
    
    def predict(self, user_idx: int, item_idx: int) -> float:
        """
        Predict the rating for a user-item pair.
        
        Args:
            user_idx (int): User index.
            item_idx (int): Item index.
            
        Returns:
            float: Predicted rating.
        """
        # Convert to tensors
        user_tensor = torch.LongTensor([user_idx])
        item_tensor = torch.LongTensor([item_idx])
        
        # Set model to evaluation mode
        self.eval()
        
        # Make prediction
        with torch.no_grad():
            prediction = self.forward(user_tensor, item_tensor).item()
        
        return prediction
    
    def recommend_items(
        self,
        user_idx: int,
        n: int = 10,
        exclude_rated: bool = True,
        rated_items: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Recommend items for a user.
        
        Args:
            user_idx (int): User index.
            n (int): Number of recommendations.
            exclude_rated (bool): Whether to exclude already rated items.
            rated_items (List[int], optional): List of items already rated by the user.
            
        Returns:
            List[Tuple[int, float]]: List of (item_idx, predicted_rating) tuples.
        """
        # Set model to evaluation mode
        self.eval()
        
        # Create tensors for all items
        num_items = self.item_embedding_gmf.num_embeddings
        user_tensor = torch.LongTensor([user_idx] * num_items)
        item_tensor = torch.LongTensor(range(num_items))
        
        # Make predictions
        with torch.no_grad():
            predictions = self.forward(user_tensor, item_tensor).numpy()
        
        # Exclude rated items if requested
        if exclude_rated and rated_items is not None:
            predictions[rated_items] = -np.inf
        
        # Get top N items
        top_indices = np.argsort(predictions)[::-1][:n]
        
        # Create recommendations list
        recommendations = [(int(idx), float(predictions[idx])) for idx in top_indices]
        
        return recommendations


class RecommenderSystem:
    """
    Recommender system that trains and evaluates recommendation models.
    """
    
    def __init__(
        self,
        data_loader,
        model_type: str = "ncf",
        embedding_dim: int = 32,
        layers: List[int] = [64, 32, 16, 8],
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        device: str = "cpu"
    ):
        """
        Initialize the recommender system.
        
        Args:
            data_loader: Data loader object.
            model_type (str): Type of model to use ("mf" or "ncf").
            embedding_dim (int): Dimension of the embedding vectors.
            layers (List[int]): List of layer dimensions for the MLP (NCF only).
            dropout (float): Dropout probability (NCF only).
            learning_rate (float): Learning rate for optimization.
            weight_decay (float): Weight decay for regularization.
            device (str): Device to use for training ("cpu" or "cuda").
        """
        self.data_loader = data_loader
        self.model_type = model_type
        self.embedding_dim = embedding_dim
        self.layers = layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
        
        # Get dataset information
        self.num_users = data_loader.n_users
        self.num_items = data_loader.n_movies
        
        # Create model
        self._create_model()
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()
    
    def _create_model(self):
        """Create the recommendation model."""
        if self.model_type == "mf":
            self.model = MatrixFactorization(
                num_users=self.num_users,
                num_items=self.num_items,
                embedding_dim=self.embedding_dim
            )
        elif self.model_type == "ncf":
            self.model = NeuralCollaborativeFiltering(
                num_users=self.num_users,
                num_items=self.num_items,
                embedding_dim=self.embedding_dim,
                layers=self.layers,
                dropout=self.dropout
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        logger.info(f"Created {self.model_type.upper()} model with "
                   f"{sum(p.numel() for p in self.model.parameters())} parameters")
    
    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int = 10,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the recommendation model.
        
        Args:
            train_loader (DataLoader): DataLoader for training data.
            num_epochs (int): Number of training epochs.
            verbose (bool): Whether to print progress.
            
        Returns:
            Dict[str, List[float]]: Training history.
        """
        # Set model to training mode
        self.model.train()
        
        # Training history
        history = {"loss": []}
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for user_indices, item_indices, ratings in train_loader:
                # Move data to device
                user_indices = user_indices.to(self.device)
                item_indices = item_indices.to(self.device)
                ratings = ratings.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(user_indices, item_indices)
                loss = self.criterion(predictions, ratings)
                
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                
                # Update statistics
                epoch_loss += loss.item()
                num_batches += 1
            
            # Calculate average loss
            avg_loss = epoch_loss / num_batches
            history["loss"].append(avg_loss)
            
            if verbose:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        logger.info(f"Training completed with final loss: {history['loss'][-1]:.4f}")
        
        return history
    
    def evaluate(
        self,
        test_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate the recommendation model.
        
        Args:
            test_loader (DataLoader): DataLoader for testing data.
            
        Returns:
            Dict[str, float]: Evaluation metrics.
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Evaluation metrics
        total_mse = 0.0
        total_mae = 0.0
        total_samples = 0
        
        # Evaluation loop
        with torch.no_grad():
            for user_indices, item_indices, ratings in test_loader:
                # Move data to device
                user_indices = user_indices.to(self.device)
                item_indices = item_indices.to(self.device)
                ratings = ratings.to(self.device)
                
                # Make predictions
                predictions = self.model(user_indices, item_indices)
                
                # Calculate metrics
                mse = ((predictions - ratings) ** 2).sum().item()
                mae = torch.abs(predictions - ratings).sum().item()
                
                # Update statistics
                total_mse += mse
                total_mae += mae
                total_samples += len(ratings)
        
        # Calculate final metrics
        rmse = np.sqrt(total_mse / total_samples)
        mae = total_mae / total_samples
        
        metrics = {
            "rmse": rmse,
            "mae": mae,
            "mse": total_mse / total_samples,
            "num_samples": total_samples
        }
        
        logger.info(f"Evaluation results: RMSE = {rmse:.4f}, MAE = {mae:.4f}")
        
        return metrics
    
    def recommend_for_user(
        self,
        user_idx: int,
        n: int = 10,
        exclude_rated: bool = True
    ) -> List[Dict]:
        """
        Generate recommendations for a user.
        
        Args:
            user_idx (int): User index.
            n (int): Number of recommendations.
            exclude_rated (bool): Whether to exclude already rated items.
            
        Returns:
            List[Dict]: List of recommendation information.
        """
        # Get items rated by the user
        user_ratings = self.data_loader.get_user_ratings(user_idx)
        rated_items = list(user_ratings.keys()) if exclude_rated else None
        
        # Get recommendations
        recommendations = self.model.recommend_items(
            user_idx=user_idx,
            n=n,
            exclude_rated=exclude_rated,
            rated_items=rated_items
        )
        
        # Create detailed recommendations
        detailed_recommendations = []
        for item_idx, predicted_rating in recommendations:
            movie_title = self.data_loader.get_movie_title(item_idx)
            movie_stats = self.data_loader.get_movie_stats(item_idx)
            
            recommendation = {
                "movie_idx": item_idx,
                "title": movie_title,
                "predicted_rating": predicted_rating,
                "avg_rating": movie_stats["avg_rating"],
                "num_ratings": movie_stats["num_ratings"]
            }
            
            detailed_recommendations.append(recommendation)
        
        return detailed_recommendations
    
    def save_model(self, path: str):
        """
        Save the model to a file.
        
        Args:
            path (str): Path to save the model.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        torch.save({
            "model_type": self.model_type,
            "embedding_dim": self.embedding_dim,
            "layers": self.layers,
            "dropout": self.dropout,
            "num_users": self.num_users,
            "num_items": self.num_items,
            "state_dict": self.model.state_dict()
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load the model from a file.
        
        Args:
            path (str): Path to load the model from.
        """
        # Check if file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load model
        checkpoint = torch.load(path, map_location=self.device)
        
        # Update model parameters
        self.model_type = checkpoint["model_type"]
        self.embedding_dim = checkpoint["embedding_dim"]
        self.layers = checkpoint["layers"]
        self.dropout = checkpoint["dropout"]
        self.num_users = checkpoint["num_users"]
        self.num_items = checkpoint["num_items"]
        
        # Create model
        self._create_model()
        
        # Load state dict
        self.model.load_state_dict(checkpoint["state_dict"])
        
        logger.info(f"Model loaded from {path}")


# Example usage
if __name__ == "__main__":
    from data_loader import MovieLensDataLoader
    
    # Create data loader
    data_loader = MovieLensDataLoader()
    
    # Get data loaders
    train_loader, test_loader = data_loader.get_data_loaders(batch_size=64)
    
    # Create recommender system
    recommender = RecommenderSystem(
        data_loader=data_loader,
        model_type="ncf",
        embedding_dim=32,
        layers=[64, 32, 16, 8],
        dropout=0.2,
        learning_rate=0.001
    )
    
    # Train model
    history = recommender.train(train_loader, num_epochs=5)
    
    # Evaluate model
    metrics = recommender.evaluate(test_loader)
    
    # Generate recommendations for a user
    user_idx = 0
    recommendations = recommender.recommend_for_user(user_idx, n=5)
    
    print(f"\nRecommendations for User {user_idx}:")
    for i, rec in enumerate(recommendations):
        print(f"  {i+1}. {rec['title']} - Predicted rating: {rec['predicted_rating']:.2f}, "
              f"Avg rating: {rec['avg_rating']:.2f} ({rec['num_ratings']} ratings)")