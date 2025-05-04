import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FaceNetModel(nn.Module):
    """
    FaceNet model for face recognition.
    
    This model generates embeddings for face images that can be used for face recognition.
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        pretrained_path: Optional[str] = None
    ):
        """
        Initialize the FaceNet model.
        
        Args:
            embedding_dim (int): Dimension of the embedding vectors.
            pretrained_path (str, optional): Path to pretrained model weights.
        """
        super(FaceNetModel, self).__init__()
        
        # Define model architecture
        self.embedding_dim = embedding_dim
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512 * 6 * 6, 1024)
        self.bn5 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, embedding_dim)
        
        # Load pretrained weights if provided
        if pretrained_path:
            self.load_pretrained(pretrained_path)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width).
            
        Returns:
            torch.Tensor: Embedding vectors of shape (batch_size, embedding_dim).
        """
        # Convolutional layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 48x48
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 24x24
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 12x12
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # 6x6
        
        # Flatten
        x = x.view(-1, 512 * 6 * 6)
        
        # Fully connected layers
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.fc2(x)
        
        # L2 normalization
        x = F.normalize(x, p=2, dim=1)
        
        return x
    
    def load_pretrained(self, pretrained_path: str) -> bool:
        """
        Load pretrained weights.
        
        Args:
            pretrained_path (str): Path to pretrained weights.
            
        Returns:
            bool: Whether loading was successful.
        """
        try:
            # Check if file exists
            if not os.path.exists(pretrained_path):
                logger.error(f"Pretrained weights file not found: {pretrained_path}")
                return False
            
            # Load weights
            state_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
            
            # Handle different state dict formats
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # Try to load state dict
            try:
                self.load_state_dict(state_dict)
                logger.info(f"Loaded pretrained weights from {pretrained_path}")
                return True
            except Exception as e:
                logger.warning(f"Error loading pretrained weights: {e}")
                
                # Try to load with a different approach (partial loading)
                model_dict = self.state_dict()
                pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
                
                if pretrained_dict:
                    model_dict.update(pretrained_dict)
                    self.load_state_dict(model_dict)
                    logger.info(f"Loaded partial pretrained weights from {pretrained_path}")
                    return True
                else:
                    logger.error("No matching keys found in pretrained weights")
                    return False
        
        except Exception as e:
            logger.error(f"Error loading pretrained weights: {e}")
            return False
    
    def get_embedding(self, x: torch.Tensor) -> np.ndarray:
        """
        Get embedding for an input tensor.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            np.ndarray: Embedding vector.
        """
        # Set model to evaluation mode
        self.eval()
        
        # Get embedding
        with torch.no_grad():
            embedding = self.forward(x)
        
        # Convert to numpy
        return embedding.cpu().numpy()


class FaceRecognitionSystem:
    """
    Face recognition system using FaceNet embeddings.
    
    This system handles training, evaluation, and inference for face recognition.
    """
    
    def __init__(
        self,
        model: FaceNetModel,
        device: torch.device = torch.device('cpu'),
        distance_threshold: float = 0.6
    ):
        """
        Initialize the face recognition system.
        
        Args:
            model (FaceNetModel): FaceNet model.
            device (torch.device): Device to use for computation.
            distance_threshold (float): Threshold for face recognition.
        """
        self.model = model
        self.device = device
        self.distance_threshold = distance_threshold
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Initialize embeddings database
        self.embeddings_db = {}
        self.label_to_name = {}
    
    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int = 10,
        learning_rate: float = 0.001,
        margin: float = 0.2,
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Train the FaceNet model.
        
        Args:
            train_loader (DataLoader): Training data loader.
            num_epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for optimization.
            margin (float): Margin for triplet loss.
            save_path (str, optional): Path to save the trained model.
            
        Returns:
            Dict[str, List[float]]: Training history.
        """
        # Set model to training mode
        self.model.train()
        
        # Define optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        triplet_loss = nn.TripletMarginLoss(margin=margin)
        
        # Training history
        history = {"loss": []}
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for images, labels in train_loader:
                # Skip if batch is too small
                if len(images) < 3:
                    continue
                
                # Move data to device
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                embeddings = self.model(images)
                
                # Create triplets
                triplets = self._create_triplets(embeddings, labels)
                
                if triplets is None:
                    continue
                
                anchors, positives, negatives = triplets
                
                # Calculate loss
                loss = triplet_loss(anchors, positives, negatives)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update statistics
                epoch_loss += loss.item()
                num_batches += 1
            
            # Calculate average loss
            avg_loss = epoch_loss / max(1, num_batches)
            history["loss"].append(avg_loss)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # Save model if requested
        if save_path:
            self._save_model(save_path)
        
        logger.info(f"Training completed with final loss: {history['loss'][-1]:.4f}")
        
        return history
    
    def _create_triplets(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Create triplets for triplet loss.
        
        Args:
            embeddings (torch.Tensor): Embedding vectors.
            labels (torch.Tensor): Labels.
            
        Returns:
            Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: Anchors, positives, and negatives.
        """
        # Get unique labels
        unique_labels = torch.unique(labels)
        
        # Skip if not enough classes
        if len(unique_labels) < 2:
            return None
        
        # Initialize triplets
        anchors = []
        positives = []
        negatives = []
        
        # Create triplets
        for label in unique_labels:
            # Get indices for this label
            label_mask = (labels == label)
            label_indices = torch.where(label_mask)[0]
            
            # Skip if not enough samples
            if len(label_indices) < 2:
                continue
            
            # Get indices for other labels
            other_mask = (labels != label)
            other_indices = torch.where(other_mask)[0]
            
            # Skip if no other samples
            if len(other_indices) == 0:
                continue
            
            # Select anchor and positive
            for i in range(len(label_indices)):
                for j in range(i + 1, len(label_indices)):
                    anchor_idx = label_indices[i]
                    positive_idx = label_indices[j]
                    
                    # Select negative
                    negative_idx = other_indices[torch.randint(0, len(other_indices), (1,)).item()]
                    
                    # Add triplet
                    anchors.append(embeddings[anchor_idx])
                    positives.append(embeddings[positive_idx])
                    negatives.append(embeddings[negative_idx])
        
        # Skip if no triplets
        if not anchors:
            return None
        
        # Convert to tensors
        anchors = torch.stack(anchors)
        positives = torch.stack(positives)
        negatives = torch.stack(negatives)
        
        return anchors, positives, negatives
    
    def evaluate(
        self,
        test_loader: DataLoader,
        dataset
    ) -> Dict[str, float]:
        """
        Evaluate the face recognition system.
        
        Args:
            test_loader (DataLoader): Testing data loader.
            dataset: Dataset object with identity information.
            
        Returns:
            Dict[str, float]: Evaluation metrics.
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Build embeddings database
        self._build_embeddings_db(test_loader, dataset)
        
        # Initialize metrics
        all_predictions = []
        all_labels = []
        
        # Evaluation loop
        with torch.no_grad():
            for images, labels in test_loader:
                # Move data to device
                images = images.to(self.device)
                labels = labels.cpu().numpy()
                
                # Get embeddings
                embeddings = self.model(images)
                
                # Recognize faces
                predictions = []
                for i, embedding in enumerate(embeddings):
                    # Get embedding as numpy array
                    embedding_np = embedding.cpu().numpy().reshape(1, -1)
                    
                    # Find closest identity
                    recognized_label = self._recognize_face(embedding_np)
                    predictions.append(recognized_label)
                
                # Update metrics
                all_predictions.extend(predictions)
                all_labels.extend(labels)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
        logger.info(f"Evaluation results: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}")
        
        return metrics
    
    def _build_embeddings_db(self, data_loader: DataLoader, dataset) -> None:
        """
        Build a database of embeddings for known identities.
        
        Args:
            data_loader (DataLoader): Data loader.
            dataset: Dataset object with identity information.
        """
        # Clear existing database
        self.embeddings_db = {}
        self.label_to_name = {}
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Process each batch
        with torch.no_grad():
            for images, labels in data_loader:
                # Move data to device
                images = images.to(self.device)
                
                # Get embeddings
                embeddings = self.model(images)
                
                # Add to database
                for i, (embedding, label) in enumerate(zip(embeddings, labels)):
                    label_int = label.item()
                    
                    # Get identity name
                    identity_name = dataset.get_identity_name(label_int)
                    self.label_to_name[label_int] = identity_name
                    
                    # Add embedding to database
                    embedding_np = embedding.cpu().numpy()
                    
                    if label_int not in self.embeddings_db:
                        self.embeddings_db[label_int] = []
                    
                    self.embeddings_db[label_int].append(embedding_np)
        
        # Average embeddings for each identity
        for label in self.embeddings_db:
            self.embeddings_db[label] = np.mean(self.embeddings_db[label], axis=0)
        
        logger.info(f"Built embeddings database with {len(self.embeddings_db)} identities")
    
    def _recognize_face(self, embedding: np.ndarray) -> int:
        """
        Recognize a face from its embedding.
        
        Args:
            embedding (np.ndarray): Face embedding.
            
        Returns:
            int: Recognized label.
        """
        if not self.embeddings_db:
            return -1
        
        # Calculate similarities with all known identities
        similarities = {}
        for label, db_embedding in self.embeddings_db.items():
            similarity = cosine_similarity(embedding, db_embedding.reshape(1, -1))[0][0]
            similarities[label] = similarity
        
        # Find the most similar identity
        best_label = max(similarities, key=similarities.get)
        best_similarity = similarities[best_label]
        
        # Check if similarity is above threshold
        if best_similarity < self.distance_threshold:
            return -1  # Unknown face
        
        return best_label
    
    def _save_model(self, save_path: str) -> None:
        """
        Save the model to a file.
        
        Args:
            save_path (str): Path to save the model.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), save_path)
        
        logger.info(f"Model saved to {save_path}")
    
    def visualize_embeddings(
        self,
        data_loader: DataLoader,
        dataset,
        output_path: str,
        max_samples: int = 500,
        figsize: Tuple[int, int] = (12, 10)
    ) -> None:
        """
        Visualize embeddings using t-SNE.
        
        Args:
            data_loader (DataLoader): Data loader.
            dataset: Dataset object with identity information.
            output_path (str): Path to save the visualization.
            max_samples (int): Maximum number of samples to visualize.
            figsize (Tuple[int, int]): Figure size.
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Collect embeddings and labels
        all_embeddings = []
        all_labels = []
        all_names = []
        
        # Process each batch
        with torch.no_grad():
            for images, labels in data_loader:
                # Move data to device
                images = images.to(self.device)
                
                # Get embeddings
                embeddings = self.model(images)
                
                # Add to lists
                for i, (embedding, label) in enumerate(zip(embeddings, labels)):
                    all_embeddings.append(embedding.cpu().numpy())
                    all_labels.append(label.item())
                    all_names.append(dataset.get_identity_name(label.item()))
                
                # Limit number of samples
                if len(all_embeddings) >= max_samples:
                    break
        
        # Convert to numpy arrays
        embeddings_np = np.array(all_embeddings)
        labels_np = np.array(all_labels)
        
        # Apply t-SNE
        logger.info("Applying t-SNE dimensionality reduction...")
        # Adjust perplexity for small datasets
        perplexity = min(30, len(embeddings_np) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(embeddings_np)
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Get unique labels
        unique_labels = np.unique(labels_np)
        
        # Plot each identity
        for label in unique_labels:
            # Get indices for this label
            indices = np.where(labels_np == label)[0]
            
            # Get name for this label
            name = dataset.get_identity_name(label)
            
            # Plot points
            plt.scatter(
                embeddings_2d[indices, 0],
                embeddings_2d[indices, 1],
                label=name,
                alpha=0.7
            )
        
        # Add legend and labels
        plt.legend(loc='best')
        plt.title('t-SNE Visualization of Face Embeddings')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.tight_layout()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save figure
        plt.savefig(output_path)
        logger.info(f"Embeddings visualization saved to {output_path}")
        
        # Close figure
        plt.close()


# Example usage
if __name__ == "__main__":
    from data_loader import create_dataloaders
    
    # Create data loaders
    train_loader, test_loader, dataset = create_dataloaders(
        data_dir="data/synthetic_faces",
        batch_size=32,
        test_split=0.2,
        min_images_per_person=5
    )
    
    # Create model
    model = FaceNetModel(embedding_dim=128)
    
    # Create face recognition system
    face_recognition = FaceRecognitionSystem(model=model)
    
    # Train model
    history = face_recognition.train(
        train_loader=train_loader,
        num_epochs=5,
        learning_rate=0.001,
        save_path="models/saved_models/facenet_trained.pt"
    )
    
    # Evaluate model
    metrics = face_recognition.evaluate(test_loader=test_loader, dataset=dataset)
    
    # Visualize embeddings
    face_recognition.visualize_embeddings(
        data_loader=test_loader,
        dataset=dataset,
        output_path="results/figures/embedding_visualization.png"
    )