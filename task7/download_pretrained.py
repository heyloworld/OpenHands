import os
import torch
import torch.nn as nn
import logging
from src.model import SRCNN

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_pretrained_model():
    """
    Create a pre-trained SRCNN model for testing.
    
    This is a simplified version of the model with weights initialized
    to demonstrate the concept without actual training.
    """
    # Create model
    model = SRCNN(num_channels=1)
    
    # Initialize with custom weights to simulate a pre-trained model
    # These are not actual trained weights, just for demonstration
    with torch.no_grad():
        # Feature extraction layer
        nn.init.normal_(model.conv1.weight, mean=0.0, std=0.001)
        nn.init.constant_(model.conv1.bias, 0.1)
        
        # Non-linear mapping layer
        nn.init.normal_(model.conv2.weight, mean=0.0, std=0.001)
        nn.init.constant_(model.conv2.bias, 0.1)
        
        # Reconstruction layer
        nn.init.normal_(model.conv3.weight, mean=0.0, std=0.001)
        nn.init.constant_(model.conv3.bias, 0.1)
    
    # Create directory
    os.makedirs("models/saved_models", exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), "models/saved_models/srcnn_model.pth")
    logger.info("Pre-trained model saved to models/saved_models/srcnn_model.pth")

if __name__ == "__main__":
    create_pretrained_model()