import os
import numpy as np
from text_generator import TextGenerator
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def save_model_weights(model_path="models/saved_models/q_learning_model.npy"):
    """
    Save the model weights to a file.
    
    Args:
        model_path (str): The path to save the model weights.
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Initialize the text generator
    generator = TextGenerator()
    
    # Get the model weights
    model_weights = {}
    for name, param in generator.model.named_parameters():
        model_weights[name] = param.detach().cpu().numpy()
    
    # Save the model weights
    np.save(model_path, model_weights)
    
    logger.info(f"Model weights saved to {model_path}")
    
    # Print some statistics
    total_params = sum(param.numel() for param in generator.model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    
    # Print the size of the saved file
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # Convert to MB
    logger.info(f"File size: {file_size:.2f} MB")

def main():
    """
    Main function to save the model weights.
    """
    save_model_weights()

if __name__ == "__main__":
    main()