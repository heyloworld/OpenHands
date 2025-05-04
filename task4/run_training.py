import os
import sys
import logging

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to run the training.
    """
    # Import the train module
    from src.train import main as train_main
    
    # Create the necessary directories
    os.makedirs('models/saved_models', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)
    
    # Run the training
    logger.info("Starting training...")
    train_main()
    logger.info("Training completed.")

if __name__ == "__main__":
    main()