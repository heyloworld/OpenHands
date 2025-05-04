import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to test the system with a different environment.
    """
    # Add the src directory to the path
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    
    # Import the main module
    from src.main import train_dqn
    
    # Create the necessary directories
    os.makedirs('models/saved_models', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)
    
    # Run the training with a different environment
    logger.info("Testing with a different environment (Acrobot-v1)...")
    try:
        train_dqn(env_name='Acrobot-v1', n_episodes=10)
    except Exception as e:
        logger.error(f"Error: {e}")
    
    logger.info("Test completed.")

if __name__ == "__main__":
    main()