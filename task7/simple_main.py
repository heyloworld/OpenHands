import os
import logging
import argparse
import torch
import matplotlib.pyplot as plt
from src.simple_data_loader import prepare_data, evaluate_model
from src.model import SRCNN, train_model, load_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main(args):
    """
    Main function to train and test the SRCNN model.
    
    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    # Create directories
    os.makedirs("models/saved_models", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"Using device: {device}")
    
    # Prepare data
    data_loader, dataset = prepare_data(
        batch_size=args.batch_size,
        scale_factor=args.scale_factor,
        patch_size=args.patch_size,
        stride=args.stride
    )
    
    # Create model
    model = SRCNN(num_channels=1)
    model.to(device)
    
    # Model path
    model_path = os.path.join("models/saved_models", "srcnn_model.pth")
    
    # Train or load model
    if not args.test_only:
        # Train model
        logger.info("Training SRCNN model...")
        losses = train_model(
            model=model,
            data_loader=data_loader,
            device=device,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            step_size=args.step_size,
            gamma=args.gamma,
            save_path=model_path
        )
        
        # Plot training loss
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig("results/figures/training_loss.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        # Load model
        model = load_model(model, model_path, device)
    
    # Evaluate model
    logger.info("Evaluating SRCNN model...")
    metrics, _ = evaluate_model(
        model=model,
        dataset=dataset,
        device=device,
        output_dir="results/figures"
    )
    
    logger.info("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SRCNN for Super-Resolution")
    
    # Data parameters
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--scale_factor", type=int, default=3, help="Scale factor for super-resolution.")
    parser.add_argument("--patch_size", type=int, default=33, help="Size of the patches to extract.")
    parser.add_argument("--stride", type=int, default=14, help="Stride for patch extraction.")
    
    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--step_size", type=int, default=30, help="Step size for learning rate scheduler.")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma for learning rate scheduler.")
    
    # Other parameters
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA.")
    parser.add_argument("--test_only", action="store_true", help="Only test the model.")
    
    args = parser.parse_args()
    
    main(args)