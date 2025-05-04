import os
import logging
import argparse
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from src.model import SRCNN

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def preprocess_image(image_path, scale_factor):
    """
    Preprocess an image for super-resolution.
    
    Args:
        image_path (str): Path to the image.
        scale_factor (int): Scale factor for super-resolution.
        
    Returns:
        tuple: LR image tensor, original HR image, and color channels.
    """
    # Load image
    hr_image = Image.open(image_path).convert('RGB')
    hr_image_np = np.array(hr_image)
    
    # Convert to YCbCr color space and extract channels
    hr_image_ycrcb = cv2.cvtColor(hr_image_np, cv2.COLOR_RGB2YCrCb)
    hr_image_y = hr_image_ycrcb[:, :, 0]
    hr_image_cb = hr_image_ycrcb[:, :, 1]
    hr_image_cr = hr_image_ycrcb[:, :, 2]
    
    # Generate LR image
    lr_image_y = cv2.resize(hr_image_y, None, fx=1.0/scale_factor, fy=1.0/scale_factor, interpolation=cv2.INTER_CUBIC)
    
    # Upscale LR image to HR size
    lr_image_y_upscaled = cv2.resize(lr_image_y, (hr_image_y.shape[1], hr_image_y.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    # Normalize image
    lr_image_y_upscaled = lr_image_y_upscaled.astype(np.float32) / 255.0
    
    # Convert to tensor
    lr_tensor = torch.from_numpy(lr_image_y_upscaled).float().unsqueeze(0).unsqueeze(0)
    
    return lr_tensor, hr_image_np, hr_image_cb, hr_image_cr

def postprocess_image(sr_tensor, cb, cr):
    """
    Postprocess the super-resolution result.
    
    Args:
        sr_tensor (torch.Tensor): Super-resolution tensor.
        cb (numpy.ndarray): Cb channel.
        cr (numpy.ndarray): Cr channel.
        
    Returns:
        numpy.ndarray: Super-resolution RGB image.
    """
    # Convert to numpy
    sr_image_y = sr_tensor.squeeze().cpu().numpy()
    
    # Denormalize
    sr_image_y = (sr_image_y * 255.0).astype(np.uint8)
    
    # Combine channels
    sr_image_ycrcb = np.stack([sr_image_y, cb, cr], axis=-1)
    
    # Convert to RGB
    sr_image_rgb = cv2.cvtColor(sr_image_ycrcb, cv2.COLOR_YCrCb2RGB)
    
    return sr_image_rgb

def test_image(model, image_path, scale_factor, output_path, device):
    """
    Test the model on a single image.
    
    Args:
        model (SRCNN): The SRCNN model.
        image_path (str): Path to the image.
        scale_factor (int): Scale factor for super-resolution.
        output_path (str): Path to save the output image.
        device (torch.device): The device to use.
    """
    # Preprocess image
    lr_tensor, hr_image, cb, cr = preprocess_image(image_path, scale_factor)
    lr_tensor = lr_tensor.to(device)
    
    # Forward pass
    with torch.no_grad():
        sr_tensor = model(lr_tensor)
    
    # Postprocess image
    sr_image = postprocess_image(sr_tensor, cb, cr)
    
    # Generate bicubic upscaled image
    bicubic_image = cv2.resize(cv2.resize(hr_image, None, fx=1.0/scale_factor, fy=1.0/scale_factor, interpolation=cv2.INTER_CUBIC), 
                              (hr_image.shape[1], hr_image.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save comparison
    plt.figure(figsize=(15, 5))
    
    # Original HR image
    plt.subplot(1, 3, 1)
    plt.imshow(hr_image)
    plt.title("Original HR")
    plt.axis('off')
    
    # Bicubic upscaled image
    plt.subplot(1, 3, 2)
    plt.imshow(bicubic_image)
    plt.title(f"Bicubic Upscaled (x{scale_factor})")
    plt.axis('off')
    
    # SRCNN SR image
    plt.subplot(1, 3, 3)
    plt.imshow(sr_image)
    plt.title("SRCNN")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Output saved to {output_path}")

def main(args):
    """
    Main function to test the SRCNN model.
    
    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model
    model = SRCNN(num_channels=1)
    
    # Load model
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    logger.info(f"Model loaded from {args.model_path}")
    
    # Test image
    test_image(
        model=model,
        image_path=args.image_path,
        scale_factor=args.scale_factor,
        output_path=args.output_path,
        device=device
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SRCNN on a single image")
    
    parser.add_argument("--model_path", type=str, default="models/saved_models/srcnn_model.pth", help="Path to the model.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image.")
    parser.add_argument("--scale_factor", type=int, default=3, help="Scale factor for super-resolution.")
    parser.add_argument("--output_path", type=str, default="results/figures/test_result.png", help="Path to save the output image.")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA.")
    
    args = parser.parse_args()
    
    main(args)