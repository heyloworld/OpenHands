import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from model import StyleTransfer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Neural Style Transfer with Perceptual Loss')
    
    parser.add_argument('--content', type=str, default='data/content.jpg',
                        help='Path to content image')
    parser.add_argument('--style', type=str, default='data/style.jpg',
                        help='Path to style image')
    parser.add_argument('--output', type=str, default='results/figures/stylized.jpg',
                        help='Path to output image')
    parser.add_argument('--content-size', type=int, default=512,
                        help='Size to resize content image to (preserving aspect ratio)')
    parser.add_argument('--style-size', type=int, default=512,
                        help='Size to resize style image to (preserving aspect ratio)')
    parser.add_argument('--content-weight', type=float, default=1.0,
                        help='Weight for content loss')
    parser.add_argument('--style-weight', type=float, default=1e6,
                        help='Weight for style loss')
    parser.add_argument('--steps', type=int, default=300,
                        help='Number of optimization steps')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate for optimization')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run the model on')
    parser.add_argument('--log-interval', type=int, default=50,
                        help='Interval for logging progress')
    parser.add_argument('--intermediate-path', type=str, 
                        default='results/figures/intermediate_results.png',
                        help='Path to save intermediate results')
    
    return parser.parse_args()

def resize_with_aspect_ratio(image_path, target_size):
    """Calculate new dimensions preserving aspect ratio."""
    from PIL import Image
    img = Image.open(image_path)
    width, height = img.size
    
    if width > height:
        new_width = target_size
        new_height = int(height * target_size / width)
    else:
        new_height = target_size
        new_width = int(width * target_size / height)
    
    return (new_width, new_height)

def plot_loss_history(loss_history, output_path):
    """Plot and save loss history."""
    plt.figure(figsize=(10, 5))
    
    # Plot total loss
    plt.subplot(1, 2, 1)
    plt.plot(loss_history['total'], label='Total Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Total Loss')
    plt.legend()
    
    # Plot content and style loss
    plt.subplot(1, 2, 2)
    plt.plot(loss_history['content'], label='Content Loss')
    plt.plot([s / 1e6 for s in loss_history['style']], label='Style Loss (scaled)')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Content and Style Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.intermediate_path), exist_ok=True)
    
    # Calculate dimensions preserving aspect ratio
    content_size = resize_with_aspect_ratio(args.content, args.content_size)
    style_size = resize_with_aspect_ratio(args.style, args.style_size)
    
    print(f"Content image will be resized to {content_size}")
    print(f"Style image will be resized to {style_size}")
    
    # Initialize style transfer
    style_transfer = StyleTransfer(
        device=args.device,
        content_weight=args.content_weight,
        style_weight=args.style_weight
    )
    
    # Perform style transfer
    result = style_transfer.transfer_style(
        content_path=args.content,
        style_path=args.style,
        output_path=args.output,
        num_steps=args.steps,
        content_size=content_size,
        style_size=style_size,
        lr=args.lr,
        log_interval=args.log_interval,
        save_intermediate=True,
        intermediate_path=args.intermediate_path
    )
    
    # Plot loss history
    loss_history_path = os.path.join(os.path.dirname(args.output), 'loss_history.png')
    plot_loss_history(result['loss_history'], loss_history_path)
    
    print(f"Stylized image saved to {args.output}")
    print(f"Intermediate results saved to {args.intermediate_path}")
    print(f"Loss history saved to {loss_history_path}")
    print(f"Processing time: {result['processing_time']:.2f} seconds")

if __name__ == '__main__':
    main()