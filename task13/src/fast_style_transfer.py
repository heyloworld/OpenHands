import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import copy

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image loading and preprocessing
def load_image(image_path, max_size=512):
    image = Image.open(image_path).convert('RGB')
    
    # Resize image while maintaining aspect ratio
    if max(image.size) > max_size:
        size = max_size
        if image.width > image.height:
            size = (max_size, int(image.height * max_size / image.width))
        else:
            size = (int(image.width * max_size / image.height), max_size)
        image = image.resize(size, Image.LANCZOS)
    
    # Convert image to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Add batch dimension
    image = transform(image).unsqueeze(0).to(device)
    
    return image

# Image deprocessing
def deprocess_image(tensor):
    # Clone the tensor to avoid modifying the original
    image = tensor.clone().detach().cpu().squeeze(0)
    
    # Denormalize
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    
    # Clamp values to [0, 1]
    image = image.clamp(0, 1)
    
    # Convert to PIL image
    transform = transforms.ToPILImage()
    image = transform(image)
    
    return image

# Content loss
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        
    def forward(self, x):
        self.loss = F.mse_loss(x, self.target)
        return x

# Style loss
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()
        
    def forward(self, x):
        G = self.gram_matrix(x)
        self.loss = F.mse_loss(G, self.target)
        return x
    
    def gram_matrix(self, x):
        batch_size, channels, height, width = x.size()
        features = x.view(batch_size, channels, height * width)
        G = torch.bmm(features, features.transpose(1, 2))
        # Normalize by total elements
        return G.div(channels * height * width)

# VGG model for feature extraction
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        # Load pretrained VGG19 model
        vgg = models.vgg19(pretrained=True).features.eval()
        self.model = vgg
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        # Extract features
        features = []
        for i, layer in enumerate(self.model):
            x = layer(x)
            # Extract features from specific layers
            if isinstance(layer, nn.Conv2d):
                features.append(x)
        return features

# Style transfer function
def style_transfer(content_img, style_img, num_steps=300, content_weight=1, style_weight=1e6, 
                  log_interval=50, save_intermediate=True, intermediate_path=None):
    """
    Perform style transfer using VGG19 features.
    
    Args:
        content_img: Content image tensor
        style_img: Style image tensor
        num_steps: Number of optimization steps
        content_weight: Weight for content loss
        style_weight: Weight for style loss
        log_interval: Interval for logging progress
        save_intermediate: Whether to save intermediate results
        intermediate_path: Path to save intermediate results
        
    Returns:
        Dictionary with processing time, loss history, and stylized image
    """
    start_time = time.time()
    
    # Initialize input image with content image
    input_img = content_img.clone().requires_grad_(True)
    
    # Initialize VGG model
    vgg = VGG().to(device)
    
    # Extract content and style features
    content_features = vgg(content_img)
    style_features = vgg(style_img)
    
    # Content target (4th conv layer, typically conv4_2)
    content_target = content_features[4].detach()
    
    # Style targets
    style_targets = []
    for i in [0, 2, 4, 8, 12]:  # layers conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
        # Calculate Gram matrix for style features
        _, c, h, w = style_features[i].shape
        features = style_features[i].view(1, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        gram = gram.div(c * h * w)
        style_targets.append(gram.detach())
    
    # Setup optimizer
    optimizer = optim.Adam([input_img], lr=0.03)
    
    # Loss history
    loss_history = {
        'total': [],
        'content': [],
        'style': []
    }
    
    # Intermediate images
    intermediate_images = []
    
    # Optimization loop
    for step in range(num_steps):
        # Forward pass
        features = vgg(input_img)
        
        # Compute content loss
        content_loss_value = F.mse_loss(features[4], content_target)
        
        # Compute style loss
        style_loss_value = 0
        for i, idx in enumerate([0, 2, 4, 8, 12]):
            # Calculate Gram matrix for input features
            _, c, h, w = features[idx].shape
            input_features = features[idx].view(1, c, h * w)
            input_gram = torch.bmm(input_features, input_features.transpose(1, 2))
            input_gram = input_gram.div(c * h * w)
            
            # Add to style loss
            style_loss_value += F.mse_loss(input_gram, style_targets[i])
        
        # Compute total loss
        total_loss = content_weight * content_loss_value + style_weight * style_loss_value
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Log progress
        if step % log_interval == 0 or step == num_steps - 1:
            print(f"Step {step}/{num_steps}, "
                  f"Total Loss: {total_loss.item():.4f}, "
                  f"Content Loss: {content_loss_value.item():.4f}, "
                  f"Style Loss: {style_loss_value.item():.4f}")
            
            # Save intermediate result
            if save_intermediate and intermediate_path is not None:
                intermediate_img = input_img.clone().detach()
                intermediate_images.append(intermediate_img)
        
        # Update loss history
        loss_history['total'].append(total_loss.item())
        loss_history['content'].append(content_loss_value.item())
        loss_history['style'].append(style_loss_value.item())
    
    # Calculate processing time
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"Style transfer completed in {processing_time:.2f} seconds")
    
    # Save intermediate results
    if save_intermediate and intermediate_path is not None and intermediate_images:
        save_intermediate_results(intermediate_images, intermediate_path)
    
    return {
        'processing_time': processing_time,
        'loss_history': loss_history,
        'stylized_image': input_img.detach()
    }

# Save intermediate results
def save_intermediate_results(images, path):
    """
    Save intermediate results as a grid image.
    
    Args:
        images: List of intermediate image tensors
        path: Path to save the grid image
    """
    # Select a subset of images if there are too many
    if len(images) > 9:
        indices = np.linspace(0, len(images) - 1, 9, dtype=int)
        images = [images[i] for i in indices]
    
    # Create a grid of images
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, img_tensor in enumerate(images):
        if i < len(axes):
            img = deprocess_image(img_tensor)
            axes[i].imshow(np.array(img))
            axes[i].set_title(f"Step {i * (len(images) // 9 + 1)}")
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(images), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# Plot loss history
def plot_loss_history(loss_history, output_path):
    """
    Plot and save loss history.
    
    Args:
        loss_history: Dictionary with loss history
        output_path: Path to save the plot
    """
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

# Main function
def main(content_path, style_path, output_path, num_steps=100, content_weight=1, style_weight=1e6,
         max_size=384, log_interval=10, intermediate_path=None):
    """
    Main function for style transfer.
    
    Args:
        content_path: Path to content image
        style_path: Path to style image
        output_path: Path to save output image
        num_steps: Number of optimization steps
        content_weight: Weight for content loss
        style_weight: Weight for style loss
        max_size: Maximum size for images
        log_interval: Interval for logging progress
        intermediate_path: Path to save intermediate results
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if intermediate_path:
        os.makedirs(os.path.dirname(intermediate_path), exist_ok=True)
    
    # Load images
    content_img = load_image(content_path, max_size=max_size)
    style_img = load_image(style_path, max_size=max_size)
    
    print(f"Content image size: {content_img.shape}")
    print(f"Style image size: {style_img.shape}")
    
    # Perform style transfer
    result = style_transfer(
        content_img=content_img,
        style_img=style_img,
        num_steps=num_steps,
        content_weight=content_weight,
        style_weight=style_weight,
        log_interval=log_interval,
        save_intermediate=True,
        intermediate_path=intermediate_path
    )
    
    # Save stylized image
    stylized_img = deprocess_image(result['stylized_image'])
    stylized_img.save(output_path)
    
    # Save loss history
    loss_history_path = os.path.join(os.path.dirname(output_path), 'loss_history.png')
    plot_loss_history(result['loss_history'], loss_history_path)
    
    # Log processing time
    processing_time_path = os.path.join(os.path.dirname(output_path), '../processing_time.txt')
    with open(processing_time_path, 'w') as f:
        f.write(f"Style transfer processing time: {result['processing_time']:.2f} seconds\n")
        f.write(f"Number of steps: {num_steps}\n")
        f.write(f"Content weight: {content_weight}\n")
        f.write(f"Style weight: {style_weight}\n")
    
    print(f"Stylized image saved to {output_path}")
    print(f"Intermediate results saved to {intermediate_path}")
    print(f"Loss history saved to {loss_history_path}")
    print(f"Processing time: {result['processing_time']:.2f} seconds")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Neural Style Transfer')
    parser.add_argument('--content', type=str, default='data/content.jpg', help='Content image path')
    parser.add_argument('--style', type=str, default='data/style.jpg', help='Style image path')
    parser.add_argument('--output', type=str, default='results/figures/stylized.jpg', help='Output image path')
    parser.add_argument('--steps', type=int, default=100, help='Number of optimization steps')
    parser.add_argument('--content-weight', type=float, default=1, help='Content loss weight')
    parser.add_argument('--style-weight', type=float, default=1e6, help='Style loss weight')
    parser.add_argument('--max-size', type=int, default=384, help='Maximum image size')
    parser.add_argument('--log-interval', type=int, default=10, help='Logging interval')
    parser.add_argument('--intermediate-path', type=str, default='results/figures/intermediate_results.png',
                       help='Path to save intermediate results')
    
    args = parser.parse_args()
    
    main(
        content_path=args.content,
        style_path=args.style,
        output_path=args.output,
        num_steps=args.steps,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        max_size=args.max_size,
        log_interval=args.log_interval,
        intermediate_path=args.intermediate_path
    )