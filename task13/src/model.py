import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time
import os
from typing import Dict, List, Tuple, Optional, Union

class VGG16FeatureExtractor(nn.Module):
    """
    VGG16 feature extractor for perceptual loss calculation.
    Extracts features from specific layers of a pre-trained VGG16 network.
    """
    def __init__(self, layers: List[str]):
        """
        Initialize the VGG16 feature extractor.
        
        Args:
            layers: List of layer names to extract features from
        """
        super(VGG16FeatureExtractor, self).__init__()
        self.layers = layers
        
        # Load pre-trained VGG16 model
        vgg16 = models.vgg16(pretrained=True).features.eval()
        
        # Freeze parameters
        for param in vgg16.parameters():
            param.requires_grad = False
            
        # Create a sequential model with the layers we need
        self.model = vgg16
        
        # Mean and std for VGG16 normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        # Layer indices for feature extraction
        self.layer_indices = {
            'relu1_2': 4,
            'relu2_2': 9,
            'relu3_3': 16,
            'relu4_3': 23,
            'relu5_3': 30
        }
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the VGG16 network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Dictionary of feature maps from specified layers
        """
        # Normalize input
        if x.device != self.mean.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)
        
        x = (x - self.mean) / self.std
        
        # Extract features
        features = {}
        for i, layer in enumerate(self.model):
            x = layer(x)
            layer_name = None
            for name, idx in self.layer_indices.items():
                if idx == i:
                    layer_name = name
                    break
            if layer_name is not None and layer_name in self.layers:
                features[layer_name] = x.clone()
            
        return features

class PerceptualLoss(nn.Module):
    """
    Perceptual loss for style transfer.
    Combines content loss and style loss using VGG16 features.
    """
    def __init__(self, 
                 content_layers: List[str] = ['relu3_3'], 
                 style_layers: List[str] = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'],
                 content_weight: float = 1.0,
                 style_weight: float = 1e6):
        """
        Initialize the perceptual loss.
        
        Args:
            content_layers: List of layer names for content loss
            style_layers: List of layer names for style loss
            content_weight: Weight for content loss
            style_weight: Weight for style loss
        """
        super(PerceptualLoss, self).__init__()
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.content_weight = content_weight
        self.style_weight = style_weight
        
        # Create feature extractor
        self.feature_extractor = VGG16FeatureExtractor(list(set(content_layers + style_layers)))
        
    def gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Gram matrix for style loss.
        
        Args:
            x: Feature map tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Gram matrix of shape (batch_size, channels, channels)
        """
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram.div(c * h * w)
    
    def content_loss(self, input_features: Dict[str, torch.Tensor], 
                    target_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute content loss.
        
        Args:
            input_features: Dictionary of feature maps from input image
            target_features: Dictionary of feature maps from target image
            
        Returns:
            Content loss
        """
        loss = 0.0
        for layer in self.content_layers:
            loss += F.mse_loss(input_features[layer], target_features[layer])
        return loss
    
    def style_loss(self, input_features: Dict[str, torch.Tensor], 
                  target_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute style loss.
        
        Args:
            input_features: Dictionary of feature maps from input image
            target_features: Dictionary of feature maps from target image
            
        Returns:
            Style loss
        """
        loss = 0.0
        for layer in self.style_layers:
            input_gram = self.gram_matrix(input_features[layer])
            target_gram = self.gram_matrix(target_features[layer])
            loss += F.mse_loss(input_gram, target_gram)
        return loss
    
    def forward(self, input_img: torch.Tensor, content_img: torch.Tensor, 
               style_img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass to compute perceptual loss.
        
        Args:
            input_img: Input image to be stylized
            content_img: Content image
            style_img: Style image
            
        Returns:
            Tuple of (total loss, content loss, style loss)
        """
        # Extract features
        input_features = self.feature_extractor(input_img)
        content_features = self.feature_extractor(content_img)
        style_features = self.feature_extractor(style_img)
        
        # Compute losses
        content_loss = self.content_loss(input_features, content_features)
        style_loss = self.style_loss(input_features, style_features)
        
        # Weighted total loss
        total_loss = self.content_weight * content_loss + self.style_weight * style_loss
        
        return total_loss, content_loss, style_loss

class StyleTransfer:
    """
    Style transfer using perceptual loss.
    """
    def __init__(self, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 content_weight: float = 1.0,
                 style_weight: float = 1e6,
                 content_layers: List[str] = ['relu3_3'],
                 style_layers: List[str] = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3']):
        """
        Initialize the style transfer.
        
        Args:
            device: Device to run the model on
            content_weight: Weight for content loss
            style_weight: Weight for style loss
            content_layers: List of layer names for content loss
            style_layers: List of layer names for style loss
        """
        self.device = device
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.content_layers = content_layers
        self.style_layers = style_layers
        
        # Create perceptual loss
        self.perceptual_loss = PerceptualLoss(
            content_layers=content_layers,
            style_layers=style_layers,
            content_weight=content_weight,
            style_weight=style_weight
        ).to(device)
        
        # Image preprocessing
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # Image postprocessing
        self.postprocess = transforms.Compose([
            transforms.Lambda(lambda x: x.clamp(0, 1)),
            transforms.ToPILImage()
        ])
        
    def load_image(self, path: str, size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Load and preprocess an image.
        
        Args:
            path: Path to the image
            size: Optional size to resize the image to
            
        Returns:
            Preprocessed image tensor
        """
        image = Image.open(path).convert('RGB')
        if size is not None:
            image = image.resize(size, Image.LANCZOS)
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        return image
    
    def save_image(self, tensor: torch.Tensor, path: str) -> None:
        """
        Save a tensor as an image.
        
        Args:
            tensor: Image tensor
            path: Path to save the image to
        """
        image = self.postprocess(tensor.squeeze(0).cpu())
        image.save(path)
        
    def transfer_style(self, 
                       content_path: str, 
                       style_path: str, 
                       output_path: str,
                       num_steps: int = 300,
                       content_size: Optional[Tuple[int, int]] = None,
                       style_size: Optional[Tuple[int, int]] = None,
                       lr: float = 0.01,
                       log_interval: int = 50,
                       save_intermediate: bool = True,
                       intermediate_path: Optional[str] = None) -> Dict[str, Union[float, List[float]]]:
        """
        Perform style transfer.
        
        Args:
            content_path: Path to content image
            style_path: Path to style image
            output_path: Path to save output image
            num_steps: Number of optimization steps
            content_size: Optional size to resize content image to
            style_size: Optional size to resize style image to
            lr: Learning rate for optimization
            log_interval: Interval for logging progress
            save_intermediate: Whether to save intermediate results
            intermediate_path: Path to save intermediate results
            
        Returns:
            Dictionary with processing time and loss history
        """
        start_time = time.time()
        
        # Load images
        content_img = self.load_image(content_path, content_size)
        style_img = self.load_image(style_path, style_size)
        
        # Initialize input image with content image
        input_img = content_img.clone().requires_grad_(True)
        
        # Setup optimizer
        optimizer = torch.optim.Adam([input_img], lr=lr)
        
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
            optimizer.zero_grad()
            
            # Compute loss
            total_loss, content_loss, style_loss = self.perceptual_loss(
                input_img, content_img, style_img
            )
            
            # Backward pass
            total_loss.backward()
            
            # Update input image
            optimizer.step()
            
            # Log progress
            if step % log_interval == 0 or step == num_steps - 1:
                print(f"Step {step}/{num_steps}, "
                      f"Total Loss: {total_loss.item():.4f}, "
                      f"Content Loss: {content_loss.item():.4f}, "
                      f"Style Loss: {style_loss.item() / self.style_weight:.4f}")
                
                # Save intermediate result
                if save_intermediate and intermediate_path is not None:
                    intermediate_img = input_img.clone().detach()
                    intermediate_images.append(intermediate_img)
            
            # Update loss history
            loss_history['total'].append(total_loss.item())
            loss_history['content'].append(content_loss.item())
            loss_history['style'].append(style_loss.item())
        
        # Save final result
        self.save_image(input_img.detach(), output_path)
        
        # Save intermediate results
        if save_intermediate and intermediate_path is not None and intermediate_images:
            self.save_intermediate_results(intermediate_images, intermediate_path)
        
        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"Style transfer completed in {processing_time:.2f} seconds")
        
        # Log processing time
        processing_time_path = os.path.join(os.path.dirname(output_path), '../processing_time.txt')
        os.makedirs(os.path.dirname(processing_time_path), exist_ok=True)
        with open(processing_time_path, 'w') as f:
            f.write(f"Style transfer processing time: {processing_time:.2f} seconds\n")
            f.write(f"Number of steps: {num_steps}\n")
            f.write(f"Content weight: {self.content_weight}\n")
            f.write(f"Style weight: {self.style_weight}\n")
        
        return {
            'processing_time': processing_time,
            'loss_history': loss_history
        }
    
    def save_intermediate_results(self, images: List[torch.Tensor], path: str) -> None:
        """
        Save intermediate results as a grid image.
        
        Args:
            images: List of intermediate image tensors
            path: Path to save the grid image
        """
        import matplotlib.pyplot as plt
        
        # Select a subset of images if there are too many
        if len(images) > 9:
            indices = np.linspace(0, len(images) - 1, 9, dtype=int)
            images = [images[i] for i in indices]
        
        # Create a grid of images
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.flatten()
        
        for i, img_tensor in enumerate(images):
            if i < len(axes):
                img = self.postprocess(img_tensor.squeeze(0).cpu())
                axes[i].imshow(np.array(img))
                axes[i].set_title(f"Step {i * (len(images) // 9 + 1)}")
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(images), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(path)
        plt.close()