import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class FashionResNet(nn.Module):
    """
    ResNet-18 model adapted for Fashion-MNIST classification.
    """
    def __init__(self, num_classes=10, pretrained=True):
        """
        Initialize the model.
        
        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pretrained weights
        """
        super(FashionResNet, self).__init__()
        
        # Load the pretrained ResNet-18 model
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.resnet = resnet18(weights=weights)
        
        # Modify the first convolutional layer to accept grayscale images (1 channel)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify the final fully connected layer for our number of classes
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 1, height, width]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, num_classes]
        """
        return self.resnet(x)

def get_model(num_classes=10, pretrained=True, device=None):
    """
    Create and return the FashionResNet model.
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        device (torch.device): Device to move the model to
        
    Returns:
        FashionResNet: The model
    """
    model = FashionResNet(num_classes=num_classes, pretrained=pretrained)
    
    if device is not None:
        model = model.to(device)
        
    return model