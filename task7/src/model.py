import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SRCNN(nn.Module):
    """
    SRCNN model for super-resolution.
    
    Architecture:
    1. Feature extraction layer with 9x9 kernels
    2. Non-linear mapping layer with 1x1 kernels
    3. Reconstruction layer with 5x5 kernels
    """
    def __init__(self, num_channels=1):
        """
        Initialize the SRCNN model.
        
        Args:
            num_channels (int): Number of input channels.
        """
        super(SRCNN, self).__init__()
        
        # Feature extraction layer
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Non-linear mapping layer
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Reconstruction layer
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize the weights using normal distribution.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
                nn.init.zeros_(m.bias.data)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor.
        """
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = self.conv3(out)
        return out

def train_model(model, data_loader, device, num_epochs=100, learning_rate=1e-4, step_size=30, gamma=0.1, save_path=None):
    """
    Train the SRCNN model.
    
    Args:
        model (SRCNN): The SRCNN model.
        data_loader (torch.utils.data.DataLoader): DataLoader for training.
        device (torch.device): The device to use.
        num_epochs (int): Number of epochs.
        learning_rate (float): Learning rate.
        step_size (int): Step size for learning rate scheduler.
        gamma (float): Gamma for learning rate scheduler.
        save_path (str, optional): Path to save the model.
        
    Returns:
        list: Training losses.
    """
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    # Training loop
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        with tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for lr_batch, hr_batch in pbar:
                # Move tensors to device
                lr_batch = lr_batch.to(device)
                hr_batch = hr_batch.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                sr_batch = model(lr_batch)
                
                # Calculate loss
                loss = criterion(sr_batch, hr_batch)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Update statistics
                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
        
        # Update learning rate
        scheduler.step()
        
        # Calculate average loss
        avg_loss = epoch_loss / len(data_loader)
        losses.append(avg_loss)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        # Save model
        if save_path and (epoch + 1) % 10 == 0:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info(f"Model saved to {save_path}")
    
    # Save final model
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        logger.info(f"Final model saved to {save_path}")
    
    return losses

def load_model(model, model_path, device):
    """
    Load a trained model.
    
    Args:
        model (SRCNN): The SRCNN model.
        model_path (str): Path to the model.
        device (torch.device): The device to use.
        
    Returns:
        SRCNN: The loaded model.
    """
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    logger.info(f"Model loaded from {model_path}")
    return model

if __name__ == "__main__":
    # Test the model
    model = SRCNN()
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")