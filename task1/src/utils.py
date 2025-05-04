import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import get_class_names

class GradCAM:
    """
    Grad-CAM implementation for model interpretability.
    
    Grad-CAM uses the gradients of any target concept flowing into the final convolutional layer
    to produce a coarse localization map highlighting important regions in the image for predicting
    the concept.
    """
    def __init__(self, model, target_layer):
        """
        Initialize GradCAM.
        
        Args:
            model (nn.Module): The model to interpret
            target_layer: The target layer to compute gradients for
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.register_hooks()
    
    def register_hooks(self):
        """
        Register forward and backward hooks.
        """
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Register the hooks
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def __call__(self, input_image, class_idx=None):
        """
        Generate Grad-CAM for the input image.
        
        Args:
            input_image (torch.Tensor): Input image tensor of shape [1, C, H, W]
            class_idx (int, optional): Class index to generate Grad-CAM for.
                                      If None, uses the predicted class.
                                      
        Returns:
            numpy.ndarray: Grad-CAM heatmap
        """
        # Forward pass
        model_output = self.model(input_image)
        
        # If class_idx is None, use the predicted class
        if class_idx is None:
            class_idx = torch.argmax(model_output).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Target for backprop
        one_hot = torch.zeros_like(model_output)
        one_hot[0, class_idx] = 1
        
        # Backward pass
        model_output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients.detach().cpu()
        activations = self.activations.detach().cpu()
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted sum of activation maps
        cam = torch.sum(weights * activations, dim=1).squeeze()
        
        # ReLU to only keep positive influences
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam) if torch.max(cam) > 0 else cam
        
        # Convert to numpy array
        cam = cam.squeeze().cpu().numpy()
        
        # For Fashion-MNIST, we'll just use the activation map as is
        # since the input is already 28x28
        # No need to resize
        
        return cam

def visualize_model_predictions(model, data_loader, device, num_images=6):
    """
    Visualize model predictions on a few images.
    
    Args:
        model (nn.Module): The model to use for predictions
        data_loader (DataLoader): Data loader to get images from
        device (torch.device): Device to run the model on
        num_images (int): Number of images to visualize
    """
    class_names = get_class_names()
    model.eval()
    
    # Get a batch of images
    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(images.to(device))
        _, preds = torch.max(outputs, 1)
    
    # Plot the images with predictions
    fig = plt.figure(figsize=(15, 3 * (num_images // 3 + 1)))
    
    for i in range(min(num_images, len(images))):
        ax = plt.subplot(num_images // 3 + 1, 3, i + 1)
        img = images[i].squeeze().numpy()
        plt.imshow(img, cmap='gray')
        
        pred_label = preds[i].item()
        true_label = labels[i].item()
        
        title_color = 'green' if pred_label == true_label else 'red'
        ax.set_title(f'Pred: {class_names[pred_label]}\nTrue: {class_names[true_label]}',
                    color=title_color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('./plots/model_predictions.png')
    plt.close()

def plot_confusion_matrix(cm, class_names):
    """
    Plot a confusion matrix.
    
    Args:
        cm (torch.Tensor): Confusion matrix
        class_names (list): List of class names
    """
    cm = cm.numpy()
    
    # Normalize the confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    fmt = '.2f'
    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm_norm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('./plots/confusion_matrix.png')
    plt.close()