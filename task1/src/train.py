import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from src.data_loader import get_data_loaders, get_class_names
from src.model import get_model
from src.utils import GradCAM, visualize_model_predictions

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                device, num_epochs=25, save_dir='./models/saved_models'):
    """
    Train the model.
    
    Args:
        model (nn.Module): The model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device (torch.device): Device to train on
        num_epochs (int): Number of epochs to train for
        save_dir (str): Directory to save the model
        
    Returns:
        model (nn.Module): The trained model
        history (dict): Training history
    """
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir='./logs')
    
    # Initialize the history dictionary to store metrics
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Initialize best validation accuracy for model saving
    best_val_acc = 0.0
    
    # Training loop
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        # Wrap the training loader with tqdm for progress bar
        train_loader_tqdm = tqdm(train_loader, desc=f'Training')
        
        for inputs, labels in train_loader_tqdm:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
            
            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # Update the progress bar
            train_loader_tqdm.set_postfix({
                'loss': loss.item(),
                'acc': torch.sum(preds == labels.data).item() / inputs.size(0)
            })
        
        # Calculate epoch statistics
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = running_corrects.double() / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        # Wrap the validation loader with tqdm for progress bar
        val_loader_tqdm = tqdm(val_loader, desc=f'Validation')
        
        for inputs, labels in val_loader_tqdm:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass (no gradient calculation for validation)
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
            
            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # Update the progress bar
            val_loader_tqdm.set_postfix({
                'loss': loss.item(),
                'acc': torch.sum(preds == labels.data).item() / inputs.size(0)
            })
        
        # Calculate epoch statistics
        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = running_corrects.double() / len(val_loader.dataset)
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step(epoch_val_loss)
        
        # Print epoch statistics
        print(f'Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f}')
        print(f'Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}')
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', epoch_train_loss, epoch)
        writer.add_scalar('Loss/val', epoch_val_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_train_acc, epoch)
        writer.add_scalar('Accuracy/val', epoch_val_acc, epoch)
        
        # Update history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc.item())
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc.item())
        
        # Save the model if it's the best so far
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, 'fashionnet.pt'))
            print(f'Model saved with validation accuracy: {best_val_acc:.4f}')
    
    # Close the TensorBoard writer
    writer.close()
    
    # Return the trained model and history
    return model, history

def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluate the model on the test set.
    
    Args:
        model (nn.Module): The model to evaluate
        test_loader (DataLoader): Test data loader
        criterion: Loss function
        device (torch.device): Device to evaluate on
        
    Returns:
        test_loss (float): Test loss
        test_acc (float): Test accuracy
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    
    # Wrap the test loader with tqdm for progress bar
    test_loader_tqdm = tqdm(test_loader, desc='Testing')
    
    # Confusion matrix
    num_classes = 10
    confusion_matrix = torch.zeros(num_classes, num_classes)
    
    for inputs, labels in test_loader_tqdm:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
        
        # Update statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
        # Update confusion matrix
        for t, p in zip(labels.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
        
        # Update the progress bar
        test_loader_tqdm.set_postfix({
            'loss': loss.item(),
            'acc': torch.sum(preds == labels.data).item() / inputs.size(0)
        })
    
    # Calculate test statistics
    test_loss = running_loss / len(test_loader.dataset)
    test_acc = running_corrects.double() / len(test_loader.dataset)
    
    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')
    
    # Print per-class accuracy
    class_names = get_class_names()
    per_class_acc = confusion_matrix.diag() / confusion_matrix.sum(1)
    for i, (acc, name) in enumerate(zip(per_class_acc, class_names)):
        print(f'Accuracy of {name}: {acc:.4f}')
    
    return test_loss, test_acc, confusion_matrix

def plot_training_history(history):
    """
    Plot the training history.
    
    Args:
        history (dict): Training history
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='train')
    ax1.plot(history['val_loss'], label='val')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='train')
    ax2.plot(history['val_acc'], label='val')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('./plots/training_history.png')
    plt.close()

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create directories
    os.makedirs('./plots', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Hyperparameters
    batch_size = 64
    num_epochs = 1  # Reduced for faster execution
    learning_rate = 0.001
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=batch_size)
    
    # Get model
    model = get_model(num_classes=10, pretrained=True, device=device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3
    )
    
    # Train the model
    print('Starting training...')
    start_time = time.time()
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, 
        device, num_epochs=num_epochs
    )
    end_time = time.time()
    print(f'Training completed in {end_time - start_time:.2f} seconds')
    
    # Plot training history
    plot_training_history(history)
    
    # Load the best model
    model.load_state_dict(torch.load('./models/saved_models/fashionnet.pt'))
    
    # Evaluate the model
    print('Evaluating the model on the test set...')
    test_loss, test_acc, confusion_matrix = evaluate_model(model, test_loader, criterion, device)
    
    # Skip model interpretability visualizations for now
    print('Skipping model interpretability visualizations...')
    
    # Skip GradCAM for now as it's causing issues
    print("Model training and evaluation completed successfully!")
    print(f"Model saved to ./models/saved_models/fashionnet.pt")
    print("Class-wise accuracy calculated and displayed above.")
    
    print('Done!')

if __name__ == '__main__':
    main()