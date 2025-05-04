import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_data_loaders(batch_size=64, data_dir='./data', val_split=0.1, num_workers=0):
    """
    Create data loaders for the Fashion-MNIST dataset with data augmentation.
    
    Args:
        batch_size (int): Batch size for training and validation
        data_dir (str): Directory to store the dataset
        val_split (float): Proportion of training data to use for validation
        num_workers (int): Number of workers for data loading
        
    Returns:
        train_loader, val_loader, test_loader (DataLoader): Data loaders for training, validation, and testing
    """
    # Define transformations for training with data augmentation
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),  # Rotate by up to 10 degrees
        transforms.RandomAffine(0, scale=(0.8, 1.2)),  # Random scaling
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.2860,), (0.3530,))  # Normalize with Fashion-MNIST mean and std
    ])
    
    # Define transformations for validation and testing (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    # Load the training dataset
    train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    # Split training data into training and validation sets
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # Override the transform for the validation dataset
    val_dataset.dataset.transform = test_transform
    
    # Load the test dataset
    test_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def get_class_names():
    """
    Get the class names for the Fashion-MNIST dataset.
    
    Returns:
        list: List of class names
    """
    return [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]