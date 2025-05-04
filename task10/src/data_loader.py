import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
from typing import Tuple, List, Dict, Optional, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FaceDataset(Dataset):
    """
    Dataset for loading face images for face recognition.
    
    This dataset loads images from a directory structure where each subdirectory
    represents a different person/identity.
    """
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[transforms.Compose] = None,
        min_images_per_person: int = 5,
        image_size: Tuple[int, int] = (96, 96)
    ):
        """
        Initialize the FaceDataset.
        
        Args:
            data_dir (str): Directory containing face images.
            transform (transforms.Compose, optional): Transformations to apply to images.
            min_images_per_person (int): Minimum number of images required per person.
            image_size (Tuple[int, int]): Size to resize images to.
        """
        self.data_dir = data_dir
        self.image_size = image_size
        
        # Set up transformations
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
        
        # Load dataset
        self.images, self.labels, self.identities = self._load_dataset(min_images_per_person)
        
        # Get number of classes
        self.num_classes = len(self.identities)
        
        logger.info(f"Loaded {len(self.images)} images of {self.num_classes} identities")
    
    def _load_dataset(self, min_images_per_person: int) -> Tuple[List[str], List[int], List[str]]:
        """
        Load the dataset from the directory.
        
        Args:
            min_images_per_person (int): Minimum number of images required per person.
            
        Returns:
            Tuple[List[str], List[int], List[str]]: Lists of image paths, labels, and identity names.
        """
        images = []
        labels = []
        identities = []
        
        # Check if data directory exists
        if not os.path.exists(self.data_dir):
            logger.error(f"Data directory {self.data_dir} does not exist")
            return images, labels, identities
        
        # Get list of subdirectories (each representing a person)
        person_dirs = [d for d in os.listdir(self.data_dir) 
                      if os.path.isdir(os.path.join(self.data_dir, d))]
        
        # Sort for reproducibility
        person_dirs.sort()
        
        # Load images for each person
        for idx, person_dir in enumerate(person_dirs):
            person_path = os.path.join(self.data_dir, person_dir)
            
            # Get image files for this person
            image_files = [f for f in os.listdir(person_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Skip if not enough images
            if len(image_files) < min_images_per_person:
                logger.warning(f"Skipping {person_dir}: only {len(image_files)} images "
                              f"(minimum {min_images_per_person} required)")
                continue
            
            # Add to dataset
            for image_file in image_files:
                image_path = os.path.join(person_path, image_file)
                images.append(image_path)
                labels.append(idx)
            
            identities.append(person_dir)
        
        return images, labels, identities
    
    def __len__(self) -> int:
        """
        Get the number of images in the dataset.
        
        Returns:
            int: Number of images.
        """
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get an item from the dataset.
        
        Args:
            idx (int): Index of the item.
            
        Returns:
            Tuple[torch.Tensor, int]: Image tensor and label.
        """
        image_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transformations
            if self.transform:
                image = self.transform(image)
            
            return image, label
        
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            
            # Return a blank image and the label
            blank_image = torch.zeros(3, self.image_size[0], self.image_size[1])
            return blank_image, label
    
    def get_identity_name(self, label: int) -> str:
        """
        Get the identity name for a label.
        
        Args:
            label (int): Label index.
            
        Returns:
            str: Identity name.
        """
        if 0 <= label < len(self.identities):
            return self.identities[label]
        return "Unknown"
    
    def get_identity_samples(self, label: int, max_samples: int = 5) -> List[str]:
        """
        Get sample image paths for an identity.
        
        Args:
            label (int): Label index.
            max_samples (int): Maximum number of samples to return.
            
        Returns:
            List[str]: List of image paths.
        """
        if 0 <= label < len(self.identities):
            # Find all images for this identity
            identity_images = [self.images[i] for i in range(len(self.images)) 
                              if self.labels[i] == label]
            
            # Return up to max_samples
            return identity_images[:max_samples]
        
        return []


class FacePreprocessor:
    """
    Preprocessor for face images.
    
    This class handles face detection, alignment, and preprocessing for face recognition.
    """
    
    def __init__(
        self,
        face_size: Tuple[int, int] = (96, 96),
        use_face_detection: bool = False,
        face_detector_path: Optional[str] = None
    ):
        """
        Initialize the FacePreprocessor.
        
        Args:
            face_size (Tuple[int, int]): Size to resize face images to.
            use_face_detection (bool): Whether to use face detection.
            face_detector_path (str, optional): Path to face detector model.
        """
        self.face_size = face_size
        self.use_face_detection = use_face_detection
        
        # Set up face detector if requested
        self.face_detector = None
        if use_face_detection:
            try:
                if face_detector_path and os.path.exists(face_detector_path):
                    # Load custom face detector
                    self.face_detector = cv2.CascadeClassifier(face_detector_path)
                else:
                    # Use default face detector
                    self.face_detector = cv2.CascadeClassifier(
                        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                    )
                
                logger.info("Face detector loaded successfully")
            
            except Exception as e:
                logger.error(f"Error loading face detector: {e}")
                self.use_face_detection = False
        
        # Set up transformations
        self.transform = transforms.Compose([
            transforms.Resize(face_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess a face image.
        
        Args:
            image_path (str): Path to the image.
            
        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Detect and crop face if requested
            if self.use_face_detection and self.face_detector is not None:
                # Convert to OpenCV format
                cv_image = np.array(image)
                cv_image = cv_image[:, :, ::-1].copy()  # RGB to BGR
                
                # Detect faces
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                
                # Crop to face if detected
                if len(faces) > 0:
                    x, y, w, h = faces[0]  # Use the first face
                    face_image = cv_image[y:y+h, x:x+w]
                    
                    # Convert back to PIL
                    image = Image.fromarray(face_image[:, :, ::-1])  # BGR to RGB
            
            # Apply transformations
            image_tensor = self.transform(image)
            
            return image_tensor
        
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            
            # Return a blank image
            return torch.zeros(3, self.face_size[0], self.face_size[1])
    
    def preprocess_batch(self, image_paths: List[str]) -> torch.Tensor:
        """
        Preprocess a batch of face images.
        
        Args:
            image_paths (List[str]): List of image paths.
            
        Returns:
            torch.Tensor: Batch of preprocessed image tensors.
        """
        # Process each image
        batch = []
        for image_path in image_paths:
            image_tensor = self.preprocess_image(image_path)
            batch.append(image_tensor)
        
        # Stack into a batch
        if batch:
            return torch.stack(batch)
        
        # Return empty batch if no images
        return torch.zeros(0, 3, self.face_size[0], self.face_size[1])


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    test_split: float = 0.2,
    min_images_per_person: int = 5,
    image_size: Tuple[int, int] = (96, 96),
    num_workers: int = 4,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, FaceDataset]:
    """
    Create training and testing data loaders.
    
    Args:
        data_dir (str): Directory containing face images.
        batch_size (int): Batch size for training.
        test_split (float): Proportion of data to use for testing.
        min_images_per_person (int): Minimum number of images required per person.
        image_size (Tuple[int, int]): Size to resize images to.
        num_workers (int): Number of worker threads for data loading.
        random_seed (int): Random seed for reproducibility.
        
    Returns:
        Tuple[DataLoader, DataLoader, FaceDataset]: Training and testing data loaders, and dataset.
    """
    # Set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Create dataset
    dataset = FaceDataset(
        data_dir=data_dir,
        min_images_per_person=min_images_per_person,
        image_size=image_size
    )
    
    # Get dataset size
    dataset_size = len(dataset)
    if dataset_size == 0:
        logger.error(f"No valid data found in {data_dir}")
        return None, None, dataset
    
    # Create indices for train/test split
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    
    split_idx = int(np.floor(test_split * dataset_size))
    train_indices, test_indices = indices[split_idx:], indices[:split_idx]
    
    # Create samplers
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers
    )
    
    logger.info(f"Created data loaders with {len(train_indices)} training samples "
               f"and {len(test_indices)} testing samples")
    
    return train_loader, test_loader, dataset


# Example usage
if __name__ == "__main__":
    # Create data loaders
    train_loader, test_loader, dataset = create_dataloaders(
        data_dir="data/synthetic_faces",
        batch_size=32,
        test_split=0.2,
        min_images_per_person=5
    )
    
    if train_loader is not None and test_loader is not None:
        # Print dataset information
        print(f"Dataset loaded with {len(dataset)} images of {dataset.num_classes} identities")
        print(f"Training batches: {len(train_loader)}, Testing batches: {len(test_loader)}")
        
        # Get a batch of images
        images, labels = next(iter(train_loader))
        print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")
        
        # Print some identity information
        for i in range(min(5, dataset.num_classes)):
            print(f"Identity {i}: {dataset.get_identity_name(i)}")
            samples = dataset.get_identity_samples(i, max_samples=2)
            print(f"  Sample images: {samples}")
    else:
        print("Failed to create data loaders")