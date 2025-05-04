import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Set5Dataset(Dataset):
    """
    Dataset class for the Set5 dataset.
    """
    def __init__(self, scale_factor=3, patch_size=33, stride=14, transform=None):
        """
        Initialize the Set5Dataset.
        
        Args:
            scale_factor (int): Scale factor for super-resolution.
            patch_size (int): Size of the patches to extract.
            stride (int): Stride for patch extraction.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        
        # Load the Set5 dataset from Hugging Face
        logger.info("Loading Set5 dataset from Hugging Face...")
        self.dataset = load_dataset("eugenesiow/Set5", trust_remote_code=True)
        
        # Extract images
        self.hr_images = []
        self.image_names = []
        
        # The dataset might have 'test' or 'validation' split
        split_name = 'validation' if 'validation' in self.dataset else 'test'
        
        for item in self.dataset[split_name]:
            image = item['image']
            self.hr_images.append(image)
            self.image_names.append(item['name'])
        
        logger.info(f"Loaded {len(self.hr_images)} images from Set5 dataset.")
        
        # Generate LR and HR patches
        self.lr_patches, self.hr_patches = self._generate_patches()
    
    def _generate_patches(self):
        """
        Generate LR and HR patches from the images.
        
        Returns:
            tuple: LR patches and HR patches.
        """
        lr_patches = []
        hr_patches = []
        
        for hr_image in self.hr_images:
            # Convert PIL Image to numpy array
            hr_image_np = np.array(hr_image)
            
            # Convert to YCbCr color space and use only the Y channel
            if len(hr_image_np.shape) == 3 and hr_image_np.shape[2] == 3:
                hr_image_y = cv2.cvtColor(hr_image_np, cv2.COLOR_RGB2YCrCb)[:, :, 0]
            else:
                hr_image_y = hr_image_np
            
            # Generate LR image
            lr_image_y = cv2.resize(hr_image_y, None, fx=1.0/self.scale_factor, fy=1.0/self.scale_factor, interpolation=cv2.INTER_CUBIC)
            lr_image_y = cv2.resize(lr_image_y, (hr_image_y.shape[1], hr_image_y.shape[0]), interpolation=cv2.INTER_CUBIC)
            
            # Extract patches
            h, w = hr_image_y.shape
            for i in range(0, h - self.patch_size + 1, self.stride):
                for j in range(0, w - self.patch_size + 1, self.stride):
                    hr_patch = hr_image_y[i:i+self.patch_size, j:j+self.patch_size]
                    lr_patch = lr_image_y[i:i+self.patch_size, j:j+self.patch_size]
                    
                    # Normalize patches
                    hr_patch = hr_patch.astype(np.float32) / 255.0
                    lr_patch = lr_patch.astype(np.float32) / 255.0
                    
                    # Add channel dimension
                    hr_patch = hr_patch[np.newaxis, ...]
                    lr_patch = lr_patch[np.newaxis, ...]
                    
                    hr_patches.append(hr_patch)
                    lr_patches.append(lr_patch)
        
        logger.info(f"Generated {len(lr_patches)} LR-HR patch pairs.")
        return lr_patches, hr_patches
    
    def __len__(self):
        """
        Return the number of patches.
        
        Returns:
            int: Number of patches.
        """
        return len(self.lr_patches)
    
    def __getitem__(self, idx):
        """
        Get a patch pair.
        
        Args:
            idx (int): Index of the patch pair.
            
        Returns:
            tuple: LR patch and HR patch.
        """
        lr_patch = self.lr_patches[idx]
        hr_patch = self.hr_patches[idx]
        
        # Convert to torch tensors
        lr_patch = torch.from_numpy(lr_patch)
        hr_patch = torch.from_numpy(hr_patch)
        
        if self.transform:
            lr_patch = self.transform(lr_patch)
            hr_patch = self.transform(hr_patch)
        
        return lr_patch, hr_patch
    
    def get_full_images(self):
        """
        Get the full images for testing.
        
        Returns:
            tuple: LR images, HR images, and image names.
        """
        lr_images = []
        hr_images = []
        
        for hr_image in self.hr_images:
            # Convert PIL Image to numpy array
            hr_image_np = np.array(hr_image)
            
            # Convert to YCbCr color space and extract channels
            if len(hr_image_np.shape) == 3 and hr_image_np.shape[2] == 3:
                hr_image_ycrcb = cv2.cvtColor(hr_image_np, cv2.COLOR_RGB2YCrCb)
                hr_image_y = hr_image_ycrcb[:, :, 0]
                hr_image_cb = hr_image_ycrcb[:, :, 1]
                hr_image_cr = hr_image_ycrcb[:, :, 2]
            else:
                hr_image_y = hr_image_np
                hr_image_cb = None
                hr_image_cr = None
            
            # Generate LR image
            lr_image_y = cv2.resize(hr_image_y, None, fx=1.0/self.scale_factor, fy=1.0/self.scale_factor, interpolation=cv2.INTER_CUBIC)
            
            # Normalize images
            hr_image_y = hr_image_y.astype(np.float32) / 255.0
            lr_image_y = lr_image_y.astype(np.float32) / 255.0
            
            # Store images and color channels
            hr_images.append({
                'y': hr_image_y,
                'cb': hr_image_cb,
                'cr': hr_image_cr,
                'original': hr_image_np
            })
            
            lr_images.append(lr_image_y)
        
        return lr_images, hr_images, self.image_names

def prepare_data(batch_size=128, scale_factor=3, patch_size=33, stride=14):
    """
    Prepare the data for training and testing.
    
    Args:
        batch_size (int): Batch size for training.
        scale_factor (int): Scale factor for super-resolution.
        patch_size (int): Size of the patches to extract.
        stride (int): Stride for patch extraction.
        
    Returns:
        tuple: Training DataLoader and the dataset.
    """
    # Create the dataset
    dataset = Set5Dataset(scale_factor=scale_factor, patch_size=patch_size, stride=stride)
    
    # Create the DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return data_loader, dataset

def evaluate_model(model, dataset, device, output_dir):
    """
    Evaluate the model on the full images.
    
    Args:
        model (torch.nn.Module): The SRCNN model.
        dataset (Set5Dataset): The dataset.
        device (torch.device): The device to use.
        output_dir (str): Directory to save the output images.
        
    Returns:
        dict: Evaluation metrics.
    """
    model.eval()
    
    # Get the full images
    lr_images, hr_images, image_names = dataset.get_full_images()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Metrics
    metrics = {
        'psnr': [],
        'ssim': []
    }
    
    # Process each image
    sr_images = []
    for i, (lr_image, hr_image, name) in enumerate(zip(lr_images, hr_images, image_names)):
        # Upscale LR image to HR size
        lr_image_upscaled = cv2.resize(lr_image, (hr_image['y'].shape[1], hr_image['y'].shape[0]), interpolation=cv2.INTER_CUBIC)
        
        # Convert to tensor
        lr_tensor = torch.from_numpy(lr_image_upscaled).float().unsqueeze(0).unsqueeze(0).to(device)
        
        # Forward pass
        with torch.no_grad():
            sr_tensor = model(lr_tensor)
        
        # Convert to numpy
        sr_image = sr_tensor.squeeze().cpu().numpy()
        
        # Calculate metrics
        psnr_value = psnr(hr_image['y'], sr_image)
        ssim_value = ssim(hr_image['y'], sr_image, data_range=1.0)
        
        metrics['psnr'].append(psnr_value)
        metrics['ssim'].append(ssim_value)
        
        logger.info(f"Image {name}: PSNR = {psnr_value:.2f} dB, SSIM = {ssim_value:.4f}")
        
        # Convert SR image back to RGB
        if hr_image['cb'] is not None and hr_image['cr'] is not None:
            sr_image_y = (sr_image * 255.0).astype(np.uint8)
            sr_image_ycrcb = np.stack([sr_image_y, hr_image['cb'], hr_image['cr']], axis=-1)
            sr_image_rgb = cv2.cvtColor(sr_image_ycrcb, cv2.COLOR_YCrCb2RGB)
        else:
            sr_image_rgb = (sr_image * 255.0).astype(np.uint8)
        
        sr_images.append(sr_image_rgb)
        
        # Save individual images
        plt.figure(figsize=(15, 5))
        
        # Original HR image
        plt.subplot(1, 3, 1)
        plt.imshow(hr_image['original'])
        plt.title(f"Original HR - {name}")
        plt.axis('off')
        
        # Bicubic upscaled LR image
        plt.subplot(1, 3, 2)
        bicubic_rgb = cv2.resize(cv2.resize(hr_image['original'], None, fx=1.0/dataset.scale_factor, fy=1.0/dataset.scale_factor, interpolation=cv2.INTER_CUBIC), 
                                (hr_image['original'].shape[1], hr_image['original'].shape[0]), interpolation=cv2.INTER_CUBIC)
        plt.imshow(bicubic_rgb)
        plt.title(f"Bicubic Upscaled (x{dataset.scale_factor})")
        plt.axis('off')
        
        # SRCNN SR image
        plt.subplot(1, 3, 3)
        plt.imshow(sr_image_rgb)
        plt.title(f"SRCNN - PSNR: {psnr_value:.2f}dB")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"sr_{name}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Calculate average metrics
    avg_psnr = np.mean(metrics['psnr'])
    avg_ssim = np.mean(metrics['ssim'])
    
    logger.info(f"Average PSNR: {avg_psnr:.2f} dB, Average SSIM: {avg_ssim:.4f}")
    
    # Create comparison figure
    create_comparison_figure(hr_images, sr_images, image_names, dataset.scale_factor, os.path.join(output_dir, "super_resolution_compare.png"))
    
    # Create results figure
    create_results_figure(hr_images, sr_images, image_names, os.path.join(output_dir, "super_resolution_results.png"))
    
    return metrics, sr_images

def create_comparison_figure(hr_images, sr_images, image_names, scale_factor, output_path):
    """
    Create a comparison figure with zoomed-in details.
    
    Args:
        hr_images (list): List of HR images.
        sr_images (list): List of SR images.
        image_names (list): List of image names.
        scale_factor (int): Scale factor for super-resolution.
        output_path (str): Path to save the figure.
    """
    plt.figure(figsize=(20, 25))
    
    for i, (hr_image, sr_image, name) in enumerate(zip(hr_images, sr_images, image_names)):
        if i >= 5:  # Limit to 5 images
            break
        
        # Get original HR image
        hr_rgb = hr_image['original']
        
        # Generate bicubic upscaled image
        bicubic_rgb = cv2.resize(cv2.resize(hr_rgb, None, fx=1.0/scale_factor, fy=1.0/scale_factor, interpolation=cv2.INTER_CUBIC), 
                                (hr_rgb.shape[1], hr_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        # Select a region of interest for zooming
        h, w = hr_rgb.shape[:2]
        # Define different ROIs for each image to highlight interesting details
        if i == 0:  # baby
            roi_x, roi_y = int(w * 0.4), int(h * 0.4)
            roi_size = int(min(w, h) * 0.2)
        elif i == 1:  # bird
            roi_x, roi_y = int(w * 0.5), int(h * 0.4)
            roi_size = int(min(w, h) * 0.2)
        elif i == 2:  # butterfly
            roi_x, roi_y = int(w * 0.6), int(h * 0.5)
            roi_size = int(min(w, h) * 0.15)
        elif i == 3:  # head
            roi_x, roi_y = int(w * 0.5), int(h * 0.3)
            roi_size = int(min(w, h) * 0.2)
        else:  # woman
            roi_x, roi_y = int(w * 0.5), int(h * 0.3)
            roi_size = int(min(w, h) * 0.15)
        
        # Ensure ROI is within image bounds
        roi_x = max(0, min(roi_x, w - roi_size))
        roi_y = max(0, min(roi_y, h - roi_size))
        
        # Extract ROIs
        hr_roi = hr_rgb[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
        bicubic_roi = bicubic_rgb[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
        sr_roi = sr_image[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
        
        # Plot full images and ROIs
        plt.subplot(5, 3, i*3 + 1)
        plt.imshow(hr_rgb)
        plt.title(f"Original HR - {name}")
        plt.axis('off')
        plt.gca().add_patch(plt.Rectangle((roi_x, roi_y), roi_size, roi_size, edgecolor='red', facecolor='none', linewidth=2))
        
        plt.subplot(5, 3, i*3 + 2)
        plt.imshow(bicubic_rgb)
        plt.title(f"Bicubic Upscaled (x{scale_factor})")
        plt.axis('off')
        plt.gca().add_patch(plt.Rectangle((roi_x, roi_y), roi_size, roi_size, edgecolor='red', facecolor='none', linewidth=2))
        
        plt.subplot(5, 3, i*3 + 3)
        plt.imshow(sr_image)
        plt.title("SRCNN")
        plt.axis('off')
        plt.gca().add_patch(plt.Rectangle((roi_x, roi_y), roi_size, roi_size, edgecolor='red', facecolor='none', linewidth=2))
        
        # Add inset zoomed ROIs
        ax1 = plt.subplot(5, 3, i*3 + 1)
        axins1 = ax1.inset_axes([0.5, 0.5, 0.47, 0.47])
        axins1.imshow(hr_roi)
        axins1.set_title("Zoom", fontsize=8)
        axins1.set_xticks([])
        axins1.set_yticks([])
        
        ax2 = plt.subplot(5, 3, i*3 + 2)
        axins2 = ax2.inset_axes([0.5, 0.5, 0.47, 0.47])
        axins2.imshow(bicubic_roi)
        axins2.set_title("Zoom", fontsize=8)
        axins2.set_xticks([])
        axins2.set_yticks([])
        
        ax3 = plt.subplot(5, 3, i*3 + 3)
        axins3 = ax3.inset_axes([0.5, 0.5, 0.47, 0.47])
        axins3.imshow(sr_roi)
        axins3.set_title("Zoom", fontsize=8)
        axins3.set_xticks([])
        axins3.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Comparison figure saved to {output_path}")

def create_results_figure(hr_images, sr_images, image_names, output_path):
    """
    Create a figure showing the super-resolution results.
    
    Args:
        hr_images (list): List of HR images.
        sr_images (list): List of SR images.
        image_names (list): List of image names.
        output_path (str): Path to save the figure.
    """
    plt.figure(figsize=(15, 10))
    
    for i, (hr_image, sr_image, name) in enumerate(zip(hr_images, sr_images, image_names)):
        if i >= 5:  # Limit to 5 images
            break
        
        plt.subplot(2, 3, i+1)
        plt.imshow(sr_image)
        plt.title(f"SRCNN - {name}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Results figure saved to {output_path}")

if __name__ == "__main__":
    # Test the data loader
    data_loader, dataset = prepare_data(batch_size=32)
    
    # Print some information
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(data_loader)}")
    
    # Get a batch
    lr_batch, hr_batch = next(iter(data_loader))
    print(f"LR batch shape: {lr_batch.shape}")
    print(f"HR batch shape: {hr_batch.shape}")
    
    # Get the full images
    lr_images, hr_images, image_names = dataset.get_full_images()
    print(f"Number of full images: {len(lr_images)}")
    print(f"Image names: {image_names}")