# Image Super-Resolution using SRCNN

This project implements the Super-Resolution Convolutional Neural Network (SRCNN) for image super-resolution using the Set5 dataset.

## Project Structure

```
.
├── src/
│   ├── data_loader.py  # Data loading and preprocessing
│   └── model.py        # SRCNN model implementation
├── models/
│   └── saved_models/   # Directory for saved models
│       └── srcnn_model.pth  # Trained SRCNN model
├── results/
│   └── figures/        # Directory for figures
│       ├── super_resolution_compare.png  # Comparison of HR, bicubic, and SRCNN
│       └── super_resolution_results.png  # Super-resolution results
├── main.py             # Main script to train and test the model
├── test.py             # Script to test the model on a single image
└── README.md           # This file
```

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- NumPy
- OpenCV
- scikit-image
- matplotlib
- Hugging Face datasets

## Implementation Details

### Data Loading and Preprocessing

The `Set5Dataset` class in `src/data_loader.py` is responsible for loading the Set5 dataset from Hugging Face, preprocessing the images, and generating LR-HR patch pairs for training. The preprocessing steps include:

1. Converting images to YCbCr color space and using only the Y channel
2. Generating LR images by downscaling and upscaling with bicubic interpolation
3. Extracting patches from the images
4. Normalizing the patches

### SRCNN Model

The `SRCNN` class in `src/model.py` implements the Super-Resolution Convolutional Neural Network with the following architecture:

1. Feature extraction layer with 9x9 kernels
2. Non-linear mapping layer with 1x1 kernels
3. Reconstruction layer with 5x5 kernels

### Training and Evaluation

The `train_model` function in `src/model.py` trains the SRCNN model using the Adam optimizer and MSE loss. The `evaluate_model` function in `src/data_loader.py` evaluates the model on the full images and calculates PSNR and SSIM metrics.

## Usage

### Training and Testing

To train and test the SRCNN model, run:

```bash
python main.py [--batch_size BATCH_SIZE] [--scale_factor SCALE_FACTOR] [--patch_size PATCH_SIZE] [--stride STRIDE] [--num_epochs NUM_EPOCHS] [--learning_rate LEARNING_RATE] [--step_size STEP_SIZE] [--gamma GAMMA] [--no_cuda] [--test_only]
```

Arguments:
- `--batch_size`: Batch size for training. Default: 128
- `--scale_factor`: Scale factor for super-resolution. Default: 3
- `--patch_size`: Size of the patches to extract. Default: 33
- `--stride`: Stride for patch extraction. Default: 14
- `--num_epochs`: Number of epochs. Default: 100
- `--learning_rate`: Learning rate. Default: 1e-4
- `--step_size`: Step size for learning rate scheduler. Default: 30
- `--gamma`: Gamma for learning rate scheduler. Default: 0.1
- `--no_cuda`: Disable CUDA.
- `--test_only`: Only test the model.

### Testing on a Single Image

To test the model on a single image, run:

```bash
python test.py --image_path IMAGE_PATH [--model_path MODEL_PATH] [--scale_factor SCALE_FACTOR] [--output_path OUTPUT_PATH] [--no_cuda]
```

Arguments:
- `--image_path`: Path to the image.
- `--model_path`: Path to the model. Default: "models/saved_models/srcnn_model.pth"
- `--scale_factor`: Scale factor for super-resolution. Default: 3
- `--output_path`: Path to save the output image. Default: "results/figures/test_result.png"
- `--no_cuda`: Disable CUDA.

## Results

The evaluation results include:

- PSNR and SSIM metrics for each image
- Comparison of original HR, bicubic upscaled, and SRCNN super-resolution images
- Zoomed-in details to highlight the improvements

## References

- Dong, C., Loy, C. C., He, K., & Tang, X. (2014). Learning a deep convolutional network for image super-resolution. In European conference on computer vision (pp. 184-199). Springer, Cham.
- Set5 dataset: https://huggingface.co/datasets/eugenesiow/Set5

## License

This project is licensed under the MIT License - see the LICENSE file for details.