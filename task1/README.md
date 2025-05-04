# Fashion-MNIST Classification with ResNet-18

This project implements a system to classify images from the Fashion-MNIST dataset using the ResNet-18 model in PyTorch.

## Features

- Data loading and preprocessing with data augmentation
- ResNet-18 model adapted for Fashion-MNIST
- Training with progress visualization using tqdm
- Model interpretability with Grad-CAM
- TensorBoard integration for monitoring training
- Confusion matrix and per-class accuracy analysis

## Project Structure

```
.
├── main.py                     # Main script to run the system
├── src/
│   ├── data_loader.py          # Data loading and preprocessing
│   ├── model.py                # ResNet-18 model definition
│   ├── train.py                # Training and evaluation code
│   └── utils.py                # Utility functions for visualization and interpretability
├── models/
│   └── saved_models/           # Directory to save trained models
├── plots/                      # Directory for generated plots
└── logs/                       # TensorBoard logs
```

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- tqdm
- matplotlib
- numpy

## Usage

To train the model:

```bash
python main.py
```

Optional arguments:
- `--batch_size`: Batch size for training (default: 64)
- `--epochs`: Number of epochs to train for (default: 10)
- `--lr`: Learning rate (default: 0.001)
- `--no_pretrained`: Do not use pretrained weights
- `--data_dir`: Directory to store the dataset (default: ./data)
- `--save_dir`: Directory to save the model (default: ./models/saved_models)

## Model Interpretability

The system includes Grad-CAM implementation for model interpretability. Grad-CAM visualizations help understand which parts of the image the model focuses on when making predictions.

## Results

After training, the system generates:
- Training and validation loss/accuracy plots
- Confusion matrix
- Model prediction visualizations
- Grad-CAM visualizations for selected test images

The trained model is saved as `fashionnet.pt` in the `models/saved_models/` directory.