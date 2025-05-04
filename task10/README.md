# Face Recognition with FaceNet

This project implements a face recognition system using the FaceNet model with the LFW dataset. The system generates facial embeddings that can be used for face recognition tasks.

## Project Structure

```
.
├── data/                       # Data directory
│   └── synthetic_faces/        # Synthetic face dataset for testing
├── models/                     # Models directory
│   └── saved_models/           # Saved model weights
│       └── facenet.pt          # FaceNet model weights
├── results/                    # Results directory
│   ├── figures/                # Figures and visualizations
│   │   └── embedding_visualization.png  # t-SNE visualization of embeddings
│   └── metrics/                # Evaluation metrics
│       └── recognition_accuracy.txt     # Face recognition accuracy
├── src/                        # Source code
│   ├── data_loader.py          # Data loading and preprocessing
│   └── model.py                # FaceNet model implementation
└── main.py                     # Main script
```

## Features

- **Face Detection and Preprocessing**: Standardizes facial images for recognition.
- **FaceNet Model**: Generates facial embeddings for face recognition.
- **Face Recognition**: Identifies faces based on embedding similarity.
- **Visualization**: Visualizes facial embeddings using t-SNE.
- **Evaluation**: Measures recognition accuracy and other metrics.

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- scikit-learn
- matplotlib
- numpy
- Pillow
- OpenCV

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/face-recognition.git
cd face-recognition

# Install dependencies
pip install torch torchvision scikit-learn matplotlib numpy pillow opencv-python-headless
```

## Usage

### Training and Evaluation

```bash
# Train and evaluate the model with default parameters
python main.py --train

# Use a pretrained model for evaluation only
python main.py

# Train with custom parameters
python main.py --train --num-epochs 20 --batch-size 64 --embedding-dim 256
```

### Command-line Arguments

#### Data Parameters
- `--data-dir`: Directory containing face images (default: "data/synthetic_faces")
- `--batch-size`: Batch size for training (default: 32)
- `--test-split`: Proportion of data to use for testing (default: 0.2)
- `--min-images-per-person`: Minimum number of images required per person (default: 5)
- `--image-size`: Size to resize images to (default: 96)
- `--num-workers`: Number of worker threads for data loading (default: 4)
- `--random-seed`: Random seed for reproducibility (default: 42)

#### Model Parameters
- `--embedding-dim`: Dimension of the embedding vectors (default: 128)
- `--model-path`: Path to save/load model (default: "models/saved_models/facenet.pt")
- `--distance-threshold`: Threshold for face recognition (default: 0.6)

#### Training Parameters
- `--train`: Train the model
- `--num-epochs`: Number of training epochs (default: 10)
- `--learning-rate`: Learning rate for optimization (default: 0.001)
- `--margin`: Margin for triplet loss (default: 0.2)
- `--no-cuda`: Disable CUDA

#### Output Parameters
- `--metrics-path`: Path to save evaluation metrics (default: "results/metrics/recognition_accuracy.txt")
- `--visualization-path`: Path to save embeddings visualization (default: "results/figures/embedding_visualization.png")
- `--max-visualization-samples`: Maximum number of samples to visualize (default: 500)

## Implementation Details

### Data Loader

The `FaceDataset` class in `src/data_loader.py` handles loading and preprocessing face images. It supports:

- Loading images from a directory structure where each subdirectory represents a different person
- Applying transformations to standardize images
- Filtering identities with too few images

The `FacePreprocessor` class provides additional preprocessing capabilities:

- Face detection using OpenCV
- Face alignment and cropping
- Normalization for the FaceNet model

### FaceNet Model

The `FaceNetModel` class in `src/model.py` implements the FaceNet architecture:

- Convolutional layers for feature extraction
- Fully connected layers for embedding generation
- L2 normalization for embedding comparison
- Support for loading pretrained weights

### Face Recognition System

The `FaceRecognitionSystem` class provides a complete face recognition pipeline:

- Training with triplet loss
- Building an embeddings database for known identities
- Face recognition based on embedding similarity
- Evaluation of recognition accuracy
- Visualization of embeddings using t-SNE

## Results

After running the system, you can find:

1. Recognition accuracy and other metrics in `results/metrics/recognition_accuracy.txt`
2. Visualization of facial embeddings in `results/figures/embedding_visualization.png`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The FaceNet model is based on the paper "FaceNet: A Unified Embedding for Face Recognition and Clustering" by Schroff et al.
- The LFW dataset is provided by the University of Massachusetts, Amherst