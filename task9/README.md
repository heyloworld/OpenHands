# Movie Recommendation System

This project implements a movie recommendation system using Neural Collaborative Filtering (NCF) and Matrix Factorization (MF) approaches on the MovieLens dataset.

## Project Structure

```
.
├── data/                       # Data directory
│   └── ml-100k/                # MovieLens 100K dataset
├── models/                     # Trained models
│   ├── ncf_model.pt            # NCF model
│   └── mf_model.pt             # MF model
├── results/                    # Results directory
│   └── metrics/                # Metrics and recommendations
│       ├── evaluation_metrics.txt      # Evaluation metrics
│       └── top_10_recommendations.txt  # Top 10 recommendations
├── src/                        # Source code
│   ├── data_loader.py          # Data loading and preprocessing
│   └── model.py                # Model implementation
└── main.py                     # Main script
```

## Features

- **Data Loading**: Loads and preprocesses the MovieLens dataset.
- **Neural Collaborative Filtering**: Implements the NCF approach for movie recommendations.
- **Matrix Factorization**: Implements a baseline MF approach for comparison.
- **Evaluation**: Evaluates the models using RMSE, MAE, and other metrics.
- **Recommendations**: Generates personalized movie recommendations for users.

## Requirements

- Python 3.6+
- PyTorch
- NumPy
- Pandas
- scikit-learn

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system

# Install dependencies
pip install torch numpy pandas scikit-learn
```

## Usage

### Training and Evaluation

```bash
# Train and evaluate the models with default parameters
python main.py

# Train with custom parameters
python main.py --num-epochs 20 --batch-size 128 --embedding-dim 64
```

### Command-line Arguments

#### Data Parameters
- `--data-path`: Path to the MovieLens dataset (default: "data/ml-100k")
- `--test-size`: Proportion of data to use for testing (default: 0.2)
- `--random-state`: Random seed for reproducibility (default: 42)

#### Model Parameters
- `--embedding-dim`: Dimension of the embedding vectors (default: 32)
- `--layers`: List of layer dimensions for the MLP (default: [64, 32, 16, 8])
- `--dropout`: Dropout probability (default: 0.2)

#### Training Parameters
- `--num-epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size for training (default: 64)
- `--learning-rate`: Learning rate for optimization (default: 0.001)
- `--weight-decay`: Weight decay for regularization (default: 0.0)
- `--no-cuda`: Disable CUDA

#### Recommendation Parameters
- `--num-recommendations`: Number of recommendations to generate (default: 10)

#### Output Parameters
- `--metrics-path`: Path to save evaluation metrics (default: "results/metrics/evaluation_metrics.txt")
- `--recommendations-path`: Path to save recommendations (default: "results/metrics/top_10_recommendations.txt")
- `--save-models`: Save trained models
- `--ncf-model-path`: Path to save NCF model (default: "models/ncf_model.pt")
- `--mf-model-path`: Path to save MF model (default: "models/mf_model.pt")

## Implementation Details

### Data Loader

The `MovieLensDataLoader` class in `src/data_loader.py` handles loading and preprocessing the MovieLens dataset. It provides methods for:

- Loading the dataset from local files or downloading it if not available
- Splitting the data into training and testing sets
- Creating user and item mappings
- Generating PyTorch DataLoader objects
- Retrieving user and movie information

### Models

The `src/model.py` file implements two recommendation models:

1. **Matrix Factorization (MF)**: A baseline approach that learns latent factors for users and items through matrix factorization.

2. **Neural Collaborative Filtering (NCF)**: A more advanced approach that combines matrix factorization with neural networks for better recommendation performance.

Both models are implemented as PyTorch modules and provide methods for training, evaluation, and generating recommendations.

### Recommender System

The `RecommenderSystem` class in `src/model.py` provides a unified interface for training and evaluating recommendation models. It handles:

- Model creation and initialization
- Training and optimization
- Evaluation using metrics like RMSE and MAE
- Generating personalized recommendations
- Saving and loading models

## Results

After training, the system generates:

1. Evaluation metrics for both NCF and MF models, saved to `results/metrics/evaluation_metrics.txt`
2. Top 10 movie recommendations for a test user, saved to `results/metrics/top_10_recommendations.txt`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The MovieLens dataset is provided by GroupLens Research
- The NCF approach is based on the paper "Neural Collaborative Filtering" by He et al.