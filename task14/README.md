# Telco Customer Churn Prediction

This project implements a machine learning system to predict customer churn using the Telco Customer Churn dataset. The system includes feature engineering, handling imbalanced data, model training with cross-validation, and comprehensive evaluation.

## Project Structure

```
.
├── data/                      # Data directory
│   └── churn.csv              # Telco Customer Churn dataset (downloaded)
├── models/                    # Directory for saved models
│   └── logistic_regression.pkl # Trained Logistic Regression model
├── results/                   # Results directory
│   ├── figures/               # Directory for figures
│   │   ├── roc_curve.png      # ROC curve
│   │   └── pr_curve.png       # Precision-Recall curve
│   └── metrics/               # Directory for metrics
│       └── classification_report.txt # Classification report
├── src/                       # Source code
│   ├── data_loader.py         # Data loading and preprocessing
│   ├── model.py               # Model implementation
│   └── train.py               # Training script
└── main.py                    # Main script to run the pipeline
```

## Features

- **Data Loading**: Downloads the Telco Customer Churn dataset from Hugging Face or loads it from a local file.
- **Feature Engineering**: Preprocesses the data, handles missing values, encodes categorical features, and selects the most relevant features.
- **Handling Imbalanced Data**: Implements oversampling (SMOTE) and undersampling techniques to handle class imbalance.
- **Model Training**: Trains a Logistic Regression model with cross-validation and hyperparameter tuning.
- **Model Evaluation**: Evaluates the model using various metrics, including accuracy, precision, recall, F1 score, and ROC AUC.
- **Visualization**: Generates ROC and Precision-Recall curves to visualize model performance.

## Requirements

- Python 3.6+
- NumPy
- Pandas
- Scikit-learn
- Imbalanced-learn (for SMOTE)
- Matplotlib
- Requests

## Usage

### Running the Pipeline

```bash
python main.py
```

### Command Line Arguments

- `--data-url`: URL to download the dataset from (default: Hugging Face URL)
- `--data-path`: Path to the dataset file (default: `data/churn.csv`)
- `--test-size`: Proportion of the dataset to include in the test split (default: 0.2)
- `--num-features`: Number of features to select (default: 15)
- `--handle-imbalance`: Method to handle imbalanced data (`smote`, `undersample`, or `none`) (default: `smote`)
- `--sampling-strategy`: Sampling strategy for handling imbalanced data (default: 1.0)
- `--model-type`: Type of model to train (default: `logistic`)
- `--cv`: Number of cross-validation folds (default: 5)
- `--class-weight`: Class weights for the model (`balanced` or `none`) (default: `balanced`)
- `--model-path`: Path to save the trained model (default: `models/logistic_regression.pkl`)
- `--metrics-path`: Path to save the classification report (default: `results/metrics/classification_report.txt`)
- `--roc-curve-path`: Path to save the ROC curve (default: `results/figures/roc_curve.png`)
- `--pr-curve-path`: Path to save the Precision-Recall curve (default: `results/figures/pr_curve.png`)
- `--random-state`: Random state for reproducibility (default: 42)
- `--verbose`: Verbosity level (default: 1)

### Example

```bash
python main.py --num-features 20 --handle-imbalance smote --cv 10
```

## Dataset

The Telco Customer Churn dataset contains information about customers of a telecommunications company, including their demographics, services they have signed up for, and whether they have churned (left the company). The dataset is available on Hugging Face at [scikit-learn/churn-prediction](https://huggingface.co/datasets/scikit-learn/churn-prediction).

## Model

The project uses a Logistic Regression model for predicting customer churn. The model is trained with cross-validation and hyperparameter tuning to find the best combination of parameters. The model is evaluated using various metrics, including accuracy, precision, recall, F1 score, and ROC AUC.

## Results

The results of the model training and evaluation are saved in the `results` directory. The classification report, which includes precision, recall, and F1 score, is saved to `results/metrics/classification_report.txt`. The ROC curve is saved to `results/figures/roc_curve.png`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.